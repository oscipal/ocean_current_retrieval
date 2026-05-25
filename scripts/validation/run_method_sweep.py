"""Drifter validation sweep across the full pipeline-method matrix.

For every scene under ``data/sentinel-1/scene*/``, every subswath containing
at least one drifter fix, and every method in the matrix, write one row per
drifter fix into a long-format CSV.  The matrix isolates the contribution of
each processing step:

  * Doppler estimation: mosaic-first (deramp → mosaic → doppler_2d_SLC) vs
    mosaic-last (per-burst doppler_2d_SLC → stitch).
  * With / without wind+wave Doppler bias removal (Mouche-2012 ERA5, or OCN
    owiRadVel).
  * With / without burst-periodic descalloping (rvl.descallop).
  * With / without OCN antenna-mispointing correction (rvlDcMiss).
  * Stokes-drift baseline (ERA5 wave) always available.

Plus two reference baselines:

  * ``ocn_product`` — our pipeline driven by OCN's measured Doppler
    (rvlDcObs as f_dc) with mispointing applied.  This is the operational
    Sentinel-1 OCN current.
  * ``glo12`` — Copernicus GLO12 model current projected onto the SAR LOS.

For each (scene, subswath) we run **four** GAMMA pipelines (mosaic ∈ {last,
first} × wave ∈ {mouche, ocn}) with ``descallop_blocks=True`` so the result
dict carries both the descalloped ``f_dca`` and the original
``f_dca_pre_descallop``.  Five processing-stage variants per (mosaic, wave)
pair are derived in memory from a single result without re-running the
pipeline.

For each (subswath, burst) hit by a drifter we additionally run one
OCN-product pipeline (``use_ocn_dc=True``) to produce the ``ocn_product``
baseline.

GAMMA prep outputs are persisted with ``--keep-products`` so re-runs reuse
them.  Heavy pipeline results are pickle-cached under
``data/drifters/method_sweep_cache/`` so re-runs only re-derive the rows.

Output
------
Long-format CSV (default ``data/drifters/method_sweep.csv``) with columns::

    scene, platform_id, time, subswath, burst, lat, lon, dist_km, inc_deg,
    v_los_drift, method, v_los_pipe, residual

Usage
-----
    python scripts/validation/run_method_sweep.py
    python scripts/validation/run_method_sweep.py --scene scene1 --no-cache
    python scripts/validation/run_method_sweep.py --results-csv my.csv
"""

from __future__ import annotations

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.sentinel_1.pipeline import (
    run_all_bursts,
    run_gamma_pipeline_from_safe,
)
from scripts.validation.run_drifter_validation import (
    DRIFTER_DATASET_AUTO,
    find_burst_for_point,
    nearest_pixel,
    project_uv_to_los,
    query_drifters,
    read_footprint_kml,
    read_overpass_time,
)
from scripts.validation.run_drifter_validation_all import (
    discover_scenes,
)


# ─── method matrix ──────────────────────────────────────────────────────────
# Stage keys, sorted so the CSV preserves the natural left-to-right reading of
# the pipeline.  Numeric prefix → sort; the rest is descriptive.
#
#   01 geom              -v_r                              (geom + sideband only)
#   02 geom_stokes       -v_r - v_stokes                    (+ Stokes)
#   03 stokes_<wave>      -v_r - v_stokes - v_wave           (+ wave bias)
#   04 stokes_<wave>_desc -v_r_desc - v_stokes - v_wave      (+ descallop)
#   05 full_<wave>        -v_r_desc - v_stokes - v_wave + v_miss   (+ mispointing)

MOSAIC_MODES = ("last", "first")
WAVE_SOURCES = ("mouche", "ocn")


def _result_at(r: dict, iy: int, ix: int) -> dict:
    """Pull the scalar values needed for method derivation from a GAMMA result.

    ``descallop_blocks=True`` is assumed so both descalloped (``f_dca``) and
    pre-descalloped (``f_dca_pre_descallop``) anomalies are present.
    """
    lam = float(r["wavelength_m"])
    f_pre = float(r["f_dca_pre_descallop"][iy, ix])
    f_des = float(r["f_dca"][iy, ix])
    return {
        "v_r_nodesc": lam / 2.0 * f_pre,
        "v_r_desc":   lam / 2.0 * f_des,
        "v_stokes":   float(r["v_stokes"][iy, ix]),
        "v_wave":     float(r["v_wave"][iy, ix]),
        "v_miss":     float(r["v_miss_ocn"][iy, ix]),
        "v_model":    float(r["v_model"][iy, ix]),
    }


def _derive_methods_for_mosaic(
    r_mouche: dict,
    r_ocn: dict,
    iy_m: int, ix_m: int,
    iy_o: int, ix_o: int,
    mosaic_tag: str,
) -> dict:
    """Return method_key -> v_los for one mosaic mode, from its (mouche, ocn) pair.

    Stages 01/02 are wave-independent, so we read them once from the mouche
    run.  Stages 03/04/05 are emitted per wave source.
    """
    M = _result_at(r_mouche, iy_m, ix_m)
    O = _result_at(r_ocn,    iy_o, ix_o)

    return {
        f"01_geom_{mosaic_tag}":              -M["v_r_nodesc"],
        f"02_geom_stokes_{mosaic_tag}":       -M["v_r_nodesc"] - M["v_stokes"],
        # + Mouche wave
        f"03_stokes_mouche_{mosaic_tag}":     -M["v_r_nodesc"] - M["v_stokes"] - M["v_wave"],
        f"04_stokes_mouche_desc_{mosaic_tag}":-M["v_r_desc"]   - M["v_stokes"] - M["v_wave"],
        f"05_full_mouche_{mosaic_tag}":       -M["v_r_desc"]   - M["v_stokes"] - M["v_wave"] + M["v_miss"],
        # + OCN owiRadVel wave
        f"03_stokes_ocn_{mosaic_tag}":        -O["v_r_nodesc"] - O["v_stokes"] - O["v_wave"],
        f"04_stokes_ocn_desc_{mosaic_tag}":   -O["v_r_desc"]   - O["v_stokes"] - O["v_wave"],
        f"05_full_ocn_{mosaic_tag}":          -O["v_r_desc"]   - O["v_stokes"] - O["v_wave"] + O["v_miss"],
    }


# ─── per-scene driver ───────────────────────────────────────────────────────

def _cache_key(scene: str, subswath: str, mosaic: str, wave: str) -> str:
    return f"{scene}_{subswath}_{mosaic}_{wave}"


def _ocn_cache_key(scene: str, subswath: str, burst_idx: int) -> str:
    return f"{scene}_{subswath}_burst{burst_idx}_ocn_product"


def _load_or_compute(
    cache_dir: Path, key: str, compute, use_cache: bool,
):
    path = cache_dir / f"{key}.pkl"
    if use_cache and path.exists():
        print(f"  [cache]    {path.name}")
        with open(path, "rb") as f:
            return pickle.load(f)
    print(f"  [compute]  {path.name}")
    obj = compute()
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return obj


def _run_gamma_one(
    scene: dict, subswath: str, mosaic: str, wave: str,
    pol: str, products_root: Path,
) -> dict:
    products_dir = products_root / f"gamma_{subswath.lower()}"
    return run_gamma_pipeline_from_safe(
        slc_safe=str(scene["safe"]), subswath=subswath,
        poeorb_path=str(scene["poeorb"]), aux_cal_path=str(scene["aux_cal"]),
        ocn_safe=str(scene["ocn_safe"]), era5_wind=str(scene["era5_wind"]),
        era5_wave=str(scene["era5_wave"]), glo12=str(scene["glo12"]),
        polarisation=pol,
        mosaic_mode=mosaic,
        wave_source=wave,
        descallop_blocks=True,
        keep_products=True,
        products_dir=str(products_dir),
    )


def _run_ocn_product(
    scene: dict, subswath: str, burst_idx: int, pol: str,
) -> dict | None:
    bursts = run_all_bursts(
        slc_safe=str(scene["safe"]), subswath=subswath,
        poeorb_path=str(scene["poeorb"]), aux_cal_path=str(scene["aux_cal"]),
        ocn_safe=str(scene["ocn_safe"]), era5_wind=str(scene["era5_wind"]),
        era5_wave=str(scene["era5_wave"]), glo12=str(scene["glo12"]),
        polarisation=pol, use_ocn_dc=True,
        burst_indices=[burst_idx],
    )
    return bursts[0] if bursts else None


def sweep_scene(
    scene: dict,
    drifter_csv: Path,
    cache_dir: Path,
    products_root: Path,
    pol: str = "vv",
    window_h: float = 6.0,
    max_dist_km: float = 10.0,
    max_dt_min: float = 30.0,
    drifter_dataset: str = DRIFTER_DATASET_AUTO,
    use_cache: bool = True,
) -> pd.DataFrame:
    label = scene["label"]
    safe = scene["safe"]
    fp_lon, fp_lat = read_footprint_kml(safe)
    t0 = read_overpass_time(safe)
    print(f"\n=== {label}  overpass {t0.isoformat()} ===")

    drifters = query_drifters(fp_lon, fp_lat, t0, window_h, drifter_csv, drifter_dataset)
    in_fp = drifters[drifters["in_footprint"]].copy()
    if not len(in_fp):
        print(f"[{label}] no drifters in footprint")
        return pd.DataFrame()
    in_fp["dt_min"] = (in_fp["time"] - pd.Timestamp(t0)).abs().dt.total_seconds() / 60.0
    in_fp = in_fp[in_fp["dt_min"] <= max_dt_min]
    in_fp = in_fp.sort_values(["platform_id", "time"]).reset_index(drop=True)
    if not len(in_fp):
        print(f"[{label}] no fixes within ±{max_dt_min:.0f} min of overpass")
        return pd.DataFrame()

    # Locate each fix's (subswath, burst).  Skip fixes that don't fall in any
    # burst polygon.
    located = []
    for _, fix in in_fp.iterrows():
        hit = find_burst_for_point(safe, fix["latitude"], fix["longitude"], pol)
        if hit is None:
            continue
        sw, bi = hit
        located.append((sw, bi, fix))
    if not located:
        print(f"[{label}] no drifter fix maps to a burst")
        return pd.DataFrame()

    subswaths = sorted({sw for sw, _, _ in located})
    print(f"[{label}] {len(located)} fix(es) across subswaths: {subswaths}")

    # Run the four GAMMA pipelines per relevant subswath (with caching).
    gamma_results: dict = {}  # (subswath, mosaic, wave) -> result dict
    for subswath in subswaths:
        for mosaic in MOSAIC_MODES:
            for wave in WAVE_SOURCES:
                key = _cache_key(label, subswath, mosaic, wave)
                gamma_results[(subswath, mosaic, wave)] = _load_or_compute(
                    cache_dir, key,
                    lambda sw=subswath, m=mosaic, w=wave: _run_gamma_one(
                        scene, sw, m, w, pol, products_root,
                    ),
                    use_cache,
                )

    # Run the OCN-product pipeline once per (subswath, burst) hit.
    ocn_product: dict = {}  # (subswath, burst) -> result dict
    bursts_hit = sorted({(sw, bi) for sw, bi, _ in located})
    for sw, bi in bursts_hit:
        key = _ocn_cache_key(label, sw, bi)
        ocn_product[(sw, bi)] = _load_or_compute(
            cache_dir, key,
            lambda sw=sw, bi=bi: _run_ocn_product(scene, sw, bi, pol),
            use_cache,
        )

    # Build rows
    rows = []
    for subswath, burst_idx, fix in located:
        lat0 = float(fix["latitude"])
        lon0 = float(fix["longitude"])

        # GAMMA results per mosaic — keyed by (mosaic, wave)
        # All four GAMMA grids share the same lat/lon (per mosaic mode) so the
        # (iy, ix) lookup is per mosaic.
        method_values: dict[str, float] = {}
        inc_for_row = float("nan")
        for mosaic in MOSAIC_MODES:
            r_m = gamma_results[(subswath, mosaic, "mouche")]
            r_o = gamma_results[(subswath, mosaic, "ocn")]
            iy_m, ix_m = nearest_pixel(r_m["lat"], r_m["lon"], lat0, lon0)
            iy_o, ix_o = nearest_pixel(r_o["lat"], r_o["lon"], lat0, lon0)
            # Distance check on the mouche grid (the two grids coincide).
            plat_lat = float(r_m["lat"][iy_m, ix_m])
            plat_lon = float(r_m["lon"][iy_m, ix_m])
            dist_km = 111.0 * np.hypot(
                plat_lat - lat0,
                (plat_lon - lon0) * np.cos(np.deg2rad(lat0)),
            )
            if dist_km > max_dist_km:
                continue
            if mosaic == "last":
                inc_for_row = float(r_m["inc"][iy_m, ix_m])
                look_az = float(r_m["look_az_rad"])
                v_los_drift = project_uv_to_los(
                    float(fix["EWCT"]), float(fix["NSCT"]),
                    inc_for_row, look_az,
                )
                dist_km_for_row = dist_km
                v_model_row = float(r_m["v_model"][iy_m, ix_m])
            mvals = _derive_methods_for_mosaic(
                r_m, r_o, iy_m, ix_m, iy_o, ix_o, mosaic,
            )
            method_values.update(mvals)

        if not method_values:
            continue

        # OCN-product baseline (one row).
        ocn_r = ocn_product.get((subswath, burst_idx))
        if ocn_r is not None:
            iy, ix = nearest_pixel(ocn_r["lat"], ocn_r["lon"], lat0, lon0)
            method_values["ocn_product"] = float(ocn_r["v_current_ocn"][iy, ix])
        else:
            method_values["ocn_product"] = float("nan")
        # GLO12 model (read from mosaic_last + mouche; same grid for all mosaic_last).
        method_values["glo12"] = v_model_row

        base = {
            "scene":       label,
            "platform_id": int(fix["platform_id"]),
            "time":        fix["time"],
            "subswath":    subswath,
            "burst":       burst_idx,
            "lat":         lat0,
            "lon":         lon0,
            "dist_km":     dist_km_for_row,
            "inc_deg":     inc_for_row,
            "v_los_drift": v_los_drift,
        }
        for method, v_pipe in method_values.items():
            rows.append({**base,
                         "method":      method,
                         "v_los_pipe":  v_pipe,
                         "residual":    v_pipe - v_los_drift})
    return pd.DataFrame(rows)


def print_method_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("\n[summary] no rows.")
        return
    g = df.groupby("method")
    stats = pd.DataFrame({
        "N":    g["residual"].count(),
        "bias": g["residual"].mean(),
        "RMSE": g["residual"].apply(lambda s: float(np.sqrt((s ** 2).mean()))),
        "r":    g.apply(lambda gg: float(np.corrcoef(gg["v_los_drift"], gg["v_los_pipe"])[0, 1])
                        if len(gg.dropna(subset=["v_los_drift", "v_los_pipe"])) > 1 else float("nan")),
    })
    # Order methods by natural sort key
    stats = stats.reindex(sorted(stats.index, key=_sort_key))
    fmt = stats.copy()
    fmt["bias"] = fmt["bias"].map(lambda v: f"{v:+.3f}")
    fmt["RMSE"] = fmt["RMSE"].map(lambda v: f"{v:.3f}")
    fmt["r"]    = fmt["r"].map(lambda v: f"{v:+.2f}")
    print("\nMethod summary (m/s):")
    print(fmt.to_string())


def _sort_key(method: str) -> tuple:
    # 01_..., 02_..., ..., then ocn_product, glo12 last
    if method == "ocn_product":
        return (98, method)
    if method == "glo12":
        return (99, method)
    try:
        return (int(method[:2]), method)
    except ValueError:
        return (50, method)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--pol", default="vv")
    p.add_argument("--window-h", type=float, default=6.0)
    p.add_argument("--max-dist-km", type=float, default=10.0)
    p.add_argument("--max-dt-min", type=float, default=30.0)
    p.add_argument("--drifter-dir", type=Path, default=Path("data/drifters"))
    p.add_argument("--drifter-dataset", default=DRIFTER_DATASET_AUTO)
    p.add_argument("--results-csv", type=Path,
                   default=Path("data/drifters/method_sweep.csv"))
    p.add_argument("--cache-dir", type=Path,
                   default=Path("data/drifters/method_sweep_cache"))
    p.add_argument("--no-cache", action="store_true",
                   help="Recompute pipelines even if a pickle exists.")
    p.add_argument("--scene", default=None,
                   help="Restrict to a single scene label (e.g. 'scene1'). "
                        "Default: sweep all discovered scenes.")
    args = p.parse_args()

    scenes = discover_scenes(args.data_root)
    if args.scene is not None:
        scenes = [s for s in scenes if s["label"] == args.scene]
    if not scenes:
        print("No scenes to sweep.")
        return

    print(f"Sweeping {len(scenes)} scene(s).")
    args.drifter_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    products_root = args.data_root / "sentinel-1"
    all_rows = []
    for s in scenes:
        missing = [k for k in ("aux_cal", "poeorb", "era5_wind", "era5_wave", "glo12")
                   if s[k] is None or not Path(s[k]).exists()]
        if missing:
            print(f"[{s['label']}] skip (missing: {','.join(missing)})")
            continue
        try:
            df = sweep_scene(
                scene=s,
                drifter_csv=args.drifter_dir / f"{s['label']}.csv",
                cache_dir=args.cache_dir,
                products_root=products_root,
                pol=args.pol, window_h=args.window_h,
                max_dist_km=args.max_dist_km, max_dt_min=args.max_dt_min,
                drifter_dataset=args.drifter_dataset,
                use_cache=not args.no_cache,
            )
        except Exception as e:
            print(f"[{s['label']}] FAILED: {e}")
            continue
        if not df.empty:
            all_rows.append(df)

    if not all_rows:
        print("\nNo rows from any scene.")
        return

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(args.results_csv, index=False)
    print(f"\nSaved {len(out)} rows to {args.results_csv}")
    print_method_summary(out)


if __name__ == "__main__":
    main()
