"""End-to-end CMEMS drifter validation of the S1 RVL pipeline.

Steps:
  1. Read scene footprint + overpass time from the SLC SAFE
  2. Query CMEMS NRT drifters in the footprint, ±window-h of overpass
  3. For each drifter fix inside the polygon, locate the (subswath, burst) that
     contains it using the annotation geolocation grid
  4. Run the pipeline for that burst, project (EWCT, NSCT) onto the line of
     sight using the same convention as metocean.project_current_onto_look, and
     report the residual vs v_current / v_current_ocn.

Single command, no intermediate files except the drifter CSV downloaded from
CMEMS.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import copernicusmarine as cm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.sentinel_1.pipeline import (
    run_pipeline,
    run_all_bursts,
    run_gamma_dop2d_pipeline,
    run_gamma_pipeline_from_safe,
)
from scripts.sentinel_1.safe_io import find_safe_files, parse_annotation
from scripts.sentinel_1.gamma_variants import gamma_doppler_mosaic_last
from scripts.sentinel_1.grid_merge import smooth_block_grid

DRIFTER_DATASET_AUTO = "auto"
DRIFTER_DATASET_NRT = "cmems_obs-ins_glo_phy-cur_nrt_drifter_irr"  # ~last 3 months
DRIFTER_DATASET_MY  = "cmems_obs-ins_glo_phy-cur_my_drifter_PT1H"  # 1993–present, hourly


def read_footprint_kml(safe_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    kml = safe_dir / "preview" / "map-overlay.kml"
    root = ET.parse(kml).getroot()
    coords_el = next(e for e in root.iter() if e.tag.endswith("coordinates"))
    pts = [p.split(",") for p in coords_el.text.strip().split()]
    lon = np.array([float(p[0]) for p in pts])
    lat = np.array([float(p[1]) for p in pts])
    return lon, lat


def read_overpass_time(safe_dir: Path) -> datetime:
    manifest = (safe_dir / "manifest.safe").read_text()
    m = re.search(r"startTime>([^<]+)<", manifest)
    if not m:
        raise ValueError(f"No startTime in {safe_dir}/manifest.safe")
    return datetime.fromisoformat(m.group(1))


def point_in_quad(lon: float, lat: float, qlon: np.ndarray, qlat: np.ndarray) -> bool:
    n = len(qlon)
    inside = False
    j = n - 1
    for i in range(n):
        if ((qlat[i] > lat) != (qlat[j] > lat)) and (
            lon < (qlon[j] - qlon[i]) * (lat - qlat[i]) / (qlat[j] - qlat[i] + 1e-12) + qlon[i]
        ):
            inside = not inside
        j = i
    return inside


def _empty_drifter_frame() -> pd.DataFrame:
    cols = ["platform_id", "time", "latitude", "longitude", "EWCT", "NSCT", "depth", "in_footprint"]
    return pd.DataFrame(columns=cols)


def _load_drifter_csv(csv_path: Path, fp_lon: np.ndarray, fp_lat: np.ndarray) -> pd.DataFrame:
    if not csv_path.exists():
        return _empty_drifter_frame()
    raw = pd.read_csv(csv_path)
    if raw.empty:
        return _empty_drifter_frame()
    wide = raw.pivot_table(
        index=["platform_id", "time", "latitude", "longitude", "depth"],
        columns="variable", values="value").reset_index()
    wide["time"] = pd.to_datetime(wide["time"], utc=True).dt.tz_convert(None)
    wide = wide.dropna(subset=["EWCT", "NSCT"])
    wide["in_footprint"] = [
        point_in_quad(lo, la, fp_lon, fp_lat)
        for lo, la in zip(wide["longitude"], wide["latitude"])
    ]
    return wide


def choose_drifter_datasets(t0: datetime, dataset_id: str) -> list[str]:
    if dataset_id != DRIFTER_DATASET_AUTO:
        return [dataset_id]
    age_days = (datetime.utcnow() - t0).total_seconds() / 86400.0
    if age_days <= 120:
        return [DRIFTER_DATASET_NRT, DRIFTER_DATASET_MY]
    return [DRIFTER_DATASET_MY, DRIFTER_DATASET_NRT]


def query_drifters(
    fp_lon: np.ndarray,
    fp_lat: np.ndarray,
    t0: datetime,
    window_h: float,
    out_csv: Path,
    dataset_id: str = DRIFTER_DATASET_MY,
) -> pd.DataFrame:
    pad = 0.1
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cached = _load_drifter_csv(out_csv, fp_lon, fp_lat)
    last_error = None
    for dataset in choose_drifter_datasets(t0, dataset_id):
        try:
            cm.subset(
                dataset_id=dataset,
                dataset_part="history",
                variables=["EWCT", "NSCT"],
                minimum_longitude=float(fp_lon.min() - pad),
                maximum_longitude=float(fp_lon.max() + pad),
                minimum_latitude=float(fp_lat.min() - pad),
                maximum_latitude=float(fp_lat.max() + pad),
                start_datetime=(t0 - timedelta(hours=window_h)).isoformat(),
                end_datetime=(t0 + timedelta(hours=window_h)).isoformat(),
                output_directory=str(out_csv.parent),
                output_filename=out_csv.stem,
                file_format="csv",
                overwrite=True,
            )
        except Exception as exc:
            last_error = exc
            print(f"[drifters] {dataset} query failed for {out_csv.stem}: {exc}")
            continue
        fresh = _load_drifter_csv(out_csv, fp_lon, fp_lat)
        if not fresh.empty:
            print(f"[drifters] {out_csv.stem}: using {dataset} ({len(fresh)} fixes)")
            return fresh
        print(f"[drifters] {out_csv.stem}: {dataset} returned no usable fixes")
    if not cached.empty:
        print(f"[drifters] {out_csv.stem}: falling back to cached CSV ({len(cached)} fixes)")
        return cached
    if last_error is not None:
        print(f"[drifters] {out_csv.stem}: no data after query failures")
    return _empty_drifter_frame()


def find_burst_for_point(
    slc_safe: Path, lat0: float, lon0: float, polarisation: str
) -> tuple[str, int] | None:
    """Return (subswath, burst_idx) whose footprint polygon contains the point.

    The geolocation grid is sparse but its first/last rows × first/last columns
    give the four ground corners of each burst.  An axis-aligned bbox check is
    too loose for tilted swaths (~25° from N) and can false-match adjacent
    subswaths; point-in-quad on the actual corners is exact.
    """
    for subswath in ("iw1", "iw2", "iw3"):
        files = find_safe_files(str(slc_safe), subswath, polarisation)
        annot = parse_annotation(files["annotation"])
        gg = annot.geoloc_grid
        lines = np.asarray(gg["line"])               # (n_lines,)
        lat = np.asarray(gg["latitude"])             # (n_lines, n_pixels)
        lon = np.asarray(gg["longitude"])
        lpb = annot.lines_per_burst
        for burst_idx in range(len(annot.bursts)):
            row_mask = (lines >= burst_idx * lpb) & (lines <= (burst_idx + 1) * lpb)
            if not row_mask.any():
                continue
            sub_lat = lat[row_mask]; sub_lon = lon[row_mask]
            qlat = np.array([sub_lat[0, 0], sub_lat[0, -1], sub_lat[-1, -1], sub_lat[-1, 0]])
            qlon = np.array([sub_lon[0, 0], sub_lon[0, -1], sub_lon[-1, -1], sub_lon[-1, 0]])
            if point_in_quad(lon0, lat0, qlon, qlat):
                return subswath, burst_idx
    return None


def project_uv_to_los(u: float, v: float, inc_deg: float, look_az_rad: float) -> float:
    return (u * np.sin(look_az_rad) + v * np.cos(look_az_rad)) * np.sin(np.deg2rad(inc_deg))


def find_gamma_products(
    t0: datetime,
    subswath: str,
    pol: str,
    gamma_root: Path = Path("data/sentinel-1"),
) -> tuple[Path, str] | None:
    """Return ``(gamma_dir, base_id)`` for a scene's GAMMA-deramped products.

    Looks for ``<gamma_root>/gamma_<subswath>/<YYYYMMDD>_<subswath>_<pol>.deramp.slc``
    where ``YYYYMMDD`` is the scene's overpass date.  Returns None if the file
    is missing (the GAMMA shell script has not been run for this scene/subswath).
    """
    gd = gamma_root / f"gamma_{subswath.lower()}"
    bid = f"{t0.strftime('%Y%m%d')}_{subswath.lower()}_{pol.lower()}"
    return (gd, bid) if (gd / f"{bid}.deramp.slc").exists() else None


def nearest_pixel(lat_grid: np.ndarray, lon_grid: np.ndarray, lat0: float, lon0: float) -> tuple[int, int]:
    coslat = np.cos(np.deg2rad(lat0))
    d2 = (lat_grid - lat0) ** 2 + ((lon_grid - lon0) * coslat) ** 2
    d2 = np.where(np.isfinite(d2), d2, np.inf)
    return np.unravel_index(int(np.argmin(d2)), d2.shape)


def validate_scene(
    safe: Path,
    poeorb: Path,
    aux_cal: Path,
    ocn_safe: Path,
    era5_wind: Path,
    era5_wave: Path,
    glo12: Path,
    drifter_csv: Path,
    pol: str = "vv",
    window_h: float = 6.0,
    max_dist_km: float = 10.0,
    max_dt_min: float = 30.0,
    scene_label: str | None = None,
    drifter_dataset: str = DRIFTER_DATASET_MY,
    engine: str = "ours",
    gamma_blsz: int = 256,
    gamma_add_demod_back: "str | bool" = "blend",
    gamma_geom_source: str = "gamma",
    gamma_wave_source: str = "mouche",
    gamma_descallop_blocks: bool = False,
    gamma_keep_products: bool = False,
    smooth_az: int = 1,
    smooth_rg: int = 1,
) -> pd.DataFrame:
    fp_lon, fp_lat = read_footprint_kml(safe)
    t0 = read_overpass_time(safe)
    label = scene_label or safe.parent.name
    print(f"[{label}] overpass {t0.isoformat()}  "
          f"footprint lon[{fp_lon.min():.2f},{fp_lon.max():.2f}] "
          f"lat[{fp_lat.min():.2f},{fp_lat.max():.2f}]")

    drifters = query_drifters(fp_lon, fp_lat, t0, window_h, drifter_csv, drifter_dataset)
    in_fp = drifters[drifters["in_footprint"]].copy()
    print(f"[{label}] drifter fixes: total={len(drifters)} in_footprint={len(in_fp)} "
          f"platforms={in_fp['platform_id'].nunique() if len(in_fp) else 0}")
    if not len(in_fp):
        return pd.DataFrame()

    in_fp["dt_min"] = (in_fp["time"] - pd.Timestamp(t0)).abs().dt.total_seconds() / 60.0
    in_fp = in_fp[in_fp["dt_min"] <= max_dt_min].copy()
    in_fp = in_fp.sort_values(["platform_id", "time"]).reset_index(drop=True)
    if not len(in_fp):
        print(f"[{label}] no fixes within ±{max_dt_min:.0f} min of overpass")
        return pd.DataFrame()

    if engine not in {"ours", "gamma"}:
        raise ValueError(f"engine must be 'ours' or 'gamma'; got {engine!r}")

    cache: dict = {}
    ocn_cache: dict = {}   # (subswath, burst_idx) -> burst dict from run_all_bursts(use_ocn_dc=True)
    rows = []
    for _, fix in in_fp.iterrows():
        hit = find_burst_for_point(safe, fix["latitude"], fix["longitude"], pol)
        if hit is None:
            continue
        subswath, burst_idx = hit
        # Cache key: per (subswath, burst) for our pipeline, per subswath for GAMMA
        # (one GAMMA mosaic-last grid covers the entire subswath).
        key = (subswath, burst_idx) if engine == "ours" else subswath
        if key not in cache:
            if engine == "ours":
                print(f"[{label}] run pipeline on {subswath} burst {burst_idx}")
                r = run_pipeline(
                    slc_safe=str(safe), subswath=subswath, burst_idx=burst_idx,
                    poeorb_path=str(poeorb), aux_cal_path=str(aux_cal),
                    ocn_safe=str(ocn_safe), era5_wind=str(era5_wind),
                    era5_wave=str(era5_wave), glo12=str(glo12),
                    polarisation=pol,
                )
            else:  # gamma
                gp = find_gamma_products(t0, subswath, pol)
                if gp is not None:
                    gamma_dir, base_id = str(gp[0]), gp[1]
                    print(f"[{label}] run GAMMA mosaic-last on {subswath} "
                          f"(cached deramp at {gamma_dir}/{base_id}.*, "
                          f"blsz={gamma_blsz}, demod_back={gamma_add_demod_back!r}, "
                          f"geom={gamma_geom_source!r})")
                else:
                    gamma_dir, base_id = None, None
                    print(f"[{label}] run GAMMA mosaic-last on {subswath} "
                          f"(in-memory prep from SAFE, blsz={gamma_blsz}, "
                          f"demod_back={gamma_add_demod_back!r}, "
                          f"geom={gamma_geom_source!r})")
                r = run_gamma_pipeline_from_safe(
                    slc_safe=str(safe), subswath=subswath,
                    poeorb_path=str(poeorb), aux_cal_path=str(aux_cal),
                    ocn_safe=str(ocn_safe), era5_wind=str(era5_wind),
                    era5_wave=str(era5_wave), glo12=str(glo12),
                    polarisation=pol,
                    blsz=gamma_blsz, add_demod_back=gamma_add_demod_back,
                    geom_source=gamma_geom_source,
                    wave_source=gamma_wave_source,
                    descallop_blocks=gamma_descallop_blocks,
                    gamma_dir=gamma_dir, base_id=base_id,
                    keep_products=gamma_keep_products,
                )
            if smooth_az > 1 or smooth_rg > 1:
                print(f"[{label}] block-grid smooth az={smooth_az} rg={smooth_rg}")
                r = smooth_block_grid(r, smooth_az=smooth_az, smooth_rg=smooth_rg)
            cache[key] = r
        r = cache[key]
        if r is None:
            continue
        iy, ix = nearest_pixel(r["lat"], r["lon"], float(fix["latitude"]), float(fix["longitude"]))
        plat_lat = float(r["lat"][iy, ix]); plat_lon = float(r["lon"][iy, ix])
        dist_km = 111.0 * np.hypot(
            plat_lat - fix["latitude"],
            (plat_lon - fix["longitude"]) * np.cos(np.deg2rad(fix["latitude"])),
        )
        if dist_km > max_dist_km:
            continue
        look_az = float(r["look_az_rad"])
        inc_deg = float(r["inc"][iy, ix])
        v_los_drift = project_uv_to_los(float(fix["EWCT"]), float(fix["NSCT"]), inc_deg, look_az)
        v_los_s1     = float(r["v_current"][iy, ix])
        v_los_s1_ocn = float(r["v_current_ocn"][iy, ix])
        v_los_glo12  = float(r["v_model"][iy, ix])

        # OCN-product current: our pipeline driven by OCN's observed Doppler
        # (rvlDcObs), sampled on the matching burst's lat/lon grid.  Same
        # quantity the poster's middle panel shows.
        ocn_key = (subswath, burst_idx)
        if ocn_key not in ocn_cache:
            try:
                print(f"[{label}] OCN-product pipeline on {subswath} burst {burst_idx}")
                ocn_bursts = run_all_bursts(
                    slc_safe=str(safe), subswath=subswath,
                    poeorb_path=str(poeorb), aux_cal_path=str(aux_cal),
                    ocn_safe=str(ocn_safe), era5_wind=str(era5_wind),
                    era5_wave=str(era5_wave), glo12=str(glo12),
                    polarisation=pol, use_ocn_dc=True,
                    burst_indices=[burst_idx],
                )
                ocn_cache[ocn_key] = ocn_bursts[0] if ocn_bursts else None
            except Exception as e:
                print(f"[{label}] OCN-product pipeline failed on {subswath} "
                      f"burst {burst_idx}: {e}")
                ocn_cache[ocn_key] = None
        ocn_r = ocn_cache[ocn_key]
        if ocn_r is None:
            v_los_ocn_product = float("nan")
        else:
            iy_o, ix_o = nearest_pixel(
                ocn_r["lat"], ocn_r["lon"],
                float(fix["latitude"]), float(fix["longitude"]),
            )
            v_los_ocn_product = float(ocn_r["v_current_ocn"][iy_o, ix_o])

        rows.append({
            "scene":              label,
            "platform_id":        int(fix["platform_id"]),
            "time":               fix["time"],
            "dt_min":             float(fix["dt_min"]),
            "subswath":           subswath,
            "burst":              burst_idx,
            "lat":                float(fix["latitude"]),
            "lon":                float(fix["longitude"]),
            "dist_km":            dist_km,
            "inc_deg":            inc_deg,
            "u_drift":            float(fix["EWCT"]),
            "v_drift":            float(fix["NSCT"]),
            "v_los_drift":        v_los_drift,
            "v_los_s1":           v_los_s1,
            "v_los_s1_ocn":       v_los_s1_ocn,
            "v_los_ocn_product":  v_los_ocn_product,
            "v_los_glo12":        v_los_glo12,
            "residual_s1":          v_los_s1            - v_los_drift,
            "residual_s1_ocn":      v_los_s1_ocn        - v_los_drift,
            "residual_ocn_product": v_los_ocn_product   - v_los_drift,
            "residual_glo12":       v_los_glo12         - v_los_drift,
        })
    return pd.DataFrame(rows)


def print_summary(out: pd.DataFrame) -> None:
    if out.empty:
        print("No usable matches.")
        return
    print()
    has_ocn_product = "v_los_ocn_product" in out.columns
    cols = [
        "scene", "platform_id", "time", "dt_min", "subswath", "burst",
        "dist_km", "v_los_drift", "v_los_s1", "v_los_s1_ocn",
    ]
    if has_ocn_product:
        cols += ["v_los_ocn_product"]
    cols += ["v_los_glo12", "residual_s1", "residual_s1_ocn"]
    if has_ocn_product:
        cols += ["residual_ocn_product"]
    cols += ["residual_glo12"]

    table = out.loc[:, cols].copy()
    table["time"] = pd.to_datetime(table["time"]).dt.strftime("%Y-%m-%d %H:%M")
    float_cols = ["dt_min", "dist_km", "v_los_drift", "v_los_s1", "v_los_s1_ocn",
                  "v_los_glo12", "residual_s1", "residual_s1_ocn", "residual_glo12"]
    if has_ocn_product:
        float_cols += ["v_los_ocn_product", "residual_ocn_product"]
    for col in float_cols:
        table[col] = table[col].map(lambda v: f"{v:.3f}")
    print("Match table:")
    print(table.to_string(index=False))
    print()

    agg = dict(
        matches=("platform_id", "size"),
        platforms=("platform_id", "nunique"),
        bias_s1=("residual_s1", "mean"),
        rmse_s1=("residual_s1", lambda s: float(np.sqrt((s**2).mean()))),
        bias_s1_ocn=("residual_s1_ocn", "mean"),
        rmse_s1_ocn=("residual_s1_ocn", lambda s: float(np.sqrt((s**2).mean()))),
        bias_glo12=("residual_glo12", "mean"),
        rmse_glo12=("residual_glo12", lambda s: float(np.sqrt((s**2).mean()))),
    )
    if has_ocn_product:
        agg["bias_ocn_product"] = ("residual_ocn_product", "mean")
        agg["rmse_ocn_product"] = ("residual_ocn_product",
                                   lambda s: float(np.sqrt((s**2).mean())))
    scene_summary = (
        out.groupby("scene", as_index=False)
        .agg(**agg)
        .sort_values("scene")
    )
    fmt_cols = ["bias_s1", "rmse_s1", "bias_s1_ocn", "rmse_s1_ocn",
                "bias_glo12", "rmse_glo12"]
    if has_ocn_product:
        fmt_cols += ["bias_ocn_product", "rmse_ocn_product"]
    for col in fmt_cols:
        scene_summary[col] = scene_summary[col].map(lambda v: f"{v:.3f}")
    print("Scene summary:")
    print(scene_summary.to_string(index=False))
    print()

    summary_cols = ["v_los_s1", "v_los_s1_ocn", "v_los_glo12"]
    if has_ocn_product:
        summary_cols.insert(2, "v_los_ocn_product")
    for col in summary_cols:
        diff = out[col] - out["v_los_drift"]
        diff = diff.dropna()
        print(f"{col:18s} − drifter:  N={len(diff):3d}  bias={diff.mean(): .3f}  "
              f"RMSE={np.sqrt((diff**2).mean()): .3f}  m/s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--safe", type=Path, required=True)
    p.add_argument("--poeorb", type=Path, required=True)
    p.add_argument("--aux-cal", type=Path, required=True)
    p.add_argument("--ocn-safe", type=Path, required=True)
    p.add_argument("--era5-wind", type=Path, required=True)
    p.add_argument("--era5-wave", type=Path, required=True)
    p.add_argument("--glo12", type=Path, required=True)
    p.add_argument("--pol", default="vv")
    p.add_argument("--window-h", type=float, default=6.0,
                   help="Hours around overpass to query CMEMS (broader than match filter)")
    p.add_argument("--drifter-csv", type=Path, default=Path("data/drifters/scene1.csv"))
    p.add_argument("--max-dist-km", type=float, default=10.0,
                   help="Reject matches where nearest pixel is farther than this")
    p.add_argument("--max-dt-min", type=float, default=30.0,
                   help="Reject fixes farther than this in time from overpass")
    p.add_argument("--drifter-dataset", default=DRIFTER_DATASET_AUTO,
                   help=f"CMEMS dataset id (default: {DRIFTER_DATASET_AUTO}). "
                        f"Auto tries {DRIFTER_DATASET_NRT} then {DRIFTER_DATASET_MY} for recent scenes, "
                        f"and the reverse order for older scenes.")
    p.add_argument("--engine", default="ours", choices=("ours", "gamma"),
                   help="Pipeline producing the SAR current: 'ours' (per-burst "
                        "run_pipeline; default) or 'gamma' (GAMMA mosaic-last "
                        "+ doppler_2d_SLC; runs GAMMA prep in-memory from the "
                        "SAFE if cached products are absent).")
    p.add_argument("--gamma-blsz", type=int, default=256,
                   help="GAMMA doppler_2d_SLC azimuth block size (engine=gamma).")
    p.add_argument("--gamma-demod-back", default="blend",
                   choices=("none", "step", "blend", "hanning"),
                   help="add_demod_back mode for gamma_doppler_mosaic_last "
                        "(engine=gamma; default 'blend').")
    p.add_argument("--gamma-geom-source", default="gamma",
                   choices=("gamma", "annotation", "poeorb"),
                   help="Geometry-Doppler source for engine=gamma (default 'gamma').")
    p.add_argument("--gamma-wave-source", default="mouche",
                   choices=("mouche", "ocn"),
                   help="Wave-Doppler bias source: 'mouche' (default; ERA5 wind + "
                        "Mouche-2012 simplification) or 'ocn' (OCN owiRadVel — "
                        "the operationally calibrated wind-wave Doppler velocity).")
    p.add_argument("--gamma-descallop", action="store_true",
                   help="Apply our azimuth-periodic descalloping (rvl.descallop) "
                        "to the GAMMA mosaic-last f_dca grid before deriving v_r.")
    p.add_argument("--gamma-keep-products", action="store_true",
                   help="Persist GAMMA prep outputs (.slc/.par/.tops_par) after "
                        "the run instead of using a tempdir.  Speeds up re-runs.")
    p.add_argument("--smooth-az", type=int, default=1,
                   help="Boxcar-smooth the result's data fields by this many "
                        "azimuth-block rows before lookup (default 1 = no smoothing). "
                        "Use to suppress burst-boundary stepping without attenuating "
                        "per-burst amplitudes (good companion to "
                        "--gamma-demod-back step/hanning).")
    p.add_argument("--smooth-rg", type=int, default=1,
                   help="Boxcar-smooth in range-block columns (default 1 = none).")
    args = p.parse_args()

    out = validate_scene(
        safe=args.safe, poeorb=args.poeorb, aux_cal=args.aux_cal,
        ocn_safe=args.ocn_safe, era5_wind=args.era5_wind, era5_wave=args.era5_wave,
        glo12=args.glo12, drifter_csv=args.drifter_csv,
        pol=args.pol, window_h=args.window_h,
        max_dist_km=args.max_dist_km, max_dt_min=args.max_dt_min,
        drifter_dataset=args.drifter_dataset,
        engine=args.engine,
        gamma_blsz=args.gamma_blsz,
        gamma_add_demod_back=args.gamma_demod_back,
        gamma_geom_source=args.gamma_geom_source,
        gamma_wave_source=args.gamma_wave_source,
        gamma_descallop_blocks=args.gamma_descallop,
        gamma_keep_products=args.gamma_keep_products,
        smooth_az=args.smooth_az,
        smooth_rg=args.smooth_rg,
    )
    print_summary(out)


if __name__ == "__main__":
    main()
