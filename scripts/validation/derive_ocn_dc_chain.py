"""Derive the OCN-f_dc method chain from cached ``ocn_product`` pickles.

The sweep's ``ocn_product`` baseline already runs ``run_all_bursts`` with
``use_ocn_dc=True`` and (by default) ``do_descallop=True``, so its result dict
carries:

  * ``v_r``           — radial velocity from the descalloped, OCN-f_dc anomaly
  * ``v_stokes``      — ERA5 Stokes drift on LOS
  * ``v_wave``        — Mouche-2012 wave Doppler bias on LOS
  * ``v_miss_ocn``    — OCN antenna-mispointing on LOS
  * ``v_current``     — −v_r − v_stokes − v_wave (no mispointing)
  * ``v_current_ocn`` — −v_r + v_miss − v_stokes − v_wave (with mispointing)

Four incremental method values per fix can be read off the same pickle (no
recompute, no GAMMA), and the last one is identical to the sweep's
``ocn_product`` row:

  * ``ocn_dc_geom``         = −v_r                                 (only OCN DC + descallop applied)
  * ``ocn_dc_stokes``       = −v_r − v_stokes                       (+ Stokes)
  * ``ocn_dc_mouche``       = −v_r − v_stokes − v_wave              (+ Mouche wave; = v_current)
  * ``ocn_dc_full``         = + v_miss_ocn                          (+ mispointing; = v_current_ocn = ocn_product)

The script reads ``data/drifters/method_sweep.csv`` for the drifter
coordinates / v_los_drift / fix indexing, then for each unique
(scene, subswath, burst) loads the matching ``..._ocn_product.pkl`` and
samples it at the nearest pixel of the drifter location.

Outputs
-------
  * Prints per-method N / bias / RMSE / r against ``v_los_drift``.
  * Writes a long-format CSV (default ``data/drifters/ocn_dc_chain.csv``)
    with the same schema as the sweep CSV so it can be concatenated or
    plotted with the same ``make_method_scatter.py``.

Usage
-----
    python scripts/validation/derive_ocn_dc_chain.py
    python scripts/validation/derive_ocn_dc_chain.py --append
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _ocn_pkl_path(cache_dir: Path, scene: str, subswath: str, burst: int) -> Path:
    return cache_dir / f"{scene}_{subswath}_burst{burst}_ocn_product.pkl"


def _nearest_pixel(lat_grid: np.ndarray, lon_grid: np.ndarray,
                   lat0: float, lon0: float) -> tuple[int, int]:
    coslat = np.cos(np.deg2rad(lat0))
    d2 = (lat_grid - lat0) ** 2 + ((lon_grid - lon0) * coslat) ** 2
    d2 = np.where(np.isfinite(d2), d2, np.inf)
    return np.unravel_index(int(np.argmin(d2)), d2.shape)


def derive_for_fix(r: dict, lat0: float, lon0: float) -> dict[str, float]:
    iy, ix = _nearest_pixel(r["lat"], r["lon"], lat0, lon0)
    v_r       = float(r["v_r"][iy, ix])         # already descalloped
    v_stokes  = float(r["v_stokes"][iy, ix])
    v_wave    = float(r["v_wave"][iy, ix])      # Mouche-2012
    v_miss    = float(r["v_miss_ocn"][iy, ix])
    return {
        "ocn_dc_geom":   -v_r,
        "ocn_dc_stokes": -v_r - v_stokes,
        "ocn_dc_mouche": -v_r - v_stokes - v_wave,
        "ocn_dc_full":   -v_r - v_stokes - v_wave + v_miss,
    }


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, sub in df.groupby("method"):
        s = sub.dropna(subset=["v_los_drift", "v_los_pipe"])
        if len(s) < 2:
            continue
        x = s["v_los_drift"].to_numpy(dtype=np.float64)
        y = s["v_los_pipe"].to_numpy(dtype=np.float64)
        diff = y - x
        r = (float(np.corrcoef(x, y)[0, 1])
             if x.std() > 0 and y.std() > 0 else float("nan"))
        rows.append({
            "method":   m,
            "N":        len(s),
            "bias":     float(diff.mean()),
            "RMSE":     float(np.sqrt((diff ** 2).mean())),
            "std_pipe": float(y.std()),
            "r":        r,
        })
    out = pd.DataFrame(rows).sort_values("method").reset_index(drop=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sweep-csv", type=Path,
                   default=Path("data/drifters/method_sweep.csv"),
                   help="Existing long-format sweep CSV (needed for fix coords + v_los_drift).")
    p.add_argument("--cache-dir", type=Path,
                   default=Path("data/drifters/method_sweep_cache"),
                   help="Directory holding the ocn_product pickles from the sweep.")
    p.add_argument("--out-csv", type=Path,
                   default=Path("data/drifters/ocn_dc_chain.csv"),
                   help="Where to write the derived chain rows.")
    p.add_argument("--append", action="store_true",
                   help="Append derived rows to the existing sweep CSV "
                        "(in addition to writing --out-csv).")
    args = p.parse_args()

    sweep = pd.read_csv(args.sweep_csv)
    # Distinct fixes: one row per (scene, platform_id, time, burst, lat, lon).
    # Pull v_los_drift, inc_deg, dist_km from any method row — they're the same.
    fix_cols = ["scene", "platform_id", "time", "subswath", "burst",
                "lat", "lon", "dist_km", "inc_deg", "v_los_drift"]
    fixes = sweep[fix_cols].drop_duplicates().reset_index(drop=True)
    print(f"Sweep CSV: {len(sweep)} rows  |  {len(fixes)} unique fixes")

    pkl_cache: dict[tuple, dict] = {}
    rows = []
    missing_pkl = 0
    for _, fx in fixes.iterrows():
        key = (fx["scene"], fx["subswath"], int(fx["burst"]))
        if key not in pkl_cache:
            path = _ocn_pkl_path(args.cache_dir, *key)
            if not path.exists():
                missing_pkl += 1
                continue
            with open(path, "rb") as f:
                pkl_cache[key] = pickle.load(f)
        r = pkl_cache[key]
        if r is None:
            continue
        derived = derive_for_fix(r, float(fx["lat"]), float(fx["lon"]))
        for method, v_pipe in derived.items():
            rows.append({
                "scene":       fx["scene"],
                "platform_id": int(fx["platform_id"]),
                "time":        fx["time"],
                "subswath":    fx["subswath"],
                "burst":       int(fx["burst"]),
                "lat":         float(fx["lat"]),
                "lon":         float(fx["lon"]),
                "dist_km":     float(fx["dist_km"]),
                "inc_deg":     float(fx["inc_deg"]),
                "v_los_drift": float(fx["v_los_drift"]),
                "method":      method,
                "v_los_pipe":  float(v_pipe),
                "residual":    float(v_pipe) - float(fx["v_los_drift"]),
            })
    if missing_pkl:
        print(f"  ({missing_pkl} fix(es) skipped — no ocn_product pickle in cache)")

    if not rows:
        print("No derivable rows.")
        return

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Saved {len(out)} rows  ->  {args.out_csv}")

    if args.append:
        combined = pd.concat([sweep, out], ignore_index=True)
        combined.to_csv(args.sweep_csv, index=False)
        print(f"Appended to {args.sweep_csv}  (now {len(combined)} rows)")

    # Print stats: derived chain alone, plus a side-by-side with ocn_product.
    print("\nOCN-f_dc chain (derived from cached ocn_product pickles):")
    stats = _summary(out)
    fmt = stats.copy()
    for c in ("bias", "std_pipe"):
        fmt[c] = fmt[c].map(lambda v: f"{v:+.3f}")
    fmt["RMSE"] = fmt["RMSE"].map(lambda v: f"{v:.3f}")
    fmt["r"]    = fmt["r"].map(lambda v: f"{v:+.3f}")
    print(fmt.to_string(index=False))

    # Sanity vs the existing ocn_product row in the sweep.
    if "ocn_product" in sweep["method"].unique():
        ocn = sweep[sweep["method"] == "ocn_product"][["v_los_drift", "v_los_pipe"]].dropna()
        diff = ocn["v_los_pipe"] - ocn["v_los_drift"]
        r = (float(np.corrcoef(ocn["v_los_drift"], ocn["v_los_pipe"])[0, 1])
             if ocn["v_los_drift"].std() > 0 and ocn["v_los_pipe"].std() > 0
             else float("nan"))
        print(f"\nSanity — sweep's ocn_product  "
              f"N={len(ocn)}  bias={diff.mean():+.3f}  "
              f"RMSE={np.sqrt((diff**2).mean()):.3f}  r={r:+.3f}")
        print("(should match the 'ocn_dc_full' row above to within rounding.)")


if __name__ == "__main__":
    main()
