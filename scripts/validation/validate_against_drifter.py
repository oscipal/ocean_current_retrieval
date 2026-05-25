"""Compare pipeline radial current at drifter locations to the drifter velocities.

Reads:
  * a pipeline .npz with arrays lat, lon, inc, look_az, v_current, v_current_ocn (and optionally v_model)
  * the drifter CSV produced by drifter_match.py

Projects each drifter (EWCT, NSCT) onto the S1 line of sight using the same
convention as scripts.sentinel_1.metocean.project_current_onto_look, then
reports the residual.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def project_uv_to_los(u: float, v: float, inc_deg: float, look_az_rad: float) -> float:
    inc_rad = np.deg2rad(inc_deg)
    return (u * np.sin(look_az_rad) + v * np.cos(look_az_rad)) * np.sin(inc_rad)


def nearest_pixel(lat_grid: np.ndarray, lon_grid: np.ndarray, lat0: float, lon0: float) -> tuple[int, int]:
    coslat = np.cos(np.deg2rad(lat0))
    d2 = ((lat_grid - lat0)) ** 2 + ((lon_grid - lon0) * coslat) ** 2
    finite = np.isfinite(d2)
    if not finite.any():
        raise ValueError("No finite pixels in grid")
    d2 = np.where(finite, d2, np.inf)
    return np.unravel_index(int(np.argmin(d2)), d2.shape)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=Path, required=True, help="Pipeline output .npz")
    p.add_argument("--drifter", type=Path, default=Path("data/drifters/scene1.csv"))
    p.add_argument("--max-dist-km", type=float, default=5.0,
                   help="Reject matches where the nearest pixel is farther than this")
    args = p.parse_args()

    d = np.load(args.npz, allow_pickle=False)
    lat = d["lat"]; lon = d["lon"]; inc = d["inc"]
    look_az = float(np.atleast_1d(d["look_az"]).ravel()[0])
    v_curr = d["v_current"] if "v_current" in d.files else None
    v_curr_ocn = d["v_current_ocn"] if "v_current_ocn" in d.files else None
    v_model = d["v_model"] if "v_model" in d.files else None
    print(f"Loaded grid {lat.shape}, look_az={np.rad2deg(look_az):.2f}°")

    raw = pd.read_csv(args.drifter)
    wide = raw.pivot_table(
        index=["platform_id", "time", "latitude", "longitude", "depth"],
        columns="variable", values="value").reset_index()
    wide["time"] = pd.to_datetime(wide["time"], utc=True).dt.tz_convert(None)
    wide = wide.dropna(subset=["EWCT", "NSCT"])
    print(f"Drifter fixes: {len(wide)}")

    rows = []
    for _, fix in wide.iterrows():
        try:
            iy, ix = nearest_pixel(lat, lon, fix["latitude"], fix["longitude"])
        except ValueError:
            continue
        plat_lat = float(lat[iy, ix]); plat_lon = float(lon[iy, ix])
        dist_km = 111.0 * np.hypot(
            plat_lat - fix["latitude"],
            (plat_lon - fix["longitude"]) * np.cos(np.deg2rad(fix["latitude"])),
        )
        if dist_km > args.max_dist_km:
            continue
        v_los_drifter = project_uv_to_los(
            float(fix["EWCT"]), float(fix["NSCT"]),
            float(inc[iy, ix]), look_az,
        )
        row = {
            "platform_id": int(fix["platform_id"]),
            "time": fix["time"],
            "lat": fix["latitude"], "lon": fix["longitude"],
            "u_drift": fix["EWCT"], "v_drift": fix["NSCT"],
            "inc_deg": float(inc[iy, ix]),
            "dist_km": dist_km,
            "v_los_drifter": v_los_drifter,
        }
        if v_curr is not None:
            row["v_los_s1"] = float(v_curr[iy, ix])
        if v_curr_ocn is not None:
            row["v_los_s1_ocn"] = float(v_curr_ocn[iy, ix])
        if v_model is not None:
            row["v_los_glo12"] = float(v_model[iy, ix])
        rows.append(row)

    if not rows:
        print("No drifter fix matched any in-grid pixel within tolerance.")
        return

    out = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda x: f"{x: .3f}")
    print()
    print(out.to_string(index=False))

    if "v_los_s1" in out.columns:
        diff = out["v_los_s1"] - out["v_los_drifter"]
        print(f"\nS1 (no mispointing) − drifter: bias = {diff.mean(): .3f} m/s, "
              f"RMSE = {np.sqrt((diff**2).mean()): .3f} m/s")
    if "v_los_s1_ocn" in out.columns:
        diff = out["v_los_s1_ocn"] - out["v_los_drifter"]
        print(f"S1 (OCN mispointing)  − drifter: bias = {diff.mean(): .3f} m/s, "
              f"RMSE = {np.sqrt((diff**2).mean()): .3f} m/s")


if __name__ == "__main__":
    main()
