"""Pull CMEMS NRT drifter fixes inside a Sentinel-1 scene footprint and time window.

First-pass validation step: just report what's available before wiring it to the RVL pipeline.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import copernicusmarine as cm

DATASET_ID = "cmems_obs-ins_glo_phy-cur_nrt_drifter_irr"


def read_footprint_kml(safe_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    kml = safe_dir / "preview" / "map-overlay.kml"
    root = ET.parse(kml).getroot()
    coords_el = root.find(".//{http://www.google.com/kml/ext/2.2}LatLonQuad/"
                          "{http://www.opengis.net/kml/2.2}coordinates")
    if coords_el is None:  # namespace fallback
        coords_el = next(e for e in root.iter() if e.tag.endswith("coordinates"))
    pts = [p.split(",") for p in coords_el.text.strip().split()]
    lon = np.array([float(p[0]) for p in pts])
    lat = np.array([float(p[1]) for p in pts])
    return lon, lat


def read_overpass_time(safe_dir: Path) -> datetime:
    import re
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


def query_drifters(
    fp_lon: np.ndarray,
    fp_lat: np.ndarray,
    t0: datetime,
    window_h: float,
    out_csv: Path,
) -> pd.DataFrame:
    pad = 0.1
    bbox = dict(
        minimum_longitude=float(fp_lon.min() - pad),
        maximum_longitude=float(fp_lon.max() + pad),
        minimum_latitude=float(fp_lat.min() - pad),
        maximum_latitude=float(fp_lat.max() + pad),
    )
    tmin = (t0 - timedelta(hours=window_h)).isoformat()
    tmax = (t0 + timedelta(hours=window_h)).isoformat()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cm.subset(
        dataset_id=DATASET_ID,
        dataset_part="history",
        variables=["EWCT", "NSCT"],
        start_datetime=tmin,
        end_datetime=tmax,
        output_directory=str(out_csv.parent),
        output_filename=out_csv.stem,
        file_format="csv",
        overwrite=True,
        **bbox,
    )

    raw = pd.read_csv(out_csv)
    if raw.empty:
        return pd.DataFrame(columns=["time", "latitude", "longitude", "EWCT", "NSCT", "platform_id"])

    wide = raw.pivot_table(
        index=["platform_id", "time", "latitude", "longitude", "depth"],
        columns="variable",
        values="value",
    ).reset_index()
    wide["time"] = pd.to_datetime(wide["time"], utc=True).dt.tz_convert(None)
    wide = wide.dropna(subset=["EWCT", "NSCT"])
    wide["in_footprint"] = [
        point_in_quad(lo, la, fp_lon, fp_lat)
        for lo, la in zip(wide["longitude"], wide["latitude"])
    ]
    return wide


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--safe", type=Path, required=True, help="Path to S1 SLC .SAFE directory")
    p.add_argument("--window-h", type=float, default=6.0, help="±hours around overpass to query")
    p.add_argument("--out", type=Path, default=Path("data/drifters/scene1.csv"))
    args = p.parse_args()

    fp_lon, fp_lat = read_footprint_kml(args.safe)
    t0 = read_overpass_time(args.safe)

    print(f"Scene overpass: {t0.isoformat()} UTC")
    print(f"Footprint bbox: lon [{fp_lon.min():.3f}, {fp_lon.max():.3f}] "
          f"lat [{fp_lat.min():.3f}, {fp_lat.max():.3f}]")
    print(f"Query window:   ±{args.window_h:.1f} h\n")

    df = query_drifters(fp_lon, fp_lat, t0, args.window_h, args.out)

    print(f"Total fixes in bbox+time: {len(df)}")
    in_fp = df[df["in_footprint"]].copy()
    print(f"Fixes inside scene polygon: {len(in_fp)}")
    if "platform_id" in in_fp.columns:
        print(f"Unique drifters in scene:   {in_fp['platform_id'].nunique()}")
    if len(in_fp):
        print("\nNearest-to-overpass fix per drifter:")
        in_fp["dt_min"] = (in_fp["time"] - pd.Timestamp(t0)).abs().dt.total_seconds() / 60.0
        nearest = in_fp.sort_values("dt_min").groupby("platform_id", as_index=False).first()
        print(nearest[["platform_id", "time", "latitude", "longitude",
                       "EWCT", "NSCT", "depth", "dt_min"]].to_string(index=False))


if __name__ == "__main__":
    main()
