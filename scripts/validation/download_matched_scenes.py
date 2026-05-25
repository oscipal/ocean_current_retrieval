"""Discover S1 scenes that intersect CMEMS drifters, download the matched ones.

Phase 1 (always runs, no auth needed):
  * Query CMEMS NRT drifters in --bbox over --start..--end
  * Query CDSE OpenSearch for S1A IW SLC scenes in same bbox/time
  * For each scene, check whether any drifter fix lies inside the footprint
    polygon within ±--max-dt-h hours of overpass
  * Print the matched list

Phase 2 (--download, needs CDSE credentials):
  For each matched scene, populate the conventional paths used by
  run_drifter_validation_all.py:
    data/sentinel-1/<scene_dir>/S1A_IW_SLC.SAFE
    data/sentinel-1/<scene_dir>/S1A_IW_OCN.SAFE
    data/sentinel-1/<POEORB *.EOF covering scene date>
    data/era5_data/<scene_dir>/{era5_wind,era5_wave,glo12}.nc

Credentials:
  CDSE  : CDSE_USER + CDSE_PASS env vars (account at https://dataspace.copernicus.eu)
  CMEMS : already configured via `copernicusmarine login`
  CDS   : ~/.cdsapirc for ERA5 (only needed if --skip-era5 is not set)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import copernicusmarine as cm
from shapely.geometry import Point, shape

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DRIFTER_DATASET_NRT = "cmems_obs-ins_glo_phy-cur_nrt_drifter_irr"  # ~last 3 months
DRIFTER_DATASET_MY  = "cmems_obs-ins_glo_phy-cur_my_drifter_PT1H"  # 1993–present, hourly
GLO12_DATASET = "cmems_mod_glo_phy_anfc_0.083deg_PT1H-m"
CDSE_ODATA = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
CDSE_DOWNLOAD = "https://download.dataspace.copernicus.eu/odata/v1/Products({pid})/$value"
CDSE_TOKEN = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"


# ── drifter side ─────────────────────────────────────────────────────────────

def query_drifters_bbox(
    bbox: tuple[float, float, float, float],  # lon_min, lon_max, lat_min, lat_max
    start: datetime,
    end: datetime,
    out_csv: Path,
    dataset_id: str = DRIFTER_DATASET_MY,
) -> pd.DataFrame:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cm.subset(
        dataset_id=dataset_id,
        dataset_part="history",
        variables=["EWCT", "NSCT"],
        minimum_longitude=bbox[0], maximum_longitude=bbox[1],
        minimum_latitude=bbox[2],  maximum_latitude=bbox[3],
        start_datetime=start.isoformat(),
        end_datetime=end.isoformat(),
        output_directory=str(out_csv.parent),
        output_filename=out_csv.stem,
        file_format="csv",
        overwrite=True,
    )
    if not out_csv.exists():
        return pd.DataFrame(columns=["platform_id", "time", "latitude", "longitude", "EWCT", "NSCT"])
    raw = pd.read_csv(out_csv)
    if raw.empty:
        return pd.DataFrame(columns=["platform_id", "time", "latitude", "longitude", "EWCT", "NSCT"])
    wide = raw.pivot_table(
        index=["platform_id", "time", "latitude", "longitude", "depth"],
        columns="variable", values="value").reset_index()
    wide["time"] = pd.to_datetime(wide["time"], utc=True).dt.tz_convert(None)
    return wide.dropna(subset=["EWCT", "NSCT"])


# ── S1 catalog ───────────────────────────────────────────────────────────────

def _bbox_polygon_wkt(bbox: tuple[float, float, float, float]) -> str:
    lon_min, lon_max, lat_min, lat_max = bbox
    return (f"POLYGON(({lon_min} {lat_min},{lon_max} {lat_min},"
            f"{lon_max} {lat_max},{lon_min} {lat_max},{lon_min} {lat_min}))")


def _odata_query(filter_expr: str, max_records: int = 10000) -> list[dict]:
    out = []
    url = CDSE_ODATA
    params = {"$filter": filter_expr, "$top": min(100, max_records), "$orderby": "ContentDate/Start"}
    while url:
        r = requests.get(url, params=params if url == CDSE_ODATA else None, timeout=60)
        r.raise_for_status()
        body = r.json()
        out.extend(body.get("value", []))
        url = body.get("@odata.nextLink") if len(out) < max_records else None
        params = None  # nextLink already encodes everything
    return out[:max_records]


def search_s1_slc(
    bbox: tuple[float, float, float, float],
    start: datetime,
    end: datetime,
    max_records: int = 10000,
) -> list[dict]:
    """Return OData products for S1A IW SLC scenes inside bbox + window."""
    poly = _bbox_polygon_wkt(bbox)
    expr = (
        "Collection/Name eq 'SENTINEL-1' and contains(Name,'S1A_IW_SLC') "
        f"and ContentDate/Start gt {start.isoformat()}.000Z "
        f"and ContentDate/Start lt {end.isoformat()}.000Z "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{poly}')"
    )
    return _odata_query(expr, max_records=max_records)


def parse_s1_feature(prod: dict) -> dict:
    return {
        "id":       prod["Id"],
        "name":     prod["Name"].removesuffix(".SAFE"),
        "start":    datetime.fromisoformat(prod["ContentDate"]["Start"].rstrip("Z")),
        "end":      datetime.fromisoformat(prod["ContentDate"]["End"].rstrip("Z")),
        "geometry": shape(prod["GeoFootprint"]),
    }


def find_matched_scenes(
    drifters: pd.DataFrame,
    scenes: list[dict],
    max_dt_h: float,
) -> list[dict]:
    matched = []
    for s in scenes:
        dt = pd.Timedelta(hours=max_dt_h)
        cand = drifters[(drifters["time"] >= s["start"] - dt) & (drifters["time"] <= s["start"] + dt)]
        if cand.empty:
            continue
        in_poly = cand[[s["geometry"].contains(Point(lo, la))
                        for lo, la in zip(cand["longitude"], cand["latitude"])]]
        if in_poly.empty:
            continue
        matched.append({
            **s,
            "n_drifter_fixes": len(in_poly),
            "n_drifter_platforms": int(in_poly["platform_id"].nunique()),
            "closest_dt_min": float((in_poly["time"] - s["start"]).abs().dt.total_seconds().min() / 60.0),
        })
    return matched


# ── CDSE download ────────────────────────────────────────────────────────────

def cdse_token(user: str, password: str) -> str:
    r = requests.post(
        CDSE_TOKEN,
        data={"client_id": "cdse-public", "grant_type": "password",
              "username": user, "password": password},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def cdse_download(
    product_id: str,
    token_fn,
    out_path: Path,
    max_attempts: int = 5,
) -> None:
    """Download a CDSE product, retrying transient network errors.

    `token_fn` is a zero-arg callable that returns a fresh bearer token (CDSE
    tokens expire ~10 min, often before a full SLC download finishes).
    """
    import time
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = CDSE_DOWNLOAD.format(pid=product_id)
    for attempt in range(1, max_attempts + 1):
        try:
            token = token_fn()
            with requests.get(url, headers={"Authorization": f"Bearer {token}"},
                              stream=True, timeout=600, allow_redirects=True) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
            return
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
                requests.exceptions.HTTPError) as e:
            if attempt == max_attempts:
                raise
            backoff = 5 * 2 ** (attempt - 1)
            print(f"  [retry {attempt}/{max_attempts}] {type(e).__name__}: {e} "
                  f"— sleeping {backoff}s then retrying")
            if out_path.exists():
                out_path.unlink()
            time.sleep(backoff)


def find_matching_ocn(slc_name: str, max_offset_s: float = 5.0) -> dict | None:
    """Find the OCN product for the *same slice* as the SLC.

    The OCN sibling shares the SLC's sensing start time. Consecutive S1 slices
    are ~25 s apart, so the query window (±30 s) can also catch the neighbouring
    slice's OCN — we therefore take the candidate whose start is *closest* to
    the SLC start, and return None if even the closest is more than
    `max_offset_s` away (no true sibling in the catalogue) rather than attach
    the wrong slice's OCN.
    """
    m = re.search(r"(\d{8}T\d{6})_(\d{8}T\d{6})", slc_name)
    if not m:
        return None
    t0 = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")
    expr = (
        "Collection/Name eq 'SENTINEL-1' and contains(Name,'S1A_IW_OCN') "
        f"and ContentDate/Start gt {(t0 - timedelta(seconds=30)).isoformat()}.000Z "
        f"and ContentDate/Start lt {(t0 + timedelta(seconds=30)).isoformat()}.000Z"
    )
    hits = _odata_query(expr, max_records=10)
    if not hits:
        return None

    def offset_s(hit: dict) -> float:
        start = datetime.fromisoformat(hit["ContentDate"]["Start"].rstrip("Z"))
        return abs((start - t0).total_seconds())

    best = min(hits, key=offset_s)
    return best if offset_s(best) <= max_offset_s else None


def unzip_to_safe(zip_path: Path, dest_dir: Path, rename_to: str) -> Path:
    """Unzip a CDSE product and rename the resulting *.SAFE folder.

    Extraction goes into a private temp subdir so that a *.SAFE folder already
    present in dest_dir (e.g. the SLC, when we are now unzipping the OCN into
    the same scene dir) can never be picked up and renamed by mistake.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest_dir / f".extract_{rename_to}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        # the temp dir contains exactly the one product just extracted
        safes = [p for p in tmp.iterdir() if p.is_dir() and p.name.endswith(".SAFE")]
        if not safes:
            raise RuntimeError(f"No .SAFE folder after unzipping {zip_path}")
        target = dest_dir / rename_to
        if target.exists():
            shutil.rmtree(target)
        safes[0].rename(target)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    zip_path.unlink()
    return target


# ── ERA5 / GLO12 per scene ───────────────────────────────────────────────────

def download_era5(scene_bbox: tuple[float, float, float, float], t0: datetime, out_dir: Path) -> None:
    """Pull ERA5 wind + wave for the scene's hour using the existing JSON-config helper."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # north, west, south, east
    cfg = {
        "download_dir": str(out_dir),
        "year": t0.year,
        "month": t0.month,
        "day": t0.day,
        "time": [f"{t0.hour:02d}:00"],
        "area": [scene_bbox[3], scene_bbox[0], scene_bbox[2], scene_bbox[1]],
    }
    cfg_path = out_dir / "era5_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "download_era5.py"), "--config", str(cfg_path)],
        check=True,
    )


def download_glo12(scene_bbox: tuple[float, float, float, float], t0: datetime, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    day_start = datetime(t0.year, t0.month, t0.day)
    cm.subset(
        dataset_id=GLO12_DATASET,
        variables=["uo", "vo", "thetao", "zos", "so"],
        minimum_longitude=scene_bbox[0] - 0.5, maximum_longitude=scene_bbox[1] + 0.5,
        minimum_latitude=scene_bbox[2] - 0.5,  maximum_latitude=scene_bbox[3] + 0.5,
        minimum_depth=0.0, maximum_depth=1.0,
        start_datetime=day_start.isoformat(),
        end_datetime=(day_start + timedelta(days=1)).isoformat(),
        output_directory=str(out_path.parent),
        output_filename=out_path.name,
        overwrite=True,
    )


def download_poeorb(t0: datetime, save_dir: Path) -> None:
    subprocess.run(
        ["eof", "-m", "S1A", "-d", t0.strftime("%Y%m%d"), "--save-dir", str(save_dir)],
        check=True,
    )


# ── orchestration ────────────────────────────────────────────────────────────

def next_scene_dir(data_root: Path) -> str:
    existing = sorted(p.name for p in (data_root / "sentinel-1").glob("scene*") if p.is_dir())
    nums = [int(re.search(r"\d+$", n).group()) for n in existing if re.search(r"\d+$", n)]
    return f"scene{(max(nums) + 1) if nums else 1}"


def scene_dir_for_overpass(data_root: Path, t0: datetime) -> str | None:
    """Return existing scene dir whose SLC overpass matches t0 (within 5s), else None."""
    for d in sorted((data_root / "sentinel-1").glob("scene*")):
        slc = d / "S1A_IW_SLC.SAFE"
        if not (slc / "manifest.safe").exists():
            continue
        m = re.search(r"startTime>([^<]+)<", (slc / "manifest.safe").read_text())
        if not m:
            continue
        if abs((datetime.fromisoformat(m.group(1)) - t0).total_seconds()) < 5:
            return d.name
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bbox", nargs=4, type=float, default=[20, 33, -38, -30],
                   metavar=("LONmin", "LONmax", "LATmin", "LATmax"))
    p.add_argument("--start", type=lambda s: datetime.fromisoformat(s),
                   default=datetime(2026, 2, 4))
    p.add_argument("--end",   type=lambda s: datetime.fromisoformat(s),
                   default=datetime(2026, 2, 25, 23, 59, 59))
    p.add_argument("--max-dt-h", type=float, default=1.0,
                   help="Max time between drifter fix and S1 overpass to call it a match")
    p.add_argument("--download", action="store_true",
                   help="Actually fetch matched scenes (otherwise discovery only)")
    p.add_argument("--skip-era5", action="store_true")
    p.add_argument("--skip-glo12", action="store_true")
    p.add_argument("--skip-poeorb", action="store_true")
    p.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--drifter-csv", type=Path,
                   default=REPO_ROOT / "data" / "drifters" / "agulhas_bbox.csv")
    p.add_argument("--manifest-out", type=Path,
                   default=REPO_ROOT / "data" / "drifters" / "matched_scenes.json")
    p.add_argument("--drifter-dataset", default=DRIFTER_DATASET_MY,
                   help=f"CMEMS dataset id (default: {DRIFTER_DATASET_MY}). "
                        f"Use {DRIFTER_DATASET_NRT} for last ~3 months.")
    args = p.parse_args()

    bbox = tuple(args.bbox)
    print(f"[discover] CMEMS drifters: bbox={bbox}  {args.start.date()}..{args.end.date()}  "
          f"dataset={args.drifter_dataset}")
    drifters = query_drifters_bbox(bbox, args.start, args.end, args.drifter_csv, args.drifter_dataset)
    print(f"[discover]   {len(drifters)} drifter fixes, "
          f"{drifters['platform_id'].nunique() if len(drifters) else 0} unique platforms")
    if drifters.empty:
        print("No drifters in window. Nothing to do.")
        return

    print(f"[discover] CDSE S1A IW SLC catalog query…")
    feats = search_s1_slc(bbox, args.start, args.end)
    scenes = [parse_s1_feature(f) for f in feats]
    print(f"[discover]   {len(scenes)} scenes in catalog")

    matched = find_matched_scenes(drifters, scenes, args.max_dt_h)
    matched.sort(key=lambda s: s["start"])
    print(f"\n[match]   {len(matched)} scenes contain a drifter within ±{args.max_dt_h:.1f} h")
    for s in matched:
        print(f"   {s['start'].isoformat()[:19]}  {s['name']}  "
              f"({s['n_drifter_fixes']} fixes / {s['n_drifter_platforms']} platforms, "
              f"closest Δt={s['closest_dt_min']:.0f} min)")

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(
        [{**{k: v for k, v in s.items() if k != "geometry"},
          "start": s["start"].isoformat(), "end": s["end"].isoformat(),
          "geometry_wkt": s["geometry"].wkt}
         for s in matched], indent=2))
    print(f"[match]   wrote {args.manifest_out}")

    if not args.download:
        print("\nDiscovery only. Re-run with --download to fetch.")
        return

    # ── download phase ────────────────────────────────────────────────────
    user = os.environ.get("CDSE_USER")
    password = os.environ.get("CDSE_PASS")
    if not user or not password:
        sys.exit("ERROR: set CDSE_USER and CDSE_PASS env vars (https://dataspace.copernicus.eu)")
    token_fn = lambda: cdse_token(user, password)
    token_fn()  # fail fast on bad creds
    print(f"\n[download] CDSE auth ok (tokens minted per request, refreshed on retry)\n")

    failed_scenes = []
    for s in matched:
        # Reuse existing scene dir if same overpass, else allocate next number
        existing = scene_dir_for_overpass(args.data_root, s["start"])
        scene = existing or next_scene_dir(args.data_root)
        scene_dir = args.data_root / "sentinel-1" / scene
        era_dir = args.data_root / "era5_data" / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        slc_target = scene_dir / "S1A_IW_SLC.SAFE"
        ocn_target = scene_dir / "S1A_IW_OCN.SAFE"

        print(f"=== {scene}  {s['name']} ===")

        try:
            if not slc_target.exists():
                print(f"  [SLC] downloading …")
                zip_path = scene_dir / f"{s['name']}.zip"
                cdse_download(s["id"], token_fn, zip_path)
                unzip_to_safe(zip_path, scene_dir, "S1A_IW_SLC.SAFE")
            else:
                print(f"  [SLC] already present, skipping")

            if not ocn_target.exists():
                print(f"  [OCN] searching CDSE for sibling …")
                ocn_feat = find_matching_ocn(s["name"])
                if ocn_feat is None:
                    print(f"  [OCN] no sibling found; skipping")
                else:
                    ocn_name = ocn_feat["Name"].removesuffix(".SAFE")
                    print(f"  [OCN] downloading {ocn_name} …")
                    zip_path = scene_dir / f"{ocn_name}.zip"
                    cdse_download(ocn_feat["Id"], token_fn, zip_path)
                    unzip_to_safe(zip_path, scene_dir, "S1A_IW_OCN.SAFE")
            else:
                print(f"  [OCN] already present, skipping")
        except Exception as e:
            print(f"  [SCENE FAILED] {type(e).__name__}: {e} — moving on")
            failed_scenes.append((scene, s["name"], str(e)))
            continue

        if not args.skip_poeorb:
            if any(scene_dir.glob("*POEORB*.EOF")):
                print(f"  [POEORB] already present, skipping")
            else:
                print(f"  [POEORB] eof for {s['start'].date()}…")
                try:
                    download_poeorb(s["start"], scene_dir)
                except subprocess.CalledProcessError as e:
                    print(f"  [POEORB] failed: {e}")

        scene_bbox = s["geometry"].bounds  # (minx, miny, maxx, maxy)
        scene_bbox = (scene_bbox[0], scene_bbox[2], scene_bbox[1], scene_bbox[3])  # lon_min,lon_max,lat_min,lat_max

        if not args.skip_era5 and not (era_dir / "era5_wind.nc").exists():
            print(f"  [ERA5] download_era5.py …")
            try:
                download_era5(scene_bbox, s["start"], era_dir)
            except subprocess.CalledProcessError as e:
                print(f"  [ERA5] failed: {e}")
        elif not args.skip_era5:
            print(f"  [ERA5] already present, skipping")

        if not args.skip_glo12 and not (era_dir / "glo12.nc").exists():
            print(f"  [GLO12] copernicusmarine subset …")
            try:
                download_glo12(scene_bbox, s["start"], era_dir / "glo12.nc")
            except Exception as e:
                print(f"  [GLO12] failed: {e}")
        elif not args.skip_glo12:
            print(f"  [GLO12] already present, skipping")

    if failed_scenes:
        print(f"\n[summary] {len(failed_scenes)} scene(s) failed (re-run the same command to retry):")
        for scene, name, err in failed_scenes:
            print(f"  {scene}  {name}\n    {err}")
    print("\nDone. Now run:  python scripts/validation/run_drifter_validation_all.py")


if __name__ == "__main__":
    main()
