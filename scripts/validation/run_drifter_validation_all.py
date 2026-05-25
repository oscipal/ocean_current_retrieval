"""Run drifter validation across every scene in data/sentinel-1/scene*/.

Conventions assumed for auto-discovery:
  data/sentinel-1/<scene>/S1A_IW_SLC.SAFE
  data/sentinel-1/<scene>/S1A_IW_OCN.SAFE
  data/sentinel-1/S1A_OPER_AUX_POEORB_*_V<start>_<stop>.EOF        (covering scene date)
  data/sentinel-1/S1A_AUX_CAL_*.SAFE                                 (singleton, shared)
  data/era5_data/<scene>/era5_wind.nc
  data/era5_data/<scene>/era5_wave.nc
  data/era5_data/<scene>/glo12.nc

Any scene missing aux files is reported and skipped.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.validation.run_drifter_validation import (
    DRIFTER_DATASET_AUTO,
    DRIFTER_DATASET_MY,
    DRIFTER_DATASET_NRT,
    find_gamma_products,
    read_overpass_time,
    validate_scene,
    print_summary,
)

POEORB_RE = re.compile(r"V(\d{8}T\d{6})_(\d{8}T\d{6})\.EOF$")


def find_poeorb(poeorb_dir: Path, t0: datetime) -> Path | None:
    for f in sorted(poeorb_dir.glob("*POEORB*.EOF")):
        m = POEORB_RE.search(f.name)
        if not m:
            continue
        start = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")
        stop = datetime.strptime(m.group(2), "%Y%m%dT%H%M%S")
        if start <= t0 <= stop:
            return f
    return None


def discover_scenes(data_root: Path) -> list[dict]:
    aux_cal = next(iter((data_root / "sentinel-1").glob("S1A_AUX_CAL_*.SAFE")), None)
    shared_poeorb_dir = data_root / "sentinel-1"
    out = []
    for scene_dir in sorted((data_root / "sentinel-1").glob("scene*")):
        if not scene_dir.is_dir():
            continue
        slc = scene_dir / "S1A_IW_SLC.SAFE"
        ocn = scene_dir / "S1A_IW_OCN.SAFE"
        if not (slc.exists() and ocn.exists()):
            continue
        try:
            t0 = read_overpass_time(slc)
        except Exception as e:
            print(f"[{scene_dir.name}] cannot read overpass time: {e}")
            continue
        # Prefer a POEORB inside the scene dir; fall back to the legacy shared
        # location so old layouts still validate.
        poeorb = find_poeorb(scene_dir, t0) or find_poeorb(shared_poeorb_dir, t0)
        era_dir = data_root / "era5_data" / scene_dir.name
        record = {
            "label":     scene_dir.name,
            "safe":      slc,
            "ocn_safe":  ocn,
            "t0":        t0,
            "aux_cal":   aux_cal,
            "poeorb":    poeorb,
            "era5_wind": era_dir / "era5_wind.nc",
            "era5_wave": era_dir / "era5_wave.nc",
            "glo12":     era_dir / "glo12.nc",
        }
        out.append(record)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--pol", default="vv")
    p.add_argument("--window-h", type=float, default=6.0)
    p.add_argument("--max-dist-km", type=float, default=10.0)
    p.add_argument("--max-dt-min", type=float, default=30.0)
    p.add_argument("--drifter-dir", type=Path, default=Path("data/drifters"))
    p.add_argument("--results-csv", type=Path, default=Path("data/drifters/validation_results.csv"))
    p.add_argument("--drifter-dataset", default=DRIFTER_DATASET_AUTO,
                   help=f"CMEMS dataset id (default: {DRIFTER_DATASET_AUTO}). "
                        f"Auto tries {DRIFTER_DATASET_NRT} and {DRIFTER_DATASET_MY} in a scene-dependent order.")
    p.add_argument("--engine", default="ours", choices=("ours", "gamma"),
                   help="Pipeline producing the SAR current: 'ours' (per-burst "
                        "run_pipeline; default) or 'gamma' (GAMMA mosaic-last + "
                        "doppler_2d_SLC; runs GAMMA prep in-memory from the "
                        "SAFE when cached products are absent).")
    p.add_argument("--gamma-blsz", type=int, default=256,
                   help="GAMMA doppler_2d_SLC azimuth block size (engine=gamma).")
    p.add_argument("--gamma-demod-back", default="blend",
                   choices=("none", "step", "blend", "hanning"),
                   help="add_demod_back mode for gamma_doppler_mosaic_last.")
    p.add_argument("--gamma-geom-source", default="gamma",
                   choices=("gamma", "annotation", "poeorb"),
                   help="Geometry-Doppler source for engine=gamma (default 'gamma').")
    p.add_argument("--gamma-wave-source", default="mouche",
                   choices=("mouche", "ocn"),
                   help="Wave-Doppler bias source: 'mouche' (default) or 'ocn' "
                        "(OCN owiRadVel — operationally calibrated).")
    p.add_argument("--gamma-descallop", action="store_true",
                   help="Apply azimuth-periodic descalloping to the GAMMA f_dca grid.")
    p.add_argument("--gamma-keep-products", action="store_true",
                   help="Persist GAMMA prep outputs after the run instead of "
                        "using a tempdir.  Speeds up re-runs.")
    p.add_argument("--smooth-az", type=int, default=1,
                   help="Boxcar-smooth result data fields by this many az-block "
                        "rows before drifter lookup (default 1 = no smoothing). "
                        "Suppresses burst-boundary stepping without amplitude "
                        "attenuation; pair with --gamma-demod-back step/hanning.")
    p.add_argument("--smooth-rg", type=int, default=1,
                   help="Boxcar-smooth in range-block columns (default 1 = none).")
    args = p.parse_args()

    scenes = discover_scenes(args.data_root)
    if not scenes:
        print("No scenes found under data/sentinel-1/scene*/")
        return

    print(f"Discovered {len(scenes)} scene(s); engine={args.engine!r}\n")
    readiness_rows = []
    for s in scenes:
        missing = [k for k in ("aux_cal", "poeorb", "era5_wind", "era5_wave", "glo12")
                   if s[k] is None or not Path(s[k]).exists()]
        details = "-" if not missing else ",".join(missing)
        if args.engine == "gamma":
            cached = [sw for sw in ("iw1", "iw2", "iw3")
                      if find_gamma_products(s["t0"], sw, args.pol) is not None]
            gamma_note = ("cached:" + ",".join(cached)) if cached else "in-memory prep"
            details = gamma_note + ("" if not missing else f"; missing {','.join(missing)}")
        readiness_rows.append({
            "scene": s["label"],
            "date": s["t0"].strftime("%Y-%m-%d"),
            "status": "ready" if not missing else "missing",
            "details": details,
        })
    print(pd.DataFrame(readiness_rows).to_string(index=False))
    print()

    args.drifter_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for s in scenes:
        missing = [k for k in ("aux_cal", "poeorb", "era5_wind", "era5_wave", "glo12")
                   if s[k] is None or not Path(s[k]).exists()]
        if missing:
            print(f"[{s['label']}] skip (missing: {','.join(missing)})")
            continue
        try:
            df = validate_scene(
                safe=s["safe"], poeorb=s["poeorb"], aux_cal=s["aux_cal"],
                ocn_safe=s["ocn_safe"], era5_wind=s["era5_wind"],
                era5_wave=s["era5_wave"], glo12=s["glo12"],
                drifter_csv=args.drifter_dir / f"{s['label']}.csv",
                pol=args.pol, window_h=args.window_h,
                max_dist_km=args.max_dist_km, max_dt_min=args.max_dt_min,
                scene_label=s["label"],
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
        except Exception as e:
            print(f"[{s['label']}] FAILED: {e}")
            continue
        if not df.empty:
            all_rows.append(df)

    if not all_rows:
        print("\nNo validation rows from any scene.")
        return

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(args.results_csv, index=False)
    print(f"\nSaved {len(out)} rows to {args.results_csv}")
    print_summary(out)


if __name__ == "__main__":
    main()
