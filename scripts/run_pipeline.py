#!/usr/bin/env python3
"""Run the Sentinel-1 ocean-current retrieval for one scene.

By default this runs the **custom pipeline**: the Doppler centroid is
estimated for each TOPS burst with GAMMA, the geometry is removed from the
precise orbit, and the sideband, descalloping, mispointing, Stokes and wave
(Mouche) corrections are applied, giving the radial surface current.

Pass ``--hybrid`` to instead run the **hybrid configuration**, which takes
the observed Doppler and the geometry from the operational Sentinel-1 ocean
product and applies the same downstream corrections.

The retrieved current is regridded onto a regular latitude/longitude grid,
saved as a NetCDF file and a quicklook PNG, and summarised on screen.

Examples
--------
    python scripts/run_pipeline.py                       # custom pipeline, scene1 iw1
    python scripts/run_pipeline.py --hybrid              # hybrid configuration
    python scripts/run_pipeline.py --scene scene1 --subswath iw2 --pol vv
    python scripts/run_pipeline.py --data-root data --out-dir data/current
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.sentinel_1.grid_merge import merge_burst_grids
from scripts.sentinel_1.pipeline import run_all_bursts, run_gamma_pipeline_from_safe

# Canonical "custom pipeline" configuration (matches the report / figures).
CUSTOM_CFG = dict(
    blsz=256,
    add_demod_back="blend",
    geom_source="gamma",
    wave_source="mouche",
    descallop_blocks=True,
)
CURRENT_FIELD = "v_current_ocn"   # radial current with the OCN mispointing term


def scene_paths(data_root: str, scene: str, subswath: str, pol: str) -> dict:
    """Locate the SLC, OCN, orbit, calibration and ERA5/GLO12 inputs."""
    s1 = os.path.join(data_root, "sentinel-1", scene)
    era = os.path.join(data_root, "era5_data", scene)

    def first(patterns, what):
        for p in patterns:
            hits = sorted(glob.glob(p))
            if hits:
                return hits[0]
        raise SystemExit(f"Could not find {what} (looked for {patterns})")

    poe = first([os.path.join(s1, "S1A_OPER_AUX_POEORB_*.EOF"),
                 os.path.join(data_root, "sentinel-1", "S1A_OPER_AUX_POEORB_*.EOF")],
                "precise orbit (POEORB .EOF)")
    aux = first([os.path.join(data_root, "sentinel-1", "S1A_AUX_CAL_*.SAFE")],
                "auxiliary calibration (AUX_CAL .SAFE)")
    return {
        "slc_safe":     os.path.join(s1, "S1A_IW_SLC.SAFE"),
        "ocn_safe":     os.path.join(s1, "S1A_IW_OCN.SAFE"),
        "poeorb_path":  poe,
        "aux_cal_path": aux,
        "era5_wind":    os.path.join(era, "era5_wind.nc"),
        "era5_wave":    os.path.join(era, "era5_wave.nc"),
        "glo12":        os.path.join(era, "glo12.nc"),
        "subswath":     subswath,
        "polarisation": pol,
    }


def run_custom(paths: dict, products_dir: str) -> list[dict]:
    """Custom pipeline: GAMMA per-burst Doppler + full correction chain."""
    result = run_gamma_pipeline_from_safe(
        slc_safe=paths["slc_safe"], subswath=paths["subswath"],
        poeorb_path=paths["poeorb_path"], aux_cal_path=paths["aux_cal_path"],
        ocn_safe=paths["ocn_safe"], era5_wind=paths["era5_wind"],
        era5_wave=paths["era5_wave"], glo12=paths["glo12"],
        polarisation=paths["polarisation"],
        keep_products=True, products_dir=products_dir,
        **CUSTOM_CFG,
    )
    return [result]


def run_hybrid(paths: dict) -> list[dict]:
    """Hybrid configuration: ocean-product Doppler + the same corrections."""
    return run_all_bursts(
        slc_safe=paths["slc_safe"], subswath=paths["subswath"],
        poeorb_path=paths["poeorb_path"], aux_cal_path=paths["aux_cal_path"],
        ocn_safe=paths["ocn_safe"], era5_wind=paths["era5_wind"],
        era5_wave=paths["era5_wave"], glo12=paths["glo12"],
        polarisation=paths["polarisation"],
        use_ocn_dc=True,
    )


def save_netcdf(out_nc: str, lat, lon, current, meta: dict) -> bool:
    try:
        import xarray as xr
    except ImportError:
        print("  (xarray not available, skipping NetCDF)")
        return False
    ds = xr.Dataset(
        {"radial_current": (("latitude", "longitude"), current)},
        coords={"latitude": lat, "longitude": lon},
        attrs=meta,
    )
    ds["radial_current"].attrs.update(units="m s-1",
                                      long_name="surface radial current")
    ds.to_netcdf(out_nc)
    return True


def quicklook(out_png: str, lat, lon, current, title: str) -> None:
    finite = current[np.isfinite(current)]
    vmax = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
    fig, ax = plt.subplots(figsize=(6.4, 6.6), constrained_layout=True)
    im = ax.pcolormesh(lon, lat, current, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       shading="auto")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02, label="radial current [m/s]")
    cb.outline.set_visible(False)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hybrid", action="store_true",
                   help="Run the hybrid configuration (ocean-product Doppler) "
                        "instead of the default custom pipeline.")
    p.add_argument("--data-root", default="data")
    p.add_argument("--scene", default="scene1")
    p.add_argument("--subswath", default="iw1", choices=("iw1", "iw2", "iw3"))
    p.add_argument("--pol", default="vv", choices=("vv", "vh"))
    p.add_argument("--resolution-deg", type=float, default=0.01,
                   help="Output grid spacing in degrees (default 0.01, ~1 km).")
    p.add_argument("--out-dir", default="data/current",
                   help="Where to write the NetCDF and quicklook PNG.")
    args = p.parse_args()

    mode = "hybrid" if args.hybrid else "custom"
    paths = scene_paths(args.data_root, args.scene, args.subswath, args.pol)
    products_dir = os.path.join(args.data_root, "sentinel-1",
                                f"gamma_{args.subswath.lower()}")

    print(f"=== {mode} pipeline | {args.scene} {args.subswath.upper()} "
          f"{args.pol.upper()} ===")
    bursts = run_hybrid(paths) if args.hybrid else run_custom(paths, products_dir)

    print("regridding onto a regular lat/lon grid ...")
    lat, lon, current = merge_burst_grids(
        bursts, variable=CURRENT_FIELD, overlap="average",
        resolution_deg=args.resolution_deg)

    finite = current[np.isfinite(current)]
    print(f"  grid {current.shape}  valid cells {finite.size}  "
          f"mean {finite.mean():+.3f}  range [{finite.min():+.3f}, "
          f"{finite.max():+.3f}] m/s")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.scene}_{args.subswath}_{args.pol}_{mode}"
    out_nc = os.path.join(args.out_dir, f"current_{tag}.nc")
    out_png = os.path.join(args.out_dir, f"current_{tag}.png")
    meta = dict(scene=args.scene, subswath=args.subswath, polarisation=args.pol,
                configuration=mode, field=CURRENT_FIELD)
    if save_netcdf(out_nc, lat, lon, current, meta):
        print(f"saved -> {out_nc}")
    quicklook(out_png, lat, lon, current,
              f"{mode} pipeline: {args.scene} {args.subswath.upper()}")
    print(f"saved -> {out_png}")


if __name__ == "__main__":
    main()
