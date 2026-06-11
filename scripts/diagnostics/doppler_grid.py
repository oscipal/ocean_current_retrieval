#!/usr/bin/env python3
"""Single-panel map of our Doppler-centroid estimate on the raw SLC.

Runs the GAMMA pipeline with our default estimator (``lag1_gamma`` =
``doppler_2d_SLC`` lag-1 autocorrelation on the deramped SLC) and plots the
resulting Doppler grid as one image — no corrections, no comparison, just the
numbers our estimation produces straight off the scene.

Fields (``--field``):

  * ``f_dc``   (default) — the **raw measured Doppler centroid** in Hz, exactly
                 what ``doppler_2d_SLC`` returns per block.  No geometry, no
                 sideband, no descalloping removed.
  * ``f_dca``  — the Doppler **anomaly** in Hz (after geometry + sideband +
                 descalloping), i.e. what actually feeds the velocity inversion.
  * ``v_r``    — radial velocity (λ/2)·f_dca in m/s.

The grid is shown on its geographic (lat/lon) extent.  The heavy pipeline
output is pickle-cached so re-runs only re-render.

Usage
-----
    python scripts/diagnostics/doppler_grid.py
    python scripts/diagnostics/doppler_grid.py --field f_dca
    python scripts/diagnostics/doppler_grid.py --subswath iw2 --no-cache
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.sentinel_1.pipeline import run_gamma_pipeline_from_safe


_FIELD_INFO = {
    "f_dc":  ("Raw Doppler centroid (doppler_2d_SLC, lag-1)", "Doppler centroid [Hz]", "Hz"),
    "f_dca": ("Doppler anomaly (geom + sideband + descallop removed)", "Doppler anomaly [Hz]", "Hz"),
    "v_r":   ("Radial velocity (λ/2)·f_dca", "Radial velocity [m/s]", "m/s"),
}


def scene_paths(data_root: str, scene: str, subswath: str, pol: str) -> dict:
    s1 = os.path.join(data_root, "sentinel-1", scene)
    era = os.path.join(data_root, "era5_data", scene)
    poe = sorted(glob.glob(os.path.join(s1, "S1A_OPER_AUX_POEORB_*.EOF")))
    if not poe:
        poe = sorted(glob.glob(
            os.path.join(data_root, "sentinel-1", "S1A_OPER_AUX_POEORB_*.EOF")))
    aux = sorted(glob.glob(
        os.path.join(data_root, "sentinel-1", "S1A_AUX_CAL_*.SAFE")))
    return {
        "slc_safe":     os.path.join(s1, "S1A_IW_SLC.SAFE"),
        "ocn_safe":     os.path.join(s1, "S1A_IW_OCN.SAFE"),
        "poeorb":       poe[0],
        "aux_cal":      aux[0],
        "era5_wind":    os.path.join(era, "era5_wind.nc"),
        "era5_wave":    os.path.join(era, "era5_wave.nc"),
        "glo12":        os.path.join(era, "glo12.nc"),
        "subswath":     subswath,
        "polarisation": pol,
    }


def run_pipeline(paths: dict) -> dict:
    return run_gamma_pipeline_from_safe(
        slc_safe=paths["slc_safe"], subswath=paths["subswath"],
        poeorb_path=paths["poeorb"], aux_cal_path=paths["aux_cal"],
        ocn_safe=paths["ocn_safe"], era5_wind=paths["era5_wind"],
        era5_wave=paths["era5_wave"], glo12=paths["glo12"],
        polarisation=paths["polarisation"],
        f_dc_method="lag1_gamma",
        add_demod_back="blend",
        wave_source="mouche",
        descallop_blocks=False,
        keep_products=True,
        products_dir=os.path.join("data", "sentinel-1",
                                  f"gamma_{paths['subswath'].lower()}"),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", default="data")
    p.add_argument("--scene", default="scene1")
    p.add_argument("--subswath", default="iw1")
    p.add_argument("--pol", default="vv")
    p.add_argument("--field", choices=tuple(_FIELD_INFO), default="f_dc")
    p.add_argument("--vmax", type=float, default=None,
                   help="Symmetric colour scale.  Default = 98th-pctile of |field|.")
    p.add_argument("--cmap", default="RdBu_r")
    p.add_argument("--cache-dir", default="plots/diagnostics/cache")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    paths = scene_paths(args.data_root, args.scene, args.subswath, args.pol)
    tag = f"{args.scene}_{args.subswath}_{args.pol}"
    cache = os.path.join(args.cache_dir, f"dopgrid_{tag}.pkl")
    out = args.out or f"plots/diagnostics/doppler_grid_{args.field}_{tag}.png"

    if not args.no_cache and os.path.exists(cache):
        print(f"[cache] {cache}")
        with open(cache, "rb") as f:
            r = pickle.load(f)
    else:
        print(f"[compute] running pipeline for {tag} ...")
        r = run_pipeline(paths)
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(r, f)

    field = np.asarray(r[args.field], dtype=np.float64)
    lat = np.asarray(r["lat"], dtype=np.float64)
    lon = np.asarray(r["lon"], dtype=np.float64)

    finite = field[np.isfinite(field)]
    title, cbar_label, unit = _FIELD_INFO[args.field]
    vmax = args.vmax if args.vmax is not None else float(
        np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0

    extent = (float(np.nanmin(lon)), float(np.nanmax(lon)),
              float(np.nanmin(lat)), float(np.nanmax(lat)))

    print(f"  field={args.field}  shape={field.shape}  "
          f"std={finite.std():.3f} {unit}  "
          f"range=[{finite.min():+.3f}, {finite.max():+.3f}] {unit}  "
          f"vmax=±{vmax:.3f}")

    fig, ax = plt.subplots(figsize=(6.4, 6.6), constrained_layout=True)
    im = ax.imshow(field, extent=extent, origin="lower", cmap=args.cmap,
                   vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
    ax.set_title(f"{title}\n{args.scene} {args.subswath.upper()} {args.pol.upper()}   "
                 f"std={finite.std():.2f} {unit}", fontsize=10)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    cb = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.02, label=cbar_label)
    cb.outline.set_visible(False)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
