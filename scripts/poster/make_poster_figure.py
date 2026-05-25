#!/usr/bin/env python3
"""Three-panel poster figure: best GAMMA result | OCN product | GLO12 model.

Best GAMMA config (from notebooks/gamma_deramp_mosaic_last.ipynb):
    blsz=256, add_demod_back='blend', geom_source='gamma',
    smoothing az=3 rg=1, current = -v_r - v_stokes - v_wave  (no mispointing).

Heavy results (the GAMMA mosaic-last run and the per-burst OCN/GLO12 pipeline)
are pickle-cached so subsequent runs only re-render the figure.

Usage
-----
    python scripts/poster/make_poster_figure.py
    python scripts/poster/make_poster_figure.py --scene scene1 --subswath iw1
    python scripts/poster/make_poster_figure.py --out poster.svg --no-cache
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.sentinel_1.grid_merge import merge_burst_grids, smooth_block_grid
from scripts.sentinel_1.pipeline import (
    run_all_bursts,
    run_gamma_pipeline_from_safe,
)


# ─── best-config knobs ──────────────────────────────────────────────────────
# Visualization-tuned config: Mouche-2012 wave bias and no descalloping.  The
# OCN-owiRadVel + descalloping combo gives the best drifter-validation numbers
# (bias ≈ −0.025 m/s) but flattens fine-scale features in the image.  Keep the
# numerical config in scripts/validation/run_drifter_validation*.py and use this
# one for the poster panel.
BLSZ              = 256
ADD_DEMOD_BACK    = "blend"
GEOM_SOURCE       = "gamma"
WAVE_SOURCE       = "mouche"         # 'mouche' (visual) or 'ocn' (numerical)
DESCALLOP_BLOCKS  = False            # True for numerical accuracy, False for visuals
SMOOTH_AZ         = 3
SMOOTH_RG         = 1
GAMMA_FIELD       = "v_current"      # = -v_r - v_stokes - v_wave  (no miss)
OCN_FIELD         = "v_current_ocn"  # OCN-product current (rvlDcObs as f_dc)


def scene_paths(data_root: str, scene: str, subswath: str, pol: str) -> dict:
    """Locate everything the pipeline needs for one scene/subswath."""
    s1_dir = os.path.join(data_root, "sentinel-1", scene)
    era_dir = os.path.join(data_root, "era5_data", scene)
    safe = os.path.join(s1_dir, "S1A_IW_SLC.SAFE")
    poeorb_hits = sorted(glob.glob(os.path.join(s1_dir, "S1A_OPER_AUX_POEORB_*.EOF")))
    if not poeorb_hits:
        poeorb_hits = sorted(glob.glob(
            os.path.join(data_root, "sentinel-1", "S1A_OPER_AUX_POEORB_*.EOF")
        ))
    aux_cal_hits = sorted(glob.glob(
        os.path.join(data_root, "sentinel-1", "S1A_AUX_CAL_*.SAFE")
    ))
    return {
        "slc_safe":   safe,
        "ocn_safe":   os.path.join(s1_dir, "S1A_IW_OCN.SAFE"),
        "poeorb":     poeorb_hits[0],
        "aux_cal":    aux_cal_hits[0],
        "era5_wind":  os.path.join(era_dir, "era5_wind.nc"),
        "era5_wave":  os.path.join(era_dir, "era5_wave.nc"),
        "glo12":      os.path.join(era_dir, "glo12.nc"),
        "subswath":   subswath,
        "polarisation": pol,
        "_scene":     scene,
    }


def cached(path: str, compute, use_cache: bool):
    if use_cache and os.path.exists(path):
        print(f"[cache] {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
    print(f"[compute] {path}")
    obj = compute()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return obj


def compute_gamma_result(paths: dict) -> dict:
    """Run the best-config GAMMA mosaic-last pipeline and apply smoothing."""
    r = run_gamma_pipeline_from_safe(
        slc_safe=paths["slc_safe"], subswath=paths["subswath"],
        poeorb_path=paths["poeorb"], aux_cal_path=paths["aux_cal"],
        ocn_safe=paths["ocn_safe"], era5_wind=paths["era5_wind"],
        era5_wave=paths["era5_wave"], glo12=paths["glo12"],
        polarisation=paths["polarisation"],
        blsz=BLSZ, add_demod_back=ADD_DEMOD_BACK, geom_source=GEOM_SOURCE,
        wave_source=WAVE_SOURCE, descallop_blocks=DESCALLOP_BLOCKS,
        keep_products=True,
        products_dir=os.path.join("data", "sentinel-1",
                                  f"gamma_{paths['subswath'].lower()}"),
    )
    return smooth_block_grid(r, smooth_az=SMOOTH_AZ, smooth_rg=SMOOTH_RG)


def compute_ocn_bursts(paths: dict) -> list[dict]:
    """OCN-product current = run_all_bursts with use_ocn_dc=True (rvlDcObs as f_dc)."""
    return run_all_bursts(
        slc_safe=paths["slc_safe"], subswath=paths["subswath"],
        poeorb_path=paths["poeorb"], aux_cal_path=paths["aux_cal"],
        ocn_safe=paths["ocn_safe"], era5_wind=paths["era5_wind"],
        era5_wave=paths["era5_wave"], glo12=paths["glo12"],
        polarisation=paths["polarisation"],
        use_ocn_dc=True,
    )


def regrid(result_or_bursts, variable: str, resolution_deg: float):
    """Wrap a single-dict result or a burst list and regrid the variable."""
    if isinstance(result_or_bursts, dict):
        bursts = [result_or_bursts]
    else:
        bursts = result_or_bursts
    return merge_burst_grids(bursts, variable=variable,
                             overlap="average", resolution_deg=resolution_deg)


def crop_to_common(grids: list[tuple], pad_deg: float = 0.0):
    """Trim every (lat, lon, field) tuple to the intersection of their lat/lon extents."""
    lat_lo = max(g[0].min() for g in grids) + pad_deg
    lat_hi = min(g[0].max() for g in grids) - pad_deg
    lon_lo = max(g[1].min() for g in grids) + pad_deg
    lon_hi = min(g[1].max() for g in grids) - pad_deg
    out = []
    for lat, lon, field in grids:
        ja = np.searchsorted(lat, lat_lo, side="left")
        jb = np.searchsorted(lat, lat_hi, side="right")
        ia = np.searchsorted(lon, lon_lo, side="left")
        ib = np.searchsorted(lon, lon_hi, side="right")
        out.append((lat[ja:jb], lon[ia:ib], field[ja:jb, ia:ib]))
    return out, (lon_lo, lon_hi, lat_lo, lat_hi)


def plot_poster(panels: list[tuple], out_svg: str, vmax: float,
                extent: tuple, cmap: str = "RdBu_r") -> None:
    """One row x three columns, shared symmetric colour scale, single colourbar."""
    fig, axes = plt.subplots(
        1, 3, figsize=(13.5, 4.6),
        gridspec_kw={"width_ratios": [1, 1, 1]},
        constrained_layout=True,
    )
    for ax, (title, field) in zip(axes, panels):
        im = ax.imshow(
            field, extent=extent, origin="lower",
            cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto",
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Longitude [deg]")
    axes[0].set_ylabel("Latitude [deg]")
    cb = fig.colorbar(im, ax=axes.tolist(), shrink=0.92, pad=0.015,
                      label="radial current [m/s]")
    cb.outline.set_visible(False)
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    print(f"saved -> {out_svg}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", default="data")
    p.add_argument("--scene", default="scene1")
    p.add_argument("--subswath", default="iw1")
    p.add_argument("--pol", default="vv")
    p.add_argument("--resolution-deg", type=float, default=0.01,
                   help="Lat/lon grid resolution for regridding (default 0.01).")
    p.add_argument("--out", default=None,
                   help="Output SVG path (default plots/poster/<scene>_<sw>.svg).")
    p.add_argument("--cache-dir", default=None,
                   help="Pickle cache directory (default plots/poster/cache).")
    p.add_argument("--no-cache", action="store_true",
                   help="Recompute even if cached pickles exist.")
    p.add_argument("--vmax", type=float, default=None,
                   help="Override the colour scale (m/s).  Default = 98th-pctile of |all three|.")
    args = p.parse_args()

    paths = scene_paths(args.data_root, args.scene, args.subswath, args.pol)
    out_svg = args.out or f"plots/poster/{args.scene}_{args.subswath}.svg"
    cache_dir = args.cache_dir or "plots/poster/cache"
    use_cache = not args.no_cache
    tag = f"{args.scene}_{args.subswath}_{args.pol}"
    # Bake the GAMMA config into the cache filename so config changes don't
    # accidentally reuse a stale pickle.
    gamma_cfg = (f"blsz{BLSZ}_{ADD_DEMOD_BACK}_geom-{GEOM_SOURCE}"
                 f"_wave-{WAVE_SOURCE}"
                 f"{'_desc' if DESCALLOP_BLOCKS else ''}"
                 f"_smooth{SMOOTH_AZ}x{SMOOTH_RG}")

    print(f"=== {tag}  ({gamma_cfg}) ===")
    gamma = cached(
        os.path.join(cache_dir, f"gamma_{tag}_{gamma_cfg}.pkl"),
        lambda: compute_gamma_result(paths),
        use_cache,
    )
    ocn = cached(
        os.path.join(cache_dir, f"ocn_{tag}.pkl"),
        lambda: compute_ocn_bursts(paths),
        use_cache,
    )

    print("regridding panels ...")
    g_lat, g_lon, g_field = regrid(gamma, GAMMA_FIELD, args.resolution_deg)
    o_lat, o_lon, o_field = regrid(ocn,   OCN_FIELD,    args.resolution_deg)
    m_lat, m_lon, m_field = regrid(ocn,   "v_model",    args.resolution_deg)

    cropped, extent = crop_to_common([
        (g_lat, g_lon, g_field),
        (o_lat, o_lon, o_field),
        (m_lat, m_lon, m_field),
    ])
    fields = [c[2] for c in cropped]

    if args.vmax is not None:
        vmax = float(args.vmax)
    else:
        flat = np.concatenate([f[np.isfinite(f)].ravel() for f in fields])
        vmax = float(np.nanpercentile(np.abs(flat), 98))
    print(f"  shared vmax = ±{vmax:.3f} m/s   extent (lon,lat) = "
          f"({extent[0]:.3f},{extent[1]:.3f}) ({extent[2]:.3f},{extent[3]:.3f})")

    panels = [
        ("Custom pipeline",    fields[0]),
        ("Sentinel-1 OCN",     fields[1]),
        ("GLO12 model",        fields[2]),
    ]
    os.makedirs(os.path.dirname(out_svg) or ".", exist_ok=True)
    plot_poster(panels, out_svg, vmax=vmax, extent=extent)


if __name__ == "__main__":
    main()
