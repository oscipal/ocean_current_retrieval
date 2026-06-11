#!/usr/bin/env python3
"""Three-panel comparison of Doppler-centroid estimators.

Each estimator is run through the **visualisation pipeline config** —
identical to ``make_poster_figure.py``: ``wave_source='mouche'``,
``descallop_blocks=False``, ``add_demod_back='blend'``, smoothing 3×1 — so
the comparison mirrors what the poster's "Custom pipeline" panel uses but
with the Doppler-centroid estimator swapped out.

By default (``--field v_current``) the plot shows the **final radial-current
field** in m/s.  Pass ``--field f_dc`` to fall back to the raw measured
Doppler centroid in Hz (the earlier diagnostic mode).

Panels (left to right):

  1. ``lag1_gamma``    — current default: GAMMA ``doppler_2d_SLC`` with
                          lag-1 autocorrelation, blsz=256.
  2. ``fft_centroid``  — pure-Python FFT spectral-centroid estimator on the
                          deramped SLC at OCN's native 233×128 grid.
  3. **Sentinel-1 OCN** — ESA's operational L2 product (rvlDcObs as f_dc in
                          v-current mode; rvlDcObs directly in f_dc mode).

Heavy pipeline outputs are pickle-cached so re-runs only re-render.

Usage
-----
    python scripts/poster/make_dop_comparison.py                       # velocity (default)
    python scripts/poster/make_dop_comparison.py --field f_dc          # raw Hz comparison
    python scripts/poster/make_dop_comparison.py --no-cache
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
from scripts.sentinel_1.ocn_product import load_ocn_safe
from scripts.sentinel_1.pipeline import run_all_bursts, run_gamma_pipeline_from_safe


_OCN_FILL = 9.9692099683868690e+36

# Visualisation config — identical to make_poster_figure.py so this comparison
# mirrors the poster's "Custom pipeline" panel exactly (modulo estimator swap).
VIS_WAVE_SOURCE      = "mouche"
VIS_DESCALLOP_BLOCKS = False
VIS_ADD_DEMOD_BACK   = "blend"
VIS_GEOM_SOURCE_LAG1 = "gamma"          # what mosaic-last + lag1 expects
VIS_SMOOTH_AZ        = 3
VIS_SMOOTH_RG        = 1


def scene_paths(data_root: str, scene: str, subswath: str, pol: str) -> dict:
    s1 = os.path.join(data_root, "sentinel-1", scene)
    era = os.path.join(data_root, "era5_data", scene)
    safe = os.path.join(s1, "S1A_IW_SLC.SAFE")
    poe = sorted(glob.glob(os.path.join(s1, "S1A_OPER_AUX_POEORB_*.EOF")))
    if not poe:
        poe = sorted(glob.glob(
            os.path.join(data_root, "sentinel-1", "S1A_OPER_AUX_POEORB_*.EOF")))
    aux = sorted(glob.glob(
        os.path.join(data_root, "sentinel-1", "S1A_AUX_CAL_*.SAFE")))
    return {
        "slc_safe":     safe,
        "ocn_safe":     os.path.join(s1, "S1A_IW_OCN.SAFE"),
        "poeorb":       poe[0],
        "aux_cal":      aux[0],
        "era5_wind":    os.path.join(era, "era5_wind.nc"),
        "era5_wave":    os.path.join(era, "era5_wave.nc"),
        "glo12":        os.path.join(era, "glo12.nc"),
        "subswath":     subswath,
        "polarisation": pol,
        "_scene":       scene,
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


def run_pipeline(paths: dict, f_dc_method: str) -> dict:
    """Run the GAMMA pipeline using the *visualisation* config (matches
    make_poster_figure.py).  3×1 block smoothing applied at the end."""
    r = run_gamma_pipeline_from_safe(
        slc_safe=paths["slc_safe"], subswath=paths["subswath"],
        poeorb_path=paths["poeorb"], aux_cal_path=paths["aux_cal"],
        ocn_safe=paths["ocn_safe"], era5_wind=paths["era5_wind"],
        era5_wave=paths["era5_wave"], glo12=paths["glo12"],
        polarisation=paths["polarisation"],
        f_dc_method=f_dc_method,
        add_demod_back=VIS_ADD_DEMOD_BACK,
        wave_source=VIS_WAVE_SOURCE,
        descallop_blocks=VIS_DESCALLOP_BLOCKS,
        keep_products=True,
        products_dir=os.path.join("data", "sentinel-1",
                                  f"gamma_{paths['subswath'].lower()}"),
    )
    return smooth_block_grid(r, smooth_az=VIS_SMOOTH_AZ, smooth_rg=VIS_SMOOTH_RG)


def run_ocn_product(paths: dict) -> list[dict]:
    """OCN-product current = our pipeline driven by rvlDcObs as f_dc.  Same
    quantity make_poster_figure.py's middle panel shows."""
    return run_all_bursts(
        slc_safe=paths["slc_safe"], subswath=paths["subswath"],
        poeorb_path=paths["poeorb"], aux_cal_path=paths["aux_cal"],
        ocn_safe=paths["ocn_safe"], era5_wind=paths["era5_wind"],
        era5_wave=paths["era5_wave"], glo12=paths["glo12"],
        polarisation=paths["polarisation"],
        use_ocn_dc=True,
    )


def load_ocn_dc_obs(paths: dict) -> dict:
    """Read OCN ``rvlDcObs`` for the given subswath + co-located lat/lon."""
    sw_idx = {"iw1": 0, "iw2": 1, "iw3": 2}[paths["subswath"].lower()]
    ocn = load_ocn_safe(paths["ocn_safe"],
                        swath=paths["subswath"], polarisation=paths["polarisation"])
    ds = ocn["rvl"]
    if "rvlSwath" in ds.dims:
        ds = ds.isel(rvlSwath=sw_idx)

    def _clean(name):
        arr = ds[name].values.astype(np.float64)
        arr[arr == _OCN_FILL] = np.nan
        return arr

    return {
        "lat":     _clean("rvlLat"),
        "lon":     _clean("rvlLon"),
        "dc_obs":  _clean("rvlDcObs"),
    }


def regrid_field(lat: np.ndarray, lon: np.ndarray, field: np.ndarray,
                 resolution_deg: float) -> tuple:
    """Reuse our existing merge helper to interpolate a single 2D grid onto
    a regular lat/lon mesh."""
    return merge_burst_grids(
        [{"lat": lat, "lon": lon, "field": field}],
        variable="field", overlap="average",
        resolution_deg=resolution_deg,
    )


def crop_to_common(grids, pad_deg: float = 0.0):
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


def plot_panels(panels, out_path: str, vmax: float, extent: tuple,
                cbar_label: str = "Doppler centroid [Hz]",
                stat_unit: str = "Hz",
                cmap: str = "RdBu_r") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6),
                             gridspec_kw={"width_ratios": [1, 1, 1]},
                             constrained_layout=True)
    fmt = "{:+.3f}" if stat_unit == "m/s" else "{:+.0f}"
    for ax, (title, field, stats) in zip(axes, panels):
        im = ax.imshow(field, extent=extent, origin="lower",
                       cmap=cmap, vmin=-vmax, vmax=vmax,
                       aspect="auto", interpolation="nearest")
        ax.set_title(
            f"{title}\nstd={stats[0]:.3f} {stat_unit}   "
            f"range=[{fmt.format(stats[1])}, {fmt.format(stats[2])}] {stat_unit}",
            fontsize=10,
        )
        ax.set_xlabel("Longitude [deg]")
    axes[0].set_ylabel("Latitude [deg]")
    cb = fig.colorbar(im, ax=axes.tolist(), shrink=0.92, pad=0.015, label=cbar_label)
    cb.outline.set_visible(False)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    print(f"saved -> {out_path}")


def _stats(field: np.ndarray) -> tuple[float, float, float]:
    f = field[np.isfinite(field)]
    if not f.size:
        return float("nan"), float("nan"), float("nan")
    return float(f.std()), float(f.min()), float(f.max())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", default="data")
    p.add_argument("--scene", default="scene1")
    p.add_argument("--subswath", default="iw1")
    p.add_argument("--pol", default="vv")
    p.add_argument("--resolution-deg", type=float, default=0.01)
    p.add_argument("--field", choices=("v_current", "f_dc"), default="v_current",
                   help="What to plot.  'v_current' (default) = final radial-current field "
                        "from the visualisation config, matches make_poster_figure.py.  "
                        "'f_dc' = raw measured Doppler centroid in Hz (diagnostic).")
    p.add_argument("--out", default=None,
                   help="Output path (default plots/poster/dop_comparison_<field>_<scene>_<sw>.svg).")
    p.add_argument("--cache-dir", default="plots/poster/cache_dop")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--vmax", type=float, default=None,
                   help="Symmetric colour scale.  Default = 98th-pctile of |all|.")
    args = p.parse_args()

    paths = scene_paths(args.data_root, args.scene, args.subswath, args.pol)
    out_svg = args.out or f"plots/poster/dop_comparison_{args.field}_{args.scene}_{args.subswath}.svg"
    use_cache = not args.no_cache
    tag = f"{args.scene}_{args.subswath}_{args.pol}"

    print(f"=== Doppler-estimator comparison ({args.field}): {tag} ===")

    if args.field == "v_current":
        # Each custom panel runs the full pipeline with the visualisation config.
        # Cache key includes 'v' so we don't collide with the f_dc-mode caches
        # (those were computed with the diagnostic config, not the visual one).
        lag1 = cached(
            os.path.join(args.cache_dir, f"v_lag1_{tag}.pkl"),
            lambda: run_pipeline(paths, "lag1_gamma"), use_cache,
        )
        fft = cached(
            os.path.join(args.cache_dir, f"v_fft_{tag}.pkl"),
            lambda: run_pipeline(paths, "fft_centroid"), use_cache,
        )
        # OCN-product baseline = our pipeline using rvlDcObs as f_dc.  Same
        # quantity make_poster_figure.py's middle panel shows.
        ocn_bursts = cached(
            os.path.join(args.cache_dir, f"v_ocn_product_{tag}.pkl"),
            lambda: run_ocn_product(paths), use_cache,
        )

        print("regridding all three onto a common lat/lon mesh ...")
        from scripts.sentinel_1.grid_merge import merge_burst_grids as _mb
        # All panels: v_current_ocn (= v_current + OCN antenna-mispointing
        # correction).  Apples-to-apples — every panel includes mispointing.
        g_lag1 = regrid_field(lag1["lat"], lag1["lon"],
                              np.asarray(lag1["v_current_ocn"], dtype=np.float32),
                              args.resolution_deg)
        g_fft  = regrid_field(fft["lat"],  fft["lon"],
                              np.asarray(fft["v_current_ocn"],  dtype=np.float32),
                              args.resolution_deg)
        g_ocn_lat, g_ocn_lon, g_ocn_field = _mb(
            ocn_bursts, variable="v_current_ocn",
            overlap="average", resolution_deg=args.resolution_deg,
        )
        g_ocn = (g_ocn_lat, g_ocn_lon, g_ocn_field)
        units, cbar_label = "m/s", "radial current [m/s]"
        panel_titles = [
            "lag1_gamma · viz config\n(GAMMA doppler_2d_SLC, with mispointing)",
            "fft_centroid · viz config\n(OCN-grid, 64×168 cells, with mispointing)",
            "Sentinel-1 OCN\n(rvlDcObs through our pipeline, with mispointing)",
        ]
        stat_unit = "m/s"

    else:  # f_dc mode (original diagnostic)
        lag1 = cached(
            os.path.join(args.cache_dir, f"lag1_{tag}.pkl"),
            lambda: run_pipeline(paths, "lag1_gamma"), use_cache,
        )
        fft = cached(
            os.path.join(args.cache_dir, f"fft_{tag}.pkl"),
            lambda: run_pipeline(paths, "fft_centroid"), use_cache,
        )
        ocn = cached(
            os.path.join(args.cache_dir, f"ocn_dc_{tag}.pkl"),
            lambda: load_ocn_dc_obs(paths), use_cache,
        )

        print("regridding all three onto a common lat/lon mesh ...")
        g_lag1 = regrid_field(lag1["lat"], lag1["lon"],
                              np.asarray(lag1["f_dc"], dtype=np.float32),
                              args.resolution_deg)
        g_fft  = regrid_field(fft["lat"],  fft["lon"],
                              np.asarray(fft["f_dc"],  dtype=np.float32),
                              args.resolution_deg)
        g_ocn  = regrid_field(ocn["lat"],  ocn["lon"],
                              np.asarray(ocn["dc_obs"], dtype=np.float32),
                              args.resolution_deg)
        units, cbar_label = "Hz", "Doppler centroid [Hz]"
        panel_titles = [
            "lag1_gamma\n(GAMMA doppler_2d_SLC, blsz=256)",
            "fft_centroid\n(OCN-grid, 64×168 cells)",
            "OCN rvlDcObs\n(reference)",
        ]
        stat_unit = "Hz"

    cropped, extent = crop_to_common([g_lag1, g_fft, g_ocn])
    fields = [c[2] for c in cropped]
    stats = [_stats(f) for f in fields]

    if args.vmax is not None:
        vmax = float(args.vmax)
    else:
        flat = np.concatenate([f[np.isfinite(f)].ravel() for f in fields])
        vmax = float(np.nanpercentile(np.abs(flat), 98))

    print(f"  vmax = ±{vmax:.3f} {units}   extent (lon,lat) "
          f"({extent[0]:.3f},{extent[1]:.3f}) ({extent[2]:.3f},{extent[3]:.3f})")
    labels = ("lag1_gamma", "fft_centroid", "OCN")
    for label, (sd, mn, mx) in zip(labels, stats):
        print(f"  {label:13s}  std={sd:7.3f}  range=[{mn:+8.3f}, {mx:+8.3f}] {stat_unit}")

    panels = [(t, f, s) for t, f, s in zip(panel_titles, fields, stats)]
    plot_panels(panels, out_svg, vmax=vmax, extent=extent, cbar_label=cbar_label,
                stat_unit=stat_unit)


if __name__ == "__main__":
    main()
