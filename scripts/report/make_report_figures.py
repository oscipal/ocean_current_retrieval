#!/usr/bin/env python3
"""Generate every figure used in the project report into notes/files/figures/.

One self-contained run produces the method figures (one per processing step),
the stage-by-stage results figure, the incremental-impact bar chart, the
drifter scatter, and the four-way comparison map.  Convention requested for
the report:

  * instrument-correction steps are shown in the **Doppler domain (Hz)**;
  * everything from the radial-velocity conversion onward is shown in
    **velocity (m/s)**.

Data sources (all already on disk, nothing re-run):
  * scene1 IW1 pipeline result  -> plots/poster/cache/gamma_...current.pkl
  * hybrid (OCN f_dc) bursts     -> plots/poster/cache/ocn_scene1_iw1_vv.pkl
  * pure OCN rvlRadVel field     -> plots/poster/cache/ocn_native_scene1_iw1_vv.pkl
  * drifter validation           -> data/drifters/validation_results.csv
  * incremental method sweep     -> data/drifters/method_sweep.csv

Usage
-----
    python scripts/report/make_report_figures.py
    python scripts/report/make_report_figures.py --only stagewise scatter
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.sentinel_1.grid_merge import merge_burst_grids

OUT_DIR = REPO_ROOT / "notes" / "files" / "figures"
CACHE = REPO_ROOT / "plots" / "poster" / "cache"
DRIFT = REPO_ROOT / "data" / "drifters"

CUR_PKL = CACHE / "gamma_scene1_iw1_vv_blsz256_blend_geom-gamma_wave-mouche_desc_smooth1x1_current.pkl"
HYB_PKL = CACHE / "ocn_scene1_iw1_vv.pkl"
OCN_PKL = CACHE / "ocn_native_scene1_iw1_vv.pkl"

LAMBDA = 0.055504158146633564          # C-band wavelength [m]
HZ2MS = LAMBDA / 2.0                    # Hz -> m/s for radial velocity
CMAP = "RdBu_r"


# ──────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────
def load(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)


def save(fig, name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out.relative_to(REPO_ROOT)}")


def pmesh(ax, field, lat, lon, vmax, label, title, cmap=CMAP, vmin=None):
    """Plot a native-grid field with true 2-D lat/lon (orientation-safe)."""
    vmin = -vmax if vmin is None else vmin
    im = ax.pcolormesh(lon, lat, field, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Longitude [deg]")
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.9, pad=0.02, label=label)
    cb.outline.set_visible(False)
    return im


def imnative(ax, field, vmax, label, title, cmap=CMAP, vmin=None):
    """Plot a native-grid field on range (x) / azimuth (y) block axes."""
    vmin = -vmax if vmin is None else vmin
    im = ax.imshow(field, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Range [block]")
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.9, pad=0.02, label=label)
    cb.outline.set_visible(False)
    return im


def pct(field, q=98):
    f = field[np.isfinite(field)]
    return float(np.nanpercentile(np.abs(f), q)) if f.size else 1.0


def regrid(result_or_list, var, res=0.01):
    rs = result_or_list if isinstance(result_or_list, list) else [result_or_list]
    return merge_burst_grids(rs, variable=var, overlap="average", resolution_deg=res)


def stats(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    d = y[m] - x[m]
    r = float(np.corrcoef(x[m], y[m])[0, 1]) if n > 1 else np.nan
    return n, float(d.mean()), float(np.sqrt((d ** 2).mean())), r


# ──────────────────────────────────────────────────────────────────────────
# method figures
# ──────────────────────────────────────────────────────────────────────────
def fig_per_burst_doppler(R):
    """Measured Doppler centroid on the native az/range grid, burst lines marked."""
    f_dc = R["f_dc"]
    n_az = f_dc.shape[0]
    fig, ax = plt.subplots(figsize=(5.6, 6.4), constrained_layout=True)
    vmax = pct(f_dc, 99)
    im = ax.imshow(f_dc, origin="lower", cmap="viridis", aspect="auto",
                   vmin=float(np.nanpercentile(f_dc, 2)),
                   vmax=float(np.nanpercentile(f_dc, 98)),
                   interpolation="nearest")
    # 10 bursts -> boundary every n_az/10 blocks.
    for b in range(1, 10):
        ax.axhline(b * n_az / 10.0, color="w", lw=0.6, ls="--", alpha=0.6)
    ax.set_xlabel("Range [block]")
    ax.set_ylabel("Azimuth [block] (10 bursts)")
    ax.set_title("Measured Doppler centroid per burst", fontsize=11)
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02, label="Doppler centroid [Hz]")
    cb.outline.set_visible(False)
    save(fig, "per_burst_doppler.png")


def fig_ocn_dcobs():
    """Observed Doppler centroid (rvlDcObs) from the ocean product, on its
    own geolocated lat/lon grid."""
    hyb = load(HYB_PKL)
    r = hyb[0] if isinstance(hyb, list) else hyb
    # Regrid rvlDcObs onto a regular lat/lon mesh (the product lat/lon carry
    # NaN fills, which pcolormesh cannot use as coordinates).
    glat, glon, f = regrid(r, "f_dc")
    cmap, vmin, vmax = _panel_scale(f)
    fig, ax = plt.subplots(figsize=(5.8, 6.2), constrained_layout=True)
    im = ax.pcolormesh(glon, glat, f, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading="auto")
    ax.set_title("Observed Doppler centroid (ocean product)", fontsize=11)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02,
                      label="Doppler centroid [Hz]")
    cb.outline.set_visible(False)
    save(fig, "ocn_dcobs.png")


def _panel_scale(f):
    """Pick a colour scale per field: sequential for one-signed fields,
    diverging and symmetric for signed fields."""
    ff = f[np.isfinite(f)]
    lo = float(np.nanpercentile(ff, 2))
    hi = float(np.nanpercentile(ff, 98))
    if lo >= 0 or hi <= 0:                     # one sign -> sequential
        return "viridis", lo, hi
    m = max(abs(lo), abs(hi))                  # signed -> diverging, symmetric
    return CMAP, -m, m


def _two_panel(corr, result, corr_title, result_title, unit, out, ylabel=True):
    """Left: the correction applied at this step.  Right: the pipeline field
    after the step.  Native range/azimuth axes; fields come straight from the
    pipeline result."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.6), constrained_layout=True)
    for ax, f, title in [(axes[0], corr, corr_title),
                         (axes[1], result, result_title)]:
        cmap, vmin, vmax = _panel_scale(f)
        imnative(ax, f, vmax=vmax, vmin=vmin, label=unit, title=title, cmap=cmap)
    if ylabel:
        axes[0].set_ylabel("Azimuth [block]")
    save(fig, out)


def fig_geometry_subtraction(R):
    """Left: geometric Doppler correction.  Right: anomaly post geometry."""
    geom = R["f_geom_poe"]
    if geom.ndim == 1:
        geom = np.tile(geom[None, :], (R["f_dc"].shape[0], 1))
    _two_panel(geom, R["f_dca_pre_descallop"],
               "Geometric Doppler correction",
               "Doppler anomaly post geometry correction",
               "Hz", "geometry_subtraction.png")


def fig_descalloping(R):
    """Left: scalloping correction.  Right: anomaly post descalloping."""
    removed = R["f_dca_pre_descallop"] - R["f_dca"]
    _two_panel(removed, R["f_dca"],
               "Scalloping correction (removed ripple)",
               "Doppler anomaly post descalloping correction",
               "Hz", "descalloping.png")


def fig_mispointing(R):
    """Left: mispointing Doppler correction.  Right: anomaly post mispointing."""
    _two_panel(R["f_miss_ocn"], R["f_dca"] - R["f_miss_ocn"],
               "Mispointing Doppler correction",
               "Doppler anomaly post mispointing correction",
               "Hz", "mispointing.png")


def fig_geophysical_debias(R):
    """Left: wave and Stokes velocity correction.  Right: current post geophysical."""
    corr = R["v_wave"] + R["v_stokes"]
    _two_panel(corr, R["v_current_ocn"],
               "Wave and Stokes velocity correction",
               "Surface current post geophysical correction",
               "m/s", "geophysical_debias.png")


def fig_current_map(R):
    glat, glon, gfield = regrid(R, "v_current_ocn")
    fig, ax = plt.subplots(figsize=(5.8, 6.2), constrained_layout=True)
    vmax = pct(gfield, 98)
    im = ax.pcolormesh(glon, glat, gfield, cmap=CMAP, vmin=-vmax, vmax=vmax,
                       shading="auto")
    ax.set_title("Gridded surface radial current", fontsize=11)
    ax.set_xlabel("Longitude [deg]"); ax.set_ylabel("Latitude [deg]")
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02, label="radial current [m/s]")
    cb.outline.set_visible(False)
    save(fig, "current_map.png")


# ──────────────────────────────────────────────────────────────────────────
# results figures
# ──────────────────────────────────────────────────────────────────────────
def fig_stagewise(R):
    """Two rows: instrument corrections (Doppler, Hz) then velocity (m/s).

    Panels (a)-(e) are native range/azimuth fields; only the final gridded
    current (f) is on a lat/lon mesh.
    """
    # Doppler-domain cumulative anomalies.
    a_geom = R["f_dca_pre_descallop"]                 # post geometry + sideband
    a_desc = R["f_dca"]                               # post descalloping
    a_miss = R["f_dca"] - R["f_miss_ocn"]             # post mispointing
    vmax_hz = max(pct(a_geom, 98), pct(a_desc, 98), pct(a_miss, 98))
    # Velocity-domain.
    u_r = -R["v_r"] + R["v_miss_ocn"]                 # instrument-corrected radial vel
    u_cur = R["v_current_ocn"]                        # post geophysical
    glat, glon, u_grid = regrid(R, "v_current_ocn")
    vmax_ms = max(pct(u_r, 98), pct(u_cur, 98), pct(u_grid, 98))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5), constrained_layout=True)
    imnative(axes[0, 0], a_geom, vmax_hz, "Hz", "(a) Anomaly post geometry and sideband correction")
    imnative(axes[0, 1], a_desc, vmax_hz, "Hz", "(b) Anomaly post descalloping correction")
    imnative(axes[0, 2], a_miss, vmax_hz, "Hz", "(c) Anomaly post mispointing correction")
    imnative(axes[1, 0], u_r, vmax_ms, "m/s", "(d) Radial velocity")
    imnative(axes[1, 1], u_cur, vmax_ms, "m/s", "(e) Current post geophysical correction")
    im = axes[1, 2].pcolormesh(glon, glat, u_grid, cmap=CMAP,
                               vmin=-vmax_ms, vmax=vmax_ms, shading="auto")
    axes[1, 2].set_title("(f) Gridded surface current", fontsize=10)
    axes[1, 2].set_xlabel("Longitude [deg]")
    axes[1, 2].set_ylabel("Latitude [deg]")
    fig.colorbar(im, ax=axes[1, 2], shrink=0.9, pad=0.02, label="m/s")
    axes[0, 0].set_ylabel("Azimuth [block]")
    axes[1, 0].set_ylabel("Azimuth [block]")
    save(fig, "results_stagewise.png")


SWEEP_CHAIN = [
    ("01_geom_last",              "Geometry\n+ sideband"),
    ("02_geom_stokes_last",       "+ Stokes"),
    ("03_stokes_mouche_last",     "+ Wave\n(Mouche)"),
    ("04_stokes_mouche_desc_last","+ Descallop"),
    ("05_full_mouche_last",       "+ Mispointing"),
]


def fig_incremental_rmse():
    d = pd.read_csv(DRIFT / "method_sweep.csv")
    labels, biases, rmses = [], [], []
    for key, lab in SWEEP_CHAIN:
        sub = d[d["method"] == key]
        res = (sub["v_los_pipe"] - sub["v_los_drift"]).to_numpy(float)
        res = res[np.isfinite(res)]
        labels.append(lab)
        biases.append(res.mean())
        rmses.append(np.sqrt((res ** 2).mean()))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    ax.bar(x - 0.2, rmses, 0.4, color="tab:red", label="RMSE")
    ax.bar(x + 0.2, np.abs(biases), 0.4, color="tab:blue", label="|bias|")
    for xi, (r_, b_) in enumerate(zip(rmses, biases)):
        ax.text(xi - 0.2, r_ + 0.01, f"{r_:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + 0.2, abs(b_) + 0.01, f"{b_:+.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Error vs drifters [m/s]")
    ax.set_title("Effect of each correction", fontsize=11)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25)
    save(fig, "results_incremental_rmse.png")


SCATTER_SERIES = [
    ("v_los_s1_ocn",      "Custom pipeline",              "tab:blue"),
    ("v_los_ocn_product", "Hybrid (ocean product Doppler)", "tab:purple"),
    ("v_los_ocn_native",  "Sentinel-1 ocean product",     "tab:orange"),
    ("v_los_glo12",       "GLO12 model",                  "tab:green"),
]


def fig_drifter_scatter():
    df = pd.read_csv(DRIFT / "validation_results.csv")
    x = df["v_los_drift"].to_numpy(float)
    lim = 1.6
    fig, ax = plt.subplots(figsize=(6.4, 6.4), constrained_layout=True)
    ax.plot([-lim, lim], [-lim, lim], "--", color="gray", lw=1.0, label="1:1")
    for col, lab, c in SCATTER_SERIES:
        if col not in df.columns:
            continue
        y = df[col].to_numpy(float)
        n, b, rm, r = stats(x, y)
        ax.scatter(x, y, s=26, color=c, alpha=0.65, edgecolors="none",
                   label=f"{lab}  N={n} bias={b:+.2f} RMSE={rm:.2f} r={r:+.2f}")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel("Drifter radial velocity [m/s]")
    ax.set_ylabel("Retrieved radial velocity [m/s]")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    save(fig, "results_drifter_scatter.png")


def fig_comparison_map(R):
    hyb = load(HYB_PKL)
    ocn = load(OCN_PKL)
    g_pipe = regrid(R, "v_current_ocn")
    g_hyb = regrid(hyb, "v_current_ocn")
    g_ocn = regrid(ocn, "field")
    g_glo = regrid(R, "v_model")
    grids = [g_pipe, g_hyb, g_ocn, g_glo]
    titles = ["Custom pipeline", "Hybrid",
              "Sentinel-1 ocean product", "GLO12"]
    # common extent
    lat_lo = max(g[0].min() for g in grids); lat_hi = min(g[0].max() for g in grids)
    lon_lo = max(g[1].min() for g in grids); lon_hi = min(g[1].max() for g in grids)
    flat = []
    for la, lo, fd in grids:
        flat.append(fd[np.isfinite(fd)])
    vmax = float(np.nanpercentile(np.abs(np.concatenate(flat)), 98))
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), constrained_layout=True)
    for ax, (la, lo, fd), t in zip(axes, grids, titles):
        im = ax.pcolormesh(lo, la, fd, cmap=CMAP, vmin=-vmax, vmax=vmax, shading="auto")
        ax.set_xlim(lon_lo, lon_hi); ax.set_ylim(lat_lo, lat_hi)
        ax.set_title(t, fontsize=10); ax.set_xlabel("Longitude [deg]")
    axes[0].set_ylabel("Latitude [deg]")
    cb = fig.colorbar(im, ax=axes.tolist(), shrink=0.9, pad=0.01, label="radial current [m/s]")
    cb.outline.set_visible(False)
    save(fig, "results_comparison_map.png")


def fig_flowchart():
    steps = [
        "Sentinel-1 IW SLC",
        "Precise orbit + TOPS deramping",
        "Doppler centroid per burst",
        "Geometry + sideband",
        "Descalloping",
        "Mispointing",
        "Radial velocity ($\\lambda/2$)",
        "Geophysical: Stokes + wave",
        "Combine bursts + grid",
    ]
    groups = ["in", "gen", "gen", "ins", "ins", "ins", "gen", "geo", "gen"]
    colors = {"in": "#dddddd", "gen": "#cfe8ff", "ins": "#ffe0b3", "geo": "#cdebc5"}
    fig, ax = plt.subplots(figsize=(4.6, 9.4), constrained_layout=True)
    ax.set_xlim(0, 1); ax.set_ylim(0, len(steps)); ax.axis("off")
    for i, (s, g) in enumerate(zip(steps, groups)):
        y = len(steps) - 1 - i
        ax.add_patch(plt.Rectangle((0.1, y + 0.12), 0.8, 0.72,
                                   facecolor=colors[g], edgecolor="k", lw=0.8))
        ax.text(0.5, y + 0.48, s, ha="center", va="center", fontsize=9.5)
        if i < len(steps) - 1:
            ax.annotate("", xy=(0.5, y - 0.04), xytext=(0.5, y + 0.12),
                        arrowprops=dict(arrowstyle="->", lw=1.0))
    save(fig, "pipeline_overview_flowchart.png")


# ──────────────────────────────────────────────────────────────────────────
FIGS = {
    "flowchart":   lambda R: fig_flowchart(),
    "perburst":    fig_per_burst_doppler,
    "ocndc":       lambda R: fig_ocn_dcobs(),
    "geometry":    fig_geometry_subtraction,
    "descallop":   fig_descalloping,
    "mispointing": fig_mispointing,
    "geophysical": fig_geophysical_debias,
    "current":     fig_current_map,
    "stagewise":   fig_stagewise,
    "incremental": lambda R: fig_incremental_rmse(),
    "scatter":     lambda R: fig_drifter_scatter(),
    "comparison":  fig_comparison_map,
}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", nargs="*", choices=list(FIGS), default=None)
    args = p.parse_args()
    R = load(CUR_PKL)
    todo = args.only or list(FIGS)
    for name in todo:
        print(f"\n### {name}")
        FIGS[name](R)
    print(f"\nAll figures in {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
