#!/usr/bin/env python3
"""Drifter-vs-pipelines scatter plot.

Reads the validation CSV produced by run_drifter_validation_all.py and overlays
three series on one axes with a 1:1 identity line:

    x = v_los_drift           (CMEMS drifter ground truth, projected on LOS)
    y = v_los_s1              (Custom pipeline)
    y = v_los_ocn_product     (Sentinel-1 OCN product)
    y = v_los_glo12           (GLO12 model)

Each series shows N / bias / RMSE / Pearson r in the legend.  The output is
an SVG suitable for poster use.

Usage
-----
    python scripts/poster/make_scatter_figure.py
    python scripts/poster/make_scatter_figure.py --csv data/drifters/validation_results.csv
    python scripts/poster/make_scatter_figure.py --out plots/poster/scatter.svg --lim 0.8
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SERIES = [
    # Custom pipeline (with OCN antenna-mispointing): GAMMA mosaic-last +
    # simplified Mouche + descalloping + mispointing.  v_los_s1_ocn column
    # (= 05_full_mouche_last in the method sweep).
    {"col": "v_los_s1_ocn",     "label": "Custom pipeline", "color": "tab:blue"},
    # Pure OCN L2 product — ESA's rvlRadVel field sampled at drifter location.
    # No re-derivation by our pipeline; this is the operational ESA current.
    {"col": "v_los_ocn_native", "label": "Sentinel-1 OCN",  "color": "tab:orange"},
    {"col": "v_los_glo12",      "label": "GLO12 model",     "color": "tab:green"},
]


def series_stats(x: np.ndarray, y: np.ndarray) -> tuple[int, float, float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 2:
        return n, float("nan"), float("nan"), float("nan")
    diff = y[mask] - x[mask]
    bias = float(diff.mean())
    rmse = float(np.sqrt((diff ** 2).mean()))
    if np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        r = float("nan")
    else:
        r = float(np.corrcoef(x[mask], y[mask])[0, 1])
    return n, bias, rmse, r


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--csv", default="data/drifters/validation_results.csv",
                   help="Validation CSV with v_los_drift + at least v_los_s1, "
                        "v_los_ocn_product, v_los_glo12 columns.")
    p.add_argument("--out", default="plots/poster/scatter.svg",
                   help="Output SVG path.")
    p.add_argument("--lim", type=float, default=None,
                   help="Symmetric axis limit (m/s).  Default = max(|all values|) * 1.05.")
    p.add_argument("--title", default="Drifter validation",
                   help="Plot title (set to '' to suppress).")
    p.add_argument("--alpha", type=float, default=0.65,
                   help="Scatter point alpha (default 0.65).")
    p.add_argument("--marker-size", type=float, default=28.0,
                   help="Scatter point size (default 28).")
    p.add_argument("--debias", action="store_true",
                   help="Subtract each series' median residual before plotting, "
                        "so points sit on the 1:1 line.  Legend reports the "
                        "constant offset removed.")
    args = p.parse_args()

    if not Path(args.csv).exists():
        raise SystemExit(
            f"CSV not found: {args.csv}\n"
            "  Run scripts/validation/run_drifter_validation_all.py first."
        )

    df = pd.read_csv(args.csv)
    if "v_los_drift" not in df.columns:
        raise SystemExit("CSV missing required column 'v_los_drift'.")
    available = [s for s in SERIES if s["col"] in df.columns]
    missing = [s["col"] for s in SERIES if s["col"] not in df.columns]
    if missing:
        print(f"warning: CSV missing columns {missing}; plotting only {[s['col'] for s in available]}")
    if not available:
        raise SystemExit("None of the expected y-columns are in the CSV.")

    x_all = df["v_los_drift"].to_numpy(dtype=np.float64)
    y_all = [df[s["col"]].to_numpy(dtype=np.float64) for s in available]

    # Optional debias: subtract each series' median residual so the cloud
    # is centred on the 1:1 line.  Reported in the legend so it's honest.
    offsets = [0.0] * len(available)
    if args.debias:
        offsets = []
        for i, y in enumerate(y_all):
            resid = y - x_all
            off = float(np.nanmedian(resid[np.isfinite(resid)])) if np.isfinite(resid).any() else 0.0
            offsets.append(off)
            y_all[i] = y - off

    finite_vals = [v for arr in [x_all, *y_all] for v in arr[np.isfinite(arr)]]
    if args.lim is not None:
        lim = float(args.lim)
    else:
        lim = float(np.nanmax(np.abs(finite_vals))) * 1.05 if finite_vals else 1.0

    fig, ax = plt.subplots(figsize=(6.2, 6.2), constrained_layout=True)
    ax.plot([-lim, lim], [-lim, lim], color="gray", lw=1.0, ls="--",
            zorder=1, label="1:1")
    ax.axhline(0, color="0.8", lw=0.6, zorder=0)
    ax.axvline(0, color="0.8", lw=0.6, zorder=0)

    for s, y, off in zip(available, y_all, offsets):
        n, bias, rmse, r = series_stats(x_all, y)
        off_tag = f"  (−{off:+.3f})" if args.debias and off != 0.0 else ""
        legend = (f"{s['label']}{off_tag}  "
                  f"N={n}  bias={bias:+.3f}  RMSE={rmse:.3f}  r={r:+.2f}")
        ax.scatter(x_all, y, s=args.marker_size, color=s["color"],
                   alpha=args.alpha, edgecolors="none", label=legend, zorder=2)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Drifter radial velocity [m/s]")
    ax.set_ylabel("Pipeline radial velocity [m/s]")
    title = args.title
    if args.debias and title:
        title = f"{title}  (per-series median residual removed)"
    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9, handletextpad=0.5)
    ax.grid(True, alpha=0.25, lw=0.6)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, format="svg", bbox_inches="tight")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
