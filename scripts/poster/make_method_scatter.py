#!/usr/bin/env python3
"""Small-multiples drifter-vs-pipeline scatter, one panel per method.

Reads the long-format CSV produced by ``run_method_sweep.py`` and renders one
scatter per ``method`` value with the drifter on x, the pipeline on y, a 1:1
identity line, and the (N, bias, RMSE, Pearson r) printed in the panel title.

Methods are laid out on a grid in the order encoded by their numeric prefix
(01_, 02_, ...), with ``ocn_product`` and ``glo12`` appended at the end as the
two reference baselines.

Usage
-----
    python scripts/poster/make_method_scatter.py
    python scripts/poster/make_method_scatter.py --csv data/drifters/method_sweep.csv
    python scripts/poster/make_method_scatter.py --out plots/poster/method_scatter.svg
    python scripts/poster/make_method_scatter.py --only-mosaic last
    python scripts/poster/make_method_scatter.py --lim 0.8 --ncols 4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASELINE_METHODS = ("ocn_product", "glo12")


def _sort_key(method: str) -> tuple:
    if method == "ocn_product":
        return (98, method)
    if method == "glo12":
        return (99, method)
    try:
        return (int(method[:2]), method)
    except ValueError:
        return (50, method)


def _series_stats(x: np.ndarray, y: np.ndarray) -> tuple[int, float, float, float]:
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


def _pretty_title(method: str) -> str:
    if method == "ocn_product":
        return "Sentinel-1 OCN product"
    if method == "glo12":
        return "GLO12 model"
    # Strip the "NN_" prefix for the heading
    body = method[3:] if len(method) > 3 and method[2] == "_" else method
    return body.replace("_", " ")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--csv", default="data/drifters/method_sweep.csv",
                   help="Long-format CSV from run_method_sweep.py.")
    p.add_argument("--out", default="plots/poster/method_scatter.svg",
                   help="Output SVG path.")
    p.add_argument("--lim", type=float, default=None,
                   help="Symmetric axis limit (m/s). Default = max(|all|) * 1.05.")
    p.add_argument("--ncols", type=int, default=3,
                   help="Number of subplot columns (default 3).")
    p.add_argument("--alpha", type=float, default=0.65)
    p.add_argument("--marker-size", type=float, default=22.0)
    p.add_argument("--only-mosaic", choices=("last", "first"), default=None,
                   help="Restrict to one mosaic mode (drops methods tagged with the other).")
    p.add_argument("--only-stage", default=None,
                   help="Restrict to one stage prefix, e.g. '05' for the full pipelines.")
    p.add_argument("--debias", action="store_true",
                   help="Subtract each method's median residual before plotting.")
    args = p.parse_args()

    if not Path(args.csv).exists():
        raise SystemExit(
            f"CSV not found: {args.csv}\n"
            "  Run scripts/validation/run_method_sweep.py first."
        )

    df = pd.read_csv(args.csv)
    required = {"method", "v_los_drift", "v_los_pipe"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing required columns: {sorted(missing)}")

    methods = sorted(df["method"].unique().tolist(), key=_sort_key)
    if args.only_mosaic is not None:
        kept = []
        for m in methods:
            if m in BASELINE_METHODS:
                kept.append(m)
            elif m.endswith(f"_{args.only_mosaic}"):
                kept.append(m)
        methods = kept
    if args.only_stage is not None:
        prefix = args.only_stage.rstrip("_")
        methods = [m for m in methods
                   if m in BASELINE_METHODS or m.startswith(f"{prefix}_")]

    if not methods:
        raise SystemExit("No methods remain after filtering.")

    finite_vals = np.concatenate([
        df["v_los_drift"].to_numpy(dtype=np.float64),
        df["v_los_pipe"].to_numpy(dtype=np.float64),
    ])
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    if args.lim is not None:
        lim = float(args.lim)
    else:
        lim = float(np.max(np.abs(finite_vals))) * 1.05 if len(finite_vals) else 1.0

    ncols = max(1, int(args.ncols))
    nrows = (len(methods) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.2 * ncols, 3.2 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for i, method in enumerate(methods):
        ax = axes_flat[i]
        sub = df[df["method"] == method]
        x = sub["v_los_drift"].to_numpy(dtype=np.float64)
        y = sub["v_los_pipe"].to_numpy(dtype=np.float64)
        off_label = ""
        if args.debias:
            resid = y - x
            mask = np.isfinite(resid)
            off = float(np.nanmedian(resid[mask])) if mask.any() else 0.0
            y = y - off
            if off != 0.0:
                off_label = f" (−{off:+.3f})"
        n, bias, rmse, r = _series_stats(x, y)
        ax.plot([-lim, lim], [-lim, lim], color="gray", ls="--", lw=0.8, zorder=1)
        ax.axhline(0, color="0.85", lw=0.5, zorder=0)
        ax.axvline(0, color="0.85", lw=0.5, zorder=0)
        ax.scatter(x, y, s=args.marker_size, alpha=args.alpha,
                   color="tab:blue", edgecolors="none", zorder=2)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"{_pretty_title(method)}{off_label}\n"
            f"N={n}  bias={bias:+.3f}  RMSE={rmse:.3f}  r={r:+.2f}",
            fontsize=9,
        )
        if i % ncols == 0:
            ax.set_ylabel("pipeline [m/s]", fontsize=8)
        if i // ncols == nrows - 1:
            ax.set_xlabel("drifter [m/s]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2, lw=0.4)

    # Hide unused subplots
    for j in range(len(methods), nrows * ncols):
        axes_flat[j].set_visible(False)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, format="svg", bbox_inches="tight")
    print(f"saved -> {args.out}  ({len(methods)} method panel(s), N_rows={len(df)})")


if __name__ == "__main__":
    main()
