#!/usr/bin/env python3
"""Unwrapped TOPS Doppler grid — our lag-1 estimator on the *raw* (non-deramped) SLC.

Estimates the Doppler centroid the way our pipeline does (lag-1 azimuth
autocorrelation, ``f = PRF/2π · arg⟨S_{n+1} S_n*⟩``) but on the **raw,
non-deramped** burst SLC instead of the deramped one.

On the raw SLC the TOPS antenna sweep makes the Doppler centroid ramp by
~5 kHz (≈11×PRF) across each burst, so a direct lag-1 estimate folds into a
sawtooth confined to [−PRF/2, +PRF/2].  Because the ramp is only a few Hz per
azimuth *line* (well below PRF/2), a fine per-line estimate can be
``np.unwrap``-ed back into the **full ±kHz swing**.

The grid is shown with **y = azimuth (burst time / line)** and **x = range
(sample)**, colour = unwrapped Doppler in kHz.  Each burst is unwrapped and
centred independently — the ~11×PRF jump at a burst boundary cannot be
unwrapped, and TOPS resets the ramp every burst anyway — so the picture is a
stack of identical fore→aft ramps, one per burst.

Usage
-----
    python scripts/diagnostics/doppler_unwrapped_grid.py
    python scripts/diagnostics/doppler_unwrapped_grid.py --burst 4          # single burst
    python scripts/diagnostics/doppler_unwrapped_grid.py --az-block 5 --rg-block 200
    python scripts/diagnostics/doppler_unwrapped_grid.py --subswath iw2 --no-center
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnostics.burst_spectrogram import _default_deramp_slc, _read_deramp_burst
from scripts.sentinel_1.rvl import _deramp_rate
from scripts.sentinel_1.safe_io import find_safe_files, parse_annotation, read_slc_burst


def block_reduce_complex(arr: np.ndarray, az_block: int, rg_block: int) -> np.ndarray:
    """Sum a complex array over non-overlapping (az_block × rg_block) tiles.

    Summing the lag-1 product ⟨S_{n+1} S_n*⟩ over a tile before taking the
    angle is exactly the multilooked lag-1 Doppler estimate (same as GAMMA's
    block estimator, just at our chosen block size)."""
    n_az, n_rg = arr.shape
    n_az -= n_az % az_block
    n_rg -= n_rg % rg_block
    arr = arr[:n_az, :n_rg]
    return arr.reshape(n_az // az_block, az_block,
                       n_rg // rg_block, rg_block).sum(axis=(1, 3))


def burst_lag1_doppler(burst: np.ndarray, prf_az: float,
                       az_block: int, rg_block: int,
                       do_unwrap: bool, center: bool) -> np.ndarray:
    """Lag-1 azimuth Doppler for one burst → Hz.

    ``do_unwrap`` unwraps along azimuth (needed pre-deramp, where the TOPS
    ramp folds the centroid past ±PRF/2).  Post-deramp the residual signal is
    narrowband, so unwrap and centring are off.
    """
    # Lag-1 product along azimuth, then multilook into blocks.
    prod = burst[1:, :] * np.conj(burst[:-1, :])          # (LPB-1, n_rg)
    blk = block_reduce_complex(prod, az_block, rg_block)  # (n_az_blk, n_rg_blk)
    phase = np.angle(blk)                                  # wrapped to (-π, π]
    if do_unwrap:
        phase = np.unwrap(phase, axis=0)                   # along azimuth
    f_hz = prf_az / (2.0 * np.pi) * phase                  # → Hz
    if center:
        f_hz = f_hz - f_hz.mean(axis=0, keepdims=True)     # centre each column on 0
    return f_hz


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--safe", type=Path,
                   default=Path("data/sentinel-1/scene1/S1A_IW_SLC.SAFE"))
    p.add_argument("--subswath", default="iw1", choices=("iw1", "iw2", "iw3"))
    p.add_argument("--pol", default="vv", choices=("vv", "vh"))
    p.add_argument("--burst", type=int, default=None,
                   help="Single burst index (0-based).  Default: all bursts stacked.")
    p.add_argument("--az-block", type=int, default=5,
                   help="Azimuth multilook block in lines (default 5; must keep "
                        "per-block Doppler step < PRF/2 so unwrap works).")
    p.add_argument("--rg-block", type=int, default=200,
                   help="Range multilook block in samples (default 200).")
    p.add_argument("--post-deramp", action="store_true",
                   help="Estimate on the GAMMA-deramped SLC instead of the raw "
                        "burst.  Residual signal is narrowband, so no unwrap / no "
                        "centring; plotted in Hz.")
    p.add_argument("--deramp-slc", type=Path, default=None,
                   help="GAMMA .deramp.slc path (post-deramp mode).  Default "
                        "inferred from the SAFE name.")
    p.add_argument("--no-center", action="store_true",
                   help="Do NOT subtract each burst's per-column mean (keep the "
                        "absolute folded-then-unwrapped level).  Pre-deramp only.")
    p.add_argument("--vmax", type=float, default=None,
                   help="Symmetric colour scale (kHz pre-deramp, Hz post-deramp).")
    p.add_argument("--cmap", default="RdBu_r")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    files = find_safe_files(str(args.safe), args.subswath, args.pol)
    annot = parse_annotation(files["annotation"])
    n_bursts = len(annot.bursts)
    LPB = annot.lines_per_burst
    ati = annot.azimuth_time_interval
    prf_az = 1.0 / ati                       # azimuth sampling rate (line rate)
    T_burst = LPB * ati

    burst_list = ([args.burst] if args.burst is not None
                  else list(range(n_bursts)))

    # Expected swing from the steering rate (mid-range).
    k_s = _deramp_rate(annot, burst_list[len(burst_list) // 2])
    k_s_mid = float(k_s[len(k_s) // 2])
    swing_hz = abs(k_s_mid) * T_burst

    print(f"SAFE:        {args.safe}")
    print(f"Subswath:    {args.subswath}  pol={args.pol}  bursts={burst_list}")
    print(f"PRF (line rate) = {prf_az:.2f} Hz   T_burst = {T_burst:.3f} s")
    if not args.post_deramp:
        print(f"Steering rate k_s (mid-rg) = {k_s_mid:+.1f} Hz/s")
        print(f"Expected chirp swing       = {swing_hz:.0f} Hz "
              f"= {swing_hz/prf_az:.2f} x PRF = {swing_hz/1000:.2f} kHz")
        print(f"Doppler step per az-block ({args.az_block} lines) = "
              f"{abs(k_s_mid)*args.az_block*ati:.1f} Hz  (must be < PRF/2 = {prf_az/2:.0f})")

    # Post-deramp mode reads the GAMMA-deramped SLC; resolve its path once.
    deramp_slc = par_path = None
    if args.post_deramp:
        deramp_slc = args.deramp_slc or _default_deramp_slc(args.subswath, args.pol, args.safe)
        par_path = Path(str(deramp_slc) + ".par")
        if not deramp_slc.exists() or not par_path.exists():
            raise SystemExit(
                f"Deramped SLC not found: {deramp_slc}\n"
                f"  Run gamma_prep_scene first or pass --deramp-slc."
            )
        print(f"Deramp SLC:  {deramp_slc}")

    do_unwrap = not args.post_deramp
    center = (not args.no_center) and (not args.post_deramp)

    cols = []
    az_centres_lines = []
    for bi in burst_list:
        if args.post_deramp:
            burst = _read_deramp_burst(deramp_slc, par_path, bi, LPB)
        else:
            burst = read_slc_burst(files["measurement"], annot, bi)
        f_hz = burst_lag1_doppler(burst, prf_az, args.az_block, args.rg_block,
                                  do_unwrap=do_unwrap, center=center)
        cols.append(f_hz)
        n_az_blk = f_hz.shape[0]
        base = bi * LPB
        az_centres_lines.append(base + (np.arange(n_az_blk) + 0.5) * args.az_block)
        print(f"  burst {bi:2d}: grid {f_hz.shape}  "
              f"swing(meas) = {f_hz.max()-f_hz.min():.0f} Hz")

    grid = np.vstack(cols)                                 # (sum_az_blk, n_rg_blk)
    az_lines = np.concatenate(az_centres_lines)

    # Pre-deramp spans ±kHz; post-deramp residual is narrowband, plot in Hz.
    if args.post_deramp:
        scale, unit, title = 1.0, "Hz", "Doppler (post-deramping)"
    else:
        scale, unit, title = 1000.0, "kHz", "Unwrapped Doppler (pre-deramping)"
    grid_disp = grid / scale

    finite = grid_disp[np.isfinite(grid_disp)]
    vmax = args.vmax if args.vmax is not None else float(np.nanpercentile(np.abs(finite), 99))

    n_rg_blk = grid.shape[1]
    rg_centres = (np.arange(n_rg_blk) + 0.5) * args.rg_block
    extent = (float(rg_centres[0]), float(rg_centres[-1]),
              float(az_lines[0]), float(az_lines[-1]))

    print(f"  full grid {grid.shape}  measured swing = "
          f"{finite.max()-finite.min():.2f} {unit}   vmax=±{vmax:.2f} {unit}")

    burst_tag = f"_b{args.burst}" if args.burst is not None else ""
    mode_tag = "postderamp" if args.post_deramp else "unwrapped"
    out = args.out or (f"plots/diagnostics/doppler_{mode_tag}_"
                       f"{args.subswath}_{args.pol}{burst_tag}.png")

    fig, ax = plt.subplots(figsize=(6.6, 7.2), constrained_layout=True)
    im = ax.imshow(grid_disp, extent=extent, origin="lower", cmap=args.cmap,
                   vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
    # Mark burst boundaries.
    if args.burst is None:
        for bi in range(1, n_bursts):
            ax.axhline(bi * LPB, color="0.4", lw=0.5, ls="--", alpha=0.5)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Range [sample]")
    ax.set_ylabel("Azimuth [line]")
    cb = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.02,
                      label=f"Doppler centroid [{unit}]")
    cb.outline.set_visible(False)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
