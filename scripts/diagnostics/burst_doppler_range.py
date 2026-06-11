#!/usr/bin/env python3
"""Pre/post-deramp Doppler-bandwidth comparison: kHz → PRF.

Shows the point your supervisor asked about — that in TOPS the *unwrapped*
Doppler swing inside one burst is on the order of several kHz (because the
antenna sweeps fore-to-aft), while after ``SLC_deramp_ScanSAR`` the residual
signal bandwidth is in the range of PRF.

Two panels with very different y-axis scales:

  * **Left (pre-deramp, schematic):** the true unwrapped chirp trajectory
    ``f(τ) = k_s · (τ − T_burst/2)`` plotted next to the same signal
    aliased into ``[−PRF/2, +PRF/2]`` (= what the FFT actually sees).
    The unwrapped chirp covers ~kHz; the aliased trajectory bounces
    between ±PRF/2 in a sawtooth as each wrap completes.  The y-axis
    runs from ``−swing/2`` to ``+swing/2`` so the contrast is obvious.
  * **Right (post-deramp):** the actual FFT spectrum of the GAMMA-
    deramped SLC.  The signal collapses into a narrow band well inside
    ±PRF/2 — that's the natural stripmap-equivalent azimuth bandwidth.

The −3, −10, −20 dB post-deramp bandwidths are measured and reported in
both Hz and PRF units.

Usage
-----
    python scripts/diagnostics/burst_doppler_range.py
    python scripts/diagnostics/burst_doppler_range.py --burst 3 --subswath iw2
    python scripts/diagnostics/burst_doppler_range.py --out plots/diagnostics/foo.png
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

from scripts.diagnostics.burst_spectrogram import (
    _default_deramp_slc,
    _read_deramp_burst,
)
from scripts.sentinel_1.rvl import _deramp_rate
from scripts.sentinel_1.safe_io import find_safe_files, parse_annotation


def measured_bandwidth(spec: np.ndarray, freqs: np.ndarray,
                       db_levels: tuple[float, ...] = (-3.0, -10.0, -20.0)
                       ) -> dict[float, tuple[float, float, float]]:
    """Return ``{dB_level: (bw_total_Hz, f_lo_Hz, f_hi_Hz)}``.

    Bandwidth is the frequency span over which the (peak-normalised) power
    spectrum stays above ``10**(dB/10)``.
    """
    sp_norm = spec / spec.max()
    out = {}
    for dB in db_levels:
        thr = 10 ** (dB / 10.0)
        mask = sp_norm >= thr
        if not mask.any():
            out[dB] = (0.0, 0.0, 0.0)
            continue
        lo = float(freqs[np.argmax(mask)])
        hi = float(freqs[len(mask) - 1 - np.argmax(mask[::-1])])
        out[dB] = (hi - lo, lo, hi)
    return out


def post_deramp_psd(burst: np.ndarray, n_fft: int = 1024,
                    rg_centre: int = 10000, rg_width: int = 200,
                    azimuth_line_time: float = 1.0 / 486.4863
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Single Hann-windowed FFT over the centre ``n_fft`` lines, averaged
    across a small range strip.  Returns ``(freqs_Hz, psd_norm_linear)``."""
    if burst.shape[0] < n_fft:
        n_fft = burst.shape[0] // 2 * 2          # nearest even smaller
    lo = max(0, burst.shape[0] // 2 - n_fft // 2)
    rg_lo = max(0, rg_centre - rg_width // 2)
    rg_hi = min(burst.shape[1], rg_lo + rg_width)
    strip = burst[lo:lo + n_fft, rg_lo:rg_hi]    # (n_fft, rg_w)
    hann = np.hanning(n_fft)[:, None]
    sp = np.fft.fftshift(np.abs(np.fft.fft(strip * hann, axis=0)) ** 2,
                         axes=0).mean(axis=1)
    freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=azimuth_line_time))
    return freqs, sp / sp.max()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--safe", type=Path,
                   default=Path("data/sentinel-1/scene1/S1A_IW_SLC.SAFE"))
    p.add_argument("--subswath", default="iw1", choices=("iw1", "iw2", "iw3"))
    p.add_argument("--pol", default="vv", choices=("vv", "vh"))
    p.add_argument("--burst", type=int, default=None,
                   help="Burst index (0-based).  Default: middle burst.")
    p.add_argument("--deramp-slc", type=Path, default=None,
                   help="GAMMA .deramp.slc path.  Default inferred from SAFE name.")
    p.add_argument("--n-fft-post", type=int, default=1024,
                   help="FFT length for the right panel's post-deramp bandwidth "
                        "measurement (default 1024, freq res PRF/1024).")
    p.add_argument("--rg-centre", type=int, default=10000)
    p.add_argument("--rg-width", type=int, default=200)
    p.add_argument("--db-floor", type=float, default=-30.0,
                   help="Colour-scale floor for the spectrogram (default -30 dB).")
    p.add_argument("--out", type=Path,
                   default=Path("plots/diagnostics/burst_doppler_range_pre_post_deramp.png"))
    args = p.parse_args()

    files = find_safe_files(str(args.safe), args.subswath, args.pol)
    annot = parse_annotation(files["annotation"])
    burst_idx = args.burst if args.burst is not None else len(annot.bursts) // 2

    deramp_slc = args.deramp_slc or _default_deramp_slc(args.subswath, args.pol, args.safe)
    par_path = Path(str(deramp_slc) + ".par")
    if not deramp_slc.exists() or not par_path.exists():
        raise SystemExit(
            f"Deramped SLC not found: {deramp_slc}\n"
            f"  Run gamma_prep_scene first or pass --deramp-slc."
        )

    LPB = annot.lines_per_burst
    ati = annot.azimuth_time_interval
    PRF = 1.0 / ati
    T_burst = LPB * ati

    # Steering rate at mid-range from the annotation.
    k_s_arr = _deramp_rate(annot, burst_idx)
    k_s_mid = float(k_s_arr[len(k_s_arr) // 2])
    unwrapped_swing = abs(k_s_mid) * T_burst                     # Hz total
    half_swing_kHz = unwrapped_swing / 2.0 / 1000.0

    print(f"SAFE:        {args.safe}")
    print(f"Subswath:    {args.subswath}  pol={args.pol}  burst={burst_idx}")
    print(f"PRF:         {PRF:.1f} Hz   burst duration = {T_burst:.3f} s")
    print(f"k_s (mid-range)              = {k_s_mid:+.1f} Hz/s")
    print(f"Unwrapped chirp swing        = {unwrapped_swing:.0f} Hz "
          f"= {unwrapped_swing/PRF:.2f} x PRF "
          f"= {unwrapped_swing/1000:.2f} kHz")
    print(f"Deramp SLC:  {deramp_slc}")

    deramp = _read_deramp_burst(deramp_slc, par_path, burst_idx, LPB)

    # Right panel: longer-FFT measurement of the deramped spectrum.
    freqs_post, sp_post = post_deramp_psd(
        deramp, n_fft=args.n_fft_post,
        rg_centre=args.rg_centre, rg_width=args.rg_width,
        azimuth_line_time=ati,
    )
    sp_post_dB = 10 * np.log10(sp_post)
    bw = measured_bandwidth(sp_post, freqs_post)
    print()
    print("Post-deramp bandwidth (measured on a centred 1024-line window):")
    for dB, (w, lo, hi) in bw.items():
        print(f"  {dB:+5.0f} dB :  {w:6.1f} Hz  ({lo:+6.1f}, {hi:+6.1f})   "
              f"= {w/PRF:5.2f} x PRF")

    # Theoretical chirp trajectory (centred so middle of burst is f=0).
    t_line = np.linspace(0, T_burst, 2000)
    f_chirp_Hz = k_s_mid * (t_line - T_burst / 2.0)
    # Same signal aliased into [-PRF/2, +PRF/2] — what the FFT actually sees.
    f_aliased_Hz = ((f_chirp_Hz + PRF / 2.0) % PRF) - PRF / 2.0

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)

    # Left: unwrapped chirp trajectory + the same signal aliased into ±PRF/2.
    ax = axes[0]
    y_max_kHz = half_swing_kHz * 1.15        # a bit of headroom
    # Highlight the PRF window so the reader sees how narrow it is.
    ax.axhspan(-PRF / 2000.0, +PRF / 2000.0, color="cyan", alpha=0.12,
               label=fr"FFT can only see ±PRF/2 = ±{PRF/2:.0f} Hz")
    # Aliased trajectory (= what an FFT of the raw burst would report).
    ax.plot(t_line, f_aliased_Hz / 1000.0, color="tab:blue", lw=1.3,
            label="aliased into ±PRF/2 (what the FFT sees)")
    # Unwrapped chirp.
    ax.plot(t_line, f_chirp_Hz / 1000.0, color="red", lw=2.2,
            label=r"true unwrapped chirp $k_s\,(\tau - T/2)$")
    # PRF/2 reference lines
    ax.axhline(+PRF / 2000.0, color="cyan", ls="--", lw=0.9)
    ax.axhline(-PRF / 2000.0, color="cyan", ls="--", lw=0.9)
    ax.set_xlim(0, T_burst)
    ax.set_ylim(-y_max_kHz, +y_max_kHz)
    ax.set_xlabel("Burst time [s]")
    ax.set_ylabel("Doppler frequency [kHz]")
    ax.set_title(f"Before deramping — true chirp swing = "
                 f"{unwrapped_swing/1000:.2f} kHz ≈ {unwrapped_swing/PRF:.1f}×PRF\n"
                 f"(the signal aliases each time it leaves the PRF window)")
    ax.legend(loc="upper right", framealpha=0.92, fontsize=9)
    ax.grid(True, alpha=0.2)

    # Right: spectrum line plot, x-axis in Hz, on its natural scale
    ax = axes[1]
    ax.plot(freqs_post, sp_post_dB, color="tab:orange", lw=1.4)
    ax.axvline(+PRF / 2.0, color="cyan", ls="--", lw=1.0,
               label=fr"±PRF/2 = ±{PRF/2:.0f} Hz")
    ax.axvline(-PRF / 2.0, color="cyan", ls="--", lw=1.0)
    # Mark −3 / −10 / −20 dB widths.
    for dB, color in [(-3, "tab:red"), (-10, "tab:purple"), (-20, "tab:brown")]:
        w, lo, hi = bw[dB]
        ax.axhline(dB, color=color, ls=":", lw=0.9, alpha=0.7)
        ax.annotate(f"{dB} dB  →  {w:.0f} Hz  ({w/PRF:.2f}×PRF)",
                    xy=(freqs_post[-1] * 0.95, dB), xytext=(0, 4),
                    textcoords="offset points",
                    ha="right", va="bottom", fontsize=9, color=color)
    ax.set_xlim(freqs_post[0], freqs_post[-1])
    ax.set_ylim(args.db_floor, 1.0)
    ax.set_xlabel("Doppler frequency [Hz]")
    ax.set_ylabel("Power [dB rel. to peak]")
    ax.set_title(f"After deramping — bandwidth in the PRF range\n"
                 f"(−3 dB = {bw[-3.0][0]:.0f} Hz, "
                 f"−10 dB = {bw[-10.0][0]:.0f} Hz)")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.25)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
