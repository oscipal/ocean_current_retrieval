#!/usr/bin/env python3
"""Pre-vs-post-deramp azimuth spectrogram for one Sentinel-1 burst.

Visual "the deramp worked" check.  Computes a short-time Fourier transform
along azimuth on the same burst before and after ``SLC_deramp_ScanSAR`` and
displays the two time-frequency maps side by side.

In TOPS the antenna steering produces a Doppler chirp whose total swing far
exceeds the PRF, so the *pre-deramp* spectrogram shows a **diagonal stripe**
that wraps around the unambiguous band [−PRF/2, +PRF/2].  After deramping,
that linear-in-time phase term is removed and the energy collapses to a
**horizontal band near 0 Hz** — the stripmap-equivalent narrowband signal.

Inputs
------
  * ``--safe``      — the Sentinel-1 SLC SAFE directory.  Default: scene1.
  * ``--deramp-slc`` — GAMMA's ``.deramp.slc`` (FCOMPLEX, big-endian).
                       Default: ``data/sentinel-1/gamma_<sw>/<YYYYMMDD>_<sw>_<pol>.deramp.slc``.
  * ``--burst``     — burst index (0-based).  Default: middle burst.
  * ``--subswath``  — ``iw1``/``iw2``/``iw3``.  Default ``iw1``.
  * ``--pol``       — ``vv`` or ``vh``.  Default ``vv``.

STFT parameters (defaults match the diagnostic plot from the May 2026
session; see ``notes/session_2026-05-27.md``):
  * 128-line Hann window
  * 32-line hop (75 % overlap)
  * 100 range bins averaged for SNR (no spatial blur in azimuth)

Usage
-----
    python scripts/diagnostics/burst_spectrogram.py
    python scripts/diagnostics/burst_spectrogram.py --burst 3 --subswath iw2
    python scripts/diagnostics/burst_spectrogram.py --out plots/diagnostics/foo.png
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

from scripts.sentinel_1.safe_io import find_safe_files, parse_annotation, read_slc_burst


def _default_deramp_slc(subswath: str, pol: str, safe: Path) -> Path:
    """Guess the GAMMA deramped SLC path from the SAFE contents.

    Mirrors the convention used by gamma_prep_scene + the shell driver:
    ``data/sentinel-1/gamma_<sw>/<YYYYMMDD>_<sw>_<pol>.deramp.slc``.

    The SAFE directory itself is often renamed to a generic ``S1A_IW_SLC.SAFE``
    in this repo, so we infer the overpass date from the measurement TIFF
    filename which preserves the timestamp.
    """
    sw, pol = subswath.lower(), pol.lower()
    # Measurement TIFFs follow s1a-<sw>-slc-<pol>-<YYYYMMDD>t<HHMMSS>-... .tiff
    import glob
    hits = sorted(glob.glob(str(safe / "measurement" / f"s1?-{sw}-slc-{pol}-*.tiff")))
    if not hits:
        raise SystemExit(
            f"No measurement TIFF for {sw}/{pol} in {safe}; pass --deramp-slc."
        )
    name = Path(hits[0]).name
    # Date token sits between the polarisation and the start-time hour, e.g.
    # ...-vv-20260205t165251-...
    parts = name.split("-")
    date = next((tok[:8] for tok in parts if len(tok) == 15 and tok[8] == "t" and tok[:8].isdigit()),
                None)
    if date is None:
        raise SystemExit(
            f"Could not parse date from TIFF name {name!r}; pass --deramp-slc."
        )
    return (REPO_ROOT / "data" / "sentinel-1" / f"gamma_{sw}" /
            f"{date}_{sw}_{pol}.deramp.slc")


def _read_deramp_burst(deramp_slc: Path, par_path: Path,
                       burst_idx: int, lines_per_burst: int) -> np.ndarray:
    """Read one burst from a GAMMA ``.deramp.slc`` (FCOMPLEX, big-endian)."""
    # Pull dimensions from the .par file (range_samples / azimuth_lines).
    rg, az = None, None
    for line in par_path.read_text().splitlines():
        k = line.split(":", 1)[0].strip()
        if k == "range_samples":
            rg = int(line.split()[1])
        elif k == "azimuth_lines":
            az = int(line.split()[1])
    if rg is None or az is None:
        raise SystemExit(f"Could not parse range_samples/azimuth_lines from {par_path}")

    mm = np.memmap(deramp_slc, dtype=">f4", mode="r", shape=(az, 2 * rg))
    block = mm[burst_idx * lines_per_burst:(burst_idx + 1) * lines_per_burst, :]
    return (block[:, 0::2] + 1j * block[:, 1::2]).astype(np.complex64)


def azimuth_spectrogram(burst: np.ndarray,
                        n_fft: int = 128,
                        hop: int = 32,
                        rg_centre: int = 10000,
                        rg_width: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-burst azimuth STFT averaged across a small range strip.

    Returns
    -------
    times : ndarray of shape (n_windows,)
        Window-centre azimuth time *relative to burst start*, in lines (you
        multiply by the azimuth-line-time to get seconds).
    norm_freqs : ndarray of shape (n_fft,)
        Normalised frequency f / PRF in [−0.5, +0.5).
    psd : ndarray of shape (n_fft, n_windows)
        Welch-style power spectrum: ``|FFT(Hann · segment)|²`` averaged
        across the range strip, fftshifted so 0 Hz is at the centre.
    """
    n_az, n_rg = burst.shape
    rg_lo = max(0, rg_centre - rg_width // 2)
    rg_hi = min(n_rg, rg_lo + rg_width)
    if rg_hi - rg_lo < 2:
        raise SystemExit(f"range strip too narrow: cols [{rg_lo},{rg_hi}) of {n_rg}")
    strip = burst[:, rg_lo:rg_hi]                              # (n_az, rg_w)

    hann = np.hanning(n_fft)
    n_win = max(0, (n_az - n_fft) // hop + 1)
    if n_win < 1:
        raise SystemExit(f"burst too short ({n_az} lines) for {n_fft}-line window")
    psd = np.zeros((n_fft, n_win), dtype=np.float64)
    for k in range(n_win):
        seg = strip[k * hop:k * hop + n_fft, :] * hann[:, None]
        sp = np.fft.fft(seg, axis=0)
        psd[:, k] = (np.abs(sp) ** 2).mean(axis=1)

    psd = np.fft.fftshift(psd, axes=0)
    norm_freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0))
    win_centres_lines = np.arange(n_win) * hop + n_fft / 2.0
    return win_centres_lines, norm_freqs, psd


def plot_two_panels(t_pre, f_pre, sp_pre,
                    t_post, f_post, sp_post,
                    azimuth_line_time: float,
                    burst_idx: int,
                    out_path: str,
                    db_floor: float = -30.0) -> None:
    """Render the two spectrograms side by side with per-panel dB normalisation."""
    t_pre_s  = t_pre  * azimuth_line_time
    t_post_s = t_post * azimuth_line_time
    sp_pre_dB  = 10 * np.log10(sp_pre  / sp_pre.max())
    sp_post_dB = 10 * np.log10(sp_post / sp_post.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4),
                             constrained_layout=True, sharey=True)
    for ax, t, f, sp_dB, title in [
        (axes[0], t_pre_s,  f_pre,  sp_pre_dB,
         f"Before deramping (burst {burst_idx})"),
        (axes[1], t_post_s, f_post, sp_post_dB,
         f"After deramping (burst {burst_idx})"),
    ]:
        im = ax.pcolormesh(t, f, sp_dB, vmin=db_floor, vmax=0,
                           cmap="viridis", shading="auto")
        ax.set_xlabel("Burst time [s]")
        ax.set_title(title)
        ax.set_xlim(t[0], t[-1])
    axes[0].set_ylabel("Normalised frequency [f / PRF]")
    axes[0].set_ylim(-0.5, 0.5)
    cb = fig.colorbar(im, ax=axes.tolist(), shrink=0.92, pad=0.015,
                      label="Power [dB rel. to panel peak]")
    cb.outline.set_visible(False)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"saved -> {out_path}")


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
    p.add_argument("--n-fft", type=int, default=128,
                   help="STFT window length in azimuth lines (default 128).")
    p.add_argument("--hop", type=int, default=32,
                   help="STFT hop in azimuth lines (default 32 = 75%% overlap).")
    p.add_argument("--rg-centre", type=int, default=10000,
                   help="Range-bin centre of the SNR-averaging strip.")
    p.add_argument("--rg-width", type=int, default=100,
                   help="Width of the range strip in samples (default 100).")
    p.add_argument("--db-floor", type=float, default=-30.0,
                   help="dB floor for the colour scale (default -30).")
    p.add_argument("--out", type=Path,
                   default=Path("plots/diagnostics/burst_spectrogram_pre_post_deramp.png"))
    args = p.parse_args()

    files = find_safe_files(str(args.safe), args.subswath, args.pol)
    annot = parse_annotation(files["annotation"])
    burst_idx = args.burst if args.burst is not None else len(annot.bursts) // 2

    deramp_slc = args.deramp_slc or _default_deramp_slc(args.subswath, args.pol, args.safe)
    par_path = Path(str(deramp_slc) + ".par")
    if not deramp_slc.exists() or not par_path.exists():
        raise SystemExit(
            f"Deramped SLC not found: {deramp_slc}\n"
            f"  Run gamma_prep_scene (or shell_commands/gamma_iw*_deramp_*.sh) first,\n"
            f"  or pass --deramp-slc to point at it explicitly."
        )

    print(f"SAFE:        {args.safe}")
    print(f"Subswath:    {args.subswath}  pol={args.pol}  burst={burst_idx}/"
          f"{len(annot.bursts)-1}")
    print(f"PRF:         {1.0/annot.azimuth_time_interval:.2f} Hz   "
          f"lines/burst={annot.lines_per_burst}")
    print(f"Deramp SLC:  {deramp_slc}")

    raw = read_slc_burst(files["measurement"], annot, burst_idx)
    deramp = _read_deramp_burst(deramp_slc, par_path, burst_idx, annot.lines_per_burst)

    print(f"reading: raw {raw.shape}, deramped {deramp.shape}")

    t_pre,  f_pre,  sp_pre  = azimuth_spectrogram(raw,
                                                  n_fft=args.n_fft, hop=args.hop,
                                                  rg_centre=args.rg_centre,
                                                  rg_width=args.rg_width)
    t_post, f_post, sp_post = azimuth_spectrogram(deramp,
                                                  n_fft=args.n_fft, hop=args.hop,
                                                  rg_centre=args.rg_centre,
                                                  rg_width=args.rg_width)
    print(f"STFT: {sp_pre.shape[1]} windows  range strip "
          f"cols [{args.rg_centre - args.rg_width // 2}, "
          f"{args.rg_centre + args.rg_width // 2})")

    plot_two_panels(t_pre, f_pre, sp_pre,
                    t_post, f_post, sp_post,
                    azimuth_line_time=annot.azimuth_time_interval,
                    burst_idx=burst_idx,
                    out_path=str(args.out),
                    db_floor=args.db_floor)


if __name__ == "__main__":
    main()
