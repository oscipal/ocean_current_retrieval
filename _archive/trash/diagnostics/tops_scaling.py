"""Visualize burst-wise sensitivity to alternative TOPS Doppler scaling choices."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scripts.sentinel_1.aux_files import apply_poeorb
from scripts.sentinel_1.safe_io import find_safe_files, parse_annotation, read_slc_burst
from scripts.sentinel_1.rvl import (
    _fm_rate_at_burst,
    _geom_doppler_poeorb,
    _steering_doppler_rate,
    apply_burst_valid_sample_mask,
    deramp_burst,
    estimate_correlation_grid,
)


def _stack_bursts(arrays: list[np.ndarray]) -> np.ndarray:
    n_cols = max(a.shape[1] for a in arrays)
    rows = []
    sep = np.full((1, n_cols), np.nan, dtype=np.float32)
    for i, arr in enumerate(arrays):
        pad = np.full((arr.shape[0], n_cols), np.nan, dtype=np.float32)
        pad[:, : arr.shape[1]] = arr
        rows.append(pad)
        if i < len(arrays) - 1:
            rows.append(sep)
    return np.vstack(rows)


def run(
    slc_safe: str,
    poeorb_path: str,
    subswath: str,
    polarisation: str,
    out_path: str,
    block_az: int,
    block_rg: int,
    stride_az: int,
    stride_rg: int,
) -> None:
    files = find_safe_files(slc_safe, subswath, polarisation)
    annot_original = parse_annotation(files["annotation"])
    annot = apply_poeorb(annot_original, poeorb_path)

    variants = {"normal": [], "scaled": [], "inverse": []}

    for burst_idx, burst in enumerate(annot.bursts):
        print(f"burst {burst_idx}")
        raw = read_slc_burst(files["measurement"], annot, burst_idx)
        deramped = deramp_burst(raw, annot, burst_idx)
        valid_mask = apply_burst_valid_sample_mask(deramped, burst)

        p0, p1, _, rg_centers = estimate_correlation_grid(
            deramped, block_az, block_rg, stride_az, stride_rg,
            valid_mask=valid_mask,
        )

        valid = np.isfinite(p0) & (p0 > 0)
        f_lag = annot.prf / (2.0 * np.pi) * np.angle(p1)
        f_lag = np.where(valid, f_lag, np.nan)

        k_a = _fm_rate_at_burst(annot, burst_idx)[rg_centers]
        k_psi = _steering_doppler_rate(annot, burst_idx)
        scale = 1.0 + k_psi / k_a
        f_geom = _geom_doppler_poeorb(annot, annot_original, burst_idx, rg_centers)

        variants["normal"].append((annot.wavelength / 2.0 * (f_lag - f_geom[None, :])).astype(np.float32))
        variants["scaled"].append((annot.wavelength / 2.0 * (f_lag * scale[None, :] - f_geom[None, :])).astype(np.float32))
        variants["inverse"].append((annot.wavelength / 2.0 * (f_lag / scale[None, :] - f_geom[None, :])).astype(np.float32))

    stacked = {name: _stack_bursts(arrs) for name, arrs in variants.items()}
    finite = np.concatenate([a[np.isfinite(a)] for a in stacked.values()])
    vmax = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 8), sharex=True, sharey=True)
    for ax, (name, arr) in zip(axes, stacked.items()):
        im = ax.imshow(arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(name)
        ax.set_xlabel("range block")
    axes[0].set_ylabel("azimuth block, bursts stacked")
    fig.colorbar(im, ax=axes, label="m/s")
    fig.suptitle(f"{subswath.upper()} {polarisation.upper()} TOPS scaling diagnostic")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("slc_safe")
    parser.add_argument("poeorb_path")
    parser.add_argument("--subswath", default="iw1")
    parser.add_argument("--pol", default="vv")
    parser.add_argument("--out", default="plots/tops_scaling_diagnostic.png")
    parser.add_argument("--block-az", type=int, default=256)
    parser.add_argument("--block-rg", type=int, default=512)
    parser.add_argument("--stride-az", type=int, default=128)
    parser.add_argument("--stride-rg", type=int, default=256)
    args = parser.parse_args()
    run(args.slc_safe, args.poeorb_path, args.subswath, args.pol, args.out,
        args.block_az, args.block_rg, args.stride_az, args.stride_rg)


if __name__ == "__main__":
    main()
