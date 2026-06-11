"""Score candidate deramp models by burst-offset stability and seam mismatch."""

from __future__ import annotations

import argparse

import numpy as np

from scripts.sentinel_1.rvl import (
    _fm_rate_at_burst,
    _steering_doppler_rate,
    apply_burst_valid_sample_mask,
    deramp_burst,
    estimate_doppler_grid_fft,
)
from scripts.sentinel_1.safe_io import find_safe_files, parse_annotation, read_slc_burst


def _summarise(arr: np.ndarray) -> tuple[float, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.nanmean(finite)), float(np.nanstd(finite))


def _fit_trend(y: np.ndarray) -> np.ndarray:
    idx = np.arange(y.size, dtype=np.float64)
    ok = np.isfinite(y)
    if np.count_nonzero(ok) < 2:
        return np.full_like(y, np.nan, dtype=np.float64)
    coeff = np.polyfit(idx[ok], y[ok], deg=1)
    return np.polyval(coeff, idx)


def _apply_chirp(raw: np.ndarray, annot, k_s: np.ndarray) -> np.ndarray:
    lpb = annot.lines_per_burst
    t_az = (
        np.arange(lpb, dtype=np.float64) - 0.5 * (lpb - 1)
    ) * annot.azimuth_time_interval
    chirp = np.exp(1j * np.pi * k_s[np.newaxis, :] * t_az[:, np.newaxis] ** 2)
    return (raw * chirp).astype(np.complex64)


def _deramp_variant(raw: np.ndarray, annot, burst_idx: int, variant: str) -> np.ndarray:
    if variant == "none":
        return raw.copy()
    if variant == "current":
        return deramp_burst(raw, annot, burst_idx)

    k_a = _fm_rate_at_burst(annot, burst_idx).astype(np.float64)
    k_psi = float(_steering_doppler_rate(annot, burst_idx))

    if variant == "plus_formula":
        k_s = -k_a * k_psi / (k_a + k_psi)
        return _apply_chirp(raw, annot, k_s)
    if variant == "neg_ka":
        return _apply_chirp(raw, annot, -k_a)
    if variant == "neg_kpsi":
        k_s = np.full_like(k_a, -k_psi)
        return _apply_chirp(raw, annot, k_s)

    raise ValueError(f"unknown variant: {variant}")


def _estimate_fft_fdc(
    data: np.ndarray,
    annot,
    burst,
    block_az: int,
    block_rg: int,
    stride_az: int,
    stride_rg: int,
) -> np.ndarray:
    work = data.copy()
    valid_mask = apply_burst_valid_sample_mask(work, burst)
    f_dc, _, _, _ = estimate_doppler_grid_fft(
        work,
        annot.prf,
        block_az,
        block_rg,
        stride_az,
        stride_rg,
        valid_mask=valid_mask,
    )
    return f_dc


def _edge_step_metric(grids: list[np.ndarray], edge_rows: int) -> tuple[float, float, int]:
    diffs: list[np.ndarray] = []
    for left, right in zip(grids[:-1], grids[1:]):
        n = min(edge_rows, left.shape[0], right.shape[0])
        if n == 0:
            continue
        left_edge = left[-n:, :]
        right_edge = right[:n, :]
        delta = right_edge - left_edge
        finite = delta[np.isfinite(delta)]
        if finite.size:
            diffs.append(finite.astype(np.float64))

    if not diffs:
        return float("nan"), float("nan"), 0

    all_diffs = np.concatenate(diffs)
    rms = float(np.sqrt(np.mean(all_diffs**2)))
    mad = float(np.median(np.abs(all_diffs)))
    return rms, mad, int(all_diffs.size)


def run(
    slc_safe: str,
    subswath: str,
    polarisation: str = "vv",
    block_az: int = 256,
    block_rg: int = 512,
    stride_az: int = 128,
    stride_rg: int = 256,
    edge_rows: int = 2,
) -> None:
    files = find_safe_files(slc_safe, subswath, polarisation)
    annot = parse_annotation(files["annotation"])
    variants = ("none", "current", "plus_formula", "neg_ka", "neg_kpsi")

    print(
        "variant,"
        "offset_rms_hz,offset_mad_hz,"
        "mean_within_burst_std_hz,"
        "edge_step_rms_hz,edge_step_mad_hz,"
        "n_edge_samples"
    )

    for variant in variants:
        burst_means = []
        burst_stds = []
        burst_grids: list[np.ndarray] = []

        for burst_idx, burst in enumerate(annot.bursts):
            raw = read_slc_burst(files["measurement"], annot, burst_idx)
            data = _deramp_variant(raw, annot, burst_idx, variant)
            f_dc = _estimate_fft_fdc(
                data,
                annot,
                burst,
                block_az,
                block_rg,
                stride_az,
                stride_rg,
            )
            mean_hz, std_hz = _summarise(f_dc)
            burst_means.append(mean_hz)
            burst_stds.append(std_hz)
            burst_grids.append(f_dc)

        means = np.array(burst_means, dtype=np.float64)
        trend = _fit_trend(means)
        offsets = means - trend
        finite_offsets = offsets[np.isfinite(offsets)]
        offset_rms = float(np.sqrt(np.mean(finite_offsets**2)))
        offset_mad = float(np.median(np.abs(finite_offsets)))
        mean_std = float(np.nanmean(np.array(burst_stds, dtype=np.float64)))
        edge_rms, edge_mad, n_edge = _edge_step_metric(burst_grids, edge_rows=edge_rows)

        print(
            f"{variant},"
            f"{offset_rms:.6f},{offset_mad:.6f},"
            f"{mean_std:.6f},"
            f"{edge_rms:.6f},{edge_mad:.6f},"
            f"{n_edge}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score candidate deramp models by burst FFT offsets and burst-edge seam mismatch."
    )
    parser.add_argument("slc_safe", help="Path to the Sentinel-1 SLC SAFE directory")
    parser.add_argument("--subswath", default="iw1", help="Subswath, e.g. iw1/iw2/iw3")
    parser.add_argument("--pol", default="vv", help="Polarisation, e.g. vv or vh")
    parser.add_argument("--block-az", type=int, default=256)
    parser.add_argument("--block-rg", type=int, default=512)
    parser.add_argument("--stride-az", type=int, default=128)
    parser.add_argument("--stride-rg", type=int, default=256)
    parser.add_argument("--edge-rows", type=int, default=2, help="Rows from each burst edge to compare")
    args = parser.parse_args()

    run(
        slc_safe=args.slc_safe,
        subswath=args.subswath,
        polarisation=args.pol,
        block_az=args.block_az,
        block_rg=args.block_rg,
        stride_az=args.stride_az,
        stride_rg=args.stride_rg,
        edge_rows=args.edge_rows,
    )


if __name__ == "__main__":
    main()
