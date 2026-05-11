"""Compare burst-wise Doppler centroid offsets under different deramp conventions."""

from __future__ import annotations

import argparse

import numpy as np
from scripts.sentinel_1.aux_files import apply_poeorb
from scripts.sentinel_1.safe_io import find_safe_files, parse_annotation, read_slc_burst
from scripts.sentinel_1.rvl import (
    _orbital_speed,
    _steering_doppler_rate,
    apply_burst_valid_sample_mask,
    correlation_to_doppler,
    deramp_burst,
    estimate_correlation_grid,
    estimate_doppler_grid_fft,
)


def _deramp_with_custom_kpsi(
    burst: np.ndarray,
    annot,
    burst_idx: int,
    kpsi_mode: str,
) -> np.ndarray:
    from scripts.sentinel_1.rvl import C_LIGHT, _fm_rate_at_burst

    lpb = annot.lines_per_burst
    k_a = _fm_rate_at_burst(annot, burst_idx).astype(np.float64)

    if kpsi_mode == "nominal":
        return deramp_burst(burst, annot, burst_idx)

    if kpsi_mode == "signed_steering":
        v_sat = _orbital_speed(annot, burst_idx)
        psi_dot = annot.azimuth_steering_rate * np.pi / 180.0
        k_psi = 2.0 * annot.radar_frequency / C_LIGHT * v_sat * psi_dot
    elif kpsi_mode == "negative_steering":
        k_psi = -_steering_doppler_rate(annot, burst_idx)
    else:
        raise ValueError(f"unknown kpsi_mode: {kpsi_mode}")

    k_s = -k_a * k_psi / (k_a - k_psi)
    t_az = (
        np.arange(lpb, dtype=np.float64) - 0.5 * (lpb - 1)
    ) * annot.azimuth_time_interval
    chirp = np.exp(1j * np.pi * k_s[np.newaxis, :] * t_az[:, np.newaxis] ** 2)
    return (burst * chirp).astype(np.complex64)


def _summarise(arr: np.ndarray) -> tuple[float, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.nanmean(finite)), float(np.nanstd(finite))


def _compute_variant_stats(
    data: np.ndarray,
    valid_mask: np.ndarray,
    annot,
    block_az: int,
    block_rg: int,
    stride_az: int,
    stride_rg: int,
) -> dict[str, float]:
    out: dict[str, float] = {}

    p0, p1, _, _ = estimate_correlation_grid(
        data,
        block_az,
        block_rg,
        stride_az,
        stride_rg,
        valid_mask=valid_mask,
    )
    f_dc_prf, _, _ = correlation_to_doppler(
        p0, p1, annot.prf, annot.wavelength, gamma_amb=None,
    )
    f_dc_radar_prf, _, _ = correlation_to_doppler(
        p0, p1, annot.radar_prf, annot.wavelength, gamma_amb=None,
    )
    fft_prf, _, _, _ = estimate_doppler_grid_fft(
        data,
        annot.prf,
        block_az,
        block_rg,
        stride_az,
        stride_rg,
        valid_mask=valid_mask,
    )
    fft_radar_prf, _, _, _ = estimate_doppler_grid_fft(
        data,
        annot.radar_prf,
        block_az,
        block_rg,
        stride_az,
        stride_rg,
        valid_mask=valid_mask,
    )

    out["cde_prf_mean"], out["cde_prf_std"] = _summarise(f_dc_prf)
    out["cde_radar_prf_mean"], out["cde_radar_prf_std"] = _summarise(f_dc_radar_prf)
    out["fft_prf_mean"], out["fft_prf_std"] = _summarise(fft_prf)
    out["fft_radar_prf_mean"], out["fft_radar_prf_std"] = _summarise(fft_radar_prf)
    return out


def run(
    slc_safe: str,
    subswath: str,
    polarisation: str,
    poeorb_path: str | None,
    block_az: int,
    block_rg: int,
    stride_az: int,
    stride_rg: int,
) -> None:
    files = find_safe_files(slc_safe, subswath, polarisation)
    annot_original = parse_annotation(files["annotation"])
    annot = apply_poeorb(annot_original, poeorb_path) if poeorb_path else annot_original

    header = [
        "burst_idx",
        "raw_cde_prf_mean", "raw_cde_prf_std",
        "raw_cde_radar_mean", "raw_cde_radar_std",
        "raw_fft_prf_mean", "raw_fft_prf_std",
        "raw_fft_radar_mean", "raw_fft_radar_std",
        "deramp_cde_prf_mean", "deramp_cde_prf_std",
        "deramp_cde_radar_mean", "deramp_cde_radar_std",
        "deramp_fft_prf_mean", "deramp_fft_prf_std",
        "deramp_fft_radar_mean", "deramp_fft_radar_std",
        "signed_cde_prf_mean", "signed_cde_prf_std",
        "signed_fft_prf_mean", "signed_fft_prf_std",
        "neg_cde_prf_mean", "neg_cde_prf_std",
        "neg_fft_prf_mean", "neg_fft_prf_std",
    ]
    print(",".join(header))

    for burst_idx, burst in enumerate(annot.bursts):
        raw = read_slc_burst(files["measurement"], annot, burst_idx)
        raw_mask = apply_burst_valid_sample_mask(raw.copy(), burst)
        raw = np.where(raw_mask, raw, 0.0).astype(np.complex64)

        deramped = deramp_burst(raw.copy(), annot, burst_idx)
        deramp_mask = apply_burst_valid_sample_mask(deramped, burst)

        signed = _deramp_with_custom_kpsi(raw.copy(), annot, burst_idx, "signed_steering")
        signed_mask = apply_burst_valid_sample_mask(signed, burst)

        neg = _deramp_with_custom_kpsi(raw.copy(), annot, burst_idx, "negative_steering")
        neg_mask = apply_burst_valid_sample_mask(neg, burst)

        raw_stats = _compute_variant_stats(
            raw, raw_mask, annot, block_az, block_rg, stride_az, stride_rg,
        )
        deramp_stats = _compute_variant_stats(
            deramped, deramp_mask, annot, block_az, block_rg, stride_az, stride_rg,
        )
        signed_stats = _compute_variant_stats(
            signed, signed_mask, annot, block_az, block_rg, stride_az, stride_rg,
        )
        neg_stats = _compute_variant_stats(
            neg, neg_mask, annot, block_az, block_rg, stride_az, stride_rg,
        )

        row = [
            str(burst_idx),
            f"{raw_stats['cde_prf_mean']:.6f}", f"{raw_stats['cde_prf_std']:.6f}",
            f"{raw_stats['cde_radar_prf_mean']:.6f}", f"{raw_stats['cde_radar_prf_std']:.6f}",
            f"{raw_stats['fft_prf_mean']:.6f}", f"{raw_stats['fft_prf_std']:.6f}",
            f"{raw_stats['fft_radar_prf_mean']:.6f}", f"{raw_stats['fft_radar_prf_std']:.6f}",
            f"{deramp_stats['cde_prf_mean']:.6f}", f"{deramp_stats['cde_prf_std']:.6f}",
            f"{deramp_stats['cde_radar_prf_mean']:.6f}", f"{deramp_stats['cde_radar_prf_std']:.6f}",
            f"{deramp_stats['fft_prf_mean']:.6f}", f"{deramp_stats['fft_prf_std']:.6f}",
            f"{deramp_stats['fft_radar_prf_mean']:.6f}", f"{deramp_stats['fft_radar_prf_std']:.6f}",
            f"{signed_stats['cde_prf_mean']:.6f}", f"{signed_stats['cde_prf_std']:.6f}",
            f"{signed_stats['fft_prf_mean']:.6f}", f"{signed_stats['fft_prf_std']:.6f}",
            f"{neg_stats['cde_prf_mean']:.6f}", f"{neg_stats['cde_prf_std']:.6f}",
            f"{neg_stats['fft_prf_mean']:.6f}", f"{neg_stats['fft_prf_std']:.6f}",
        ]
        print(",".join(row))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare per-burst f_dc offsets across deramp and PRF conventions.",
    )
    parser.add_argument("slc_safe")
    parser.add_argument("--subswath", default="iw1")
    parser.add_argument("--pol", default="vv")
    parser.add_argument("--poeorb", default=None)
    parser.add_argument("--block-az", type=int, default=256)
    parser.add_argument("--block-rg", type=int, default=512)
    parser.add_argument("--stride-az", type=int, default=128)
    parser.add_argument("--stride-rg", type=int, default=256)
    args = parser.parse_args()

    run(
        slc_safe=args.slc_safe,
        subswath=args.subswath,
        polarisation=args.pol,
        poeorb_path=args.poeorb,
        block_az=args.block_az,
        block_rg=args.block_rg,
        stride_az=args.stride_az,
        stride_rg=args.stride_rg,
    )


if __name__ == "__main__":
    main()
