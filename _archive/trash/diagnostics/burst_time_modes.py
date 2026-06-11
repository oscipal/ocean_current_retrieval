"""Compare burst-wise Doppler results under different burst time conventions."""

from __future__ import annotations

import argparse
from datetime import timedelta

import numpy as np

from scripts.sentinel_1.rvl import (
    _interpolate_orbit,
    apply_burst_valid_sample_mask,
    correlation_to_doppler,
    deramp_burst,
    estimate_correlation_grid,
    estimate_doppler_grid_fft,
)
from scripts.sentinel_1.safe_io import find_safe_files, parse_annotation, read_slc_burst, slant_range_time_vector


def _nearest_estimate_at_time(estimates, t):
    diffs = [abs((e.azimuth_time - t).total_seconds()) for e in estimates]
    return estimates[int(np.argmin(diffs))]


def _eval_poly(poly: np.ndarray, t0: float, tau: np.ndarray) -> np.ndarray:
    dt = tau - t0
    return sum(c * dt**k for k, c in enumerate(poly))


def _burst_time(annot, burst_idx: int, mode: str):
    burst = annot.bursts[burst_idx]
    if mode == "azimuth":
        return burst.azimuth_time
    if mode == "sensing":
        return burst.sensing_time
    if mode == "center":
        half_dt = 0.5 * (annot.lines_per_burst - 1) * annot.azimuth_time_interval
        return burst.azimuth_time + timedelta(seconds=half_dt)
    raise ValueError(f"unknown mode: {mode}")


def _fm_rate_at_time_mode(annot, burst_idx: int, mode: str) -> np.ndarray:
    t = _burst_time(annot, burst_idx, mode)
    afr = _nearest_estimate_at_time(annot.azimuth_fm_rates, t)
    tau = slant_range_time_vector(annot)
    return _eval_poly(afr.poly, afr.t0, tau)


def _steering_doppler_rate_at_time_mode(annot, burst_idx: int, mode: str) -> float:
    t = _burst_time(annot, burst_idx, mode)
    _, vel = _interpolate_orbit(annot, t)
    v_sat = float(np.linalg.norm(vel))
    psi_dot = abs(annot.azimuth_steering_rate) * np.pi / 180.0
    return 2.0 * annot.radar_frequency / 299_792_458.0 * v_sat * psi_dot


def _deramp_burst_time_mode(raw: np.ndarray, annot, burst_idx: int, mode: str) -> np.ndarray:
    if mode == "azimuth":
        return deramp_burst(raw, annot, burst_idx)

    lpb = annot.lines_per_burst
    k_a = _fm_rate_at_time_mode(annot, burst_idx, mode).astype(np.float64)
    k_psi = _steering_doppler_rate_at_time_mode(annot, burst_idx, mode)
    k_s = -k_a * k_psi / (k_a - k_psi)
    t_az = (
        np.arange(lpb, dtype=np.float64) - 0.5 * (lpb - 1)
    ) * annot.azimuth_time_interval
    chirp = np.exp(1j * np.pi * k_s[np.newaxis, :] * t_az[:, np.newaxis] ** 2)
    return (raw * chirp).astype(np.complex64)


def _geom_doppler_ann_time_mode(annot, burst_idx: int, rg_centers: np.ndarray, mode: str) -> np.ndarray:
    t = _burst_time(annot, burst_idx, mode)
    dc = _nearest_estimate_at_time(annot.dc_estimates, t)
    tau = slant_range_time_vector(annot)[rg_centers]
    return _eval_poly(dc.geometry_poly, dc.t0, tau).astype(np.float32)


def _summarise(arr: np.ndarray) -> tuple[float, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.nanmean(finite)), float(np.nanstd(finite))


def run(
    slc_safe: str,
    subswath: str,
    polarisation: str = "vv",
    estimator: str = "cde",
    block_az: int = 256,
    block_rg: int = 512,
    stride_az: int = 128,
    stride_rg: int = 256,
) -> None:
    files = find_safe_files(slc_safe, subswath, polarisation)
    annot = parse_annotation(files["annotation"])
    modes = ("azimuth", "sensing", "center")

    header = [
        "burst_idx",
        "burst_azimuth_time",
        "burst_sensing_time",
    ]
    for mode in modes:
        header.extend(
            [
                f"{mode}_f_dc_mean_hz",
                f"{mode}_f_dc_std_hz",
                f"{mode}_f_geom_mean_hz",
                f"{mode}_fm_mean_hz_per_s",
                f"{mode}_kpsi_hz_per_s",
                f"{mode}_dc_record_time",
                f"{mode}_fm_record_time",
            ]
        )
    print(",".join(header))

    for burst_idx, burst in enumerate(annot.bursts):
        raw = read_slc_burst(files["measurement"], annot, burst_idx)
        row = [str(burst_idx), burst.azimuth_time.isoformat(), burst.sensing_time.isoformat()]

        for mode in modes:
            deramped = _deramp_burst_time_mode(raw, annot, burst_idx, mode)
            valid_mask = apply_burst_valid_sample_mask(deramped, burst)

            if estimator == "fft":
                f_dc, _, _, rg_centers = estimate_doppler_grid_fft(
                    deramped,
                    annot.prf,
                    block_az,
                    block_rg,
                    stride_az,
                    stride_rg,
                    valid_mask=valid_mask,
                )
            else:
                p0, p1, _, rg_centers = estimate_correlation_grid(
                    deramped,
                    block_az,
                    block_rg,
                    stride_az,
                    stride_rg,
                    valid_mask=valid_mask,
                )
                f_dc, _, _ = correlation_to_doppler(p0, p1, annot.prf, annot.wavelength)

            f_geom = _geom_doppler_ann_time_mode(annot, burst_idx, rg_centers, mode)
            f_dc_mean, f_dc_std = _summarise(f_dc)
            f_geom_mean, _ = _summarise(f_geom)

            t_mode = _burst_time(annot, burst_idx, mode)
            dc = _nearest_estimate_at_time(annot.dc_estimates, t_mode)
            afr = _nearest_estimate_at_time(annot.azimuth_fm_rates, t_mode)
            k_a = _fm_rate_at_time_mode(annot, burst_idx, mode)
            k_psi = _steering_doppler_rate_at_time_mode(annot, burst_idx, mode)

            row.extend(
                [
                    f"{f_dc_mean:.6f}",
                    f"{f_dc_std:.6f}",
                    f"{f_geom_mean:.6f}",
                    f"{float(np.nanmean(k_a)):.6f}",
                    f"{k_psi:.6f}",
                    dc.azimuth_time.isoformat(),
                    afr.azimuth_time.isoformat(),
                ]
            )

        print(",".join(row))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare burst-wise Doppler estimates under different burst time conventions."
    )
    parser.add_argument("slc_safe", help="Path to the Sentinel-1 SLC SAFE directory")
    parser.add_argument("--subswath", default="iw1", help="Subswath, e.g. iw1/iw2/iw3")
    parser.add_argument("--pol", default="vv", help="Polarisation, e.g. vv or vh")
    parser.add_argument("--estimator", choices=("cde", "fft"), default="cde")
    parser.add_argument("--block-az", type=int, default=256)
    parser.add_argument("--block-rg", type=int, default=512)
    parser.add_argument("--stride-az", type=int, default=128)
    parser.add_argument("--stride-rg", type=int, default=256)
    args = parser.parse_args()

    run(
        slc_safe=args.slc_safe,
        subswath=args.subswath,
        polarisation=args.pol,
        estimator=args.estimator,
        block_az=args.block_az,
        block_rg=args.block_rg,
        stride_az=args.stride_az,
        stride_rg=args.stride_rg,
    )


if __name__ == "__main__":
    main()
