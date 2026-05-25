"""Compare the current deramp against ESA TOPS deramp and deramp+demod formulas."""

from __future__ import annotations

import argparse
from datetime import timedelta

import numpy as np

from scripts.sentinel_1.rvl import (
    apply_burst_valid_sample_mask,
    deramp_burst,
    estimate_doppler_grid_fft,
)
from scripts.sentinel_1.safe_io import find_safe_files, parse_annotation, read_slc_burst, slant_range_time_vector

C_LIGHT = 299_792_458.0


def _eval_poly(poly: np.ndarray, t0: float, tau: np.ndarray) -> np.ndarray:
    dt = tau - t0
    return sum(c * dt**k for k, c in enumerate(poly))


def _nearest_estimate_at_time(estimates, t):
    diffs = [abs((e.azimuth_time - t).total_seconds()) for e in estimates]
    return estimates[int(np.argmin(diffs))]


def _burst_mid_time(annot, burst_idx: int):
    burst = annot.bursts[burst_idx]
    half_dt = 0.5 * (annot.lines_per_burst - 1) * annot.azimuth_time_interval
    return burst.azimuth_time + timedelta(seconds=half_dt)


def _interpolate_velocity(annot, t) -> np.ndarray:
    t0 = annot.orbit_times[0]
    times = np.array([(ot - t0).total_seconds() for ot in annot.orbit_times], dtype=np.float64)
    ts = (t - t0).total_seconds()
    return np.array(
        [np.interp(ts, times, [v[i] for v in annot.orbit_velocities]) for i in range(3)],
        dtype=np.float64,
    )


def _eta_axis(annot) -> np.ndarray:
    lpb = annot.lines_per_burst
    return (np.arange(lpb, dtype=np.float64) - 0.5 * (lpb - 1)) * annot.azimuth_time_interval


def _esa_phase(raw: np.ndarray, annot, burst_idx: int, demodulate: bool) -> np.ndarray:
    burst_mid = _burst_mid_time(annot, burst_idx)
    tau = slant_range_time_vector(annot).astype(np.float64)
    eta = _eta_axis(annot)[:, np.newaxis]

    afr = _nearest_estimate_at_time(annot.azimuth_fm_rates, burst_mid)
    dc = _nearest_estimate_at_time(annot.dc_estimates, burst_mid)

    k_a = _eval_poly(afr.poly, afr.t0, tau).astype(np.float64)
    f_dc = _eval_poly(dc.data_poly, dc.t0, tau).astype(np.float64)

    vel = _interpolate_velocity(annot, burst_mid)
    v_s = float(np.linalg.norm(vel))
    k_psi = annot.azimuth_steering_rate * np.pi / 180.0
    k_s = (2.0 * v_s / C_LIGHT) * annot.radar_frequency * k_psi
    k_t = (k_a * k_s) / (k_a - k_s)

    eta_c = -f_dc / k_a
    eta_ref = eta_c - eta_c[annot.n_samples // 2]

    phase = -np.pi * k_t[np.newaxis, :] * (eta - eta_ref[np.newaxis, :]) ** 2
    if demodulate:
        phase = phase - 2.0 * np.pi * f_dc[np.newaxis, :] * (eta - eta_ref[np.newaxis, :])

    phi = np.exp(1j * phase)
    return (raw * phi).astype(np.complex64)


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


def _edge_step_metric(grids: list[np.ndarray], edge_rows: int) -> tuple[float, float]:
    diffs: list[np.ndarray] = []
    for left, right in zip(grids[:-1], grids[1:]):
        n = min(edge_rows, left.shape[0], right.shape[0])
        if n == 0:
            continue
        delta = right[:n, :] - left[-n:, :]
        finite = delta[np.isfinite(delta)]
        if finite.size:
            diffs.append(finite.astype(np.float64))
    if not diffs:
        return float("nan"), float("nan")
    all_diffs = np.concatenate(diffs)
    rms = float(np.sqrt(np.mean(all_diffs**2)))
    mad = float(np.median(np.abs(all_diffs)))
    return rms, mad


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
    variants = ("none", "current", "esa_eq1", "esa_eq8")

    print(
        "variant,"
        "offset_rms_hz,offset_mad_hz,"
        "mean_within_burst_std_hz,"
        "edge_step_rms_hz,edge_step_mad_hz"
    )

    for variant in variants:
        means = []
        stds = []
        grids: list[np.ndarray] = []

        for burst_idx, burst in enumerate(annot.bursts):
            raw = read_slc_burst(files["measurement"], annot, burst_idx)
            if variant == "none":
                data = raw.copy()
            elif variant == "current":
                data = deramp_burst(raw, annot, burst_idx)
            elif variant == "esa_eq1":
                data = _esa_phase(raw, annot, burst_idx, demodulate=False)
            elif variant == "esa_eq8":
                data = _esa_phase(raw, annot, burst_idx, demodulate=True)
            else:
                raise ValueError(variant)

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
            means.append(mean_hz)
            stds.append(std_hz)
            grids.append(f_dc)

        means_arr = np.array(means, dtype=np.float64)
        trend = _fit_trend(means_arr)
        offsets = means_arr - trend
        finite_offsets = offsets[np.isfinite(offsets)]
        offset_rms = float(np.sqrt(np.mean(finite_offsets**2)))
        offset_mad = float(np.median(np.abs(finite_offsets)))
        mean_std = float(np.nanmean(np.array(stds, dtype=np.float64)))
        edge_rms, edge_mad = _edge_step_metric(grids, edge_rows=edge_rows)

        print(
            f"{variant},"
            f"{offset_rms:.6f},{offset_mad:.6f},"
            f"{mean_std:.6f},"
            f"{edge_rms:.6f},{edge_mad:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the current deramp against ESA Eq. 1 and Eq. 8 implementations."
    )
    parser.add_argument("slc_safe", help="Path to the Sentinel-1 SLC SAFE directory")
    parser.add_argument("--subswath", default="iw1", help="Subswath, e.g. iw1/iw2/iw3")
    parser.add_argument("--pol", default="vv", help="Polarisation, e.g. vv or vh")
    parser.add_argument("--block-az", type=int, default=256)
    parser.add_argument("--block-rg", type=int, default=512)
    parser.add_argument("--stride-az", type=int, default=128)
    parser.add_argument("--stride-rg", type=int, default=256)
    parser.add_argument("--edge-rows", type=int, default=2)
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
