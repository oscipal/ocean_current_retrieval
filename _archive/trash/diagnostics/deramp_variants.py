"""Compare burst-wise FFT Doppler offsets under alternative deramp models."""

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

    if variant == "flip_chirp":
        return np.conj(deramp_burst(raw, annot, burst_idx)).astype(np.complex64)
    if variant == "flip_kpsi":
        k_psi = -k_psi
        k_s = -k_a * k_psi / (k_a - k_psi)
        return _apply_chirp(raw, annot, k_s)
    if variant == "plus_formula":
        k_s = -k_a * k_psi / (k_a + k_psi)
        return _apply_chirp(raw, annot, k_s)
    if variant == "neg_ka":
        return _apply_chirp(raw, annot, -k_a)
    if variant == "pos_ka":
        return _apply_chirp(raw, annot, k_a)
    if variant == "neg_kpsi":
        k_s = np.full_like(k_a, -k_psi)
        return _apply_chirp(raw, annot, k_s)
    if variant == "pos_kpsi":
        k_s = np.full_like(k_a, k_psi)
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


def run(
    slc_safe: str,
    subswath: str,
    polarisation: str = "vv",
    block_az: int = 256,
    block_rg: int = 512,
    stride_az: int = 128,
    stride_rg: int = 256,
) -> None:
    files = find_safe_files(slc_safe, subswath, polarisation)
    annot = parse_annotation(files["annotation"])
    variants = (
        "none",
        "current",
        "flip_kpsi",
        "plus_formula",
        "neg_ka",
        "pos_ka",
        "neg_kpsi",
        "pos_kpsi",
    )

    rows: list[dict[str, float | str | int]] = []
    for burst_idx, burst in enumerate(annot.bursts):
        raw = read_slc_burst(files["measurement"], annot, burst_idx)
        row: dict[str, float | str | int] = {
            "burst_idx": burst_idx,
            "burst_time": burst.azimuth_time.isoformat(),
        }

        for variant in variants:
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
            row[f"{variant}_mean_hz"] = mean_hz
            row[f"{variant}_std_hz"] = std_hz

        rows.append(row)

    for variant in variants:
        values = np.array([float(r[f"{variant}_mean_hz"]) for r in rows], dtype=np.float64)
        trend = _fit_trend(values)
        detrended = values - trend
        for row, offset in zip(rows, detrended):
            row[f"{variant}_offset_hz"] = float(offset)

    header = ["burst_idx", "burst_time"]
    for variant in variants:
        header.extend(
            [
                f"{variant}_mean_hz",
                f"{variant}_std_hz",
                f"{variant}_offset_hz",
            ]
        )
    print(",".join(header))

    for row in rows:
        values = [str(row["burst_idx"]), str(row["burst_time"])]
        for variant in variants:
            values.extend(
                [
                    f"{row[f'{variant}_mean_hz']:.6f}",
                    f"{row[f'{variant}_std_hz']:.6f}",
                    f"{row[f'{variant}_offset_hz']:.6f}",
                ]
            )
        print(",".join(values))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare burst-wise FFT Doppler offsets under alternative deramp models."
    )
    parser.add_argument("slc_safe", help="Path to the Sentinel-1 SLC SAFE directory")
    parser.add_argument("--subswath", default="iw1", help="Subswath, e.g. iw1/iw2/iw3")
    parser.add_argument("--pol", default="vv", help="Polarisation, e.g. vv or vh")
    parser.add_argument("--block-az", type=int, default=256)
    parser.add_argument("--block-rg", type=int, default=512)
    parser.add_argument("--stride-az", type=int, default=128)
    parser.add_argument("--stride-rg", type=int, default=256)
    args = parser.parse_args()

    run(
        slc_safe=args.slc_safe,
        subswath=args.subswath,
        polarisation=args.pol,
        block_az=args.block_az,
        block_rg=args.block_rg,
        stride_az=args.stride_az,
        stride_rg=args.stride_rg,
    )


if __name__ == "__main__":
    main()
