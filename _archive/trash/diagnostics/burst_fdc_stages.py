"""Compare burst-wise Doppler offsets across raw, deramped, CDE, and FFT stages."""

from __future__ import annotations

import argparse

import numpy as np

from scripts.sentinel_1.rvl import (
    apply_burst_valid_sample_mask,
    correlation_to_doppler,
    deramp_burst,
    estimate_correlation_grid,
    estimate_doppler_grid_fft,
)
from scripts.sentinel_1.safe_io import find_safe_files, parse_annotation, read_slc_burst


def _summarise(arr: np.ndarray) -> tuple[float, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.nanmean(finite)), float(np.nanstd(finite))


def _estimate_fdc(
    data: np.ndarray,
    annot,
    burst,
    estimator: str,
    block_az: int,
    block_rg: int,
    stride_az: int,
    stride_rg: int,
) -> np.ndarray:
    work = data.copy()
    valid_mask = apply_burst_valid_sample_mask(work, burst)

    if estimator == "fft":
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

    p0, p1, _, _ = estimate_correlation_grid(
        work,
        block_az,
        block_rg,
        stride_az,
        stride_rg,
        valid_mask=valid_mask,
    )
    f_dc, _, _ = correlation_to_doppler(p0, p1, annot.prf, annot.wavelength)
    return f_dc


def _fit_trend(y: np.ndarray) -> np.ndarray:
    idx = np.arange(y.size, dtype=np.float64)
    ok = np.isfinite(y)
    if np.count_nonzero(ok) < 2:
        return np.full_like(y, np.nan, dtype=np.float64)
    coeff = np.polyfit(idx[ok], y[ok], deg=1)
    return np.polyval(coeff, idx)


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

    rows: list[dict[str, float | str | int]] = []
    for burst_idx, burst in enumerate(annot.bursts):
        raw = read_slc_burst(files["measurement"], annot, burst_idx)
        deramped = deramp_burst(raw, annot, burst_idx)

        raw_cde = _estimate_fdc(raw, annot, burst, "cde", block_az, block_rg, stride_az, stride_rg)
        raw_fft = _estimate_fdc(raw, annot, burst, "fft", block_az, block_rg, stride_az, stride_rg)
        deramp_cde = _estimate_fdc(deramped, annot, burst, "cde", block_az, block_rg, stride_az, stride_rg)
        deramp_fft = _estimate_fdc(deramped, annot, burst, "fft", block_az, block_rg, stride_az, stride_rg)

        raw_cde_mean, raw_cde_std = _summarise(raw_cde)
        raw_fft_mean, raw_fft_std = _summarise(raw_fft)
        deramp_cde_mean, deramp_cde_std = _summarise(deramp_cde)
        deramp_fft_mean, deramp_fft_std = _summarise(deramp_fft)

        rows.append(
            {
                "burst_idx": burst_idx,
                "burst_time": burst.azimuth_time.isoformat(),
                "raw_cde_mean_hz": raw_cde_mean,
                "raw_cde_std_hz": raw_cde_std,
                "raw_fft_mean_hz": raw_fft_mean,
                "raw_fft_std_hz": raw_fft_std,
                "deramp_cde_mean_hz": deramp_cde_mean,
                "deramp_cde_std_hz": deramp_cde_std,
                "deramp_fft_mean_hz": deramp_fft_mean,
                "deramp_fft_std_hz": deramp_fft_std,
            }
        )

    for key in ("raw_cde_mean_hz", "raw_fft_mean_hz", "deramp_cde_mean_hz", "deramp_fft_mean_hz"):
        values = np.array([float(r[key]) for r in rows], dtype=np.float64)
        trend = _fit_trend(values)
        detrended = values - trend
        for row, trend_i, detrended_i in zip(rows, trend, detrended):
            base = key.replace("_mean_hz", "")
            row[f"{base}_trend_hz"] = float(trend_i)
            row[f"{base}_offset_hz"] = float(detrended_i)

    header = [
        "burst_idx",
        "burst_time",
        "raw_cde_mean_hz",
        "raw_cde_std_hz",
        "raw_cde_offset_hz",
        "raw_fft_mean_hz",
        "raw_fft_std_hz",
        "raw_fft_offset_hz",
        "deramp_cde_mean_hz",
        "deramp_cde_std_hz",
        "deramp_cde_offset_hz",
        "deramp_fft_mean_hz",
        "deramp_fft_std_hz",
        "deramp_fft_offset_hz",
    ]
    print(",".join(header))
    for row in rows:
        print(
            ",".join(
                [
                    str(row["burst_idx"]),
                    str(row["burst_time"]),
                    f"{row['raw_cde_mean_hz']:.6f}",
                    f"{row['raw_cde_std_hz']:.6f}",
                    f"{row['raw_cde_offset_hz']:.6f}",
                    f"{row['raw_fft_mean_hz']:.6f}",
                    f"{row['raw_fft_std_hz']:.6f}",
                    f"{row['raw_fft_offset_hz']:.6f}",
                    f"{row['deramp_cde_mean_hz']:.6f}",
                    f"{row['deramp_cde_std_hz']:.6f}",
                    f"{row['deramp_cde_offset_hz']:.6f}",
                    f"{row['deramp_fft_mean_hz']:.6f}",
                    f"{row['deramp_fft_std_hz']:.6f}",
                    f"{row['deramp_fft_offset_hz']:.6f}",
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare burst-wise Doppler offsets before and after deramping with CDE and FFT estimators."
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
