"""
Measure f_dc vs. azimuth position within each burst.

If deramping is correct, f_dc should be approximately FLAT along azimuth
within each burst (only geographic spatial variation, which is slow).

A linear RAMP in f_dc vs. azimuth eta (slope >> 0.1 Hz/s) indicates a
residual chirp from incorrect k_s — the dominant burst-offset mechanism.

Output columns:
  burst_idx, burst_mean_hz, first_third_hz, mid_third_hz, last_third_hz,
  azimuth_slope_hz_per_s, azimuth_slope_std_hz_per_s

Run:
    python -m scripts.diagnostics.burst_az_profile <slc_safe> [--subswath iw1]
"""
from __future__ import annotations

import argparse
import numpy as np

from scripts.sentinel_1.rvl import (
    apply_burst_valid_sample_mask,
    correlation_to_doppler,
    deramp_burst,
    estimate_correlation_grid,
)
from scripts.sentinel_1.safe_io import find_safe_files, parse_annotation, read_slc_burst


def _nanmean(a: np.ndarray) -> float:
    v = a[np.isfinite(a)]
    return float(np.mean(v)) if v.size else float("nan")


def run(
    slc_safe: str,
    subswath: str,
    polarisation: str = "vv",
    block_az: int = 64,
    stride_az: int = 32,
    block_rg: int = 512,
    stride_rg: int = 256,
    deramp_method: str = "current",
) -> None:
    files = find_safe_files(slc_safe, subswath, polarisation)
    annot = parse_annotation(files["annotation"])
    ati = annot.azimuth_time_interval

    header = (
        "burst_idx,burst_mean_hz,"
        "first_third_hz,mid_third_hz,last_third_hz,"
        "azimuth_slope_hz_per_s,slope_r2"
    )
    print(header)

    for burst_idx, burst in enumerate(annot.bursts):
        raw = read_slc_burst(files["measurement"], annot, burst_idx)
        deramped = deramp_burst(raw, annot, burst_idx, deramp_method=deramp_method)
        apply_burst_valid_sample_mask(deramped, burst)

        p0, p1, az_c, _ = estimate_correlation_grid(
            deramped, block_az, block_rg, stride_az, stride_rg,
        )
        f_dc, _, _ = correlation_to_doppler(p0, p1, annot.prf, annot.wavelength)

        # az_c is a list of azimuth block-centre line indices
        az_c_arr = np.asarray(az_c, dtype=np.float64)
        # per-azimuth-row mean across range
        n_az = f_dc.shape[0]
        row_means = np.array([_nanmean(f_dc[i, :]) for i in range(n_az)])
        eta = (az_c_arr - 0.5 * (annot.lines_per_burst - 1)) * ati  # [s] from burst centre

        # Third-split
        valid = np.isfinite(row_means)
        n_valid = np.count_nonzero(valid)
        burst_mean = float(np.nanmean(row_means))

        if n_valid >= 3:
            t3 = n_valid // 3
            idxs = np.where(valid)[0]
            first_mean = float(np.mean(row_means[idxs[:t3]]))
            mid_mean   = float(np.mean(row_means[idxs[t3:2*t3]]))
            last_mean  = float(np.mean(row_means[idxs[2*t3:]]))

            # Linear regression: f_dc = slope * eta + intercept
            x = eta[valid]
            y = row_means[valid]
            if x.size >= 2:
                coeffs = np.polyfit(x, y, deg=1)
                slope = float(coeffs[0])
                y_hat = np.polyval(coeffs, x)
                ss_res = float(np.sum((y - y_hat) ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            else:
                slope, r2 = float("nan"), float("nan")
        else:
            first_mean = mid_mean = last_mean = float("nan")
            slope = r2 = float("nan")

        print(
            f"{burst_idx},{burst_mean:.4f},"
            f"{first_mean:.4f},{mid_mean:.4f},{last_mean:.4f},"
            f"{slope:.6f},{r2:.4f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="f_dc azimuth profile within each burst.")
    p.add_argument("slc_safe")
    p.add_argument("--subswath", default="iw1")
    p.add_argument("--pol", default="vv")
    p.add_argument("--block-az", type=int, default=64,
                   help="Azimuth block size (small = finer profile, default 64)")
    p.add_argument("--stride-az", type=int, default=32)
    p.add_argument("--block-rg", type=int, default=512)
    p.add_argument("--stride-rg", type=int, default=256)
    p.add_argument("--deramp", default="current", choices=["current", "esa_eq1"])
    args = p.parse_args()
    run(
        slc_safe=args.slc_safe,
        subswath=args.subswath,
        polarisation=args.pol,
        block_az=args.block_az,
        stride_az=args.stride_az,
        block_rg=args.block_rg,
        stride_rg=args.stride_rg,
        deramp_method=args.deramp,
    )


if __name__ == "__main__":
    main()
