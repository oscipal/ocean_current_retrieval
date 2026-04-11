#!/usr/bin/env python3
"""
Single-burst RVL pipeline: SAFE directory → OCN-level Doppler / radial velocity.

Implements the ESA OCN RVL algorithm (Engen & Johnsen, DI-MPC-RVL-0534) for a
single selected TOPS burst, removing all multi-burst complexity as a diagnostic
baseline.

Steps executed
--------------
  I   Deramp the burst (Section 5.4, eq. 1–3).
      No spectral windowing (eq. 4) or coherent merging (eq. 5–6) is required
      for a single burst with no overlap partner.
  II  Estimate azimuth correlation coefficients p0, p1 per block (Section 5.5.1,
      eqs. 12–13 — two-stage Hanning-windowed estimator).
  III Doppler centroid; subtract annotation geometry Doppler (Section 5.7).
      A single burst has one unambiguous geometry-DC polynomial so no blending
      is needed.
  V   Geolocate the output grid (Section 5.10).

Steps deliberately omitted
--------------------------
  Step I eq. 4–6  (burst spectral window + merge) — not applicable.
  Step IV         (descalloping) — burst-periodic detection requires ≥2 bursts.

Usage
-----
  python s1_rvl_burst.py <SAFE_DIR> <SUBSWATH> <BURST_IDX> [options]

  positional arguments:
    safe_dir              Path to the .SAFE directory
    subswath              iw1 | iw2 | iw3
    burst_idx             0-based burst index

  optional arguments:
    --pol VV|VH           Polarisation (default: vv)
    --block-az N          Estimation block height [lines]  (default: 256)
    --block-rg N          Estimation block width  [samples](default: 512)
    --stride-az N         Azimuth stride           (default: 128)
    --stride-rg N         Range stride             (default: 256)
    --out FILE            Save result as NetCDF (default: burst_rvl.nc)
    --plot                Show a quick-look figure after processing
"""

from __future__ import annotations

import argparse
import sys
import os

import numpy as np
import xarray as xr

# Allow running as a top-level script from any working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.s1_io import find_safe_files, parse_annotation, read_slc_burst
from scripts.s1_rvl import (
    deramp_burst,
    estimate_correlation_grid,
    correlation_to_doppler,
    _geom_doppler_annotation,
    _geolocate_grid,
)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def compute_rvl_burst(
    safe_dir: str,
    subswath: str,
    burst_idx: int,
    polarisation: str = 'vv',
    block_az: int = 256,
    block_rg: int = 512,
    stride_az: int = 128,
    stride_rg: int = 256,
) -> xr.Dataset:
    """
    Compute the RVL for a single TOPS burst.

    Parameters
    ----------
    safe_dir    : str   Path to the .SAFE directory.
    subswath    : str   'iw1', 'iw2', or 'iw3'.
    burst_idx   : int   0-based burst index within the subswath.
    polarisation: str   'vv' (default) or 'vh'.
    block_az    : int   Estimation block height in azimuth lines.
    block_rg    : int   Estimation block width in range samples.
    stride_az   : int   Block stride in azimuth.
    stride_rg   : int   Block stride in range.

    Returns
    -------
    xr.Dataset with variables:
        doppler_hz  [Hz]    Doppler centroid after geometry subtraction
        radial_vel  [m/s]   Radial surface velocity
        snr         [-]     Signal-to-noise ratio estimate
    Coordinates:
        latitude, longitude, incidence_angle
        az_pixel (full-scene line index), rg_pixel
    """
    files = find_safe_files(safe_dir, subswath, polarisation)
    annot = parse_annotation(files['annotation'])

    burst = annot.bursts[burst_idx]

    # ------------------------------------------------------------------
    # Step I — deramp (eq. 1–3)
    # ------------------------------------------------------------------
    raw      = read_slc_burst(files['measurement'], annot, burst_idx)
    deramped = deramp_burst(raw, annot, burst_idx)

    # Zero lines that the annotation flags as invalid (ramp-up / ramp-down edges).
    # These are all-zero in the raw SLC; zeroing them here ensures the
    # min_valid_fraction check in estimate_correlation_grid works correctly.
    valid_lines = burst.first_valid_sample != -1
    deramped[~valid_lines, :] = 0.0

    # ------------------------------------------------------------------
    # Step II — azimuth correlation coefficients (eqs. 12–13)
    # ------------------------------------------------------------------
    p0, p1, az_local, rg_centers = estimate_correlation_grid(
        deramped, block_az, block_rg, stride_az, stride_rg,
    )

    # az_local is burst-local [0, linesPerBurst).
    # Convert to full-scene line indices for geolocation.
    ati = annot.azimuth_time_interval
    az0_full = int(round(
        (burst.azimuth_time - annot.first_line_time).total_seconds() / ati
    ))
    az_full = az_local + az0_full

    # ------------------------------------------------------------------
    # Step III — Doppler centroid, subtract geometry Doppler (eq. 20)
    # ------------------------------------------------------------------
    f_dc, _, snr = correlation_to_doppler(p0, p1, annot.prf, annot.wavelength)

    # Single burst → one geometry polynomial, no blending.
    f_geom = _geom_doppler_annotation(annot, burst_idx, rg_centers).astype(np.float32)
    f_dca  = f_dc - f_geom[np.newaxis, :]      # broadcast over azimuth cells

    v_r = (annot.wavelength / 2.0 * f_dca).astype(np.float32)

    # ------------------------------------------------------------------
    # Step V — geolocation (Section 5.10)
    # ------------------------------------------------------------------
    lat, lon, inc = _geolocate_grid(annot, az_full, rg_centers)

    dims = ('az_cell', 'rg_cell')

    # Broadcast f_geom to 2-D for storage
    f_geom_2d = np.broadcast_to(f_geom[np.newaxis, :], f_dca.shape).astype(np.float32).copy()

    return xr.Dataset(
        {
            'doppler_hz': xr.DataArray(
                f_dca, dims=dims,
                attrs={'long_name': 'Doppler centroid (geometry subtracted)',
                       'units': 'Hz'},
            ),
            'doppler_obs': xr.DataArray(
                f_dc, dims=dims,
                attrs={'long_name': 'Observed Doppler centroid (raw, before geometry subtraction)',
                       'units': 'Hz'},
            ),
            'doppler_geo': xr.DataArray(
                f_geom_2d, dims=dims,
                attrs={'long_name': 'Geometry Doppler from annotation polynomial',
                       'units': 'Hz'},
            ),
            'radial_vel': xr.DataArray(
                v_r, dims=dims,
                attrs={'long_name': 'Radial surface velocity',
                       'units': 'm s-1'},
            ),
            'snr': xr.DataArray(
                snr, dims=dims,
                attrs={'long_name': 'Signal-to-noise ratio estimate'},
            ),
        },
        coords={
            'latitude':        (dims, lat),
            'longitude':       (dims, lon),
            'incidence_angle': (dims, inc),
            'az_pixel':        ('az_cell', az_full),
            'rg_pixel':        ('rg_cell', rg_centers),
        },
        attrs={
            'subswath':            annot.subswath,
            'polarisation':        annot.polarisation,
            'burst_index':         burst_idx,
            'burst_azimuth_time':  str(burst.azimuth_time),
            'radar_frequency_hz':  annot.radar_frequency,
            'wavelength_m':        annot.wavelength,
            'prf_hz':              annot.prf,
            'block_az_samples':    block_az,
            'block_rg_samples':    block_rg,
            'stride_az':           stride_az,
            'stride_rg':           stride_rg,
            'algorithm_ref':       'Engen & Johnsen, DI-MPC-RVL-0534, steps I/II/III/V',
        },
    )


# ---------------------------------------------------------------------------
# Quick-look plot
# ---------------------------------------------------------------------------

def _plot(ds: xr.Dataset, burst_idx: int) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Single-burst RVL — burst {burst_idx}')

    for ax, var, label, cmap in zip(
        axes,
        ['doppler_hz', 'radial_vel', 'snr'],
        ['Doppler centroid [Hz]', 'Radial velocity [m/s]', 'SNR'],
        ['RdBu_r', 'RdBu_r', 'viridis'],
    ):
        data = ds[var].values
        vmax = np.nanpercentile(np.abs(data), 98) if var != 'snr' else np.nanpercentile(data, 98)
        vmin = -vmax if var != 'snr' else 0

        im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.set_xlabel('Range cell')
        ax.set_ylabel('Azimuth cell')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Single-burst Sentinel-1 RVL: SAFE → OCN product',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('safe_dir',   help='Path to the .SAFE directory')
    p.add_argument('subswath',   help='iw1 | iw2 | iw3')
    p.add_argument('burst_idx',  type=int, help='0-based burst index')
    p.add_argument('--pol',      default='vv', help='Polarisation: vv or vh')
    p.add_argument('--block-az', type=int, default=256, metavar='N')
    p.add_argument('--block-rg', type=int, default=512, metavar='N')
    p.add_argument('--stride-az', type=int, default=128, metavar='N')
    p.add_argument('--stride-rg', type=int, default=256, metavar='N')
    p.add_argument('--out',      default='burst_rvl.nc',
                   help='Output NetCDF path')
    p.add_argument('--plot',     action='store_true',
                   help='Show quick-look figure')
    return p


if __name__ == '__main__':
    args = _build_parser().parse_args()

    print(f'Processing {args.subswath.upper()} burst {args.burst_idx} …')
    ds = compute_rvl_burst(
        safe_dir    = args.safe_dir,
        subswath    = args.subswath,
        burst_idx   = args.burst_idx,
        polarisation= args.pol,
        block_az    = args.block_az,
        block_rg    = args.block_rg,
        stride_az   = args.stride_az,
        stride_rg   = args.stride_rg,
    )

    ds.to_netcdf(args.out)
    print(f'Saved → {args.out}')
    print(ds)

    if args.plot:
        _plot(ds, args.burst_idx)
