#!/usr/bin/env python3
"""Run the diagnostic single-burst Sentinel-1 RVL retrieval path."""

from __future__ import annotations

import argparse
import numpy as np
import xarray as xr

from .safe_io import find_safe_files, parse_annotation, read_slc_burst
from .rvl import (
    deramp_burst,
    estimate_correlation_grid,
    correlation_to_doppler,
    _geom_doppler_annotation,
    _geom_doppler_poeorb,
    _geolocate_grid,
    _interpolate_orbit,
    compute_gamma_ambiguity,
    compute_sideband_bias,
    apply_burst_valid_sample_mask,
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
    deramp_method: str = 'current',
    aux_cal_path: str | None = None,
    poeorb_path: str | None = None,
) -> xr.Dataset:
    """
    Compute the RVL for a single TOPS burst.

    Parameters
    ----------
    safe_dir    : str   Path to the .SAFE directory.
    subswath    : str   'iw1', 'iw2', or 'iw3'.
    burst_idx   : int   0-based burst index within the subswath.
    polarisation: str   'vv' (default) or 'vh'.
    block_az      : int   Estimation block height in azimuth lines.
    block_rg      : int   Estimation block width in range samples.
    stride_az     : int   Block stride in azimuth.
    stride_rg     : int   Block stride in range.
    deramp_method : str
        SAFE-based deramp method used in Step I.
    aux_cal_path  : str or None
        Path to the AUX_CAL .SAFE directory.  When provided applies the
        varpi_delta correction, sideband bias, and attitude mispointing
        correction (Section 5.5.1 eq. 9, Section 5.6).
    poeorb_path   : str or None
        Path to a POEORB / RESORB .EOF file.  When provided computes the
        full POEORB-based geometry Doppler and subtracts it directly from
        f_dc, bypassing the annotation polynomial entirely.

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
    from .aux_files import apply_poeorb, parse_aux_cal

    files = find_safe_files(safe_dir, subswath, polarisation)
    annot          = parse_annotation(files['annotation'])
    annot_original = annot   # preserve original SVs for POEORB differential

    # Optional: replace orbit SVs with precise POEORB
    if poeorb_path is not None:
        annot = apply_poeorb(annot, poeorb_path)

    # Optional: load AUX_CAL antenna pattern
    aap = None
    if aux_cal_path is not None:
        aap = parse_aux_cal(aux_cal_path, subswath, polarisation)

    burst = annot.bursts[burst_idx]

    # Effective velocity (orbital speed at burst centre)
    _, vel = _interpolate_orbit(annot, burst.azimuth_time)
    v_eff  = float(np.linalg.norm(vel))

    # AUX_CAL scalars (computed once for this burst)
    # Use hardware PRF (not azimuth output frequency) for AUX_CAL ambiguity corrections.
    gamma_amb  = compute_gamma_ambiguity(aap, annot.radar_prf, v_eff, annot.wavelength) \
                 if aap is not None else None
    f_sideband = compute_sideband_bias(aap, 0.0, annot.radar_prf, v_eff, annot.wavelength) \
                 if aap is not None else 0.0

    # ------------------------------------------------------------------
    # Step I — deramp (eq. 1–3)
    # ------------------------------------------------------------------
    raw      = read_slc_burst(files['measurement'], annot, burst_idx)
    deramped = deramp_burst(raw, annot, burst_idx, deramp_method=deramp_method)

    # Zero samples outside the annotation's per-line valid range.
    valid_mask = apply_burst_valid_sample_mask(deramped, burst)

    # ------------------------------------------------------------------
    # Step II — correlation coefficients with optional varpi_delta (eq. 9)
    # ------------------------------------------------------------------
    p0, p1, az_local, rg_centers = estimate_correlation_grid(
        deramped, block_az, block_rg, stride_az, stride_rg,
        valid_mask=valid_mask,
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
    f_dc, _, snr = correlation_to_doppler(
        p0, p1, annot.prf, annot.wavelength, gamma_amb=gamma_amb,
    )

    # Geometry Doppler: POEORB (preferred) or annotation polynomial (fallback).
    # With POEORB the full geometry is computed from the precise orbit directly;
    # the annotation polynomial does not appear in the output formula.
    if poeorb_path is not None:
        f_geom_poe = _geom_doppler_poeorb(
            annot, annot_original, burst_idx, rg_centers,
        ).astype(np.float32)
        f_geom_ann = _geom_doppler_annotation(annot_original, burst_idx, rg_centers).astype(np.float32)
        mispointing_source = 'poeorb'
    else:
        f_geom_ann = _geom_doppler_annotation(annot, burst_idx, rg_centers).astype(np.float32)
        f_geom_poe = f_geom_ann
        mispointing_source = 'none'

    f_dca = f_dc - f_geom_poe[np.newaxis, :]   # broadcast over azimuth cells

    # Subtract sideband bias (scalar → broadcast)
    f_dca = f_dca - float(f_sideband)

    # Diagnostic: mispointing differential (POEORB − annotation)
    f_miss_arr = (f_geom_poe - f_geom_ann).astype(np.float32)


    v_r = (annot.wavelength / 2.0 * f_dca).astype(np.float32)

    # ------------------------------------------------------------------
    # Step V — geolocation (Section 5.10)
    # ------------------------------------------------------------------
    lat, lon, inc = _geolocate_grid(annot, az_full, rg_centers)

    dims = ('az_cell', 'rg_cell')

    f_geom_ann_2d = np.broadcast_to(f_geom_ann[np.newaxis, :], f_dca.shape).astype(np.float32).copy()
    f_geom_poe_2d = np.broadcast_to(f_geom_poe[np.newaxis, :], f_dca.shape).astype(np.float32).copy()
    f_miss_2d     = np.broadcast_to(f_miss_arr[np.newaxis, :], f_dca.shape).astype(np.float32).copy()

    return xr.Dataset(
        {
            'doppler_hz': xr.DataArray(
                f_dca, dims=dims,
                attrs={'long_name': 'Doppler centroid anomaly (all corrections applied)',
                       'units': 'Hz'},
            ),
            'doppler_obs': xr.DataArray(
                f_dc, dims=dims,
                attrs={'long_name': 'Observed Doppler centroid (raw, before subtraction)',
                       'units': 'Hz'},
            ),
            'doppler_geo': xr.DataArray(
                f_geom_ann_2d, dims=dims,
                attrs={'long_name': 'Geometry Doppler — annotation polynomial',
                       'units': 'Hz'},
            ),
            'doppler_geo_poeorb': xr.DataArray(
                f_geom_poe_2d, dims=dims,
                attrs={'long_name': 'Geometry Doppler — POEORB numerical (0 if no POEORB)',
                       'units': 'Hz'},
            ),
            'doppler_miss': xr.DataArray(
                f_miss_2d, dims=dims,
                attrs={'long_name': 'POEORB mispointing correction (f_geom_poeorb − f_geom_ann)',
                       'units': 'Hz'},
            ),
            'radial_vel': xr.DataArray(
                v_r, dims=dims,
                attrs={'long_name': 'Radial surface velocity (all corrections applied)',
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
            'deramp_method':       deramp_method,
            'poeorb_applied':      poeorb_path is not None,
            'aux_cal_applied':     aux_cal_path is not None,
            'mispointing_source':  mispointing_source,
            'mispointing_hz':      float(np.mean(f_miss_arr)),
            'gamma_ambiguity':     float(gamma_amb) if gamma_amb is not None else float('nan'),
            'sideband_bias_hz':    float(f_sideband),
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
    p.add_argument('--block-az',  type=int, default=256, metavar='N')
    p.add_argument('--block-rg',  type=int, default=512, metavar='N')
    p.add_argument('--stride-az', type=int, default=128, metavar='N')
    p.add_argument('--stride-rg', type=int, default=256, metavar='N')
    p.add_argument('--deramp-method', choices=('current', 'esa_eq1'), default='current',
                   help='SAFE-based deramp method for Step I')
    p.add_argument('--aux-cal',   default=None, metavar='SAFE',
                   help='AUX_CAL .SAFE directory (enables varpi_delta / sideband / mispointing)')
    p.add_argument('--poeorb',    default=None, metavar='EOF',
                   help='POEORB or RESORB .EOF file (replaces annotation orbit SVs)')
    p.add_argument('--out',       default='burst_rvl.nc',
                   help='Output NetCDF path')
    p.add_argument('--plot',      action='store_true',
                   help='Show quick-look figure')
    return p


if __name__ == '__main__':
    args = _build_parser().parse_args()

    print(f'Processing {args.subswath.upper()} burst {args.burst_idx} …')
    ds = compute_rvl_burst(
        safe_dir     = args.safe_dir,
        subswath     = args.subswath,
        burst_idx    = args.burst_idx,
        polarisation = args.pol,
        block_az     = args.block_az,
        block_rg     = args.block_rg,
        stride_az    = args.stride_az,
        stride_rg    = args.stride_rg,
        deramp_method = args.deramp_method,
        aux_cal_path = args.aux_cal,
        poeorb_path  = args.poeorb,
    )

    ds.to_netcdf(args.out)
    print(f'Saved → {args.out}')
    print(ds)

    if args.plot:
        _plot(ds, args.burst_idx)
