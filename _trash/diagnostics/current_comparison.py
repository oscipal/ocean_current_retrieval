#!/usr/bin/env python3
"""Compare burst-level RVL retrievals against OCN, ERA5, and GLO12 corrections."""

from __future__ import annotations

import argparse

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scripts.sentinel_1.metocean import (
    compute_stokes_radial,
    compute_wave_doppler_bias,
    load_era5_wave,
    load_era5_wind,
    load_glo12_current,
    load_ocn_rvl,
    match_to_sar_grid,
    project_current_onto_look,
)
from scripts.sentinel_1.aux_files import apply_poeorb
from scripts.sentinel_1.burst_pipeline import compute_rvl_burst


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def compute_current_velocity(
    slc_safe:     str,
    ocn_safe:     str | None,
    subswath:     str,
    burst_idx:    int,
    era5_wind:    str,
    era5_wave:    str,
    glo12:        str,
    polarisation: str  = 'vv',
    block_az:     int  = 256,
    block_rg:     int  = 512,
    stride_az:    int  = 128,
    stride_rg:    int  = 256,
    aux_cal_path: str | None = None,
    poeorb_path:  str | None = None,
) -> dict:
    """
    Returns a dict with keys:
      v_sar_current   SAR-derived current-only radial velocity [m/s]
      v_ocn_current   OCN-derived current-only radial velocity [m/s]  (NaN if no OCN)
      v_model         glo12 model current projected onto look  [m/s]
      v_stokes        Stokes drift radial component            [m/s]
      v_wave          Wave Doppler bias                        [m/s]
      v_miss          Mispointing correction used for SAR      [m/s]
      our_ds          full xr.Dataset from compute_rvl_burst
      lat, lon, inc   SAR grid coordinates
      mispointing_source  'ocn' | 'attitude' | 'none'

    Mispointing correction
    ----------------------
    When *poeorb_path* is provided, ``compute_rvl_burst`` computes the full
    POEORB-based geometry Doppler and subtracts it directly from f_dc.  The
    annotation polynomial is NOT used in this path.  The resulting ``our_rv``
    is already mispointing-free, and v_miss is recorded for reference only.

    When only *ocn_safe* is provided (no POEORB), mispointing is NOT corrected
    in ``our_rv``; the v_miss from OCN rvlDcMiss is recorded for comparison.

    Final current formula (both cases):
        v_sar_current = −our_rv − v_stokes − v_wave
    """
    # ---- Step 1: single-burst RVL (with optional aux corrections) --------
    print('  Running single-burst RVL …')
    our_ds = compute_rvl_burst(
        safe_dir=slc_safe, subswath=subswath, burst_idx=burst_idx,
        polarisation=polarisation,
        block_az=block_az, block_rg=block_rg,
        stride_az=stride_az, stride_rg=stride_rg,
        aux_cal_path=aux_cal_path,
        poeorb_path=poeorb_path,
    )
    acq_time = our_ds.attrs['burst_azimuth_time']
    our_lat  = our_ds['latitude'].values
    our_lon  = our_ds['longitude'].values
    our_inc  = our_ds['incidence_angle'].values
    our_rv   = our_ds['radial_vel'].values      # already has aux corrections if applied

    # ---- Step 2: look direction ------------------------------------------
    # Prefer heading from OCN; fall back to orbit-derived heading.
    print('  Computing look direction …')
    if ocn_safe is not None:
        ocn_raw = load_ocn_rvl(ocn_safe, subswath, polarisation)
        heading = float(np.nanmean(ocn_raw['heading']))
    else:
        ocn_raw = None
        # Derive heading from POEORB or annotation orbit state vectors
        from scripts.sentinel_1.safe_io import find_safe_files, parse_annotation
        from scripts.sentinel_1.rvl import _interpolate_orbit
        ann_path = find_safe_files(slc_safe, subswath, polarisation)['annotation']
        _ann = parse_annotation(ann_path)
        if poeorb_path is not None:
            _ann = apply_poeorb(_ann, poeorb_path)
        from scripts.sentinel_1.safe_io import _iso_to_datetime
        burst_t = _iso_to_datetime(acq_time)
        pos, vel = _interpolate_orbit(_ann, burst_t)
        # Heading = azimuth of velocity vector (degrees clockwise from North)
        # Convert ECEF velocity to ENU at the sub-satellite point for a
        # correct heading regardless of latitude.
        lat_s = np.arcsin(pos[2] / np.linalg.norm(pos))
        lon_s = np.arctan2(pos[1], pos[0])
        e_east  = np.array([-np.sin(lon_s),  np.cos(lon_s), 0.0])
        e_north = np.array([-np.sin(lat_s) * np.cos(lon_s),
                            -np.sin(lat_s) * np.sin(lon_s),
                             np.cos(lat_s)])
        v_e = float(np.dot(vel, e_east))
        v_n = float(np.dot(vel, e_north))
        heading = float(np.rad2deg(np.arctan2(v_e, v_n)))
    look_az = np.deg2rad(heading + 90.0)           # right-looking
    print(f'    heading={heading:.1f} deg  ->  look_az={np.rad2deg(look_az):.1f} deg')

    # ---- Step 3: mispointing note ------------------------------------------
    # When POEORB was provided, compute_rvl_burst used the full POEORB-based
    # geometry Doppler (f_dc - f_geom_poeorb) so our_rv is already corrected.
    # When only OCN is available, our_rv used the annotation polynomial which
    # does NOT include mispointing — v_miss from OCN rvlDcMiss is for reference.
    # In either case v_miss is NOT added to the formula below; it is stored for
    # diagnostic purposes only.
    wavelength = our_ds.attrs['wavelength_m']
    if ocn_raw is not None:
        dc_miss_ocn = match_to_sar_grid(
            our_lat, our_lon, ocn_raw['lat'], ocn_raw['lon'], ocn_raw['dc_miss'],
        )
        v_miss = (wavelength / 2.0 * dc_miss_ocn).astype(np.float32)
        mispointing_source = 'ocn'
    elif our_ds.attrs.get('mispointing_source') in ('poeorb', 'attitude'):
        f_miss_hz = float(our_ds.attrs.get('mispointing_hz', 0.0))
        v_miss = np.full(our_lat.shape, wavelength / 2.0 * f_miss_hz, dtype=np.float32)
        mispointing_source = our_ds.attrs['mispointing_source'] + '_ref_only'
    else:
        v_miss = np.zeros(our_lat.shape, dtype=np.float32)
        mispointing_source = 'none'
    print(f'  Mispointing for reference ({mispointing_source}): '
          f'[{np.nanmin(v_miss):.3f}, {np.nanmax(v_miss):.3f}] m/s (NOT applied to our_rv)')

    # ---- Step 4: Stokes drift correction ---------------------------------
    print('  Loading ERA5 Stokes drift …')
    wave = load_era5_wave(era5_wave, acq_time)
    print(f'    ERA5 wave time: {wave["time"]}')
    v_stokes = compute_stokes_radial(wave, our_lat, our_lon, our_inc, look_az)
    print(f'    v_stokes range: [{np.nanmin(v_stokes):.3f}, {np.nanmax(v_stokes):.3f}] m/s')

    # ---- Step 5: wave Doppler bias ---------------------------------------
    print('  Loading ERA5 wind …')
    wind = load_era5_wind(era5_wind, acq_time)
    print(f'    ERA5 wind time: {wind["time"]}')
    v_wave = compute_wave_doppler_bias(wind, our_lat, our_lon, our_inc, look_az)
    print(f'    v_wave range: [{np.nanmin(v_wave):.3f}, {np.nanmax(v_wave):.3f}] m/s')

    # ---- Step 6: current-only SAR velocity (away-from-satellite = positive) -
    # our_rv = λ/2 · f_dca  (toward satellite = positive).
    # Apply OCN mispointing when available, matching the walkthrough notebook:
    #   v_current = −our_rv + v_miss − v_stokes − v_wave
    if ocn_raw is not None:
        v_sar_current = (-our_rv + v_miss - v_stokes - v_wave).astype(np.float32)
    else:
        v_sar_current = (-our_rv - v_stokes - v_wave).astype(np.float32)

    # ---- Step 7: OCN current-only velocity (only when OCN product available)
    if ocn_raw is not None:
        ocn_rv_matched = match_to_sar_grid(
            our_lat, our_lon, ocn_raw['lat'], ocn_raw['lon'], ocn_raw['rad_vel'],
        )
        v_ocn_current = (ocn_rv_matched - v_stokes - v_wave).astype(np.float32)
    else:
        ocn_rv_matched = np.full(our_lat.shape, np.nan, dtype=np.float32)
        v_ocn_current  = np.full(our_lat.shape, np.nan, dtype=np.float32)

    # ---- Step 8: glo12 reference current ---------------------------------
    print('  Loading glo12 model current …')
    model = load_glo12_current(glo12, acq_time)
    print(f'    glo12 time: {model["time"]}')
    v_model = project_current_onto_look(model, our_lat, our_lon, our_inc, look_az)
    print(f'    v_model range: [{np.nanmin(v_model):.3f}, {np.nanmax(v_model):.3f}] m/s')

    return {
        'v_sar_current':      v_sar_current.astype(np.float32),
        'v_ocn_current':      v_ocn_current.astype(np.float32),
        'v_model':            v_model,
        'v_stokes':           v_stokes,
        'v_wave':             v_wave,
        'v_miss':             v_miss,
        'ocn_rv':             ocn_rv_matched,
        'our_ds':             our_ds,
        'lat':                our_lat,
        'lon':                our_lon,
        'inc':                our_inc,
        'mispointing_source': mispointing_source,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _map(ax, lon, lat, data, title, vmin, vmax, cmap='RdBu_r'):
    sc = ax.scatter(lon.ravel(), lat.ravel(), c=data.ravel(),
                    s=6, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    plt.colorbar(sc, ax=ax, label='m/s', fraction=0.04, pad=0.02)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('Lon'); ax.set_ylabel('Lat')


def _scatter(ax, x, y, xlabel, ylabel):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size == 0:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                transform=ax.transAxes)
        return
    lim = np.nanpercentile(np.abs(np.concatenate([x, y])), 99)
    ax.scatter(x, y, s=3, alpha=0.4, rasterized=True)
    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=0.8)
    r    = np.corrcoef(x, y)[0, 1]
    bias = float(np.mean(y - x))
    rmse = float(np.sqrt(np.mean((y - x)**2)))
    ax.set_title(f'r={r:.3f}  bias={bias:.3f}  RMSE={rmse:.3f}', fontsize=8)
    ax.set_xlabel(xlabel, fontsize=8); ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)


def make_figure(result: dict, subswath: str, burst_idx: int,
                out_path: str, show: bool) -> None:
    lat = result['lat']; lon = result['lon']
    v_sar = result['v_sar_current']
    v_ocn = result['v_ocn_current']
    v_mod = result['v_model']
    v_sto = result['v_stokes']
    v_wav = result['v_wave']
    v_mis = result['v_miss']

    all_cur = np.concatenate([
        v_sar[np.isfinite(v_sar)], v_ocn[np.isfinite(v_ocn)],
        v_mod[np.isfinite(v_mod)],
    ])
    vmax_c = float(np.nanpercentile(np.abs(all_cur), 98)) if all_cur.size else 1.0
    vmax_c = max(vmax_c, 0.1)

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        f'Ocean current radial velocity  —  {subswath.upper()} burst {burst_idx}\n'
        f'Corrections: mispointing (OCN rvlDcMiss) + Stokes drift (ERA5) + '
        f'wave Doppler bias (ERA5, Mouche 2012)',
        fontsize=10,
    )
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.35)

    # Row 0: the three current estimates
    _map(fig.add_subplot(gs[0, 0]), lon, lat, v_sar,
         'SAR current [m/s]\n(ours, corrected)', -vmax_c, vmax_c)
    _map(fig.add_subplot(gs[0, 1]), lon, lat, v_ocn,
         'OCN current [m/s]\n(corrected)', -vmax_c, vmax_c)
    _map(fig.add_subplot(gs[0, 2]), lon, lat, v_mod,
         'glo12 model current [m/s]\n(reference)', -vmax_c, vmax_c)

    # Scatter: SAR vs model and OCN vs model
    ax_s1 = fig.add_subplot(gs[0, 3])
    _scatter(ax_s1, v_mod.ravel(), v_sar.ravel(), 'glo12 [m/s]', 'SAR [m/s]')
    ax_s1.set_title('SAR vs model\n' + ax_s1.get_title(), fontsize=8)

    # Row 1: applied corrections
    vmax_cor = float(np.nanpercentile(
        np.abs(np.concatenate([
            v_mis[np.isfinite(v_mis)], v_sto[np.isfinite(v_sto)],
            v_wav[np.isfinite(v_wav)],
        ])), 98
    )) if np.isfinite(v_mis).any() else 0.5

    _map(fig.add_subplot(gs[1, 0]), lon, lat, v_mis,
         'Mispointing correction [m/s]', 0, vmax_cor * 1.2, cmap='Oranges')
    _map(fig.add_subplot(gs[1, 1]), lon, lat, v_sto,
         'Stokes drift (look) [m/s]', -vmax_cor, vmax_cor)
    _map(fig.add_subplot(gs[1, 2]), lon, lat, v_wav,
         'Wave Doppler bias [m/s]', -vmax_cor, vmax_cor)

    ax_s2 = fig.add_subplot(gs[1, 3])
    _scatter(ax_s2, v_mod.ravel(), v_ocn.ravel(), 'glo12 [m/s]', 'OCN [m/s]')
    ax_s2.set_title('OCN vs model\n' + ax_s2.get_title(), fontsize=8)

    # Row 2: SAR vs OCN current comparison
    diff = v_sar - v_ocn
    dmax = float(np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 98)) if np.isfinite(diff).any() else 0.5

    _map(fig.add_subplot(gs[2, 0]), lon, lat, result['ocn_rv'],
         'OCN raw rvlRadVel [m/s]', -vmax_c * 3, vmax_c * 3)
    _map(fig.add_subplot(gs[2, 1]), lon, lat,
         result['our_ds']['radial_vel'].values,
         'SAR raw radial_vel [m/s]', -vmax_c * 3, vmax_c * 3)
    _map(fig.add_subplot(gs[2, 2]), lon, lat, diff,
         'SAR current − OCN current [m/s]', -dmax, dmax)

    ax_s3 = fig.add_subplot(gs[2, 3])
    _scatter(ax_s3, v_ocn.ravel(), v_sar.ravel(), 'OCN current [m/s]', 'SAR current [m/s]')
    ax_s3.set_title('SAR vs OCN current\n' + ax_s3.get_title(), fontsize=8)

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Figure saved → {out_path}')
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='SAR radial velocity → ocean current (corrected)',
    )
    p.add_argument('slc_safe')
    p.add_argument('subswath')
    p.add_argument('burst_idx', type=int)
    p.add_argument('--ocn-safe',  default=None, metavar='SAFE',
                   help='OCN .SAFE (optional; provides heading + mispointing from rvlDcMiss)')
    p.add_argument('--era5-wind', required=True, metavar='FILE')
    p.add_argument('--era5-wave', required=True, metavar='FILE')
    p.add_argument('--glo12',     required=True, metavar='FILE')
    p.add_argument('--aux-cal',   default=None, metavar='SAFE',
                   help='AUX_CAL .SAFE (enables varpi_delta / sideband / mispointing)')
    p.add_argument('--poeorb',    default=None, metavar='EOF',
                   help='POEORB / RESORB .EOF (replaces annotation orbit SVs)')
    p.add_argument('--pol',       default='vv')
    p.add_argument('--block-az',  type=int, default=256)
    p.add_argument('--block-rg',  type=int, default=512)
    p.add_argument('--stride-az', type=int, default=128)
    p.add_argument('--stride-rg', type=int, default=256)
    p.add_argument('--out',       default='current_comparison.png')
    p.add_argument('--no-plot',   action='store_true', dest='no_plot')
    return p


if __name__ == '__main__':
    args = _build_parser().parse_args()

    SW   = args.subswath
    BIDX = args.burst_idx

    print(f'\n=== {SW.upper()} burst {BIDX} ===')
    result = compute_current_velocity(
        slc_safe     = args.slc_safe,
        ocn_safe     = args.ocn_safe,
        subswath     = SW,
        burst_idx    = BIDX,
        era5_wind    = args.era5_wind,
        era5_wave    = args.era5_wave,
        glo12        = args.glo12,
        polarisation = args.pol,
        block_az     = args.block_az,
        block_rg     = args.block_rg,
        stride_az    = args.stride_az,
        stride_rg    = args.stride_rg,
        aux_cal_path = args.aux_cal,
        poeorb_path  = args.poeorb,
    )

    print('\nSummary:')
    for k, label in [
        ('v_miss',        'Mispointing correction'),
        ('v_stokes',      'Stokes drift (look)   '),
        ('v_wave',        'Wave Doppler bias      '),
        ('v_sar_current', 'SAR current estimate   '),
        ('v_ocn_current', 'OCN current estimate   '),
        ('v_model',       'glo12 reference        '),
    ]:
        v = result[k]
        print(f'  {label}: [{np.nanmin(v):.3f}, {np.nanmax(v):.3f}] m/s  '
              f'(mean={np.nanmean(v):.3f})')

    make_figure(
        result    = result,
        subswath  = SW,
        burst_idx = BIDX,
        out_path  = args.out,
        show      = not args.no_plot,
    )
