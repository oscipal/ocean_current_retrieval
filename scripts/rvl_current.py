#!/usr/bin/env python3
"""
Decompose SAR radial velocity into ocean-current-only component.

Three corrections are subtracted from the observed radial velocity:

  1. Antenna mispointing  (λ/2) · f_miss
     Source: OCN rvlDcMiss — same satellite attitude as our estimate.

  2. Surface Stokes drift  projected onto radar look direction
     Source: ERA5 ust / vst

  3. Wave Doppler bias  — C-band VV empirical model (Mouche et al. 2012)
     v_wave = a(θ_inc) · U10 · cos(wind_to_look_angle)
     a(θ) = 0.025 + 0.001 · (θ − 30)  [m/s per m/s wind, θ in degrees]
     Source: ERA5 u10 / v10

The same Stokes + wave corrections are applied to the OCN rvlRadVel.
The OCN product already subtracts mispointing internally (rvlDcMiss), so
only Stokes + wave are subtracted there.

Reference: CMEMS glo12 uo/vo projected onto the radar look direction.

Look direction:  look_az = heading + 90°  (right-looking SAR)
Slant projection: v_r = (uo·sin(look_az) + vo·cos(look_az)) · sin(θ_inc)

Usage
-----
  python rvl_current.py <SLC_SAFE> <OCN_SAFE> <SUBSWATH> <BURST_IDX>
         --era5-wind  FILE  --era5-wave FILE  --glo12 FILE
         [--pol vv] [--block-az N] [--block-rg N]
         [--stride-az N] [--stride-rg N]
         [--out FILE] [--no-plot]
"""

from __future__ import annotations

import argparse
import sys
import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.s1_rvl_burst import compute_rvl_burst
from scripts.s1_ocn_product import load_ocn_safe

_OCN_FILL  = -999.0
_SWATH_IDX = {'iw1': 0, 'iw2': 1, 'iw3': 2}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _interp2d(lat_grid, lon_grid, values, query_lat, query_lon):
    """
    Bilinear interpolation of a regular (lat, lon) field onto scattered points.
    lat_grid and lon_grid must be 1-D and ascending.
    """
    interp = RegularGridInterpolator(
        (lat_grid, lon_grid), values,
        method='linear', bounds_error=False, fill_value=np.nan,
    )
    pts = np.column_stack([query_lat.ravel(), query_lon.ravel()])
    return interp(pts).reshape(query_lat.shape)


def _select_time(ds: xr.Dataset, acq_time: str, time_dim: str = 'valid_time'):
    """Select the time step in *ds* closest to *acq_time* (ISO string)."""
    import pandas as pd
    target = np.datetime64(pd.Timestamp(acq_time).tz_localize(None))
    times  = ds[time_dim].values.astype('datetime64[ns]')
    tidx   = int(np.argmin(np.abs(times - target)))
    return ds.isel({time_dim: tidx})


# ---------------------------------------------------------------------------
# ERA5 loaders
# ---------------------------------------------------------------------------

def load_era5_wind(path: str, acq_time: str) -> dict:
    """
    Load ERA5 10-m wind (u10, v10) at the hour closest to *acq_time*.
    Returns dict with lat (ascending), lon, u10, v10 as 2-D arrays.
    """
    ds = _select_time(xr.open_dataset(path), acq_time)
    lat = ds['latitude'].values.astype(np.float64)
    lon = ds['longitude'].values.astype(np.float64)
    # ERA5 lat is descending — flip to ascending for RegularGridInterpolator
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        u10 = ds['u10'].values[::-1, :].astype(np.float64)
        v10 = ds['v10'].values[::-1, :].astype(np.float64)
    else:
        u10 = ds['u10'].values.astype(np.float64)
        v10 = ds['v10'].values.astype(np.float64)
    return {'lat': lat, 'lon': lon, 'u10': u10, 'v10': v10,
            'time': str(ds['valid_time'].values)}


def load_era5_wave(path: str, acq_time: str) -> dict:
    """
    Load ERA5 surface Stokes drift (ust, vst) and wave params at *acq_time*.
    """
    ds = _select_time(xr.open_dataset(path), acq_time)
    lat = ds['latitude'].values.astype(np.float64)
    lon = ds['longitude'].values.astype(np.float64)
    flip = lat[0] > lat[-1]
    if flip:
        lat = lat[::-1]
    def _get(name):
        v = ds[name].values.astype(np.float64)
        return v[::-1, :] if flip else v
    return {
        'lat': lat, 'lon': lon,
        'ust': _get('ust'), 'vst': _get('vst'),
        'swh': _get('swh'), 'mwp': _get('mwp'), 'mwd': _get('mwd'),
        'time': str(ds['valid_time'].values),
    }


# ---------------------------------------------------------------------------
# Correction functions
# ---------------------------------------------------------------------------

def compute_stokes_radial(
    wave: dict,
    our_lat: np.ndarray,
    our_lon: np.ndarray,
    our_inc: np.ndarray,
    look_az_rad: float,
) -> np.ndarray:
    """
    Project ERA5 surface Stokes drift onto the radar slant-range direction.

    Convention: positive = surface moves AWAY from satellite (same as OCN
    rvlRadVel and the glo12 model projection).

      v_stokes_r = (ust · sin(look_az) + vst · cos(look_az)) · sin(θ_inc)

    The sin(θ_inc) factor converts the horizontal velocity projection to a
    slant-range velocity, consistent with the λ/2 · f_dc convention used by
    the SAR Doppler measurement and the OCN rvlRadVel product.
    """
    ust = _interp2d(wave['lat'], wave['lon'], wave['ust'], our_lat, our_lon)
    vst = _interp2d(wave['lat'], wave['lon'], wave['vst'], our_lat, our_lon)
    inc_rad = np.deg2rad(our_inc.astype(np.float64))
    return ((ust * np.sin(look_az_rad) + vst * np.cos(look_az_rad))
            * np.sin(inc_rad)).astype(np.float32)


def compute_wave_doppler_bias(
    wind: dict,
    our_lat: np.ndarray,
    our_lon: np.ndarray,
    our_inc: np.ndarray,
    look_az_rad: float,
) -> np.ndarray:
    """
    C-band VV empirical wave Doppler bias (Mouche et al. 2012, simplified).

        v_wave = a(θ_inc) · U10 · cos(wind_dir_to − look_az)

        a(θ) = 0.025 + 0.001 · (θ − 30)   [m/s per m/s, θ in degrees]

    wind_dir_to is the direction the wind blows TOWARD (from u10, v10).

    Convention: positive = wave-induced Doppler bias when wind/waves move AWAY
    from the satellite (same as OCN rvlRadVel).  The formula follows Mouche's
    definition where v_wave > 0 for wind blowing in the look direction.
    """
    u10 = _interp2d(wind['lat'], wind['lon'], wind['u10'], our_lat, our_lon)
    v10 = _interp2d(wind['lat'], wind['lon'], wind['v10'], our_lat, our_lon)

    U10         = np.sqrt(u10**2 + v10**2)
    wind_dir_to = np.arctan2(u10, v10)          # clockwise from North
    delta_phi   = wind_dir_to - look_az_rad     # wind-to-look angle

    a = 0.025 + 0.001 * (our_inc.astype(np.float64) - 30.0)
    a = np.clip(a, 0.015, 0.05)                 # keep in physically valid range

    return (a * U10 * np.cos(delta_phi)).astype(np.float32)


# ---------------------------------------------------------------------------
# OCN RVL loader
# ---------------------------------------------------------------------------

def load_ocn_rvl(ocn_safe: str, subswath: str, polarisation: str) -> dict:
    """Load OCN RVL fields for one subswath, fill → NaN."""
    ocn = load_ocn_safe(ocn_safe, swath=subswath, polarisation=polarisation)
    ds  = ocn['rvl']
    sw  = _SWATH_IDX[subswath.lower()]
    if 'rvlSwath' in ds.dims:
        ds = ds.isel(rvlSwath=sw)

    def _clean(name):
        arr = ds[name].values.astype(np.float64)
        arr[arr == _OCN_FILL] = np.nan
        return arr

    return {
        'lat':      _clean('rvlLat'),
        'lon':      _clean('rvlLon'),
        'rad_vel':  _clean('rvlRadVel'),
        'dc_miss':  _clean('rvlDcMiss'),
        'heading':  _clean('rvlHeading'),
        'inc':      _clean('rvlIncidenceAngle'),
    }


# ---------------------------------------------------------------------------
# Spatial matching (nearest neighbour)
# ---------------------------------------------------------------------------

def match_to_sar_grid(
    our_lat: np.ndarray,
    our_lon: np.ndarray,
    ref_lat: np.ndarray,
    ref_lon: np.ndarray,
    ref_data: np.ndarray,
    max_dist_deg: float = 0.05,
) -> np.ndarray:
    """Nearest-neighbour resample of ref_data onto the SAR grid."""
    from scipy.spatial import KDTree
    our_pts = np.column_stack([our_lat.ravel(), our_lon.ravel()])
    ref_pts = np.column_stack([ref_lat.ravel(), ref_lon.ravel()])
    valid   = np.isfinite(ref_pts).all(axis=1)
    tree    = KDTree(ref_pts[valid])
    dists, idx = tree.query(our_pts, workers=-1)
    mapped = ref_data.ravel()[np.where(valid)[0][idx]].reshape(our_lat.shape).astype(np.float32)
    mapped[dists.reshape(our_lat.shape) > max_dist_deg] = np.nan
    return mapped


# ---------------------------------------------------------------------------
# glo12 current loader
# ---------------------------------------------------------------------------

def load_glo12_current(path: str, acq_time: str) -> dict:
    """Load glo12 surface uo/vo at the hour closest to *acq_time*."""
    import pandas as pd
    ds = xr.open_dataset(path).isel(depth=0)
    target = np.datetime64(pd.Timestamp(acq_time).tz_localize(None))
    times  = ds['time'].values.astype('datetime64[ns]')
    tidx   = int(np.argmin(np.abs(times - target)))
    ds     = ds.isel(time=tidx)

    lat = ds['latitude'].values.astype(np.float64)
    lon = ds['longitude'].values.astype(np.float64)

    def _masked(name):
        v = ds[name].values.astype(np.float64)
        v[np.abs(v) > 1e10] = np.nan
        v[v < -100]          = np.nan   # catch common fill values
        return v

    return {
        'lat': lat, 'lon': lon,
        'uo': _masked('uo'), 'vo': _masked('vo'),
        'time': str(ds['time'].values),
    }


def project_current_onto_look(
    model: dict,
    our_lat: np.ndarray,
    our_lon: np.ndarray,
    our_inc: np.ndarray,
    look_az_rad: float,
) -> np.ndarray:
    """
    Project glo12 (uo, vo) onto the radar slant-range direction.

        v_r = (uo · sin(look_az) + vo · cos(look_az)) · sin(θ_inc)
    """
    uo = _interp2d(model['lat'], model['lon'], model['uo'], our_lat, our_lon)
    vo = _interp2d(model['lat'], model['lon'], model['vo'], our_lat, our_lon)
    inc_rad = np.deg2rad(our_inc.astype(np.float64))
    v_r = (uo * np.sin(look_az_rad) + vo * np.cos(look_az_rad)) * np.sin(inc_rad)
    return v_r.astype(np.float32)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def compute_current_velocity(
    slc_safe:    str,
    ocn_safe:    str,
    subswath:    str,
    burst_idx:   int,
    era5_wind:   str,
    era5_wave:   str,
    glo12:       str,
    polarisation: str  = 'vv',
    block_az:    int   = 256,
    block_rg:    int   = 512,
    stride_az:   int   = 128,
    stride_rg:   int   = 256,
) -> dict:
    """
    Returns a dict with keys:
      v_sar_current   SAR-derived current-only radial velocity [m/s]
      v_ocn_current   OCN-derived current-only radial velocity [m/s]
      v_model         glo12 model current projected onto look  [m/s]
      v_stokes        Stokes drift radial component            [m/s]
      v_wave          Wave Doppler bias                        [m/s]
      v_miss          Mispointing correction                   [m/s]
      our_ds          full xr.Dataset from compute_rvl_burst
      lat, lon, inc   SAR grid coordinates
    """
    # ---- Step 1: single-burst RVL ----------------------------------------
    print('  Running single-burst RVL …')
    our_ds = compute_rvl_burst(
        safe_dir=slc_safe, subswath=subswath, burst_idx=burst_idx,
        polarisation=polarisation,
        block_az=block_az, block_rg=block_rg,
        stride_az=stride_az, stride_rg=stride_rg,
    )
    acq_time = our_ds.attrs['burst_azimuth_time']
    our_lat  = our_ds['latitude'].values
    our_lon  = our_ds['longitude'].values
    our_inc  = our_ds['incidence_angle'].values
    our_rv   = our_ds['radial_vel'].values          # (f_dc - f_geom) * λ/2

    # ---- Step 2: look direction ------------------------------------------
    print('  Computing look direction …')
    ocn_raw  = load_ocn_rvl(ocn_safe, subswath, polarisation)
    heading  = float(np.nanmean(ocn_raw['heading']))
    look_az  = np.deg2rad(heading + 90.0)           # right-looking
    print(f'    heading={heading:.1f}°  →  look_az={np.rad2deg(look_az):.1f}°')

    # ---- Step 3: mispointing correction (from OCN, matched to our grid) --
    print('  Loading mispointing from OCN …')
    wavelength = our_ds.attrs['wavelength_m']
    dc_miss_ocn = match_to_sar_grid(
        our_lat, our_lon, ocn_raw['lat'], ocn_raw['lon'], ocn_raw['dc_miss'],
    )
    v_miss = (wavelength / 2.0 * dc_miss_ocn).astype(np.float32)
    print(f'    v_miss range: [{np.nanmin(v_miss):.3f}, {np.nanmax(v_miss):.3f}] m/s')

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
    # our_rv = λ/2 · f_dca uses "toward satellite = positive" convention.
    # OCN rvlRadVel and the glo12 model use "away from satellite = positive".
    # Convert to the away convention and subtract each bias term:
    #   v_current_away = −our_rv + v_miss_toward + v_stokes_away − v_wave_away
    #
    # v_miss: OCN rvlDcMiss is positive when mispointing shifts f_dc toward
    #   satellite → v_miss = +λ/2·rvlDcMiss is in toward-positive, so it
    #   cancels against −our_rv.
    # v_stokes: already in away-positive (positive = Stokes drifts away).
    # v_wave: Mouche formula is away-positive (positive = wave drift away).
    v_sar_current = (-our_rv + v_miss - v_stokes - v_wave).astype(np.float32)

    # ---- Step 7: OCN current-only velocity -------------------------------
    # OCN rvlRadVel already subtracts mispointing internally and is in
    # "away from satellite = positive" convention.  Subtract Stokes and wave.
    ocn_rv_matched = match_to_sar_grid(
        our_lat, our_lon, ocn_raw['lat'], ocn_raw['lon'], ocn_raw['rad_vel'],
    )
    v_ocn_current = (ocn_rv_matched - v_stokes - v_wave).astype(np.float32)

    # ---- Step 8: glo12 reference current ---------------------------------
    print('  Loading glo12 model current …')
    model = load_glo12_current(glo12, acq_time)
    print(f'    glo12 time: {model["time"]}')
    v_model = project_current_onto_look(model, our_lat, our_lon, our_inc, look_az)
    print(f'    v_model range: [{np.nanmin(v_model):.3f}, {np.nanmax(v_model):.3f}] m/s')

    return {
        'v_sar_current':  v_sar_current.astype(np.float32),
        'v_ocn_current':  v_ocn_current.astype(np.float32),
        'v_model':        v_model,
        'v_stokes':       v_stokes,
        'v_wave':         v_wave,
        'v_miss':         v_miss,
        'ocn_rv':         ocn_rv_matched,
        'our_ds':         our_ds,
        'lat':            our_lat,
        'lon':            our_lon,
        'inc':            our_inc,
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
    p.add_argument('ocn_safe')
    p.add_argument('subswath')
    p.add_argument('burst_idx', type=int)
    p.add_argument('--era5-wind', required=True, metavar='FILE')
    p.add_argument('--era5-wave', required=True, metavar='FILE')
    p.add_argument('--glo12',     required=True, metavar='FILE')
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

    SLC  = args.slc_safe
    OCN  = args.ocn_safe
    SW   = args.subswath
    BIDX = args.burst_idx

    print(f'\n=== {SW.upper()} burst {BIDX} ===')
    result = compute_current_velocity(
        slc_safe    = SLC,
        ocn_safe    = OCN,
        subswath    = SW,
        burst_idx   = BIDX,
        era5_wind   = args.era5_wind,
        era5_wave   = args.era5_wave,
        glo12       = args.glo12,
        polarisation= args.pol,
        block_az    = args.block_az,
        block_rg    = args.block_rg,
        stride_az   = args.stride_az,
        stride_rg   = args.stride_rg,
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
