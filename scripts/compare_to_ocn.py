#!/usr/bin/env python3
"""
Compare a single-burst RVL estimate to the Sentinel-1 Level-2 OCN product
and to a CMEMS model current (glo12 uo/vo).

Comparisons produced:
  - Observed Doppler    : our doppler_obs  vs  OCN rvlDcObs
  - Geometry Doppler    : our doppler_geo  vs  OCN rvlDcGeo
  - Radial velocity     : our radial_vel   vs  OCN rvlRadVel
  - Model current       : our radial_vel   vs  glo12 uo/vo projected onto look

Usage
-----
  python compare_to_ocn.py <SLC_SAFE> <OCN_SAFE> <SUBSWATH> <BURST_IDX> [options]

  positional arguments:
    slc_safe    Path to the Level-1 SLC .SAFE directory
    ocn_safe    Path to the Level-2 OCN .SAFE directory
    subswath    iw1 | iw2 | iw3
    burst_idx   0-based burst index

  optional arguments:
    --pol VV|VH       Polarisation (default: vv)
    --block-az N      Estimation block height [lines]   (default: 256)
    --block-rg N      Estimation block width  [samples] (default: 512)
    --stride-az N     Azimuth stride                    (default: 128)
    --stride-rg N     Range stride                      (default: 256)
    --glo12 FILE      CMEMS glo12 NetCDF (uo/vo) for model comparison
    --out FILE        Save figure (default: comparison.png)
    --no-plot         Skip display, only save
"""

from __future__ import annotations

import argparse
import sys
import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial import KDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.s1_rvl_burst import compute_rvl_burst
from scripts.s1_ocn_product import load_ocn_safe


# ---------------------------------------------------------------------------
# OCN loader — extract RVL for one subswath, mask fill values
# ---------------------------------------------------------------------------

_SWATH_IDX = {'iw1': 0, 'iw2': 1, 'iw3': 2}
_OCN_FILL  = -999.0


def _get_ocn_heading(ocn_safe: str, subswath: str, polarisation: str) -> float:
    """Return mean satellite heading [deg] from the OCN rvlHeading variable."""
    ocn = load_ocn_safe(ocn_safe, swath=subswath, polarisation=polarisation)
    ds  = ocn['rvl']
    sw  = _SWATH_IDX[subswath.lower()]
    if 'rvlSwath' in ds.dims:
        ds = ds.isel(rvlSwath=sw)
    arr = ds['rvlHeading'].values.astype(float)
    arr[arr == _OCN_FILL] = np.nan
    return float(np.nanmean(arr))


def load_ocn_rvl(ocn_safe: str, subswath: str, polarisation: str) -> dict:
    """
    Load RVL fields from the OCN SAFE for one subswath, fill-masked to NaN.

    Returns a dict with keys: lat, lon, dc_obs, dc_geo, rad_vel, snr — all
    shape (n_az, n_rg) float64 arrays with NaN where fill.
    """
    ocn = load_ocn_safe(ocn_safe, swath=subswath, polarisation=polarisation)
    ds  = ocn['rvl']

    # The combined IW NetCDF keeps a 3rd swath dimension; select the right one.
    sw = _SWATH_IDX[subswath.lower()]
    if 'rvlSwath' in ds.dims:
        ds = ds.isel(rvlSwath=sw)

    def _clean(name: str) -> np.ndarray:
        arr = ds[name].values.astype(np.float64)
        arr[arr == _OCN_FILL] = np.nan
        return arr

    return {
        'lat':     _clean('rvlLat'),
        'lon':     _clean('rvlLon'),
        'dc_obs':  _clean('rvlDcObs'),
        'dc_geo':  _clean('rvlDcGeo'),
        'rad_vel': _clean('rvlRadVel'),
        'snr':     _clean('rvlSnr'),
    }


# ---------------------------------------------------------------------------
# Spatial matching
# ---------------------------------------------------------------------------

def nearest_match(
    our_lat: np.ndarray,  # (n_az, n_rg)
    our_lon: np.ndarray,
    ocn_lat: np.ndarray,  # (m_az, m_rg)
    ocn_lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each cell in our grid, find the nearest OCN cell index.

    Returns flat indices into ocn_lat/ocn_lon, and a distance mask that
    rejects matches farther than 0.05° (~5 km) as outside-burst.
    """
    our_pts = np.column_stack([our_lat.ravel(), our_lon.ravel()])
    ocn_pts = np.column_stack([ocn_lat.ravel(), ocn_lon.ravel()])

    valid_ocn = np.isfinite(ocn_pts).all(axis=1)
    ocn_pts_valid = ocn_pts[valid_ocn]
    valid_ocn_idx = np.where(valid_ocn)[0]

    tree = KDTree(ocn_pts_valid)
    dists, local_idx = tree.query(our_pts, workers=-1)

    flat_idx   = valid_ocn_idx[local_idx]
    dist_mask  = dists < 0.05          # degrees

    return flat_idx, dist_mask


# ---------------------------------------------------------------------------
# Model current loader and look-direction projection
# ---------------------------------------------------------------------------

def load_model_current(glo12_path: str, acq_time: str) -> dict:
    """
    Load glo12 uo/vo at the hour closest to *acq_time* (ISO string).

    Returns dict with keys: lat, lon, uo, vo — 2-D float64 arrays
    covering the full file extent (lat/lon subset is done later during
    interpolation onto the SAR grid).
    """
    import pandas as pd
    ds = xr.open_dataset(glo12_path).isel(depth=0)
    target = pd.Timestamp(acq_time)
    tidx   = int(np.argmin(np.abs(ds.time.values - np.datetime64(target))))
    ds     = ds.isel(time=tidx)

    uo = ds['uo'].values.astype(np.float64)
    vo = ds['vo'].values.astype(np.float64)
    uo[uo > 1e10] = np.nan
    vo[vo > 1e10] = np.nan

    return {
        'lat': ds['latitude'].values,
        'lon': ds['longitude'].values,
        'uo':  uo,
        'vo':  vo,
        'time': str(ds.time.values),
    }


def project_model_onto_look(
    model: dict,
    our_lat: np.ndarray,
    our_lon: np.ndarray,
    our_inc: np.ndarray,
    our_heading: float,
) -> np.ndarray:
    """
    Bilinearly interpolate model uo/vo onto the SAR grid and project onto
    the radar slant-range direction.

    For a right-looking SAR:
        look_az = heading + 90°   (clockwise from North, in the horizontal plane)

    The slant-range component of a horizontal surface velocity (uo, vo) is:
        v_r = (uo · sin(look_az) + vo · cos(look_az)) · sin(θ_inc)

    Parameters
    ----------
    model       : dict from load_model_current
    our_lat/lon : SAR grid coordinates, shape (n_az, n_rg)
    our_inc     : incidence angle [deg], same shape
    our_heading : mean satellite heading [deg from North, clockwise]

    Returns
    -------
    v_r_model : float32, shape (n_az, n_rg)
    """
    from scipy.interpolate import RegularGridInterpolator

    look_az = np.deg2rad(our_heading + 90.0)   # rad, clockwise from North

    # Model grid is regular; interpolate uo and vo separately.
    # RegularGridInterpolator expects ascending axes.
    lat_asc = model['lat']
    lon_asc = model['lon']

    interp_uo = RegularGridInterpolator(
        (lat_asc, lon_asc), model['uo'],
        method='linear', bounds_error=False, fill_value=np.nan,
    )
    interp_vo = RegularGridInterpolator(
        (lat_asc, lon_asc), model['vo'],
        method='linear', bounds_error=False, fill_value=np.nan,
    )

    pts   = np.column_stack([our_lat.ravel(), our_lon.ravel()])
    uo_g  = interp_uo(pts).reshape(our_lat.shape)
    vo_g  = interp_vo(pts).reshape(our_lat.shape)

    inc_rad = np.deg2rad(our_inc)
    v_r = (uo_g * np.sin(look_az) + vo_g * np.cos(look_az)) * np.sin(inc_rad)

    return v_r.astype(np.float32)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _map_panel(ax, lon, lat, data, title, cmap, vmin, vmax, unit):
    sc = ax.scatter(lon.ravel(), lat.ravel(), c=data.ravel(),
                    s=4, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    plt.colorbar(sc, ax=ax, label=unit, fraction=0.04, pad=0.02)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('Lon')
    ax.set_ylabel('Lat')


def _scatter_panel(ax, x, y, xlabel, ylabel):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size == 0:
        ax.text(0.5, 0.5, 'no overlap', ha='center', va='center', transform=ax.transAxes)
        return
    lim  = np.nanpercentile(np.abs(np.concatenate([x, y])), 99)
    ax.scatter(x, y, s=1, alpha=0.3, rasterized=True)
    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=0.8, label='1:1')
    r    = np.corrcoef(x, y)[0, 1]
    bias = np.mean(y - x)
    rmse = np.sqrt(np.mean((y - x) ** 2))
    ax.set_title(f'r={r:.3f}  bias={bias:.2f}  RMSE={rmse:.2f}', fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.legend(fontsize=7)


def make_figure(
    our: xr.Dataset,
    ocn: dict,
    subswath: str,
    burst_idx: int,
    out_path: str,
    show: bool,
    v_r_model: np.ndarray | None = None,
    model_time: str = '',
) -> None:
    our_lat = our['latitude'].values
    our_lon = our['longitude'].values

    flat_idx, dist_mask = nearest_match(our_lat, our_lon, ocn['lat'], ocn['lon'])

    def _ocn_matched(key: str) -> np.ndarray:
        vals = ocn[key].ravel()[flat_idx].reshape(our_lat.shape).astype(np.float32)
        vals[~dist_mask.reshape(our_lat.shape)] = np.nan
        return vals

    ocn_dc_obs  = _ocn_matched('dc_obs')
    ocn_dc_geo  = _ocn_matched('dc_geo')
    ocn_rad_vel = _ocn_matched('rad_vel')

    our_dc_obs  = our['doppler_obs'].values
    our_dc_geo  = our['doppler_geo'].values
    our_rad_vel = our['radial_vel'].values

    # ---- Figure 1: SAR vs OCN comparison ----
    fig1 = plt.figure(figsize=(18, 14))
    fig1.suptitle(
        f'Single-burst RVL vs ESA OCN  —  {subswath.upper()} burst {burst_idx}',
        fontsize=11,
    )
    gs1 = gridspec.GridSpec(3, 4, figure=fig1, hspace=0.45, wspace=0.35)

    rows = [
        ('Observed Doppler [Hz]',  our_dc_obs,  ocn_dc_obs,  'RdBu_r', 'Hz'),
        ('Geometry Doppler [Hz]',  our_dc_geo,  ocn_dc_geo,  'RdBu_r', 'Hz'),
        ('Radial velocity [m/s]',  our_rad_vel, ocn_rad_vel, 'RdBu_r', 'm/s'),
    ]

    for row, (label, ours, theirs, cmap, unit) in enumerate(rows):
        vmax = np.nanpercentile(np.abs(np.concatenate(
            [ours[np.isfinite(ours)], theirs[np.isfinite(theirs)]]
        )), 98) if (np.isfinite(ours).any() and np.isfinite(theirs).any()) else 1.0

        ax_our = fig1.add_subplot(gs1[row, 0])
        _map_panel(ax_our, our_lon, our_lat, ours,
                   f'Ours — {label}', cmap, -vmax, vmax, unit)

        ax_ocn = fig1.add_subplot(gs1[row, 1])
        _map_panel(ax_ocn, our_lon, our_lat, theirs,
                   f'OCN — {label}', cmap, -vmax, vmax, unit)

        ax_diff = fig1.add_subplot(gs1[row, 2])
        diff = ours - theirs
        dmax = np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 98) if np.isfinite(diff).any() else 1.0
        _map_panel(ax_diff, our_lon, our_lat, diff,
                   f'Ours − OCN  [{unit}]', 'RdBu_r', -dmax, dmax, unit)

        ax_sc = fig1.add_subplot(gs1[row, 3])
        _scatter_panel(ax_sc, theirs.ravel(), ours.ravel(),
                       f'OCN {unit}', f'Ours {unit}')
        ax_sc.set_title(f'{label}\n' + ax_sc.get_title(), fontsize=8)

    ocn_path = out_path.replace('.png', '_vs_ocn.png')
    fig1.savefig(ocn_path, dpi=150, bbox_inches='tight')
    print(f'Figure saved → {ocn_path}')
    if show:
        plt.show()
    plt.close(fig1)

    # ---- Figure 2: SAR vs model current (if provided) ----
    if v_r_model is None:
        return

    fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig2.suptitle(
        f'Radial velocity: SAR vs glo12 model  —  {subswath.upper()} burst {burst_idx}'
        + (f'\nModel time: {model_time}' if model_time else ''),
        fontsize=11,
    )

    all_vals = np.concatenate([
        our_rad_vel[np.isfinite(our_rad_vel)],
        v_r_model[np.isfinite(v_r_model)],
    ])
    vmax = np.nanpercentile(np.abs(all_vals), 98) if all_vals.size > 0 else 1.0

    _map_panel(axes[0], our_lon, our_lat, our_rad_vel,
               'SAR radial velocity [m/s]', 'RdBu_r', -vmax, vmax, 'm/s')

    _map_panel(axes[1], our_lon, our_lat, v_r_model,
               'Model current — look projection [m/s]', 'RdBu_r', -vmax, vmax, 'm/s')

    diff = our_rad_vel - v_r_model
    dmax = np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 98) if np.isfinite(diff).any() else 1.0
    _map_panel(axes[2], our_lon, our_lat, diff,
               'SAR − model [m/s]\n(includes wave+wind bias)', 'RdBu_r', -dmax, dmax, 'm/s')

    # Inset scatter
    ax_sc = axes[2].inset_axes([0.62, 0.05, 0.36, 0.36])
    _scatter_panel(ax_sc, v_r_model.ravel(), our_rad_vel.ravel(), 'Model', 'SAR')
    ax_sc.tick_params(labelsize=6)
    ax_sc.set_title(ax_sc.get_title(), fontsize=6)
    ax_sc.set_xlabel('Model [m/s]', fontsize=6)
    ax_sc.set_ylabel('SAR [m/s]', fontsize=6)

    plt.tight_layout()
    model_path = out_path.replace('.png', '_vs_model.png')
    fig2.savefig(model_path, dpi=150, bbox_inches='tight')
    print(f'Figure saved → {model_path}')
    if show:
        plt.show()
    plt.close(fig2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Compare single-burst RVL to ESA OCN product and/or glo12 model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('slc_safe',  help='Level-1 SLC .SAFE directory')
    p.add_argument('ocn_safe',  help='Level-2 OCN .SAFE directory')
    p.add_argument('subswath',  help='iw1 | iw2 | iw3')
    p.add_argument('burst_idx', type=int, help='0-based burst index')
    p.add_argument('--pol',       default='vv')
    p.add_argument('--block-az',  type=int, default=256, metavar='N')
    p.add_argument('--block-rg',  type=int, default=512, metavar='N')
    p.add_argument('--stride-az', type=int, default=128, metavar='N')
    p.add_argument('--stride-rg', type=int, default=256, metavar='N')
    p.add_argument('--glo12',     default=None, metavar='FILE',
                   help='CMEMS glo12 NetCDF (uo/vo) for model current comparison')
    p.add_argument('--out',       default='comparison.png')
    p.add_argument('--no-plot',   action='store_true', dest='no_plot')
    return p


if __name__ == '__main__':
    args = _build_parser().parse_args()

    print(f'Running single-burst RVL: {args.subswath.upper()} burst {args.burst_idx} …')
    our_ds = compute_rvl_burst(
        safe_dir    = args.slc_safe,
        subswath    = args.subswath,
        burst_idx   = args.burst_idx,
        polarisation= args.pol,
        block_az    = args.block_az,
        block_rg    = args.block_rg,
        stride_az   = args.stride_az,
        stride_rg   = args.stride_rg,
    )
    print(our_ds)

    print('Loading OCN product …')
    ocn_rvl = load_ocn_rvl(args.ocn_safe, args.subswath, args.pol)

    # Optional: model current projection
    v_r_model  = None
    model_time = ''
    if args.glo12:
        print('Loading glo12 model current …')
        acq_time = our_ds.attrs['burst_azimuth_time']
        model    = load_model_current(args.glo12, acq_time)
        model_time = model['time']
        print(f'  Selected model time step: {model_time}')
        ocn_heading = _get_ocn_heading(args.ocn_safe, args.subswath, args.pol)
        print(f'  Satellite heading: {ocn_heading:.1f}°')
        v_r_model = project_model_onto_look(
            model    = model,
            our_lat  = our_ds['latitude'].values,
            our_lon  = our_ds['longitude'].values,
            our_inc  = our_ds['incidence_angle'].values,
            our_heading = ocn_heading,
        )
        print(f'  Model v_r range: [{np.nanmin(v_r_model):.3f}, {np.nanmax(v_r_model):.3f}] m/s')

    print('Matching grids and plotting …')
    make_figure(
        our        = our_ds,
        ocn        = ocn_rvl,
        subswath   = args.subswath,
        burst_idx  = args.burst_idx,
        out_path   = args.out,
        show       = not args.no_plot,
        v_r_model  = v_r_model,
        model_time = model_time,
    )
