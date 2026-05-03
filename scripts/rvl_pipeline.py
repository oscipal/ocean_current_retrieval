"""
Single-burst RVL pipeline — exact replication of rvl_pipeline_walkthrough.ipynb.

Steps
-----
  I    Deramp TOPS burst
  II   Estimate p0, p1 correlation grid
  III  Doppler centroid → subtract POEORB geometry + sideband bias → f_dca → v_r
  V    Geolocate
  +    Look direction from orbit
  +    Stokes drift (ERA5) + wave Doppler bias (ERA5, Mouche 2012)
  +    OCN rvlDcMiss mispointing correction → v_current_ocn
  +    GLO12 model comparison

Usage (CLI)
-----------
  python -m scripts.rvl_pipeline  <SLC_SAFE> <SUBSWATH> <BURST_IDX>
         --poeorb FILE --aux-cal FILE --ocn-safe FILE
         --era5-wind FILE --era5-wave FILE --glo12 FILE
         [--pol vv] [--out FILE]

Usage (import)
--------------
  from scripts.rvl_pipeline import run_pipeline
  result = run_pipeline(slc_safe, subswath, burst_idx, ...)
  # result['v_current_ocn']  — fully corrected ocean current radial velocity
  # result['v_current']      — without OCN mispointing (diagnostic)
  # result['v_model']        — GLO12 reference
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.s1_io import find_safe_files, parse_annotation, read_slc_burst, _iso_to_datetime, _nearest_estimate
from scripts.s1_aux import parse_aux_cal, apply_poeorb, parse_aux_ins
from scripts.s1_rvl import (
    deramp_burst,
    merge_bursts,
    estimate_correlation_grid,
    correlation_to_doppler,
    _fm_rate_at_burst,
    _steering_doppler_rate,
    apply_burst_valid_sample_mask,
    compute_gamma_ambiguity,
    compute_sideband_bias,
    _geom_doppler_annotation,
    _geom_doppler_poeorb,
    _blended_geom_doppler_annotation,
    _geolocate_grid,
    _interpolate_orbit,
    compute_mispointing_doppler,
    mispointing_doppler_from_yaw,
)
from scripts.s1_ocn_product import load_ocn_safe
from scripts.rvl_current import (
    load_era5_wave,
    load_era5_wind,
    compute_stokes_radial,
    compute_wave_doppler_bias,
    load_glo12_current,
    project_current_onto_look,
    match_to_sar_grid,
)

_OCN_FILL  = -999.0
_SWATH_IDX = {'iw1': 0, 'iw2': 1, 'iw3': 2}


def to_regular_grid(
    lat: np.ndarray,
    lon: np.ndarray,
    data: np.ndarray,
    resolution_deg: float = 0.01,
    method: str = 'linear',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Regrid scattered SAR estimation points onto a regular lat/lon grid.

    Parameters
    ----------
    lat, lon    : 2D arrays of block-centre coordinates
    data        : 2D array of values to regrid (same shape as lat/lon)
    resolution_deg : grid spacing in degrees (default 0.01 ≈ 1 km)
    method      : 'linear' (default) or 'nearest'

    Returns
    -------
    grid_lat : 1D array of latitude coordinates
    grid_lon : 1D array of longitude coordinates
    grid_data : 2D array, shape (len(grid_lat), len(grid_lon)), NaN outside convex hull
    """
    from scipy.interpolate import griddata

    lat_f = lat.ravel()
    lon_f = lon.ravel()
    data_f = data.ravel()

    valid = np.isfinite(lat_f) & np.isfinite(lon_f) & np.isfinite(data_f)

    grid_lat = np.arange(lat_f[valid].min(), lat_f[valid].max() + resolution_deg, resolution_deg)
    grid_lon = np.arange(lon_f[valid].min(), lon_f[valid].max() + resolution_deg, resolution_deg)

    mesh_lon, mesh_lat = np.meshgrid(grid_lon, grid_lat)

    grid_data = griddata(
        points=(lon_f[valid], lat_f[valid]),
        values=data_f[valid],
        xi=(mesh_lon, mesh_lat),
        method=method,
    ).astype(np.float32)

    return grid_lat, grid_lon, grid_data


def run_pipeline(
    slc_safe:     str,
    subswath:     str,
    burst_idx:    int,
    poeorb_path:  str,
    aux_cal_path: str,
    ocn_safe:     str,
    era5_wind:    str,
    era5_wave:    str,
    glo12:        str,
    polarisation: str = 'vv',
    block_az:     int = 256,
    block_rg:     int = 512,
    stride_az:    int = 128,
    stride_rg:    int = 256,
    tops_scaling: str = 'none',
    use_ocn_dc:   bool = False,
) -> dict:
    """
    Run the full pipeline from the walkthrough notebook.

    Returns a dict with all intermediate and final arrays:
      v_r             SAR radial velocity [m/s], toward-satellite positive
      v_stokes        Stokes drift projected onto look [m/s]
      v_wave          Wave Doppler bias [m/s]
      v_miss_ocn      OCN mispointing correction [m/s]
      v_current       Current without mispointing: -v_r - v_stokes - v_wave
      v_current_ocn   Fully corrected: -v_r + v_miss_ocn - v_stokes - v_wave
      v_model         GLO12 model current projected onto look [m/s]
      lat, lon, inc   SAR grid coordinates
      f_dca           Doppler centroid anomaly [Hz]
      snr             Signal-to-noise ratio
    """

    # ── Load annotation, apply POEORB ────────────────────────────────────────
    print(f'Loading annotation for {subswath.upper()} burst {burst_idx} …')
    files          = find_safe_files(slc_safe, subswath, polarisation)
    annot_original = parse_annotation(files['annotation'])
    annot          = apply_poeorb(annot_original, poeorb_path)
    burst          = annot.bursts[burst_idx]
    print(f'  Burst time : {burst.azimuth_time}')
    print(f'  PRF        : {annot.prf:.3f} Hz')
    print(f'  Wavelength : {annot.wavelength*100:.2f} cm')

    # ── AUX_CAL ──────────────────────────────────────────────────────────────
    aap        = parse_aux_cal(aux_cal_path, subswath, polarisation)
    _, vel     = _interpolate_orbit(annot, burst.azimuth_time)
    v_eff      = float(np.linalg.norm(vel))
    gamma_amb  = compute_gamma_ambiguity(aap, annot.radar_prf, v_eff, annot.wavelength)
    f_sideband = compute_sideband_bias(aap, 0.0, annot.radar_prf, v_eff, annot.wavelength)
    print(f'  γ_amb={gamma_amb:.5f}  f_sideband={f_sideband:.3f} Hz')

    # ── Step I: Deramp ───────────────────────────────────────────────────────
    print('Step I: deramping …')
    raw      = read_slc_burst(files['measurement'], annot, burst_idx)
    deramped = deramp_burst(raw, annot, burst_idx)
    valid_mask = apply_burst_valid_sample_mask(deramped, burst)

    # ── Step II: Correlation coefficients ───────────────────────────────────
    print('Step II: estimating p0, p1 …')
    p0, p1, az_local, rg_centers = estimate_correlation_grid(
        deramped, block_az, block_rg, stride_az, stride_rg,
        valid_mask=valid_mask,
    )

    # Full-scene azimuth indices for geolocation
    ati      = annot.azimuth_time_interval
    az0_full = int(round(
        (burst.azimuth_time - annot.first_line_time).total_seconds() / ati
    ))
    az_full = az_local + az0_full

    # ── Step III: Doppler centroid → DCA ────────────────────────────────────
    print('Step III: Doppler centroid and geometry subtraction …')
    if tops_scaling not in ('none', 'multiply', 'divide'):
        raise ValueError("tops_scaling must be 'none', 'multiply', or 'divide'")

    f_dc, _, snr = correlation_to_doppler(
        p0, p1, annot.prf, annot.wavelength, gamma_amb=gamma_amb,
    )
    k_a = _fm_rate_at_burst(annot, burst_idx)[rg_centers]
    k_psi = _steering_doppler_rate(annot, burst_idx)
    tops_scale = (1.0 + k_psi / k_a).astype(np.float32)
    if tops_scaling == 'multiply':
        f_dc = (f_dc * tops_scale[np.newaxis, :]).astype(np.float32)
    elif tops_scaling == 'divide':
        f_dc = (f_dc / tops_scale[np.newaxis, :]).astype(np.float32)

    f_geom_ann = _geom_doppler_annotation(annot_original, burst_idx, rg_centers).astype(np.float32)
    f_geom_poe = _geom_doppler_poeorb(annot, annot_original, burst_idx, rg_centers).astype(np.float32)

    f_dca = f_dc - f_geom_poe[np.newaxis, :]
    f_dca = f_dca - float(f_sideband)

    v_r = (annot.wavelength / 2.0 * f_dca).astype(np.float32)
    print(f'  v_r median: {float(np.nanmedian(v_r)):+.4f} m/s')

    # ── Step V: Geolocation ──────────────────────────────────────────────────
    print('Step V: geolocation …')
    lat, lon, inc = _geolocate_grid(annot, az_full, rg_centers)

    # ── Look direction from orbit ────────────────────────────────────────────
    acq_time = str(burst.azimuth_time)
    burst_t  = _iso_to_datetime(acq_time)
    pos_sv, vel_sv = _interpolate_orbit(annot, burst_t)
    lat_s = np.arcsin(pos_sv[2] / np.linalg.norm(pos_sv))
    lon_s = np.arctan2(pos_sv[1], pos_sv[0])
    e_east  = np.array([-np.sin(lon_s),  np.cos(lon_s), 0.0])
    e_north = np.array([-np.sin(lat_s) * np.cos(lon_s),
                        -np.sin(lat_s) * np.sin(lon_s),
                         np.cos(lat_s)])
    heading = float(np.rad2deg(np.arctan2(np.dot(vel_sv, e_east), np.dot(vel_sv, e_north))))
    look_az = np.deg2rad(heading + 90.0)
    print(f'  Heading: {heading:.1f}°   Look azimuth: {np.rad2deg(look_az):.1f}°')

    # ── Stokes drift ─────────────────────────────────────────────────────────
    print('Loading ERA5 Stokes drift …')
    wave     = load_era5_wave(era5_wave, acq_time)
    v_stokes = compute_stokes_radial(wave, lat, lon, inc, look_az)
    print(f'  v_stokes range: [{np.nanmin(v_stokes):.4f}, {np.nanmax(v_stokes):.4f}] m/s')

    # ── Wave Doppler bias ─────────────────────────────────────────────────────
    print('Loading ERA5 wind …')
    wind   = load_era5_wind(era5_wind, acq_time)
    v_wave = compute_wave_doppler_bias(wind, lat, lon, inc, look_az)
    print(f'  v_wave range: [{np.nanmin(v_wave):.4f}, {np.nanmax(v_wave):.4f}] m/s')

    # ── Current without mispointing ───────────────────────────────────────────
    v_current = (-v_r - v_stokes - v_wave).astype(np.float32)

    # ── OCN rvlDcMiss mispointing (+ optional rvlDcObs) ──────────────────────
    print('Loading OCN …')
    ocn    = load_ocn_safe(ocn_safe, swath=subswath, polarisation=polarisation)
    ds_rvl = ocn['rvl']
    sw     = _SWATH_IDX[subswath.lower()]
    if 'rvlSwath' in ds_rvl.dims:
        ds_rvl = ds_rvl.isel(rvlSwath=sw)

    def _clean(name):
        arr = ds_rvl[name].values.astype(np.float64)
        arr[arr == _OCN_FILL] = np.nan
        return arr

    ocn_lat     = _clean('rvlLat')
    ocn_lon     = _clean('rvlLon')
    ocn_dc_miss = _clean('rvlDcMiss')
    ocn_dc_obs  = _clean('rvlDcObs')

    f_miss_ocn = match_to_sar_grid(lat, lon, ocn_lat, ocn_lon, ocn_dc_miss)
    v_miss_ocn = (annot.wavelength / 2.0 * f_miss_ocn).astype(np.float32)
    f_dc_ocn   = match_to_sar_grid(lat, lon, ocn_lat, ocn_lon, ocn_dc_obs)
    print(f'  v_miss_ocn range: [{np.nanmin(v_miss_ocn):.4f}, {np.nanmax(v_miss_ocn):.4f}] m/s')

    if use_ocn_dc:
        f_dc   = f_dc_ocn.astype(np.float32)
        f_dca  = (f_dc - f_geom_poe[np.newaxis, :] - float(f_sideband)).astype(np.float32)
        v_r    = (annot.wavelength / 2.0 * f_dca).astype(np.float32)
        v_current = (-v_r - v_stokes - v_wave).astype(np.float32)
        print(f'  using rvlDcObs as f_dc; v_r median: {float(np.nanmedian(v_r)):+.4f} m/s')

    # ── Fully corrected current ───────────────────────────────────────────────
    v_current_ocn = (-v_r + v_miss_ocn - v_stokes - v_wave).astype(np.float32)

    # ── GLO12 comparison ──────────────────────────────────────────────────────
    print('Loading GLO12 …')
    model   = load_glo12_current(glo12, acq_time)
    v_model = project_current_onto_look(model, lat, lon, inc, look_az)

    # Stats
    rmse_vs_glo12 = float('nan')
    bias_vs_glo12 = float('nan')
    r_vs_glo12    = float('nan')
    for label, vc in [('no mispointing', v_current), ('OCN mispointing', v_current_ocn)]:
        mask = np.isfinite(vc) & np.isfinite(v_model)
        if mask.sum() > 1:
            diff = vc[mask] - v_model[mask]
            bias = float(np.mean(diff))
            rmse = float(np.sqrt(np.mean(diff**2)))
            r    = float(np.corrcoef(vc[mask], v_model[mask])[0, 1])
            print(f'  vs GLO12 ({label}):  bias={bias:+.4f}  RMSE={rmse:.4f}  r={r:.4f}')
            if label == 'OCN mispointing':
                rmse_vs_glo12 = rmse
                bias_vs_glo12 = bias
                r_vs_glo12    = r

    return {
        'v_r':            v_r,
        'v_stokes':       v_stokes,
        'v_wave':         v_wave,
        'v_miss_ocn':     v_miss_ocn,
        'f_miss_ocn':     f_miss_ocn,
        'v_current':      v_current,
        'v_current_ocn':  v_current_ocn,
        'v_model':        v_model,
        'lat':            lat,
        'lon':            lon,
        'inc':            inc,
        'f_dca':          f_dca,
        'f_dc':           f_dc,
        'f_dc_ocn':       f_dc_ocn,
        'tops_scale':     tops_scale,
        'tops_scaling':   tops_scaling,
        'snr':            snr,
        'f_geom_ann':     f_geom_ann,
        'f_geom_poe':     f_geom_poe,
        'look_az_rad':    look_az,
        'heading_deg':    heading,
        'wavelength_m':   annot.wavelength,
        'subswath':       subswath,
        'burst_idx':      burst_idx,
        'acq_time':       acq_time,
        'rmse_vs_glo12':  rmse_vs_glo12,
        'bias_vs_glo12':  bias_vs_glo12,
        'r_vs_glo12':     r_vs_glo12,
        'p0':             p0,
        'p1':             np.angle(p1),
    }


def run_all_bursts(
    slc_safe:     str,
    subswath:     str,
    poeorb_path:  str,
    aux_cal_path: str,
    ocn_safe:     str,
    era5_wind:    str,
    era5_wave:    str,
    glo12:        str,
    polarisation: str = 'vv',
    burst_indices: list[int] | None = None,
    tops_scaling: str = 'none',
    merge_first:  bool = False,
    use_ocn_dc:   bool = False,
    **kwargs,
) -> list[dict]:
    """
    Run run_pipeline for every burst in the subswath and return a list of results.

    Parameters
    ----------
    burst_indices : list of int or None
        Which bursts to process. If None, all bursts in the subswath are processed.
        Ignored when merge_first=True.
    tops_scaling  : 'none', 'multiply', or 'divide'
        Optional TOPS focused-data Doppler scale test forwarded to run_pipeline.
        Ignored when merge_first=True.
    use_ocn_dc    : bool
        When True, replace the lag-1 correlation f_dc with rvlDcObs interpolated
        from the OCN product onto the SAR estimation grid. Geometry subtraction
        and all downstream corrections still use our own pipeline values.
    merge_first   : bool
        When True, deramp and coherently merge all bursts into one continuous SLC
        image first (using merge_bursts), then run the full pipeline on the merged
        image. This avoids burst-boundary discontinuities in the Doppler estimation
        at the cost of slightly more memory. Geometry corrections use the blended
        annotation polynomial (or nearest-burst POEORB differential when poeorb_path
        is provided). Returns a single-element list so all downstream functions
        (merge_burst_grids, plot_comparison) remain compatible.
    **kwargs      : forwarded to run_pipeline (block_az, block_rg, stride_az, stride_rg)
    """
    if use_ocn_dc:
        return _run_ocn_pipeline(
            slc_safe=slc_safe, subswath=subswath,
            poeorb_path=poeorb_path, aux_cal_path=aux_cal_path,
            ocn_safe=ocn_safe, era5_wind=era5_wind, era5_wave=era5_wave,
            glo12=glo12, polarisation=polarisation,
        )

    if merge_first:
        return _run_merged_pipeline(
            slc_safe=slc_safe, subswath=subswath,
            poeorb_path=poeorb_path, aux_cal_path=aux_cal_path,
            ocn_safe=ocn_safe, era5_wind=era5_wind, era5_wave=era5_wave,
            glo12=glo12, polarisation=polarisation, use_ocn_dc=use_ocn_dc, **kwargs,
        )

    files = find_safe_files(slc_safe, subswath, polarisation)
    annot = parse_annotation(files['annotation'])
    n_bursts = len(annot.bursts)

    if burst_indices is None:
        burst_indices = list(range(n_bursts))

    print(f'Processing {len(burst_indices)} bursts for {subswath.upper()} …')
    results = []
    for bidx in burst_indices:
        print(f'\n─── Burst {bidx} / {n_bursts - 1} ───────────────────────────')
        r = run_pipeline(
            slc_safe=slc_safe, subswath=subswath, burst_idx=bidx,
            poeorb_path=poeorb_path, aux_cal_path=aux_cal_path,
            ocn_safe=ocn_safe, era5_wind=era5_wind, era5_wave=era5_wave,
            glo12=glo12, polarisation=polarisation, **kwargs,
            tops_scaling=tops_scaling, use_ocn_dc=use_ocn_dc,
        )
        results.append(r)

    return results


def _run_ocn_pipeline(
    slc_safe:     str,
    subswath:     str,
    poeorb_path:  str,
    aux_cal_path: str,
    ocn_safe:     str,
    era5_wind:    str,
    era5_wave:    str,
    glo12:        str,
    polarisation: str = 'vv',
) -> list[dict]:
    """
    Process purely on the OCN product grid — no SLC reading or deramping.

    Uses rvlDcObs as f_dc and rvlDcGeo as the geometry reference, both already
    on the OCN's merged output grid. Applies our own sideband, Stokes, wave, and
    GLO12 corrections on top. Returns a single-element list.
    """
    from scipy.interpolate import NearestNDInterpolator

    print(f'use_ocn_dc: working on OCN grid for {subswath.upper()} (no SLC processing) …')
    files          = find_safe_files(slc_safe, subswath, polarisation)
    annot_original = parse_annotation(files['annotation'])
    annot          = apply_poeorb(annot_original, poeorb_path)

    # Sideband bias only — no gamma_amb (not estimating from lag-1)
    aap        = parse_aux_cal(aux_cal_path, subswath, polarisation)
    mid_burst  = len(annot.bursts) // 2
    _, vel_mid = _interpolate_orbit(annot, annot.bursts[mid_burst].azimuth_time)
    v_eff      = float(np.linalg.norm(vel_mid))
    f_sideband = compute_sideband_bias(aap, 0.0, annot.radar_prf, v_eff, annot.wavelength)
    print(f'  f_sideband={f_sideband:.3f} Hz')

    # Load OCN product
    print('Loading OCN …')
    ocn    = load_ocn_safe(ocn_safe, swath=subswath, polarisation=polarisation)
    ds_rvl = ocn['rvl']
    sw     = _SWATH_IDX[subswath.lower()]
    if 'rvlSwath' in ds_rvl.dims:
        ds_rvl = ds_rvl.isel(rvlSwath=sw)

    def _clean(name):
        arr = ds_rvl[name].values.astype(np.float64)
        arr[arr == _OCN_FILL] = np.nan
        return arr

    lat         = _clean('rvlLat').astype(np.float32)
    lon         = _clean('rvlLon').astype(np.float32)
    ocn_dc_obs  = _clean('rvlDcObs').astype(np.float32)
    ocn_dc_geo  = _clean('rvlDcGeo').astype(np.float32)
    ocn_dc_miss = _clean('rvlDcMiss').astype(np.float32)

    f_dc  = ocn_dc_obs
    f_dca = (f_dc - ocn_dc_geo - float(f_sideband)).astype(np.float32)
    v_r   = (annot.wavelength / 2.0 * f_dca).astype(np.float32)
    print(f'  v_r median: {float(np.nanmedian(v_r)):+.4f} m/s')

    # Incidence angle at OCN grid points via annotation geolocation grid
    g       = annot.geoloc_grid
    lat_g   = g['latitude'].ravel()
    lon_g   = g['longitude'].ravel()
    inc_g   = g['incidence_angle'].ravel()
    valid_g = np.isfinite(lat_g) & np.isfinite(lon_g) & np.isfinite(inc_g)
    interp_inc = NearestNDInterpolator(
        np.column_stack([lat_g[valid_g], lon_g[valid_g]]), inc_g[valid_g],
    )
    lat_flat = lat.ravel()
    lon_flat = lon.ravel()
    inc_flat = np.full(lat_flat.shape, np.nan, dtype=np.float32)
    valid_q  = np.isfinite(lat_flat) & np.isfinite(lon_flat)
    if valid_q.any():
        inc_flat[valid_q] = interp_inc(lat_flat[valid_q], lon_flat[valid_q])
    inc = inc_flat.reshape(lat.shape)

    # Look direction from mid-scene orbit state
    burst    = annot.bursts[mid_burst]
    acq_time = str(burst.azimuth_time)
    burst_t  = _iso_to_datetime(acq_time)
    pos_sv, vel_sv = _interpolate_orbit(annot, burst_t)
    lat_s = np.arcsin(pos_sv[2] / np.linalg.norm(pos_sv))
    lon_s = np.arctan2(pos_sv[1], pos_sv[0])
    e_east  = np.array([-np.sin(lon_s),  np.cos(lon_s), 0.0])
    e_north = np.array([-np.sin(lat_s) * np.cos(lon_s),
                        -np.sin(lat_s) * np.sin(lon_s),
                         np.cos(lat_s)])
    heading = float(np.rad2deg(np.arctan2(np.dot(vel_sv, e_east), np.dot(vel_sv, e_north))))
    look_az = np.deg2rad(heading + 90.0)
    print(f'  Heading: {heading:.1f}°   Look azimuth: {np.rad2deg(look_az):.1f}°')

    # Stokes drift + wave Doppler bias
    print('Loading ERA5 Stokes drift …')
    wave     = load_era5_wave(era5_wave, acq_time)
    v_stokes = compute_stokes_radial(wave, lat, lon, inc, look_az)
    print('Loading ERA5 wind …')
    wind   = load_era5_wind(era5_wind, acq_time)
    v_wave = compute_wave_doppler_bias(wind, lat, lon, inc, look_az)

    v_current = (-v_r - v_stokes - v_wave).astype(np.float32)

    f_miss_ocn = ocn_dc_miss
    v_miss_ocn = (annot.wavelength / 2.0 * f_miss_ocn).astype(np.float32)
    print(f'  v_miss_ocn range: [{np.nanmin(v_miss_ocn):.4f}, {np.nanmax(v_miss_ocn):.4f}] m/s')

    v_current_ocn = (-v_r + v_miss_ocn - v_stokes - v_wave).astype(np.float32)

    print('Loading GLO12 …')
    model   = load_glo12_current(glo12, acq_time)
    v_model = project_current_onto_look(model, lat, lon, inc, look_az)

    bias_vs_glo12, rmse_vs_glo12, r_vs_glo12 = _compute_stats(v_current_ocn, v_model)
    print(f'  vs GLO12 (OCN DC, OCN mispointing):  '
          f'bias={bias_vs_glo12:+.4f}  RMSE={rmse_vs_glo12:.4f}  r={r_vs_glo12:.4f}')

    nan_grid = np.full_like(f_dca, np.nan)
    return [{
        'v_r':            v_r,
        'v_stokes':       v_stokes,
        'v_wave':         v_wave,
        'v_miss_ocn':     v_miss_ocn,
        'f_miss_ocn':     f_miss_ocn,
        'v_current':      v_current,
        'v_current_ocn':  v_current_ocn,
        'v_model':        v_model,
        'lat':            lat,
        'lon':            lon,
        'inc':            inc,
        'f_dca':          f_dca,
        'f_dc':           f_dc,
        'f_dc_ocn':       f_dc,
        'tops_scale':     np.ones(f_dc.shape[1], dtype=np.float32),
        'tops_scaling':   'none',
        'snr':            nan_grid,
        'f_geom_ann':     ocn_dc_geo,
        'f_geom_poe':     ocn_dc_geo,
        'look_az_rad':    look_az,
        'heading_deg':    heading,
        'wavelength_m':   annot.wavelength,
        'subswath':       subswath,
        'burst_idx':      'ocn',
        'acq_time':       acq_time,
        'rmse_vs_glo12':  rmse_vs_glo12,
        'bias_vs_glo12':  bias_vs_glo12,
        'r_vs_glo12':     r_vs_glo12,
        'p0':             nan_grid,
        'p1':             nan_grid,
    }]


def _run_merged_pipeline(
    slc_safe:     str,
    subswath:     str,
    poeorb_path:  str,
    aux_cal_path: str,
    ocn_safe:     str,
    era5_wind:    str,
    era5_wave:    str,
    glo12:        str,
    polarisation: str = 'vv',
    block_az:     int = 256,
    block_rg:     int = 512,
    stride_az:    int = 128,
    stride_rg:    int = 256,
    use_ocn_dc:   bool = False,
) -> list[dict]:
    """
    Merge all bursts first, then run the full pipeline on the continuous image.
    Returns a single-element list for compatibility with merge_burst_grids etc.
    """
    print(f'merge_first: deramping and merging all {subswath.upper()} bursts …')
    files          = find_safe_files(slc_safe, subswath, polarisation)
    annot_original = parse_annotation(files['annotation'])
    annot          = apply_poeorb(annot_original, poeorb_path)

    aap        = parse_aux_cal(aux_cal_path, subswath, polarisation)
    mid_burst  = len(annot.bursts) // 2
    _, vel_mid = _interpolate_orbit(annot, annot.bursts[mid_burst].azimuth_time)
    v_eff      = float(np.linalg.norm(vel_mid))
    gamma_amb  = compute_gamma_ambiguity(aap, annot.radar_prf, v_eff, annot.wavelength)
    f_sideband = compute_sideband_bias(aap, 0.0, annot.radar_prf, v_eff, annot.wavelength)
    print(f'  γ_amb={gamma_amb:.5f}  f_sideband={f_sideband:.3f} Hz')

    # Step I: deramp, window and merge all bursts
    print('Step I: merging bursts …')
    I_c = merge_bursts(annot, files['measurement'])

    # Step II: correlation on the full merged image
    print('Step II: estimating p0, p1 …')
    p0, p1, az_centers, rg_centers = estimate_correlation_grid(
        I_c, block_az, block_rg, stride_az, stride_rg,
    )

    # Step III: Doppler centroid
    print('Step III: Doppler centroid and geometry subtraction …')
    f_dc, _, snr = correlation_to_doppler(
        p0, p1, annot.prf, annot.wavelength, gamma_amb=gamma_amb,
    )

    # Geometry subtraction — blended annotation polynomial over the full scene
    f_geom_ann = _blended_geom_doppler_annotation(annot, az_centers, rg_centers)

    if poeorb_path is not None:
        # Nearest-burst POEORB differential, assigned per block row
        ati = annot.azimuth_time_interval
        az_burst_starts = [
            int(round((b.azimuth_time - annot.first_line_time).total_seconds() / ati))
            for b in annot.bursts
        ]
        lpb = annot.lines_per_burst
        burst_f_geom_poe = [
            _geom_doppler_poeorb(annot, annot_original, j, rg_centers).astype(np.float32)
            for j in range(len(annot.bursts))
        ]
        f_geom_poe = np.zeros((len(az_centers), len(rg_centers)), dtype=np.float32)
        for i, az in enumerate(az_centers):
            j_best = int(np.argmin([
                abs(int(az) - (az0 + lpb // 2)) for az0 in az_burst_starts
            ]))
            f_geom_poe[i, :] = burst_f_geom_poe[j_best]
        f_geom = f_geom_poe
    else:
        f_geom_poe = f_geom_ann.copy()
        f_geom     = f_geom_ann

    f_dca = (f_dc - f_geom - float(f_sideband)).astype(np.float32)
    v_r   = (annot.wavelength / 2.0 * f_dca).astype(np.float32)
    print(f'  v_r median: {float(np.nanmedian(v_r)):+.4f} m/s')

    # Step V: geolocation using global az_centers from the merged image
    print('Step V: geolocation …')
    lat, lon, inc = _geolocate_grid(annot, az_centers, rg_centers)

    # Look direction from mid-scene orbit state
    burst    = annot.bursts[mid_burst]
    acq_time = str(burst.azimuth_time)
    burst_t  = _iso_to_datetime(acq_time)
    pos_sv, vel_sv = _interpolate_orbit(annot, burst_t)
    lat_s = np.arcsin(pos_sv[2] / np.linalg.norm(pos_sv))
    lon_s = np.arctan2(pos_sv[1], pos_sv[0])
    e_east  = np.array([-np.sin(lon_s),  np.cos(lon_s), 0.0])
    e_north = np.array([-np.sin(lat_s) * np.cos(lon_s),
                        -np.sin(lat_s) * np.sin(lon_s),
                         np.cos(lat_s)])
    heading = float(np.rad2deg(np.arctan2(np.dot(vel_sv, e_east), np.dot(vel_sv, e_north))))
    look_az = np.deg2rad(heading + 90.0)
    print(f'  Heading: {heading:.1f}°   Look azimuth: {np.rad2deg(look_az):.1f}°')

    # Stokes drift
    print('Loading ERA5 Stokes drift …')
    wave     = load_era5_wave(era5_wave, acq_time)
    v_stokes = compute_stokes_radial(wave, lat, lon, inc, look_az)

    # Wave Doppler bias
    print('Loading ERA5 wind …')
    wind   = load_era5_wind(era5_wind, acq_time)
    v_wave = compute_wave_doppler_bias(wind, lat, lon, inc, look_az)

    # Current without mispointing
    v_current = (-v_r - v_stokes - v_wave).astype(np.float32)

    # OCN rvlDcMiss (+ optional rvlDcObs)
    print('Loading OCN …')
    ocn    = load_ocn_safe(ocn_safe, swath=subswath, polarisation=polarisation)
    ds_rvl = ocn['rvl']
    sw     = _SWATH_IDX[subswath.lower()]
    if 'rvlSwath' in ds_rvl.dims:
        ds_rvl = ds_rvl.isel(rvlSwath=sw)

    def _clean(name):
        arr = ds_rvl[name].values.astype(np.float64)
        arr[arr == _OCN_FILL] = np.nan
        return arr

    ocn_lat     = _clean('rvlLat')
    ocn_lon     = _clean('rvlLon')
    ocn_dc_miss = _clean('rvlDcMiss')
    ocn_dc_obs  = _clean('rvlDcObs')

    f_miss_ocn = match_to_sar_grid(lat, lon, ocn_lat, ocn_lon, ocn_dc_miss)
    v_miss_ocn = (annot.wavelength / 2.0 * f_miss_ocn).astype(np.float32)
    f_dc_ocn   = match_to_sar_grid(lat, lon, ocn_lat, ocn_lon, ocn_dc_obs)
    print(f'  v_miss_ocn range: [{np.nanmin(v_miss_ocn):.4f}, {np.nanmax(v_miss_ocn):.4f}] m/s')

    if use_ocn_dc:
        f_dc      = f_dc_ocn.astype(np.float32)
        f_dca     = (f_dc - f_geom - float(f_sideband)).astype(np.float32)
        v_r       = (annot.wavelength / 2.0 * f_dca).astype(np.float32)
        v_current = (-v_r - v_stokes - v_wave).astype(np.float32)
        print(f'  using rvlDcObs as f_dc; v_r median: {float(np.nanmedian(v_r)):+.4f} m/s')

    # Fully corrected current
    v_current_ocn = (-v_r + v_miss_ocn - v_stokes - v_wave).astype(np.float32)

    # GLO12
    print('Loading GLO12 …')
    model   = load_glo12_current(glo12, acq_time)
    v_model = project_current_onto_look(model, lat, lon, inc, look_az)

    # Stats
    bias_vs_glo12, rmse_vs_glo12, r_vs_glo12 = _compute_stats(v_current_ocn, v_model)
    print(f'  vs GLO12 (merged, OCN mispointing):  '
          f'bias={bias_vs_glo12:+.4f}  RMSE={rmse_vs_glo12:.4f}  r={r_vs_glo12:.4f}')

    return [{
        'v_r':            v_r,
        'v_stokes':       v_stokes,
        'v_wave':         v_wave,
        'v_miss_ocn':     v_miss_ocn,
        'f_miss_ocn':     f_miss_ocn,
        'v_current':      v_current,
        'v_current_ocn':  v_current_ocn,
        'v_model':        v_model,
        'lat':            lat,
        'lon':            lon,
        'inc':            inc,
        'f_dca':          f_dca,
        'f_dc':           f_dc,
        'f_dc_ocn':       f_dc_ocn,
        'tops_scale':     np.ones(len(rg_centers), dtype=np.float32),
        'tops_scaling':   'none',
        'snr':            snr,
        'f_geom_ann':     f_geom_ann,
        'f_geom_poe':     f_geom_poe,
        'look_az_rad':    look_az,
        'heading_deg':    heading,
        'wavelength_m':   annot.wavelength,
        'subswath':       subswath,
        'burst_idx':      'merged',
        'acq_time':       acq_time,
        'rmse_vs_glo12':  rmse_vs_glo12,
        'bias_vs_glo12':  bias_vs_glo12,
        'r_vs_glo12':     r_vs_glo12,
        'p0':             p0,
        'p1':             np.angle(p1),
    }]


def merge_burst_grids(
    results:        list[dict],
    variable:       str   = 'v_current_ocn',
    overlap:        str   = 'average',
    resolution_deg: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge per-burst results onto a single regular lat/lon grid.

    All burst points are pooled before interpolation so that gaps between
    adjacent bursts are filled naturally by the triangulation.  The overlap
    strategy only affects pixels where two or more bursts actually overlap.

    Parameters
    ----------
    results        : list of dicts from run_pipeline / run_all_bursts
    variable       : which key to merge (default 'v_current_ocn')
    overlap        : how to handle overlapping pixels
                     'average'   — mean of all valid bursts
                     'first'     — first burst that has a valid value
                     'last'      — last burst that has a valid value
                     'best_rmse' — burst with lowest rmse_vs_glo12 wins
    resolution_deg : output grid spacing in degrees

    Returns
    -------
    grid_lat  : 1D array
    grid_lon  : 1D array
    merged    : 2D array, shape (len(grid_lat), len(grid_lon))
    """
    from scipy.interpolate import griddata

    if overlap not in ('average', 'first', 'last', 'best_rmse'):
        raise ValueError(f"overlap must be 'average', 'first', 'last', or 'best_rmse'")

    # Common grid covering all bursts
    all_lat = np.concatenate([r['lat'].ravel() for r in results])
    all_lon = np.concatenate([r['lon'].ravel() for r in results])
    valid   = np.isfinite(all_lat) & np.isfinite(all_lon)
    grid_lat = np.arange(all_lat[valid].min(), all_lat[valid].max() + resolution_deg, resolution_deg)
    grid_lon = np.arange(all_lon[valid].min(), all_lon[valid].max() + resolution_deg, resolution_deg)
    mesh_lon, mesh_lat = np.meshgrid(grid_lon, grid_lat)

    n = len(results)

    if overlap == 'average':
        # Pool all points and interpolate once — naturally fills inter-burst gaps
        all_data = np.concatenate([r[variable].ravel() for r in results])
        ok = np.isfinite(all_lat) & np.isfinite(all_lon) & np.isfinite(all_data)
        merged = griddata(
            points=(all_lon[ok], all_lat[ok]),
            values=all_data[ok],
            xi=(mesh_lon, mesh_lat),
            method='linear',
        ).astype(np.float32)
        return grid_lat, grid_lon, merged

    # For other strategies: interpolate each burst separately, then merge.
    # Gaps between bursts (pixels NaN in all layers) are filled at the end
    # using a combined interpolation so no seam remains.
    layers = np.full((n, len(grid_lat), len(grid_lon)), np.nan, dtype=np.float32)
    for i, r in enumerate(results):
        lat_f  = r['lat'].ravel()
        lon_f  = r['lon'].ravel()
        data_f = r[variable].ravel()
        ok     = np.isfinite(lat_f) & np.isfinite(lon_f) & np.isfinite(data_f)
        if ok.sum() < 3:
            continue
        layers[i] = griddata(
            points=(lon_f[ok], lat_f[ok]),
            values=data_f[ok],
            xi=(mesh_lon, mesh_lat),
            method='linear',
        ).astype(np.float32)

    valid_mask = np.isfinite(layers)

    if overlap == 'first':
        merged = np.full((len(grid_lat), len(grid_lon)), np.nan, dtype=np.float32)
        for i in range(n):
            fill = valid_mask[i] & ~np.isfinite(merged)
            merged[fill] = layers[i][fill]

    elif overlap == 'last':
        merged = np.full((len(grid_lat), len(grid_lon)), np.nan, dtype=np.float32)
        for i in range(n):
            merged[valid_mask[i]] = layers[i][valid_mask[i]]

    elif overlap == 'best_rmse':
        rmse_vals = np.array([r.get('rmse_vs_glo12', np.inf) for r in results])
        order = np.argsort(rmse_vals)[::-1]   # worst first so best overwrites
        merged = np.full((len(grid_lat), len(grid_lon)), np.nan, dtype=np.float32)
        for i in order:
            merged[valid_mask[i]] = layers[i][valid_mask[i]]

    # Fill any remaining gaps between bursts using the pooled interpolation
    gaps = ~np.isfinite(merged)
    if gaps.any():
        all_data = np.concatenate([r[variable].ravel() for r in results])
        ok = np.isfinite(all_lat) & np.isfinite(all_lon) & np.isfinite(all_data)
        combined = griddata(
            points=(all_lon[ok], all_lat[ok]),
            values=all_data[ok],
            xi=(mesh_lon[gaps], mesh_lat[gaps]),
            method='linear',
        ).astype(np.float32)
        merged[gaps] = combined

    return grid_lat, grid_lon, merged


def merge_model_grid(
    results:        list[dict],
    resolution_deg: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate the GLO12 model values from all bursts onto a common regular grid.
    Uses the same pooled approach as merge_burst_grids so the extent matches exactly.
    """
    from scipy.interpolate import griddata

    all_lat  = np.concatenate([r['lat'].ravel()     for r in results])
    all_lon  = np.concatenate([r['lon'].ravel()     for r in results])
    all_data = np.concatenate([r['v_model'].ravel() for r in results])

    ok = np.isfinite(all_lat) & np.isfinite(all_lon) & np.isfinite(all_data)
    grid_lat = np.arange(all_lat[ok].min(), all_lat[ok].max() + resolution_deg, resolution_deg)
    grid_lon = np.arange(all_lon[ok].min(), all_lon[ok].max() + resolution_deg, resolution_deg)
    mesh_lon, mesh_lat = np.meshgrid(grid_lon, grid_lat)

    grid_model = griddata(
        points=(all_lon[ok], all_lat[ok]),
        values=all_data[ok],
        xi=(mesh_lon, mesh_lat),
        method='linear',
    ).astype(np.float32)

    return grid_lat, grid_lon, grid_model


def write_burst_kml(
    results: list[dict],
    out_path: str = 'burst_polygons.kml',
) -> None:
    """
    Write a KML file with one polygon per burst, using the border of the
    lat/lon geolocation grid.  Coloured by burst index (blue → red).
    """
    import colorsys

    def _border(lat, lon):
        """Return (lons, lats) of the grid perimeter, closed."""
        top    = (lon[0,  :],   lat[0,  :])
        right  = (lon[1:, -1],  lat[1:, -1])
        bottom = (lon[-1, -2::-1], lat[-1, -2::-1])
        left   = (lon[-2:0:-1, 0], lat[-2:0:-1, 0])
        lons = np.concatenate([top[0], right[0], bottom[0], left[0], [top[0][0]]])
        lats = np.concatenate([top[1], right[1], bottom[1], left[1], [top[1][0]]])
        return lons, lats

    n = len(results)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        '<Document>',
        f'  <name>Burst polygons ({results[0]["subswath"].upper()})</name>',
    ]

    for i, r in enumerate(results):
        frac = i / max(n - 1, 1)
        # blue (hue=0.67) → red (hue=0.0)
        h, s, v = (1.0 - frac) * 0.67, 1.0, 1.0
        rgb = colorsys.hsv_to_rgb(h, s, v)
        # KML colour: aabbggrr
        kml_col  = f'88{int(rgb[2]*255):02x}{int(rgb[1]*255):02x}{int(rgb[0]*255):02x}'
        kml_line = f'ff{int(rgb[2]*255):02x}{int(rgb[1]*255):02x}{int(rgb[0]*255):02x}'

        style_id = f'burst{r["burst_idx"]}'
        lines += [
            f'  <Style id="{style_id}">',
            f'    <LineStyle><color>{kml_line}</color><width>2</width></LineStyle>',
            f'    <PolyStyle><color>{kml_col}</color></PolyStyle>',
            f'  </Style>',
        ]

        lons, lats = _border(r['lat'], r['lon'])
        coords = ' '.join(f'{lo:.6f},{la:.6f},0' for lo, la in zip(lons, lats))

        bias = r['bias_vs_glo12']
        rmse = r['rmse_vs_glo12']
        rv   = r['r_vs_glo12']
        lines += [
            f'  <Placemark>',
            f'    <name>Burst {r["burst_idx"]}</name>',
            f'    <description>bias={bias:+.4f} m/s  RMSE={rmse:.4f} m/s  r={rv:.4f}</description>',
            f'    <styleUrl>#{style_id}</styleUrl>',
            f'    <Polygon><outerBoundaryIs><LinearRing>',
            f'      <coordinates>{coords}</coordinates>',
            f'    </LinearRing></outerBoundaryIs></Polygon>',
            f'  </Placemark>',
        ]

    lines += ['</Document>', '</kml>']

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'KML saved → {out_path}')


def _compute_stats(sar: np.ndarray, model: np.ndarray) -> tuple[float, float, float]:
    mask = np.isfinite(sar) & np.isfinite(model)
    if mask.sum() < 2:
        return float('nan'), float('nan'), float('nan')
    diff = sar[mask] - model[mask]
    bias = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    r    = float(np.corrcoef(sar[mask], model[mask])[0, 1])
    return bias, rmse, r


def plot_comparison(
    results:       list[dict],
    overlap:       str   = 'average',
    resolution_deg: float = 0.01,
    vmax:          float | None = None,
    out_path:      str   | None = None,
    variable:      str   = 'v_current_ocn',
) -> None:
    """
    Three-panel figure: merged SAR current | GLO12 model | scatter,
    plus a per-burst stats table printed below.

    Parameters
    ----------
    results        : list of dicts from run_all_bursts
    overlap        : merge strategy passed to merge_burst_grids
    resolution_deg : grid resolution in degrees
    vmax           : colour scale limit (auto if None)
    out_path       : save figure to this path if given
    variable       : which variable to merge and compare (default 'v_current_ocn')
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    print('Merging SAR …')
    grid_lat, grid_lon, merged_sar = merge_burst_grids(
        results, overlap=overlap, resolution_deg=resolution_deg, variable=variable
    )
    print('Merging GLO12 …')
    _, _, merged_model = merge_model_grid(results, resolution_deg=resolution_deg)

    extent = [grid_lon.min(), grid_lon.max(), grid_lat.min(), grid_lat.max()]

    if vmax is None:
        finite = np.concatenate([
            merged_sar[np.isfinite(merged_sar)],
            merged_model[np.isfinite(merged_model)],
        ])
        vmax = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 0.5

    # Total stats
    total_bias, total_rmse, total_r = _compute_stats(merged_sar, merged_model)

    fig = plt.figure(figsize=(16, 6))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.35)

    # SAR map
    ax0 = fig.add_subplot(gs[0])
    im0 = ax0.imshow(merged_sar, extent=extent, origin='lower',
                     cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    plt.colorbar(im0, ax=ax0, label='m/s', fraction=0.046, pad=0.04)
    ax0.set_title(f'SAR ocean current ({overlap})\nbias={total_bias:+.3f}  RMSE={total_rmse:.3f}  r={total_r:.3f}')
    ax0.set_xlabel('Longitude [°]'); ax0.set_ylabel('Latitude [°]')

    # GLO12 map
    ax1 = fig.add_subplot(gs[1])
    im1 = ax1.imshow(merged_model, extent=extent, origin='lower',
                     cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    plt.colorbar(im1, ax=ax1, label='m/s', fraction=0.046, pad=0.04)
    ax1.set_title('GLO12 model current')
    ax1.set_xlabel('Longitude [°]'); ax1.set_ylabel('Latitude [°]')

    # Scatter
    ax2 = fig.add_subplot(gs[2])
    sar_f   = merged_sar[np.isfinite(merged_sar) & np.isfinite(merged_model)]
    model_f = merged_model[np.isfinite(merged_sar) & np.isfinite(merged_model)]
    ax2.scatter(model_f, sar_f, s=2, alpha=0.3, rasterized=True)
    lim = vmax * 1.05
    ax2.plot([-lim, lim], [-lim, lim], 'k--', lw=1)
    ax2.set_xlim(-lim, lim); ax2.set_ylim(-lim, lim)
    ax2.set_aspect('equal')
    ax2.set_xlabel('GLO12 [m/s]'); ax2.set_ylabel('SAR [m/s]')
    ax2.set_title(f'SAR vs GLO12 (total)\nbias={total_bias:+.3f}  RMSE={total_rmse:.3f}  r={total_r:.3f}')

    plt.suptitle('Ocean current radial velocity — SAR vs GLO12', fontsize=12)
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Figure saved → {out_path}')
    plt.show()

    # Per-burst stats table
    print(f'\n{"Burst":>6}  {"bias [m/s]":>10}  {"RMSE [m/s]":>10}  {"r":>6}')
    print('─' * 40)
    for r in results:
        print(f'  {r["burst_idx"]:>4}  '
              f'{r["bias_vs_glo12"]:>+10.4f}  '
              f'{r["rmse_vs_glo12"]:>10.4f}  '
              f'{r["r_vs_glo12"]:>6.4f}')
    print('─' * 40)
    print(f'{"total":>6}  {total_bias:>+10.4f}  {total_rmse:>10.4f}  {total_r:>6.4f}')


def plot_pipeline_steps(
    slc_safe:      str,
    subswath:      str,
    poeorb_path:   str,
    aux_cal_path:  str,
    polarisation:  str = 'vv',
    block_az:      int = 256,
    block_rg:      int = 512,
    stride_az:     int = 128,
    stride_rg:     int = 256,
    burst_indices: list[int] | None = None,
    out_path:      str | None = None,
) -> None:
    """
    Diagnostic plot: four panels showing per-burst pipeline intermediates,
    all bursts stitched top-to-bottom (burst 0 at top).

    Panels
    ------
    1. Raw SLC amplitude [dB]       — downsampled to block-grid resolution
    2. Deramped amplitude [dB]      — invalid lines zeroed before downsampling
    3. Block SNR                    — from estimate_correlation_grid
    4. Raw f_dc [Hz]                — before geometry subtraction
    """
    import matplotlib.pyplot as plt

    files = find_safe_files(slc_safe, subswath, polarisation)
    annot = parse_annotation(files['annotation'])
    annot = apply_poeorb(annot, poeorb_path)
    aap   = parse_aux_cal(aux_cal_path, subswath, polarisation)

    if burst_indices is None:
        burst_indices = list(range(len(annot.bursts)))

    amp_raw_list = []
    amp_drp_list = []
    snr_list     = []
    fdc_list     = []
    n_rows_list  = []   # number of azimuth block-rows per burst

    for b_idx in burst_indices:
        burst = annot.bursts[b_idx]
        print(f'  Burst {b_idx}: reading …')

        raw      = read_slc_burst(files['measurement'], annot, b_idx)
        deramped = deramp_burst(raw, annot, b_idx)
        valid_mask = apply_burst_valid_sample_mask(deramped, burst)

        p0, p1, az_centers, rg_centers = estimate_correlation_grid(
            deramped, block_az, block_rg, stride_az, stride_rg,
            valid_mask=valid_mask,
        )

        _, vel    = _interpolate_orbit(annot, burst.azimuth_time)
        gamma_amb = compute_gamma_ambiguity(
            aap, annot.radar_prf, float(np.linalg.norm(vel)), annot.wavelength,
        )

        f_dc, _, snr = correlation_to_doppler(
            p0, p1, annot.prf, annot.wavelength, gamma_amb=gamma_amb,
        )

        # Sample raw and deramped at block-center positions to match block-grid dims
        ix = np.ix_(az_centers, rg_centers)
        amp_r = 20.0 * np.log10(np.abs(raw[ix]) + 1e-10)
        amp_d = 20.0 * np.log10(np.abs(deramped[ix]) + 1e-10)

        amp_raw_list.append(amp_r.astype(np.float32))
        amp_drp_list.append(amp_d.astype(np.float32))
        snr_list.append(snr.astype(np.float32))
        fdc_list.append(f_dc.astype(np.float32))
        n_rows_list.append(amp_r.shape[0])

    # Stack all bursts top-to-bottom with a 1-row NaN separator
    def _stack(arrays):
        n_cols = arrays[0].shape[1]
        sep    = np.full((1, n_cols), np.nan, dtype=np.float32)
        parts  = []
        for i, a in enumerate(arrays):
            parts.append(a.astype(np.float32))
            if i < len(arrays) - 1:
                parts.append(sep)
        return np.vstack(parts)

    stacked = {
        'amp_raw': _stack(amp_raw_list),
        'amp_drp': _stack(amp_drp_list),
        'snr':     _stack(snr_list),
        'fdc':     _stack(fdc_list),
    }

    def _clim(arr):
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return 0.0, 1.0
        return float(np.percentile(finite, 2)), float(np.percentile(finite, 98))

    amp_vmin, amp_vmax = _clim(stacked['amp_raw'])
    snr_vmin, snr_vmax = _clim(stacked['snr'])
    fdc_finite = stacked['fdc'][np.isfinite(stacked['fdc'])]
    fdc_lim    = float(np.percentile(np.abs(fdc_finite), 98)) if fdc_finite.size else 1.0

    titles   = ['Raw amplitude [dB]', 'Deramped amplitude [dB]', 'Block SNR', 'f_dc [Hz] (raw)']
    keys     = ['amp_raw', 'amp_drp', 'snr', 'fdc']
    cmaps    = ['gray', 'gray', 'plasma', 'RdBu_r']
    vlims    = [
        (amp_vmin, amp_vmax),
        (amp_vmin, amp_vmax),
        (snr_vmin, snr_vmax),
        (-fdc_lim,  fdc_lim),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, max(6, len(burst_indices) * 1.2)))
    for ax, title, key, cmap, (vmin, vmax) in zip(axes, titles, keys, cmaps, vlims):
        im = ax.imshow(stacked[key], cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect='auto', interpolation='nearest')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Range block')
        ax.set_ylabel('Azimuth block')

        # Burst boundary lines and index labels
        row = 0
        for i, (b_idx, n_rows) in enumerate(zip(burst_indices, n_rows_list)):
            ax.text(-1, row + n_rows / 2.0, f'B{b_idx}',
                    va='center', ha='right', fontsize=7, clip_on=False)
            row += n_rows
            if i < len(burst_indices) - 1:
                ax.axhline(row - 0.5, color='yellow', lw=0.8, ls='--', alpha=0.8)
                row += 1   # NaN gap row

    plt.suptitle(f'{subswath.upper()} — pipeline diagnostic steps', fontsize=11)
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Figure saved → {out_path}')
    plt.show()


def diagnose_burst_doppler(
    slc_safe:      str,
    subswath:      str,
    poeorb_path:   str,
    aux_cal_path:  str,
    polarisation:  str = 'vv',
    block_az:      int = 256,
    block_rg:      int = 512,
    stride_az:     int = 128,
    stride_rg:     int = 256,
    burst_indices: list[int] | None = None,
) -> None:
    """
    Print a per-burst diagnostic table that distinguishes deramping errors
    (wrong k_s → residual FM chirp) from correlation/masking errors.

    Columns
    -------
    burst   burst index
    k_a     FM rate at mid-range [Hz/s]  — from annotation nearest-neighbour
    k_psi   steering Doppler rate [Hz/s] — scalar per burst
    k_s     deramp rate at mid-range [Hz/s]
    n_valid valid azimuth lines in burst
    fdc_top median f_dc of upper half of azimuth blocks [Hz]
    fdc_bot median f_dc of lower half of azimuth blocks [Hz]
    top-bot difference: large value → residual FM chirp (deramping bug)
    """
    files = find_safe_files(slc_safe, subswath, polarisation)
    annot = parse_annotation(files['annotation'])
    annot = apply_poeorb(annot, poeorb_path)
    aap   = parse_aux_cal(aux_cal_path, subswath, polarisation)

    if burst_indices is None:
        burst_indices = list(range(len(annot.bursts)))

    n_rg_mid = annot.n_samples // 2

    hdr = (f'{"burst":>5}  {"k_a [Hz/s]":>12}  {"k_psi [Hz/s]":>13}  '
           f'{"k_s [Hz/s]":>11}  {"n_valid":>7}  '
           f'{"fdc_top [Hz]":>12}  {"fdc_bot [Hz]":>12}  {"top-bot [Hz]":>12}')
    print(hdr)
    print('─' * len(hdr))

    for b_idx in burst_indices:
        burst = annot.bursts[b_idx]

        # Deramping parameters
        k_a_vec = _fm_rate_at_burst(annot, b_idx)            # (n_rg,)
        k_a     = float(k_a_vec[n_rg_mid])
        k_psi   = float(_steering_doppler_rate(annot, b_idx))
        k_s     = float(-k_a * k_psi / (k_a - k_psi))

        n_valid = int(np.count_nonzero(burst.first_valid_sample != -1))

        # f_dc estimation
        _, vel    = _interpolate_orbit(annot, burst.azimuth_time)
        gamma_amb = compute_gamma_ambiguity(
            aap, annot.radar_prf, float(np.linalg.norm(vel)), annot.wavelength,
        )

        raw      = read_slc_burst(files['measurement'], annot, b_idx)
        deramped = deramp_burst(raw, annot, b_idx)
        vmask    = apply_burst_valid_sample_mask(deramped, burst)

        p0, p1, az_centers, rg_centers = estimate_correlation_grid(
            deramped, block_az, block_rg, stride_az, stride_rg,
            valid_mask=vmask,
        )

        f_dc, _, _ = correlation_to_doppler(
            p0, p1, annot.prf, annot.wavelength, gamma_amb=gamma_amb,
        )

        n_az_blocks = f_dc.shape[0]
        half = max(1, n_az_blocks // 2)
        fdc_top = float(np.nanmedian(f_dc[:half, :]))
        fdc_bot = float(np.nanmedian(f_dc[half:, :]))
        diff    = fdc_top - fdc_bot

        print(f'{b_idx:>5}  {k_a:>12.2f}  {k_psi:>13.2f}  '
              f'{k_s:>11.2f}  {n_valid:>7}  '
              f'{fdc_top:>12.2f}  {fdc_bot:>12.2f}  {diff:>+12.2f}')


def plot_dc_estimates(
    slc_safe:     str,
    subswath:     str,
    polarisation: str = 'vv',
    out_path:     str | None = None,
) -> None:
    """
    Stacked-burst image of ESA's annotation DC polynomials (no SLC processing).

    Three panels:
      Left   — f_geom [Hz]: orbit-predicted Doppler centroid
      Centre — f_data [Hz]: ESA's data-derived Doppler centroid
      Right  — f_data − f_geom [Hz]: anomaly (ocean surface + any pipeline artifact)

    Also prints a table: burst, which dc_estimate was selected, mid-range values.
    """
    import matplotlib.pyplot as plt

    files = find_safe_files(slc_safe, subswath, polarisation)
    annot = parse_annotation(files['annotation'])
    n_rg  = annot.n_samples
    tau   = annot.slant_range_time_start + np.arange(n_rg) / annot.range_sampling_rate

    # Downsample range for display to avoid OOM (full res ~21 k samples per burst)
    rg_step = max(1, n_rg // 500)
    tau_ds  = tau[::rg_step]
    n_ds    = len(tau_ds)
    rows_per_burst = 10   # display height per burst in the image

    geom_imgs, data_imgs, diff_imgs = [], [], []
    sep = np.full((1, n_ds), np.nan, dtype=np.float32)

    hdr = (f'{"burst":>5}  {"dc_idx":>6}  {"f_geom_mid [Hz]":>16}  '
           f'{"f_data_mid [Hz]":>16}  {"diff_mid [Hz]":>14}')
    print(hdr)
    print('─' * len(hdr))

    for b_idx, burst in enumerate(annot.bursts):
        dc     = _nearest_estimate(annot.dc_estimates, burst.azimuth_time)
        dc_idx = annot.dc_estimates.index(dc)
        dt     = tau - dc.t0                                            # (n_rg,) full res
        f_geom = sum(c * dt**k for k, c in enumerate(dc.geometry_poly))
        f_data = sum(c * dt**k for k, c in enumerate(dc.data_poly))

        f_geom_ds = f_geom[::rg_step].astype(np.float32)
        f_data_ds = f_data[::rg_step].astype(np.float32)
        geom_imgs.append(np.tile(f_geom_ds, (rows_per_burst, 1)))
        data_imgs.append(np.tile(f_data_ds, (rows_per_burst, 1)))
        diff_imgs.append(np.tile((f_data_ds - f_geom_ds), (rows_per_burst, 1)))
        if b_idx < len(annot.bursts) - 1:
            geom_imgs.append(sep)
            data_imgs.append(sep)
            diff_imgs.append(sep)

        mid = n_rg // 2
        print(f'{b_idx:>5}  {dc_idx:>6}  {float(f_geom[mid]):>16.2f}  '
              f'{float(f_data[mid]):>16.2f}  {float(f_data[mid] - f_geom[mid]):>+14.2f}')

    G  = np.vstack(geom_imgs)
    D  = np.vstack(data_imgs)
    Df = np.vstack(diff_imgs)

    fig, axes = plt.subplots(1, 3, figsize=(16, max(6, len(annot.bursts))),
                             sharex=True, sharey=True)
    panels = [
        (G,  'f_geom — orbit predicted'),
        (D,  'f_data — ESA measured'),
        (Df, 'f_data − f_geom'),
    ]
    for ax, (arr, title) in zip(axes, panels):
        finite = arr[np.isfinite(arr)]
        vmax = float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0
        im = ax.imshow(arr, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       aspect='auto', interpolation='nearest')
        ax.set_title(title)
        ax.set_xlabel('range sample')
        plt.colorbar(im, ax=ax, label='Hz', shrink=0.6)
    axes[0].set_ylabel('azimuth line (bursts stacked)')
    fig.suptitle(f'{subswath.upper()} {polarisation.upper()} — annotation DC polynomials')
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Figure saved → {out_path}')
    plt.show()


def diagnose_mispointing_aux_ins(
    slc_safe:     str,
    subswath:     str,
    aux_ins_path: str,
    poeorb_path:  str | None = None,
    polarisation: str = 'vv',
    out_path:     str | None = None,
) -> None:
    """
    Compare per-burst mispointing Doppler (from attitude yaw deviation vs. nominal
    ZDS yaw) against the annotation DC polynomials.

    Mispointing Doppler: f_miss = (2/λ) · |v_sat| · (ψ_actual − ψ_ZDS)
    where ψ_ZDS is the nominal Zero-Doppler Steering yaw computed from the orbit.

    Prints:
        burst  f_miss [Hz]  f_data [Hz]  f_geom [Hz]  f_dca [Hz]  residual [Hz]

    residual = f_dca - f_miss is what remains unexplained by attitude mispointing.
    """
    import matplotlib.pyplot as plt

    files = find_safe_files(slc_safe, subswath, polarisation)
    annot_orig = parse_annotation(files['annotation'])
    annot = apply_poeorb(annot_orig, poeorb_path) if poeorb_path else annot_orig

    ins = parse_aux_ins(aux_ins_path)
    print(f'AUX_INS  referenceAntennaAngle = {ins.reference_antenna_angle_deg:.2f}°')
    if not annot.attitude:
        print('WARNING: no attitude records in annotation — f_miss will be zero.')
    print()

    n_rg = annot.n_samples
    tau  = annot.slant_range_time_start + np.arange(n_rg) / annot.range_sampling_rate
    mid  = n_rg // 2

    f_miss_arr, f_data_arr, f_geom_arr = [], [], []
    for b_idx, burst in enumerate(annot.bursts):
        dc     = _nearest_estimate(annot.dc_estimates, burst.azimuth_time)
        dt_mid = tau[mid] - dc.t0
        f_geom = float(sum(c * dt_mid**k for k, c in enumerate(dc.geometry_poly)))
        f_data = float(sum(c * dt_mid**k for k, c in enumerate(dc.data_poly)))
        f_miss = mispointing_doppler_from_yaw(annot, b_idx)

        f_miss_arr.append(f_miss)
        f_data_arr.append(f_data)
        f_geom_arr.append(f_geom)

    f_miss_arr = np.array(f_miss_arr)
    f_data_arr = np.array(f_data_arr)
    f_geom_arr = np.array(f_geom_arr)
    f_dca_arr  = f_data_arr - f_geom_arr
    residual   = f_dca_arr - f_miss_arr

    hdr = (f'{"burst":>5}  {"f_miss [Hz]":>12}  {"f_data [Hz]":>12}  '
           f'{"f_geom [Hz]":>12}  {"f_dca [Hz]":>11}  {"residual [Hz]":>14}')
    print(hdr)
    print('─' * len(hdr))
    for i in range(len(annot.bursts)):
        print(f'{i:>5}  {f_miss_arr[i]:>12.2f}  {f_data_arr[i]:>12.2f}  '
              f'{f_geom_arr[i]:>12.2f}  {f_dca_arr[i]:>11.2f}  {residual[i]:>+14.2f}')

    # ── plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    bursts = np.arange(len(annot.bursts))

    ax1.plot(bursts, f_data_arr,  'o-', label='f_data (annotation)')
    ax1.plot(bursts, f_geom_arr,  's--', label='f_geom (orbit)')
    ax1.plot(bursts, f_miss_arr,  '^-', label='f_miss (AUX_INS attitude)')
    ax1.set_xlabel('Burst index')
    ax1.set_ylabel('Doppler [Hz]')
    ax1.set_title('Doppler components per burst')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.bar(bursts - 0.2, f_dca_arr,  width=0.4, label='f_dca = f_data − f_geom', alpha=0.7)
    ax2.bar(bursts + 0.2, f_miss_arr, width=0.4, label='f_miss (AUX_INS)', alpha=0.7)
    ax2.set_xlabel('Burst index')
    ax2.set_ylabel('Doppler [Hz]')
    ax2.set_title('f_dca vs mispointing Doppler')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'{subswath.upper()} {polarisation.upper()} — mispointing diagnostic (AUX_INS)')
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'\nFigure saved → {out_path}')
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser():
    p = argparse.ArgumentParser(
        description='RVL pipeline — exact replication of rvl_pipeline_walkthrough.ipynb',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('slc_safe')
    p.add_argument('subswath')
    p.add_argument('burst_idx', type=int)
    p.add_argument('--poeorb',    required=True, metavar='EOF')
    p.add_argument('--aux-cal',   required=True, metavar='SAFE')
    p.add_argument('--ocn-safe',  required=True, metavar='SAFE')
    p.add_argument('--era5-wind', required=True, metavar='FILE')
    p.add_argument('--era5-wave', required=True, metavar='FILE')
    p.add_argument('--glo12',     required=True, metavar='FILE')
    p.add_argument('--pol',       default='vv')
    p.add_argument('--block-az',  type=int, default=256)
    p.add_argument('--block-rg',  type=int, default=512)
    p.add_argument('--stride-az', type=int, default=128)
    p.add_argument('--stride-rg', type=int, default=256)
    p.add_argument('--tops-scaling', choices=('none', 'multiply', 'divide'), default='none',
                   help='Optional TOPS focused-data Doppler scale test')
    p.add_argument('--out',       default=None, metavar='FILE',
                   help='Save result dict as .npz (optional)')
    return p


if __name__ == '__main__':
    args = _build_parser().parse_args()
    result = run_pipeline(
        slc_safe     = args.slc_safe,
        subswath     = args.subswath,
        burst_idx    = args.burst_idx,
        poeorb_path  = args.poeorb,
        aux_cal_path = args.aux_cal,
        ocn_safe     = args.ocn_safe,
        era5_wind    = args.era5_wind,
        era5_wave    = args.era5_wave,
        glo12        = args.glo12,
        polarisation = args.pol,
        block_az     = args.block_az,
        block_rg     = args.block_rg,
        stride_az    = args.stride_az,
        stride_rg    = args.stride_rg,
        tops_scaling = args.tops_scaling,
    )
    if args.out:
        np.savez(args.out, **{k: v for k, v in result.items()
                              if isinstance(v, np.ndarray)})
        print(f'Saved arrays → {args.out}')
