"""Run the main Sentinel-1 RVL retrieval pipeline and its merged-burst variants."""

from __future__ import annotations

import argparse

import numpy as np

from scripts.diagnostics.pipeline_diagnostics import (
    diagnose_burst_doppler as _diagnose_burst_doppler,
    diagnose_mispointing_aux_ins as _diagnose_mispointing_aux_ins,
    plot_dc_estimates as _plot_dc_estimates,
    plot_pipeline_steps as _plot_pipeline_steps,
)
from .metocean import (
    compute_stokes_radial,
    compute_wave_doppler_bias,
    load_era5_wave,
    load_era5_wind,
    load_glo12_current,
    match_to_sar_grid,
    project_current_onto_look,
)
from .grid_merge import (
    compute_stats as _compute_stats_impl,
    merge_burst_grids as _merge_burst_grids,
    merge_model_grid as _merge_model_grid,
    write_burst_kml as _write_burst_kml,
)
from .aux_files import apply_poeorb, parse_aux_cal, parse_aux_ins
from .safe_io import _iso_to_datetime, _nearest_estimate, find_safe_files, parse_annotation, read_slc_burst
from .ocn_product import load_ocn_safe
from .rvl import (
    deramp_burst,
    merge_bursts,
    merge_gamma_bursts,
    estimate_correlation_grid,
    estimate_doppler_grid_fft,
    correlation_to_doppler,
    _fm_rate_at_burst,
    _steering_doppler_rate,
    apply_burst_valid_sample_mask,
    compute_gamma_ambiguity,
    compute_sideband_bias,
    _geom_doppler_annotation,
    _geom_doppler_poeorb,
    _blended_geom_doppler_annotation,
    _blended_geom_doppler,
    _blend_burst_profiles,
    _geolocate_grid,
    _interpolate_orbit,
    compute_mispointing_doppler,
    mispointing_doppler_from_yaw,
)
from .plots import plot_comparison as _plot_comparison_impl

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
    estimator:    str = 'cde',
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

    # ── Step II: Correlation / Doppler estimation ────────────────────────────
    if estimator == 'fft':
        print('Step II: estimating Doppler via FFT spectral centroid …')
        f_dc, snr, az_local, rg_centers = estimate_doppler_grid_fft(
            deramped, annot.prf, block_az, block_rg, stride_az, stride_rg,
            valid_mask=valid_mask,
        )
        p0 = np.ones_like(f_dc, dtype=np.float32)
        p1 = np.full_like(f_dc, np.nan, dtype=np.float32)
    else:
        print('Step II: estimating p0, p1 …')
        p0, p1, az_local, rg_centers = estimate_correlation_grid(
            deramped, block_az, block_rg, stride_az, stride_rg,
            valid_mask=valid_mask,
        )
        f_dc, _, snr = correlation_to_doppler(
            p0, p1, annot.prf, annot.wavelength, gamma_amb=gamma_amb,
        )

    # Full-scene azimuth indices for geolocation
    ati      = annot.azimuth_time_interval
    az0_full = int(round(
        (burst.azimuth_time - annot.first_line_time).total_seconds() / ati
    ))
    az_full = az_local + az0_full

    # ── Step III: Doppler centroid → DCA ────────────────────────────────────
    print('Step III: geometry subtraction …')
    if tops_scaling not in ('none', 'multiply', 'divide'):
        raise ValueError("tops_scaling must be 'none', 'multiply', or 'divide'")
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
    estimator:    str = 'cde',
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
            glo12=glo12, polarisation=polarisation, use_ocn_dc=use_ocn_dc,
            estimator=estimator, **kwargs,
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
            tops_scaling=tops_scaling, use_ocn_dc=use_ocn_dc, estimator=estimator,
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
    estimator:    str = 'cde',
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

    # Step II: Doppler estimation on the full merged image
    if estimator == 'fft':
        print('Step II: estimating Doppler via FFT spectral centroid …')
        f_dc, snr, az_centers, rg_centers = estimate_doppler_grid_fft(
            I_c, annot.prf, block_az, block_rg, stride_az, stride_rg,
        )
        p0 = np.ones_like(f_dc, dtype=np.float32)
        p1 = np.full_like(f_dc, np.nan, dtype=np.float32)
    else:
        print('Step II: estimating p0, p1 …')
        p0, p1, az_centers, rg_centers = estimate_correlation_grid(
            I_c, block_az, block_rg, stride_az, stride_rg,
        )
        f_dc, _, snr = correlation_to_doppler(
            p0, p1, annot.prf, annot.wavelength, gamma_amb=gamma_amb,
        )

    # Step III: Doppler centroid and geometry subtraction
    print('Step III: geometry subtraction …')

    # Geometry subtraction — blended annotation polynomial over the full scene
    f_geom_ann = _blended_geom_doppler_annotation(annot, az_centers, rg_centers)

    if poeorb_path is not None:
        burst_f_geom_poe_arr = np.stack([
            _geom_doppler_poeorb(annot, annot_original, j, rg_centers).astype(np.float64)
            for j in range(len(annot.bursts))
        ], axis=0)
        # Blend with partition-of-unity weights to avoid a step at burst boundaries
        f_geom_poe = _blend_burst_profiles(annot, az_centers, burst_f_geom_poe_arr)
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


def run_gamma_merged_pipeline(
    slc_paths:     list[str],
    slc_par_paths: list[str],
    annotation_xml: str,
    subswath:      str,
    poeorb_path:   str | None,
    aux_cal_path:  str,
    ocn_safe:      str,
    era5_wind:     str,
    era5_wave:     str,
    glo12:         str,
    polarisation:  str = 'vv',
    block_az:      int = 256,
    block_rg:      int = 512,
    stride_az:     int = 128,
    stride_rg:     int = 256,
    use_ocn_dc:    bool = False,
    estimator:     str = 'cde',
) -> dict:
    """
    Run the merged RVL pipeline starting from GAMMA per-burst SLC files.

    This mirrors `_run_merged_pipeline()`, except Step I uses pre-deramped
    GAMMA burst SLCs instead of reading and deramping the SAFE measurement
    TIFF internally.
    """
    print(f'gamma_merged: merging {len(slc_paths)} deramped {subswath.upper()} bursts …')
    annot_original = parse_annotation(annotation_xml)
    annot = apply_poeorb(annot_original, poeorb_path) if poeorb_path is not None else annot_original

    aap = parse_aux_cal(aux_cal_path, subswath, polarisation)
    mid_burst = len(annot.bursts) // 2
    _, vel_mid = _interpolate_orbit(annot, annot.bursts[mid_burst].azimuth_time)
    v_eff = float(np.linalg.norm(vel_mid))
    gamma_amb = compute_gamma_ambiguity(aap, annot.radar_prf, v_eff, annot.wavelength)
    f_sideband = compute_sideband_bias(aap, 0.0, annot.radar_prf, v_eff, annot.wavelength)
    print(f'  γ_amb={gamma_amb:.5f}  f_sideband={f_sideband:.3f} Hz')

    print('Step I: merging pre-deramped GAMMA bursts …')
    I_c = merge_gamma_bursts(slc_paths, slc_par_paths)

    if estimator == 'fft':
        print('Step II: estimating Doppler via FFT spectral centroid …')
        f_dc, snr, az_centers, rg_centers = estimate_doppler_grid_fft(
            I_c, annot.prf, block_az, block_rg, stride_az, stride_rg,
        )
        p0 = np.ones_like(f_dc, dtype=np.float32)
        p1 = np.full_like(f_dc, np.nan, dtype=np.float32)
    else:
        print('Step II: estimating p0, p1 …')
        p0, p1, az_centers, rg_centers = estimate_correlation_grid(
            I_c, block_az, block_rg, stride_az, stride_rg,
        )
        f_dc, _, snr = correlation_to_doppler(
            p0, p1, annot.prf, annot.wavelength, gamma_amb=gamma_amb,
        )

    print('Step III: geometry subtraction …')
    f_geom_ann = _blended_geom_doppler_annotation(annot_original, az_centers, rg_centers)

    if poeorb_path is not None:
        burst_f_geom_poe_arr = np.stack([
            _geom_doppler_poeorb(annot, annot_original, j, rg_centers).astype(np.float64)
            for j in range(len(annot.bursts))
        ], axis=0)
        f_geom_poe = _blend_burst_profiles(annot, az_centers, burst_f_geom_poe_arr)
        f_geom = f_geom_poe
    else:
        # Fall back to the Doppler polynomials stored in the GAMMA burst .par files.
        f_geom_poe = _blended_geom_doppler(slc_par_paths, az_centers, rg_centers)
        f_geom = f_geom_poe

    f_dca = (f_dc - f_geom - float(f_sideband)).astype(np.float32)
    v_r = (annot.wavelength / 2.0 * f_dca).astype(np.float32)
    print(f'  v_r median: {float(np.nanmedian(v_r)):+.4f} m/s')

    print('Step V: geolocation …')
    lat, lon, inc = _geolocate_grid(annot, az_centers, rg_centers)

    burst = annot.bursts[mid_burst]
    acq_time = str(burst.azimuth_time)
    burst_t = _iso_to_datetime(acq_time)
    pos_sv, vel_sv = _interpolate_orbit(annot, burst_t)
    lat_s = np.arcsin(pos_sv[2] / np.linalg.norm(pos_sv))
    lon_s = np.arctan2(pos_sv[1], pos_sv[0])
    e_east = np.array([-np.sin(lon_s), np.cos(lon_s), 0.0])
    e_north = np.array([
        -np.sin(lat_s) * np.cos(lon_s),
        -np.sin(lat_s) * np.sin(lon_s),
        np.cos(lat_s),
    ])
    heading = float(np.rad2deg(np.arctan2(np.dot(vel_sv, e_east), np.dot(vel_sv, e_north))))
    look_az = np.deg2rad(heading + 90.0)
    print(f'  Heading: {heading:.1f}°   Look azimuth: {np.rad2deg(look_az):.1f}°')

    print('Loading ERA5 Stokes drift …')
    wave = load_era5_wave(era5_wave, acq_time)
    v_stokes = compute_stokes_radial(wave, lat, lon, inc, look_az)

    print('Loading ERA5 wind …')
    wind = load_era5_wind(era5_wind, acq_time)
    v_wave = compute_wave_doppler_bias(wind, lat, lon, inc, look_az)

    v_current = (-v_r - v_stokes - v_wave).astype(np.float32)

    print('Loading OCN …')
    ocn = load_ocn_safe(ocn_safe, swath=subswath, polarisation=polarisation)
    ds_rvl = ocn['rvl']
    sw = _SWATH_IDX[subswath.lower()]
    if 'rvlSwath' in ds_rvl.dims:
        ds_rvl = ds_rvl.isel(rvlSwath=sw)

    def _clean(name):
        arr = ds_rvl[name].values.astype(np.float64)
        arr[arr == _OCN_FILL] = np.nan
        return arr

    ocn_lat = _clean('rvlLat')
    ocn_lon = _clean('rvlLon')
    ocn_dc_miss = _clean('rvlDcMiss')
    ocn_dc_obs = _clean('rvlDcObs')

    f_miss_ocn = match_to_sar_grid(lat, lon, ocn_lat, ocn_lon, ocn_dc_miss)
    v_miss_ocn = (annot.wavelength / 2.0 * f_miss_ocn).astype(np.float32)
    f_dc_ocn = match_to_sar_grid(lat, lon, ocn_lat, ocn_lon, ocn_dc_obs)
    print(f'  v_miss_ocn range: [{np.nanmin(v_miss_ocn):.4f}, {np.nanmax(v_miss_ocn):.4f}] m/s')

    if use_ocn_dc:
        f_dc = f_dc_ocn.astype(np.float32)
        f_dca = (f_dc - f_geom - float(f_sideband)).astype(np.float32)
        v_r = (annot.wavelength / 2.0 * f_dca).astype(np.float32)
        v_current = (-v_r - v_stokes - v_wave).astype(np.float32)
        print(f'  using rvlDcObs as f_dc; v_r median: {float(np.nanmedian(v_r)):+.4f} m/s')

    v_current_ocn = (-v_r + v_miss_ocn - v_stokes - v_wave).astype(np.float32)

    print('Loading GLO12 …')
    model = load_glo12_current(glo12, acq_time)
    v_model = project_current_onto_look(model, lat, lon, inc, look_az)

    bias_vs_glo12, rmse_vs_glo12, r_vs_glo12 = _compute_stats(v_current_ocn, v_model)
    print(f'  vs GLO12 (gamma merged, OCN mispointing):  '
          f'bias={bias_vs_glo12:+.4f}  RMSE={rmse_vs_glo12:.4f}  r={r_vs_glo12:.4f}')

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
        'tops_scale':     np.ones(len(rg_centers), dtype=np.float32),
        'tops_scaling':   'none',
        'snr':            snr,
        'f_geom_ann':     f_geom_ann,
        'f_geom_poe':     f_geom_poe,
        'look_az_rad':    look_az,
        'heading_deg':    heading,
        'wavelength_m':   annot.wavelength,
        'subswath':       subswath,
        'burst_idx':      'gamma_merged',
        'acq_time':       acq_time,
        'rmse_vs_glo12':  rmse_vs_glo12,
        'bias_vs_glo12':  bias_vs_glo12,
        'r_vs_glo12':     r_vs_glo12,
        'p0':             p0,
        'p1':             np.angle(p1),
    }


def merge_burst_grids(
    results:        list[dict],
    variable:       str   = 'v_current_ocn',
    overlap:        str   = 'average',
    resolution_deg: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility wrapper for the dedicated merge module."""
    return _merge_burst_grids(results, variable=variable, overlap=overlap, resolution_deg=resolution_deg)


def merge_model_grid(
    results:        list[dict],
    resolution_deg: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility wrapper for the dedicated merge module."""
    return _merge_model_grid(results, resolution_deg=resolution_deg)


def write_burst_kml(
    results: list[dict],
    out_path: str = 'burst_polygons.kml',
) -> None:
    """Compatibility wrapper for the dedicated merge module."""
    _write_burst_kml(results, out_path=out_path)


def _compute_stats(sar: np.ndarray, model: np.ndarray) -> tuple[float, float, float]:
    """Compatibility wrapper for the dedicated merge module."""
    return _compute_stats_impl(sar, model)


def plot_comparison(
    results:       list[dict],
    overlap:       str   = 'average',
    resolution_deg: float = 0.01,
    vmax:          float | None = None,
    out_path:      str   | None = None,
    variable:      str   = 'v_current_ocn',
) -> None:
    """Compatibility wrapper for the dedicated visualization module."""
    _plot_comparison_impl(
        results,
        overlap=overlap,
        resolution_deg=resolution_deg,
        vmax=vmax,
        out_path=out_path,
        variable=variable,
    )


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
    """Compatibility wrapper for the diagnostics module."""
    _plot_pipeline_steps(
        slc_safe,
        subswath,
        poeorb_path,
        aux_cal_path,
        polarisation=polarisation,
        block_az=block_az,
        block_rg=block_rg,
        stride_az=stride_az,
        stride_rg=stride_rg,
        burst_indices=burst_indices,
        out_path=out_path,
    )


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
    """Compatibility wrapper for the diagnostics module."""
    _diagnose_burst_doppler(
        slc_safe,
        subswath,
        poeorb_path,
        aux_cal_path,
        polarisation=polarisation,
        block_az=block_az,
        block_rg=block_rg,
        stride_az=stride_az,
        stride_rg=stride_rg,
        burst_indices=burst_indices,
    )


def plot_dc_estimates(
    slc_safe:     str,
    subswath:     str,
    polarisation: str = 'vv',
    out_path:     str | None = None,
) -> None:
    """Compatibility wrapper for the diagnostics module."""
    _plot_dc_estimates(slc_safe, subswath, polarisation=polarisation, out_path=out_path)


def diagnose_mispointing_aux_ins(
    slc_safe:     str,
    subswath:     str,
    aux_ins_path: str,
    poeorb_path:  str | None = None,
    polarisation: str = 'vv',
    out_path:     str | None = None,
) -> None:
    """Compatibility wrapper for the diagnostics module."""
    _diagnose_mispointing_aux_ins(
        slc_safe,
        subswath,
        aux_ins_path,
        poeorb_path=poeorb_path,
        polarisation=polarisation,
        out_path=out_path,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser():
    p = argparse.ArgumentParser(
        description='Sentinel-1 RVL retrieval pipeline',
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
