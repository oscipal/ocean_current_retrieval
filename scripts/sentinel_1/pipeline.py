"""Run the main Sentinel-1 RVL retrieval pipeline and its merged-burst variants."""

from __future__ import annotations

import argparse
import os

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
    compute_wave_doppler_bias_ocn,
    load_era5_wave,
    load_era5_wind,
    load_glo12_current,
    load_ocn_wave_velocity,
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
    descallop,
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


def _maybe_descallop_grid(
    f_dca: np.ndarray,
    snr: np.ndarray,
    annot,
    stride_az: int,
    do_descallop: bool,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Apply burst-periodic descalloping when enough burst timing information is available."""
    f_dca_pre = f_dca.copy()
    if not do_descallop or len(annot.bursts) < 2:
        return f_dca, f_dca_pre, None

    burst_dt_s = (
        annot.bursts[1].azimuth_time - annot.bursts[0].azimuth_time
    ).total_seconds()
    burst_period_rows = burst_dt_s / annot.azimuth_time_interval / stride_az
    if not np.isfinite(burst_period_rows) or burst_period_rows <= 0:
        return f_dca, f_dca_pre, None

    # Require at least 6 inter-burst periods (≈ 3+ bursts in the data) before
    # attempting descalloping.  For IW mode burst_period_rows ≈ 2.4, so this
    # means at least ~15 output rows.  Single-burst output grids (~12 rows, ≈ 5
    # periods) appear to cover multiple periods but the scalloping there repeats
    # at the burst DURATION frequency (not the inter-burst frequency), so the
    # model is wrong and the fit introduces artifacts rather than removing them.
    if f_dca.shape[0] < int(np.ceil(6.0 * burst_period_rows)):
        return f_dca, f_dca_pre, burst_period_rows

    return descallop(f_dca, snr, burst_period_rows), f_dca_pre, burst_period_rows


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
    deramp_method: str = 'esa_eq1',
    do_descallop: bool = True,
    tops_scaling: str = 'none',
    use_ocn_dc:   bool = False,
    estimator:    str = 'fft',
    add_data_poly: bool = False,
    deramped_override: np.ndarray | None = None,
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
    #print(f'Loading annotation for {subswath.upper()} burst {burst_idx} …')
    files          = find_safe_files(slc_safe, subswath, polarisation)
    annot_original = parse_annotation(files['annotation'])
    annot          = apply_poeorb(annot_original, poeorb_path)
    burst          = annot.bursts[burst_idx]
    #print(f'  Burst time : {burst.azimuth_time}')
    #print(f'  PRF        : {annot.prf:.3f} Hz')
    #print(f'  Wavelength : {annot.wavelength*100:.2f} cm')

    # ── AUX_CAL ──────────────────────────────────────────────────────────────
    aap        = parse_aux_cal(aux_cal_path, subswath, polarisation)
    _, vel     = _interpolate_orbit(annot, burst.azimuth_time)
    v_eff      = float(np.linalg.norm(vel))
    gamma_amb  = compute_gamma_ambiguity(aap, annot.radar_prf, v_eff, annot.wavelength)
    f_sideband = compute_sideband_bias(aap, 0.0, annot.radar_prf, v_eff, annot.wavelength)
    #print(f'  γ_amb={gamma_amb:.5f}  f_sideband={f_sideband:.3f} Hz')

    # ── Step I: Deramp ───────────────────────────────────────────────────────
    #print('Step I: deramping …')
    if deramped_override is not None:
        # Pre-deramped burst supplied (e.g. GAMMA SLC_deramp_ScanSAR output):
        # skip the SAFE measurement TIFF read and our own deramp.
        deramped = deramped_override.astype(np.complex64)
    else:
        raw      = read_slc_burst(files['measurement'], annot, burst_idx)
        deramped = deramp_burst(raw, annot, burst_idx, deramp_method=deramp_method)
    valid_mask = apply_burst_valid_sample_mask(deramped, burst)

    # ── Step II: Correlation / Doppler estimation ────────────────────────────
    if estimator == 'fft':
        #print('Step II: estimating Doppler via FFT spectral centroid …')
        f_dc, snr, az_local, rg_centers = estimate_doppler_grid_fft(
            deramped, annot.prf, block_az, block_rg, stride_az, stride_rg,
            valid_mask=valid_mask,
        )
        p0 = np.ones_like(f_dc, dtype=np.float32)
        p1 = np.full_like(f_dc, np.nan, dtype=np.float32)
    else:
        #print('Step II: estimating p0, p1 …')
        p0, p1, az_local, rg_centers = estimate_correlation_grid(
            deramped, block_az, block_rg, stride_az, stride_rg,
            valid_mask=valid_mask,
        )
        f_dc, _, snr = correlation_to_doppler(
            p0, p1, annot.prf, annot.wavelength, gamma_amb=gamma_amb,
        )

    if add_data_poly:
        dc_est = _nearest_estimate(annot.dc_estimates, burst.azimuth_time)
        tau_at_rg = annot.slant_range_time_start + rg_centers / annot.range_sampling_rate
        data_poly_at_rg = np.array(
            sum(c * (tau_at_rg - dc_est.t0) ** k for k, c in enumerate(dc_est.data_poly)),
            dtype=np.float32,
        )
        f_dc = (f_dc + data_poly_at_rg[np.newaxis, :]).astype(np.float32)
    else:
        data_poly_at_rg = None

    # Full-scene azimuth indices for geolocation
    ati      = annot.azimuth_time_interval
    az0_full = int(round(
        (burst.azimuth_time - annot.first_line_time).total_seconds() / ati
    ))
    az_full = az_local + az0_full

    # ── Step III: Doppler centroid → DCA ────────────────────────────────────
    #print('Step III: geometry subtraction …')
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

    f_dca, f_dca_pre_descallop, burst_period_rows = _maybe_descallop_grid(
        f_dca.astype(np.float32),
        snr,
        annot,
        stride_az=stride_az,
        do_descallop=do_descallop,
    )
    v_r = (annot.wavelength / 2.0 * f_dca).astype(np.float32)
    #print(f'  v_r median: {float(np.nanmedian(v_r)):+.4f} m/s')

    # ── Step V: Geolocation ──────────────────────────────────────────────────
    #print('Step V: geolocation …')
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
    #print(f'  Heading: {heading:.1f}°   Look azimuth: {np.rad2deg(look_az):.1f}°')

    # ── Stokes drift ─────────────────────────────────────────────────────────
    #print('Loading ERA5 Stokes drift …')
    wave     = load_era5_wave(era5_wave, acq_time)
    v_stokes = compute_stokes_radial(wave, lat, lon, inc, look_az)
    #print(f'  v_stokes range: [{np.nanmin(v_stokes):.4f}, {np.nanmax(v_stokes):.4f}] m/s')

    # ── Wave Doppler bias ─────────────────────────────────────────────────────
    #print('Loading ERA5 wind …')
    wind   = load_era5_wind(era5_wind, acq_time)
    v_wave = compute_wave_doppler_bias(wind, lat, lon, inc, look_az)
    #print(f'  v_wave range: [{np.nanmin(v_wave):.4f}, {np.nanmax(v_wave):.4f}] m/s')

    # ── Current without mispointing ───────────────────────────────────────────
    v_current = (-v_r - v_stokes - v_wave).astype(np.float32)

    # ── OCN rvlDcMiss mispointing (+ optional rvlDcObs) ──────────────────────
    #print('Loading OCN …')
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
    #print(f'  v_miss_ocn range: [{np.nanmin(v_miss_ocn):.4f}, {np.nanmax(v_miss_ocn):.4f}] m/s')

    if use_ocn_dc:
        f_dc   = f_dc_ocn.astype(np.float32)
        f_dca  = (f_dc - f_geom_poe[np.newaxis, :] - float(f_sideband)).astype(np.float32)
        v_r    = (annot.wavelength / 2.0 * f_dca).astype(np.float32)
        v_current = (-v_r - v_stokes - v_wave).astype(np.float32)
        #print(f'  using rvlDcObs as f_dc; v_r median: {float(np.nanmedian(v_r)):+.4f} m/s')

    # ── Fully corrected current ───────────────────────────────────────────────
    v_current_ocn = (-v_r + v_miss_ocn - v_stokes - v_wave).astype(np.float32)

    # ── GLO12 comparison ──────────────────────────────────────────────────────
    #print('Loading GLO12 …')
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
            #print(f'  vs GLO12 ({label}):  bias={bias:+.4f}  RMSE={rmse:.4f}  r={r:.4f}')
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
        'f_dca_pre_descallop': f_dca_pre_descallop,
        'f_dc':           f_dc,
        'f_dc_ocn':       f_dc_ocn,
        'f_data_poly':    data_poly_at_rg,
        'tops_scale':     tops_scale,
        'tops_scaling':   tops_scaling,
        'deramp_method':  deramp_method,
        'do_descallop':   do_descallop,
        'burst_period_rows': burst_period_rows,
        'snr':            snr,
        'f_geom_ann':     f_geom_ann,
        'f_geom_poe':     f_geom_poe,
        'look_az_rad':    look_az,
        'heading_deg':    heading,
        'wavelength_m':   annot.wavelength,
        'sideband_bias_hz': float(f_sideband),
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
    deramp_method: str = 'esa_eq1',
    do_descallop: bool = True,
    tops_scaling: str = 'none',
    merge_first:  bool = False,
    use_ocn_dc:   bool = False,
    estimator:    str = 'fft',
    add_data_poly: bool = False,
    **kwargs,
) -> list[dict]:
    """
    Run run_pipeline for every burst in the subswath and return a list of results.

    Parameters
    ----------
    burst_indices : list of int or None
        Which bursts to process. If None, all bursts in the subswath are processed.
        Ignored when merge_first=True.
    deramp_method : 'current' or 'esa_eq1'
        SAFE-based deramp method used in Step I.
    do_descallop  : bool
        Apply burst-periodic descalloping when enough azimuth extent is available.
    tops_scaling  : 'none', 'multiply', or 'divide'
        Optional TOPS focused-data Doppler scale test forwarded to run_pipeline.
        Ignored when merge_first=True.
    use_ocn_dc    : bool
        When True, replace the lag-1 correlation f_dc with rvlDcObs interpolated
        from the OCN product onto the SAR estimation grid. Geometry subtraction
        and all downstream corrections still use our own pipeline values.
    add_data_poly : bool
        When True, add the annotation `data_poly` (the IPF's DC polynomial) on
        top of the lag-1 f_dc. Experimental; off by default — adds a small
        ~5 Hz tilt and a positive offset without bridging the lag-1 vs OCN
        amplitude gap.
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
            glo12=glo12, polarisation=polarisation, deramp_method=deramp_method,
            do_descallop=do_descallop, use_ocn_dc=use_ocn_dc,
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
            glo12=glo12, polarisation=polarisation, deramp_method=deramp_method,
            do_descallop=do_descallop, **kwargs,
            tops_scaling=tops_scaling, use_ocn_dc=use_ocn_dc, estimator=estimator,
            add_data_poly=add_data_poly,
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
    deramp_method: str = 'current',
    do_descallop: bool = True,
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
    I_c = merge_bursts(annot, files['measurement'], deramp_method=deramp_method)

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
    f_dca, f_dca_pre_descallop, burst_period_rows = _maybe_descallop_grid(
        f_dca,
        snr,
        annot,
        stride_az=stride_az,
        do_descallop=do_descallop,
    )
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
        f_dca, f_dca_pre_descallop, burst_period_rows = _maybe_descallop_grid(
            f_dca,
            snr,
            annot,
            stride_az=stride_az,
            do_descallop=do_descallop,
        )
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
        'f_dca_pre_descallop': f_dca_pre_descallop,
        'f_dc':           f_dc,
        'f_dc_ocn':       f_dc_ocn,
        'tops_scale':     np.ones(len(rg_centers), dtype=np.float32),
        'tops_scaling':   'none',
        'deramp_method':  deramp_method,
        'do_descallop':   do_descallop,
        'burst_period_rows': burst_period_rows,
        'snr':            snr,
        'f_geom_ann':     f_geom_ann,
        'f_geom_poe':     f_geom_poe,
        'look_az_rad':    look_az,
        'heading_deg':    heading,
        'wavelength_m':   annot.wavelength,
        'sideband_bias_hz': float(f_sideband),
        'subswath':       subswath,
        'burst_idx':      'merged',
        'acq_time':       acq_time,
        'rmse_vs_glo12':  rmse_vs_glo12,
        'bias_vs_glo12':  bias_vs_glo12,
        'r_vs_glo12':     r_vs_glo12,
        'p0':             p0,
        'p1':             np.angle(p1),
    }]



def run_gamma_dop2d_pipeline(
    dop2d_npz:      "str | dict",
    annotation_xml: str,
    subswath:       str,
    poeorb_path:    str | None,
    aux_cal_path:   str,
    ocn_safe:       str,
    era5_wind:      str,
    era5_wave:      str,
    glo12:          str,
    polarisation:   str = 'vv',
    use_ocn_dc:     bool = False,
    geom_source:    str = 'gamma',
    wave_source:    str = 'mouche',
    descallop_blocks: bool = False,
) -> dict:
    """
    Run the corrections pipeline starting from a pre-computed GAMMA dop2d .npz.

    The .npz must contain fd_measured, fd_model, fd_diff, range_m, az_time_s
    as produced by the gamma_iw1_deramp_merge_doppler.sh shell script.

    The sideband bias, geolocation, Stokes drift, wave Doppler, OCN mispointing
    and GLO12 comparison are applied exactly as in run_pipeline.

    Parameters
    ----------
    geom_source : 'gamma', 'poeorb', or 'annotation'
        How the geometry Doppler is removed to form the anomaly f_dca:

        * 'gamma'      — trust GAMMA's fd_diff (= fd_measured - fd_model, where
          fd_model is the deramped SLC's .par Doppler polynomial). For the
          mosaic-first npz fd_model is all-zeros, so fd_diff is the raw
          deramped Doppler with no geometry removed; for the mosaic-last npz
          the burst .par polynomial is removed, leaving a near-zero-mean
          residual. Neither equals the true geometry Doppler.
        * 'poeorb'     — ignore fd_diff; subtract our own POEORB-based geometry
          Doppler (``_geom_doppler_poeorb``, blended across bursts) from
          fd_measured. This is the physically calibrated subtraction and the
          one used by run_pipeline.
        * 'annotation' — subtract our blended annotation geometryDcPolynomial
          from fd_measured.

    Returns a dict compatible with merge_burst_grids / plot_comparison (same
    schema as run_pipeline).
    """
    _C = 299_792_458.0

    if isinstance(dop2d_npz, dict):
        print('gamma_dop2d: using in-memory Doppler result')
        d = dop2d_npz
    else:
        print(f'gamma_dop2d: loading pre-computed Doppler from {dop2d_npz} …')
        d = np.load(dop2d_npz)
    fd_measured = np.asarray(d['fd_measured'], dtype=np.float32)   # (n_blocks, n_rg)
    fd_model    = np.asarray(d['fd_model'],    dtype=np.float32)
    fd_diff     = np.asarray(d['fd_diff'],     dtype=np.float32)   # = measured - model
    range_m     = np.asarray(d['range_m'])                         # slant range [m]
    az_time_s   = np.asarray(d['az_time_s'])                       # seconds from mosaic start

    annot_original = parse_annotation(annotation_xml)
    annot = apply_poeorb(annot_original, poeorb_path) if poeorb_path is not None else annot_original

    aap = parse_aux_cal(aux_cal_path, subswath, polarisation)
    mid_burst = len(annot.bursts) // 2
    _, vel_mid = _interpolate_orbit(annot, annot.bursts[mid_burst].azimuth_time)
    v_eff = float(np.linalg.norm(vel_mid))
    gamma_amb = compute_gamma_ambiguity(aap, annot.radar_prf, v_eff, annot.wavelength)
    f_sideband = compute_sideband_bias(aap, 0.0, annot.radar_prf, v_eff, annot.wavelength)
    print(f'  γ_amb={gamma_amb:.5f}  f_sideband={f_sideband:.3f} Hz')

    # Convert physical coordinates → pixel indices for _geolocate_grid
    r0_annot = annot.slant_range_time_start * _C / 2.0
    rg_centers = np.round((range_m - r0_annot) / annot.range_pixel_spacing).astype(int)
    az_centers = np.round(az_time_s / annot.azimuth_time_interval).astype(int)

    # ── Geometry subtraction → Doppler anomaly f_dca ─────────────────────────
    # geom_source selects what is removed from the measured Doppler fd_measured.
    f_dc = fd_measured
    if geom_source == 'gamma':
        # Trust GAMMA's fd_diff (= fd_measured - fd_model).
        f_geom_poe = fd_model
        f_dca      = (fd_diff - float(f_sideband)).astype(np.float32)
    elif geom_source == 'annotation':
        # Our blended annotation geometryDcPolynomial.
        f_geom_poe = _blended_geom_doppler_annotation(
            annot_original, az_centers, rg_centers).astype(np.float32)
        f_dca      = (fd_measured - f_geom_poe - float(f_sideband)).astype(np.float32)
    elif geom_source == 'poeorb':
        # Our POEORB-based geometry Doppler, blended across bursts — the same
        # subtraction run_pipeline applies.
        burst_geom = np.stack([
            _geom_doppler_poeorb(annot, annot_original, j, rg_centers).astype(np.float64)
            for j in range(len(annot.bursts))
        ], axis=0)
        f_geom_poe = _blend_burst_profiles(annot, az_centers, burst_geom).astype(np.float32)
        f_dca      = (fd_measured - f_geom_poe - float(f_sideband)).astype(np.float32)
    else:
        raise ValueError("geom_source must be 'gamma', 'poeorb', or 'annotation'")

    f_dca_pre_descallop = f_dca.copy()
    if descallop_blocks and len(annot.bursts) >= 2:
        burst_dt_s = (annot.bursts[1].azimuth_time
                      - annot.bursts[0].azimuth_time).total_seconds()
        burst_period_rows = burst_dt_s / annot.azimuth_time_interval / float(d['blsz_lines'])
        # No SNR available from doppler_2d_SLC; use uniform weights so the
        # SNR-weighted mean degenerates to a plain row mean.
        snr_uniform = np.ones_like(f_dca, dtype=np.float32)
        f_dca = descallop(f_dca, snr_uniform, burst_period_rows)
        delta = float(np.nanmedian(np.abs(f_dca - f_dca_pre_descallop)))
        print(f'  descallop_blocks: burst_period_rows={burst_period_rows:.2f}  '
              f'median |Δf_dca|={delta:.3f} Hz')

    v_r = (annot.wavelength / 2.0 * f_dca).astype(np.float32)
    print(f'  geom_source={geom_source!r}  v_r median: {float(np.nanmedian(v_r)):+.4f} m/s')

    print('Step V: geolocation …')
    lat, lon, inc = _geolocate_grid(annot, az_centers, rg_centers)

    burst    = annot.bursts[mid_burst]
    acq_time = str(burst.azimuth_time)
    burst_t  = _iso_to_datetime(acq_time)
    pos_sv, vel_sv = _interpolate_orbit(annot, burst_t)
    lat_s = np.arcsin(pos_sv[2] / np.linalg.norm(pos_sv))
    lon_s = np.arctan2(pos_sv[1], pos_sv[0])
    e_east  = np.array([-np.sin(lon_s),  np.cos(lon_s), 0.0])
    e_north = np.array([
        -np.sin(lat_s) * np.cos(lon_s),
        -np.sin(lat_s) * np.sin(lon_s),
         np.cos(lat_s),
    ])
    heading = float(np.rad2deg(np.arctan2(np.dot(vel_sv, e_east), np.dot(vel_sv, e_north))))
    look_az = np.deg2rad(heading + 90.0)
    print(f'  Heading: {heading:.1f}°   Look azimuth: {np.rad2deg(look_az):.1f}°')

    print('Loading ERA5 Stokes drift …')
    wave     = load_era5_wave(era5_wave, acq_time)
    v_stokes = compute_stokes_radial(wave, lat, lon, inc, look_az)

    if wave_source == 'ocn':
        print('Loading OCN owiRadVel as wave bias …')
        owi    = load_ocn_wave_velocity(ocn_safe, subswath, polarisation)
        v_wave = compute_wave_doppler_bias_ocn(owi, lat, lon)
    elif wave_source == 'mouche':
        print('Loading ERA5 wind …')
        wind   = load_era5_wind(era5_wind, acq_time)
        v_wave = compute_wave_doppler_bias(wind, lat, lon, inc, look_az)
    else:
        raise ValueError(f"wave_source must be 'mouche' or 'ocn'; got {wave_source!r}")
    print(f'  wave_source={wave_source!r}  v_wave median: '
          f'{float(np.nanmedian(v_wave)):+.4f} m/s')

    v_current = (-v_r - v_stokes - v_wave).astype(np.float32)

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
        f_dca  = (f_dc - f_geom_poe - float(f_sideband)).astype(np.float32)
        v_r    = (annot.wavelength / 2.0 * f_dca).astype(np.float32)
        v_current = (-v_r - v_stokes - v_wave).astype(np.float32)
        print(f'  using rvlDcObs as f_dc; v_r median: {float(np.nanmedian(v_r)):+.4f} m/s')

    v_current_ocn = (-v_r + v_miss_ocn - v_stokes - v_wave).astype(np.float32)

    print('Loading GLO12 …')
    model   = load_glo12_current(glo12, acq_time)
    v_model = project_current_onto_look(model, lat, lon, inc, look_az)

    bias_vs_glo12, rmse_vs_glo12, r_vs_glo12 = _compute_stats(v_current_ocn, v_model)
    print(f'  vs GLO12 (gamma dop2d, OCN mispointing):  '
          f'bias={bias_vs_glo12:+.4f}  RMSE={rmse_vs_glo12:.4f}  r={r_vs_glo12:.4f}')

    nan_grid = np.full_like(f_dca, np.nan)
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
        'f_dca_pre_descallop': f_dca_pre_descallop,
        'f_dc':           f_dc,
        'f_dc_ocn':       f_dc_ocn,
        'f_geom_poe':     f_geom_poe,
        'geom_source':    geom_source,
        'wave_source':    wave_source,
        'descallop_blocks': bool(descallop_blocks),
        'tops_scale':     np.ones(len(rg_centers), dtype=np.float32),
        'tops_scaling':   'none',
        'snr':            nan_grid,
        'look_az_rad':    look_az,
        'heading_deg':    heading,
        'wavelength_m':   annot.wavelength,
        'sideband_bias_hz': float(f_sideband),
        'subswath':       subswath,
        'burst_idx':      'gamma_dop2d',
        'acq_time':       acq_time,
        'rmse_vs_glo12':  rmse_vs_glo12,
        'bias_vs_glo12':  bias_vs_glo12,
        'r_vs_glo12':     r_vs_glo12,
        'p0':             nan_grid,
        'p1':             nan_grid,
    }


def run_gamma_pipeline_from_safe(
    slc_safe:        str,
    subswath:        str,
    poeorb_path:     str | None,
    aux_cal_path:    str,
    ocn_safe:        str,
    era5_wind:       str,
    era5_wave:       str,
    glo12:           str,
    polarisation:    str = 'vv',
    blsz:            int = 256,
    add_demod_back:  "str | bool" = 'blend',
    geom_source:    str | None = None,
    wave_source:    str = 'mouche',
    descallop_blocks: bool = False,
    use_ocn_dc:     bool = False,
    mosaic_mode:    str = 'last',
    gamma_dir:      str | None = None,
    base_id:        str | None = None,
    keep_products:  bool = False,
    products_dir:   str | None = None,
) -> dict:
    """End-to-end GAMMA pipeline starting from a SAFE.

    Runs the GAMMA prep (``par_S1_SLC`` → ``S1_OPOD_vec`` → ``SLC_deramp_ScanSAR``)
    into a tempdir, then either:

    * ``mosaic_mode='last'``  — estimates the Doppler centroid per deramped
      burst (``doppler_2d_SLC``) and stitches in azimuth.  Burst-par geometry
      Doppler removal is available via ``geom_source='gamma'``.
    * ``mosaic_mode='first'`` — additionally runs ``SLC_mosaic_ScanSAR`` and
      estimates ``doppler_2d_SLC`` once on the deramped mosaic.  GAMMA's
      ``fd_model`` is all-zeros for that mode, so geometry must be supplied
      externally; the default ``geom_source`` becomes ``'poeorb'``.

    The in-memory Doppler result is then fed through
    :func:`run_gamma_dop2d_pipeline` for the geometry / Stokes / wave /
    OCN-mispointing / GLO12 corrections.  All GAMMA scratch files are deleted
    on return.

    If ``gamma_dir`` and ``base_id`` are given AND the relevant cached product
    exists (``deramp.slc`` for ``mosaic_mode='last'``, ``deramp.mosaic.slc``
    for ``'first'``), the deramping/mosaic step is skipped.  Set
    ``keep_products=True`` (with optional ``products_dir``) to persist the
    GAMMA outputs for reuse on subsequent runs.

    Returns the same dict schema as :func:`run_gamma_dop2d_pipeline`.
    """
    from .gamma_variants import (
        gamma_doppler_mosaic_first,
        gamma_doppler_mosaic_last,
        gamma_prep_scene,
    )

    if mosaic_mode not in ('last', 'first'):
        raise ValueError(f"mosaic_mode must be 'last' or 'first'; got {mosaic_mode!r}")

    # Per-mode sensible geometry-Doppler source.  GAMMA's fd_model is the
    # burst-par polynomial for mosaic_last (correct subtraction) and all-zeros
    # for mosaic_first (so 'gamma' would leave geometry in).  Allow override.
    if geom_source is None:
        geom_source = 'gamma' if mosaic_mode == 'last' else 'poeorb'

    files = find_safe_files(slc_safe, subswath, polarisation)
    annotation_xml = files['annotation']

    cache_marker = (
        f"{base_id}.deramp.slc" if mosaic_mode == 'last'
        else f"{base_id}.deramp.mosaic.slc"
    ) if base_id is not None else None
    existing = (
        gamma_dir is not None and cache_marker is not None
        and os.path.exists(os.path.join(gamma_dir, cache_marker))
    )

    def _doppler_and_corrections(gd: str, bid: str) -> dict:
        if mosaic_mode == 'last':
            dop2d = gamma_doppler_mosaic_last(
                blsz=blsz, gamma_dir=gd, base_id=bid,
                add_demod_back=add_demod_back, return_dict=True,
            )
        else:
            dop2d = gamma_doppler_mosaic_first(
                blsz=blsz, gamma_dir=gd, base_id=bid, return_dict=True,
            )
        return run_gamma_dop2d_pipeline(
            dop2d_npz=dop2d, annotation_xml=annotation_xml, subswath=subswath,
            poeorb_path=poeorb_path, aux_cal_path=aux_cal_path,
            ocn_safe=ocn_safe, era5_wind=era5_wind, era5_wave=era5_wave,
            glo12=glo12, polarisation=polarisation,
            use_ocn_dc=use_ocn_dc, geom_source=geom_source,
            wave_source=wave_source, descallop_blocks=descallop_blocks,
        )

    if existing:
        print(f"gamma_from_safe: reusing cached {mosaic_mode!r} products "
              f"{gamma_dir}/{base_id}.*")
        return _doppler_and_corrections(gamma_dir, base_id)

    prep_out = products_dir if keep_products else None
    with gamma_prep_scene(
        slc_safe=slc_safe, subswath=subswath, poeorb_path=poeorb_path,
        polarisation=polarisation, out_dir=prep_out, base_id=base_id,
        build_mosaic=(mosaic_mode == 'first'),
    ) as prep:
        return _doppler_and_corrections(prep["gamma_dir"], prep["base_id"])



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
    p.add_argument('--deramp-method', choices=('current', 'esa_eq1'), default='current',
                   help='SAFE-based deramp method for Step I')
    p.add_argument('--no-descallop', action='store_true',
                   help='Disable burst-periodic descalloping')
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
        deramp_method = args.deramp_method,
        do_descallop = not args.no_descallop,
        tops_scaling = args.tops_scaling,
    )
    if args.out:
        np.savez(args.out, **{k: v for k, v in result.items()
                              if isinstance(v, np.ndarray)})
        print(f'Saved arrays → {args.out}')
