def run_gamma_merged_pipeline(
    slc_paths:     list[str] | str,
    slc_par_paths: list[str] | str,
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
    Run the merged RVL pipeline starting from GAMMA SLC data.

    Accepts either a list of per-burst SLC paths (merged with Hanning
    windows via merge_gamma_bursts) or a single pre-mosaicked SLC path
    produced by SLC_mosaic_ScanSAR (read directly with read_slc).

    This mirrors `_run_merged_pipeline()`, except Step I uses pre-deramped
    GAMMA SLC data instead of reading and deramping the SAFE measurement
    TIFF internally.
    """
    _is_mosaic = isinstance(slc_paths, str)
    _n = 'mosaic' if _is_mosaic else f'{len(slc_paths)} bursts'
    print(f'gamma_merged: loading deramped {subswath.upper()} {_n} …')
    annot_original = parse_annotation(annotation_xml)
    annot = apply_poeorb(annot_original, poeorb_path) if poeorb_path is not None else annot_original

    aap = parse_aux_cal(aux_cal_path, subswath, polarisation)
    mid_burst = len(annot.bursts) // 2
    _, vel_mid = _interpolate_orbit(annot, annot.bursts[mid_burst].azimuth_time)
    v_eff = float(np.linalg.norm(vel_mid))
    gamma_amb = compute_gamma_ambiguity(aap, annot.radar_prf, v_eff, annot.wavelength)
    f_sideband = compute_sideband_bias(aap, 0.0, annot.radar_prf, v_eff, annot.wavelength)
    print(f'  γ_amb={gamma_amb:.5f}  f_sideband={f_sideband:.3f} Hz')

    print('Step I: loading pre-deramped GAMMA SLC …')
    if _is_mosaic:
        from ..gamma_io import read_slc as _read_slc
        I_c = _read_slc(slc_paths, slc_par_paths).astype(np.complex64)
    else:
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
        # Fall back to Doppler polynomial(s) stored in the GAMMA .par file(s).
        if _is_mosaic:
            f_geom_poe = _geom_doppler_at_pixels(slc_par_paths, rg_centers).astype(np.float32)
        else:
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
        'burst_idx':      'gamma_mosaic' if _is_mosaic else 'gamma_merged',
        'acq_time':       acq_time,
        'rmse_vs_glo12':  rmse_vs_glo12,
        'bias_vs_glo12':  bias_vs_glo12,
        'r_vs_glo12':     r_vs_glo12,
        'p0':             p0,
        'p1':             np.angle(p1),
    }


