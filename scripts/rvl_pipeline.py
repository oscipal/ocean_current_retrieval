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

from scripts.s1_io import find_safe_files, parse_annotation, read_slc_burst, _iso_to_datetime
from scripts.s1_aux import parse_aux_cal, apply_poeorb
from scripts.s1_rvl import (
    deramp_burst,
    estimate_correlation_grid,
    correlation_to_doppler,
    compute_gamma_ambiguity,
    compute_sideband_bias,
    _geom_doppler_annotation,
    _geom_doppler_poeorb,
    _geolocate_grid,
    _interpolate_orbit,
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
    valid_lines = burst.first_valid_sample != -1
    deramped[~valid_lines, :] = 0.0

    # ── Step II: Correlation coefficients ───────────────────────────────────
    print('Step II: estimating p0, p1 …')
    p0, p1, az_local, rg_centers = estimate_correlation_grid(
        deramped, block_az, block_rg, stride_az, stride_rg,
    )

    # Full-scene azimuth indices for geolocation
    ati      = annot.azimuth_time_interval
    az0_full = int(round(
        (burst.azimuth_time - annot.first_line_time).total_seconds() / ati
    ))
    az_full = az_local + az0_full

    # ── Step III: Doppler centroid → DCA ────────────────────────────────────
    print('Step III: Doppler centroid and geometry subtraction …')
    f_dc, _, snr = correlation_to_doppler(
        p0, p1, annot.prf, annot.wavelength, gamma_amb=gamma_amb,
    )

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

    # ── OCN rvlDcMiss mispointing ─────────────────────────────────────────────
    print('Loading OCN rvlDcMiss …')
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

    f_miss_ocn = match_to_sar_grid(lat, lon, ocn_lat, ocn_lon, ocn_dc_miss)
    v_miss_ocn = (annot.wavelength / 2.0 * f_miss_ocn).astype(np.float32)
    print(f'  v_miss_ocn range: [{np.nanmin(v_miss_ocn):.4f}, {np.nanmax(v_miss_ocn):.4f}] m/s')

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
    **kwargs,
) -> list[dict]:
    """
    Run run_pipeline for every burst in the subswath and return a list of results.

    Parameters
    ----------
    burst_indices : list of int or None
        Which bursts to process. If None, all bursts in the subswath are processed.
    **kwargs      : forwarded to run_pipeline (block_az, block_rg, stride_az, stride_rg)
    """
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
        )
        results.append(r)

    return results


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
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    print('Merging SAR …')
    grid_lat, grid_lon, merged_sar = merge_burst_grids(
        results, overlap=overlap, resolution_deg=resolution_deg,
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
    )
    if args.out:
        np.savez(args.out, **{k: v for k, v in result.items()
                              if isinstance(v, np.ndarray)})
        print(f'Saved arrays → {args.out}')
