"""
Sentinel-1 IW SLC SAFE reader.

Parses annotation XML and reads complex SLC bursts from GeoTIFF measurement files.
Calibration and noise LUT parsing for sigma0 computation are also included.
"""

import os
import glob
import numpy as np
import rasterio
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from scipy.interpolate import RectBivariateSpline


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_safe_files(safe_dir: str, subswath: str, polarisation: str) -> dict:
    """
    Locate annotation, measurement, calibration, and noise XML/TIFF for one
    subswath + polarisation combination inside a SAFE directory.

    Parameters
    ----------
    safe_dir : str
        Path to the .SAFE directory.
    subswath : str
        One of 'iw1', 'iw2', 'iw3' (case-insensitive).
    polarisation : str
        'vv' or 'vh' (case-insensitive).

    Returns
    -------
    dict with keys 'annotation', 'measurement', 'calibration', 'noise'
    """
    sw = subswath.lower()
    pol = polarisation.lower()

    def _find(pattern):
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"No file matching {pattern}")
        return matches[0]

    return {
        'annotation': _find(os.path.join(safe_dir, 'annotation',
                                         f's1?-{sw}-slc-{pol}-*.xml')),
        'measurement': _find(os.path.join(safe_dir, 'measurement',
                                          f's1?-{sw}-slc-{pol}-*.tiff')),
        'calibration': _find(os.path.join(safe_dir, 'annotation', 'calibration',
                                          f'calibration-s1?-{sw}-slc-{pol}-*.xml')),
        'noise':       _find(os.path.join(safe_dir, 'annotation', 'calibration',
                                          f'noise-s1?-{sw}-slc-{pol}-*.xml')),
    }


# ---------------------------------------------------------------------------
# ISO time helper
# ---------------------------------------------------------------------------

def _iso_to_datetime(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _iso_to_unix(s: str) -> float:
    return _iso_to_datetime(s).timestamp()


# ---------------------------------------------------------------------------
# Annotation dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BurstInfo:
    idx: int
    azimuth_time: datetime
    byte_offset: int
    first_valid_sample: np.ndarray   # shape (linesPerBurst,), -1 = invalid
    last_valid_sample: np.ndarray    # shape (linesPerBurst,)
    sensing_time: datetime


@dataclass
class DcEstimate:
    azimuth_time: datetime
    t0: float                         # reference slant-range time [s]
    geometry_poly: np.ndarray         # orbit-predicted DC coefficients
    data_poly: np.ndarray             # measured DC coefficients
    rms_error: float


@dataclass
class AzimuthFmRate:
    azimuth_time: datetime
    t0: float
    poly: np.ndarray                  # FM-rate polynomial coefficients


@dataclass
class S1Annotation:
    polarisation: str
    subswath: str
    radar_frequency: float            # Hz
    wavelength: float                 # m
    azimuth_steering_rate: float      # deg/s
    range_sampling_rate: float        # Hz
    slant_range_time_start: float     # s  (near range)
    range_pixel_spacing: float        # m
    azimuth_pixel_spacing: float      # m
    azimuth_time_interval: float      # s
    prf: float                        # Hz (azimuth frequency)
    n_samples: int
    n_lines: int
    lines_per_burst: int
    samples_per_burst: int
    first_line_time: datetime
    bursts: list                      # list[BurstInfo]
    dc_estimates: list                # list[DcEstimate]
    azimuth_fm_rates: list            # list[AzimuthFmRate]
    geoloc_grid: dict                 # 'line','pixel','lat','lon','inc_angle' arrays


# ---------------------------------------------------------------------------
# Annotation parser
# ---------------------------------------------------------------------------

def parse_annotation(xml_path: str) -> S1Annotation:
    """Parse a Sentinel-1 IW SLC annotation XML into an S1Annotation object."""
    root = ET.parse(xml_path).getroot()

    pi = root.find('.//generalAnnotation/productInformation')
    ii = root.find('.//imageAnnotation/imageInformation')

    radar_freq = float(pi.findtext('radarFrequency'))
    wavelength = 3e8 / radar_freq
    az_steer_rate = float(pi.findtext('azimuthSteeringRate'))   # deg/s
    rsr = float(pi.findtext('rangeSamplingRate'))

    srt_start = float(ii.findtext('slantRangeTime'))
    rps = float(ii.findtext('rangePixelSpacing'))
    aps = float(ii.findtext('azimuthPixelSpacing'))
    ati = float(ii.findtext('azimuthTimeInterval'))
    prf = float(ii.findtext('azimuthFrequency'))
    n_samples = int(ii.findtext('numberOfSamples'))
    n_lines = int(ii.findtext('numberOfLines'))
    first_line_time = _iso_to_datetime(ii.findtext('productFirstLineUtcTime'))

    # Bursts
    st = root.find('.//swathTiming')
    lpb = int(st.findtext('linesPerBurst'))
    spb = int(st.findtext('samplesPerBurst'))

    bursts = []
    for idx, b in enumerate(root.findall('.//swathTiming/burstList/burst')):
        fvs_text = b.findtext('firstValidSample').split()
        lvs_text = b.findtext('lastValidSample').split()
        bursts.append(BurstInfo(
            idx=idx,
            azimuth_time=_iso_to_datetime(b.findtext('azimuthTime')),
            byte_offset=int(b.findtext('byteOffset')),
            first_valid_sample=np.array([int(x) for x in fvs_text], dtype=np.int32),
            last_valid_sample=np.array([int(x) for x in lvs_text], dtype=np.int32),
            sensing_time=_iso_to_datetime(b.findtext('sensingTime')),
        ))

    # Doppler centroid estimates
    dc_estimates = []
    for dc in root.findall('.//dopplerCentroid/dcEstimateList/dcEstimate'):
        geom_poly_txt = dc.findtext('geometryDcPolynomial')
        data_poly_txt = dc.findtext('dataDcPolynomial')
        if geom_poly_txt is None or data_poly_txt is None:
            continue
        dc_estimates.append(DcEstimate(
            azimuth_time=_iso_to_datetime(dc.findtext('azimuthTime')),
            t0=float(dc.findtext('t0')),
            geometry_poly=np.array([float(x) for x in geom_poly_txt.split()]),
            data_poly=np.array([float(x) for x in data_poly_txt.split()]),
            rms_error=float(dc.findtext('dataDcRmsError')),
        ))

    # Azimuth FM rates
    azimuth_fm_rates = []
    for afr in root.findall('.//generalAnnotation/azimuthFmRateList/azimuthFmRate'):
        poly_txt = afr.findtext('azimuthFmRatePolynomial')
        if poly_txt is None:
            continue
        azimuth_fm_rates.append(AzimuthFmRate(
            azimuth_time=_iso_to_datetime(afr.findtext('azimuthTime')),
            t0=float(afr.findtext('t0')),
            poly=np.array([float(x) for x in poly_txt.split()]),
        ))

    # Geolocation grid
    lines, pixels, lats, lons, incs = [], [], [], [], []
    for ggp in root.findall('.//geolocationGrid/geolocationGridPointList/geolocationGridPoint'):
        lines.append(int(ggp.findtext('line')))
        pixels.append(int(ggp.findtext('pixel')))
        lats.append(float(ggp.findtext('latitude')))
        lons.append(float(ggp.findtext('longitude')))
        incs.append(float(ggp.findtext('incidenceAngle')))

    unique_lines = sorted(set(lines))
    unique_pixels = sorted(set(pixels))
    n_gl = len(unique_lines)
    n_gp = len(unique_pixels)
    lat_arr = np.array(lats).reshape(n_gl, n_gp)
    lon_arr = np.array(lons).reshape(n_gl, n_gp)
    inc_arr = np.array(incs).reshape(n_gl, n_gp)

    geoloc_grid = {
        'line': np.array(unique_lines),
        'pixel': np.array(unique_pixels),
        'latitude': lat_arr,
        'longitude': lon_arr,
        'incidence_angle': inc_arr,
    }

    pol = root.findtext('.//adsHeader/polarisation').upper()
    sw = root.findtext('.//adsHeader/swath').upper()

    return S1Annotation(
        polarisation=pol,
        subswath=sw,
        radar_frequency=radar_freq,
        wavelength=wavelength,
        azimuth_steering_rate=az_steer_rate,
        range_sampling_rate=rsr,
        slant_range_time_start=srt_start,
        range_pixel_spacing=rps,
        azimuth_pixel_spacing=aps,
        azimuth_time_interval=ati,
        prf=prf,
        n_samples=n_samples,
        n_lines=n_lines,
        lines_per_burst=lpb,
        samples_per_burst=spb,
        first_line_time=first_line_time,
        bursts=bursts,
        dc_estimates=dc_estimates,
        azimuth_fm_rates=azimuth_fm_rates,
        geoloc_grid=geoloc_grid,
    )


# ---------------------------------------------------------------------------
# SLC reader
# ---------------------------------------------------------------------------

def read_slc_full(tiff_path: str) -> np.ndarray:
    """
    Read an entire Sentinel-1 SLC GeoTIFF as a complex64 array.

    S1 SLCs are stored as CInt16 (two int16 values per pixel).

    Returns
    -------
    np.ndarray, shape (n_lines, n_samples), dtype complex64
    """
    with rasterio.open(tiff_path) as src:
        data = src.read(1)   # rasterio returns complex dtype for CInt16
    return data.astype(np.complex64)


def read_slc_burst(tiff_path: str, annot: S1Annotation, burst_idx: int) -> np.ndarray:
    """
    Read a single burst from a Sentinel-1 SLC GeoTIFF.

    IW SLC stores all bursts sequentially in azimuth; burst i occupies
    lines [i*lpb : (i+1)*lpb].

    Returns
    -------
    np.ndarray, shape (linesPerBurst, samplesPerBurst), dtype complex64
    """
    lpb = annot.lines_per_burst
    with rasterio.open(tiff_path) as src:
        window = rasterio.windows.Window(
            col_off=0, row_off=burst_idx * lpb,
            width=annot.samples_per_burst, height=lpb,
        )
        data = src.read(1, window=window)
    return data.astype(np.complex64)


# ---------------------------------------------------------------------------
# Calibration LUT parser
# ---------------------------------------------------------------------------

def parse_calibration_lut(cal_xml: str, lut_name: str = 'sigmaNought') -> dict:
    """
    Parse a Sentinel-1 calibration XML into a range-varying LUT.

    Parameters
    ----------
    cal_xml : str
        Path to calibration-*.xml.
    lut_name : str
        One of 'sigmaNought', 'betaNought', 'gamma', 'dn'.

    Returns
    -------
    dict with 'line', 'pixel', 'values' arrays (for interpolation).
    """
    root = ET.parse(cal_xml).getroot()
    lines, pixels, values = [], [], []
    for cv in root.findall('.//calibrationVectorList/calibrationVector'):
        line = int(cv.findtext('line'))
        pix = np.array([int(x) for x in cv.findtext('pixel').split()], dtype=np.float32)
        vals = np.array([float(x) for x in cv.findtext(lut_name).split()], dtype=np.float32)
        lines.append(line)
        pixels.append(pix)
        values.append(vals)

    return {
        'line': np.array(lines, dtype=np.float32),
        'pixel': pixels,       # list of arrays (constant spacing assumed)
        'values': values,
    }


def interpolate_lut(lut: dict, n_lines: int, n_samples: int) -> np.ndarray:
    """
    Bilinearly interpolate a calibration LUT onto the full pixel grid.

    Parameters
    ----------
    lut : dict
        Output of parse_calibration_lut.
    n_lines, n_samples : int
        Target grid dimensions.

    Returns
    -------
    np.ndarray, shape (n_lines, n_samples), dtype float32
    """
    cal_lines = lut['line']                    # shape (n_cvec,)
    pix_ref = lut['pixel'][0].astype(float)    # assume uniform pixel positions
    values = np.vstack([v.astype(float) for v in lut['values']])  # (n_cvec, n_pix)

    spline = RectBivariateSpline(cal_lines.astype(float), pix_ref, values, kx=1, ky=1)
    line_idx = np.arange(n_lines, dtype=float)
    pix_idx = np.arange(n_samples, dtype=float)
    return spline(line_idx, pix_idx).astype(np.float32)


# ---------------------------------------------------------------------------
# Geolocation helpers
# ---------------------------------------------------------------------------

def interpolate_geoloc(annot: S1Annotation, field: str) -> np.ndarray:
    """
    Bilinearly interpolate a geolocation grid field ('latitude', 'longitude',
    or 'incidence_angle') onto the full pixel grid (n_lines, n_samples).
    """
    g = annot.geoloc_grid
    data = g[field].astype(float)
    spline = RectBivariateSpline(
        g['line'].astype(float),
        g['pixel'].astype(float),
        data,
        kx=1, ky=1,
    )
    return spline(
        np.arange(annot.n_lines, dtype=float),
        np.arange(annot.n_samples, dtype=float),
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# DC polynomial evaluation
# ---------------------------------------------------------------------------

def eval_dc_poly(poly: np.ndarray, t0: float, tau: np.ndarray) -> np.ndarray:
    """
    Evaluate a Doppler centroid polynomial:
        f_dc(tau) = sum_k  poly[k] * (tau - t0)^k

    Parameters
    ----------
    poly : np.ndarray
        Polynomial coefficients [c0, c1, c2, ...].
    t0 : float
        Reference slant-range time [s].
    tau : np.ndarray
        Slant-range time vector [s], shape (n_samples,).

    Returns
    -------
    np.ndarray, shape (n_samples,)
    """
    dt = tau - t0
    return sum(c * dt**k for k, c in enumerate(poly))


def slant_range_time_vector(annot: S1Annotation) -> np.ndarray:
    """Return slant-range time for each range sample [s]."""
    dt_r = 1.0 / annot.range_sampling_rate
    return annot.slant_range_time_start + np.arange(annot.n_samples) * dt_r


def _nearest_estimate(estimates, burst_time: datetime):
    """Return the estimate whose azimuth_time is closest to burst_time."""
    diffs = [abs((e.azimuth_time - burst_time).total_seconds()) for e in estimates]
    return estimates[int(np.argmin(diffs))]
