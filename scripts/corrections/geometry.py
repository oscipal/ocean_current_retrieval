"""
Geometry-induced Doppler centroid estimation.

Two sources are supported:
  - BIOMASS / Sentinel-1 annotation XML  (get_dc_estimates, estimate_geom_doppler)
  - GAMMA .slc.par doppler polynomials   (geom_dc_map) — preferred for per-burst work
"""

import numpy as np
import xml.etree.ElementTree as ET
from ..io import iso_to_unix, parse_slc_par


def get_dc_estimates(annot_xml: str, sat_id: str) -> list[tuple]:
    """
    Parse geometry Doppler polynomial estimates from an annotation XML.

    Parameters
    ----------
    annot_xml : str
        Path to annotation XML.
    sat_id : str | s1 or bio
        Which satellites annotation file is passed, Sentinel-1 (s1) or Biomass (bio)

    Returns
    -------
    list of (az_unix, t0, coeffs) tuples:
        az_unix : float  — azimuth time of estimate [Unix seconds]
        t0      : float  — polynomial reference slant-range time [s]
        coeffs  : np.ndarray [c0..c4] — polynomial coefficients
    """
    dc_estimates = []
    root = ET.parse(annot_xml).getroot()

    if sat_id == 's1':
        field_name = ".//dopplerCentroid//dcEstimateList//dcEstimate"
        poly_field = 'geometryDcPolynomial'
    elif sat_id == 'bio':
        field_name = ".//dopplerParameters//dcEstimateList//dcEstimate"
        poly_field = "geometryDCPolynomial"
    else:
        raise ValueError('Only "s1" or "bio" accepted as sat_id parameters')

    for dc in root.findall(field_name):
        az_iso = dc.findtext("azimuthTime")
        t0 = dc.findtext("t0")
        poly = dc.findtext(poly_field)

        if az_iso is None or t0 is None or poly is None:
            continue

        dc_estimates.append((
            iso_to_unix(az_iso),
            float(t0),
            np.array([float(x) for x in poly.split()], dtype=np.float64),
        ))

    return dc_estimates


def estimate_geom_doppler_bio(
    annot_xml: str,
    doppler_img: np.ndarray = None,
) -> np.ndarray:
    """
    Compute the geometry-induced Doppler centroid from annotation polynomials.

    Parameters
    ----------
    annot_xml : str
        Path to BIOMASS annotation XML.
    doppler_img : np.ndarray or None
        If provided, the geometry DC is coregistered (resampled) to the same
        pixel grid as this DC image and cropped to its shape.
        If None, returns the raw (n_estimates × n_samples) array.

    Returns
    -------
    np.ndarray
        If doppler_img is None : shape (n_estimates, n_range_samples)
        If doppler_img given   : shape == doppler_img.shape, coregistered
    """
    root = ET.parse(annot_xml).getroot()
    dc_estimates = get_dc_estimates(annot_xml, 'bio')

    t_r0 = float(root.findtext(".//firstSampleSlantRangeTime"))
    dt_r = float(root.findtext(".//rangeTimeInterval"))
    n_samples = int(root.findtext(".//numberOfSamples"))
    n_lines = int(root.findtext(".//numberOfLines"))
    t_a0 = iso_to_unix(root.findtext(".//firstLineAzimuthTime"))
    dt_a = float(root.findtext(".//azimuthTimeInterval"))

    tau = t_r0 + np.arange(n_samples) * dt_r

    # Evaluate polynomials over range for each DC estimate
    geom_doppler = np.zeros((len(dc_estimates), n_samples))
    for i, (_, t0, coeffs) in enumerate(dc_estimates):
        dt = tau - t0
        geom_doppler[i] = sum(c * dt**k for k, c in enumerate(coeffs))

    if doppler_img is None:
        return geom_doppler

    # Coregister to the pixel grid of doppler_img
    az_times = np.array([e[0] for e in dc_estimates])
    az_interval = az_times[1] - az_times[0]

    geom_coregistered = np.full((n_lines, n_samples), np.nan)
    start_az_idx = 0
    active = False

    for i, (az_0, _, _) in enumerate(dc_estimates):
        if not active and np.abs(az_0 - t_a0) <= az_interval:
            active = True

        if active:
            az_idx = int((az_interval - (t_a0 - az_0)) // dt_a)
            geom_coregistered[start_az_idx:az_idx + 1, :] = geom_doppler[i]
            start_az_idx = az_idx + 1

    return geom_coregistered[:doppler_img.shape[0], :doppler_img.shape[1]]


def estimate_geom_doppler_s1_burst(
    slc_par_path: str,
    win_az: int,
    win_rg: int,
    stride_az: int,
    stride_rg: int,
) -> np.ndarray:
    """
    Compute the geometry DC map for one burst from GAMMA .slc.par polynomials,
    tiled to match the output shape of fft_doppler / cde_doppler.

    GAMMA burst extraction (S1_BURST_tab) deramped the TOPS azimuth steering,
    so the time-varying poly_dot / poly_ddot terms are already compensated.
    Only the static doppler_polynomial (function of slant range) is applied.

    Parameters
    ----------
    slc_par_path : str
        Path to a single-burst GAMMA .slc.par file.
    win_az, win_rg : int
        Window sizes used in the Doppler estimator.
    stride_az, stride_rg : int
        Strides used in the Doppler estimator.

    Returns
    -------
    np.ndarray, shape (n_az*stride_az, n_rg*stride_rg)
        Geometry DC [Hz] tiled to pixel resolution.
    """
    par = parse_slc_par(slc_par_path)

    n_az_lines  = int(par['azimuth_lines'][0])
    n_rg_pixels = int(par['range_samples'][0])
    dr = float(par['range_pixel_spacing'][0])   # range pixel spacing [m]

    coeffs = np.array(par['doppler_polynomial'][:4], dtype=float)

    def eval_poly(c, x):
        return sum(ci * x**k for k, ci in enumerate(c))

    n_az = (n_az_lines  - win_az) // stride_az + 1
    n_rg = (n_rg_pixels - win_rg) // stride_rg + 1

    # window centres as range offset from near range [m] — GAMMA polynomial variable
    rg_centers = (np.arange(n_rg) * stride_rg + win_rg // 2) * dr
    f_geom_rg = eval_poly(coeffs, rg_centers)   # shape (n_rg,), constant in az

    geom_map = np.zeros((n_az * stride_az, n_rg * stride_rg))
    for j in range(n_az):
        for i in range(n_rg):
            geom_map[j*stride_az:(j+1)*stride_az,
                     i*stride_rg:(i+1)*stride_rg] = f_geom_rg[i]

    return geom_map
