"""
Geometry-induced Doppler centroid estimation from BIOMASS annotation XML.

The satellite's motion introduces a Doppler shift that must be subtracted
from the measured DC to isolate surface-motion contributions (DCA).
"""

import numpy as np
import xml.etree.ElementTree as ET
from .io import iso_to_unix


def get_dc_estimates(annot_xml: str) -> list[tuple]:
    """
    Parse geometry Doppler polynomial estimates from a BIOMASS annotation XML.

    Returns
    -------
    list of (az_unix, t0, coeffs) tuples:
        az_unix : float  — azimuth time of estimate [Unix seconds]
        t0      : float  — polynomial reference slant-range time [s]
        coeffs  : np.ndarray [c0..c4] — polynomial coefficients
    """
    dc_estimates = []
    root = ET.parse(annot_xml).getroot()

    for dc in root.findall(".//dopplerParameters//dcEstimateList//dcEstimate"):
        az_iso = dc.findtext("azimuthTime")
        t0 = dc.findtext("t0")
        poly = dc.findtext("geometryDCPolynomial")

        if az_iso is None or t0 is None or poly is None:
            continue

        dc_estimates.append((
            iso_to_unix(az_iso),
            float(t0),
            np.array([float(x) for x in poly.split()], dtype=np.float64),
        ))

    return dc_estimates


def estimate_geom_doppler(
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
    dc_estimates = get_dc_estimates(annot_xml)

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
