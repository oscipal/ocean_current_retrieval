"""
I/O utilities for reading GAMMA RS SLC files and BIOMASS annotation XML.
"""

import numpy as np
from datetime import datetime, timezone


def iso_to_unix(iso_str: str) -> float:
    """Convert ISO8601 string (no timezone) to Unix timestamp (UTC)."""
    dt = datetime.fromisoformat(iso_str).replace(tzinfo=timezone.utc)
    return dt.timestamp()


def parse_slc_par(par_file: str) -> dict:
    """Parse a GAMMA RS .slc.par file into a dictionary."""
    params = {}
    with open(par_file, 'r') as f:
        for line in f:
            if ':' in line:
                key, _, val = line.partition(':')
                params[key.strip()] = val.strip().split()
    return params


def read_slc(slc_file: str, slc_par_file: str) -> np.ndarray:
    """
    Read a GAMMA RS big-endian float32 complex SLC file.

    Returns
    -------
    np.ndarray, shape (azimuth_lines, range_samples), dtype complex64
    """
    par = parse_slc_par(slc_par_file)
    range_samples = int(par['range_samples'][0])
    azimuth_lines = int(par['azimuth_lines'][0])

    raw = np.fromfile(slc_file, dtype=np.dtype('>f4'))
    complex_data = raw[0::2] + 1j * raw[1::2]
    return complex_data.reshape(azimuth_lines, range_samples)


def eval_poly_horner(coeffs: np.ndarray, t: float, t0: float) -> float:
    """
    Evaluate polynomial sum_k coeffs[k] * (t - t0)^k using Horner's method.
    coeffs = [c0, c1, c2, ...] (ascending order).
    """
    dt = t - t0
    y = 0.0
    for c in reversed(coeffs):
        y = y * dt + c
    return float(y)
