"""
Sentinel-1 auxiliary file parsers.

Provides
--------
AzimuthAntennaPattern   dataclass — AAP from AUX_CAL, with interpolation helpers
parse_aux_cal           read AAP for one swath/polarisation from an AUX_CAL SAFE
parse_poeorb            read orbit state vectors from a POEORB/RESORB .EOF file
apply_poeorb            replace annotation orbit SVs with the precise POEORB data
"""

from __future__ import annotations

import glob
import os
import xml.etree.ElementTree as ET
from copy import copy
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Azimuth Antenna Pattern
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AzimuthAntennaPattern:
    """
    One-way azimuth antenna pattern from AUX_CAL.

    Attributes
    ----------
    swath        : str   e.g. 'IW1'
    polarisation : str   e.g. 'VV'
    angle_step_deg : float  angular increment between samples [deg] (0.005°)
    angles_deg   : (N,) array  azimuth angles, centred at 0 [deg]
    gain_db      : (N,) array  one-way gain values [dB], 0 dB at peak
    """
    swath:           str
    polarisation:    str
    angle_step_deg:  float
    angles_deg:      np.ndarray
    gain_db:         np.ndarray

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    def gain_linear(self) -> np.ndarray:
        """One-way linear power gain (peak = 1)."""
        return 10.0 ** (self.gain_db / 10.0)

    def two_way_gain_linear(self) -> np.ndarray:
        """Two-way (round-trip) linear power gain (peak = 1)."""
        return self.gain_linear() ** 2

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def at_angle_deg(self, query_deg: np.ndarray) -> np.ndarray:
        """
        Interpolate one-way linear gain at arbitrary azimuth angles [deg].
        Angles outside ±1° return 0 (far sidelobes, negligible contribution).
        """
        return np.interp(
            query_deg, self.angles_deg, self.gain_linear(),
            left=0.0, right=0.0,
        )

    def two_way_at_angle_deg(self, query_deg: np.ndarray) -> np.ndarray:
        """Interpolate two-way linear gain at azimuth angles [deg]."""
        g = self.at_angle_deg(query_deg)
        return g ** 2

    def at_doppler_hz(
        self,
        f_hz: np.ndarray,
        v_eff: float,
        wavelength: float,
    ) -> np.ndarray:
        """
        Interpolate one-way linear gain at Doppler frequencies [Hz].

        Conversion: f = (2 v_eff / λ) · sin(ψ)  →  ψ = arcsin(f λ / (2 v_eff))

        Parameters
        ----------
        f_hz      : Doppler frequencies to query [Hz]
        v_eff     : effective (orbital) satellite speed [m/s]
        wavelength: radar wavelength [m]
        """
        sin_arg = np.asarray(f_hz, dtype=np.float64) * wavelength / (2.0 * v_eff)
        sin_arg = np.clip(sin_arg, -1.0, 1.0)
        psi_deg = np.rad2deg(np.arcsin(sin_arg))
        return self.at_angle_deg(psi_deg)

    def two_way_at_doppler_hz(
        self,
        f_hz: np.ndarray,
        v_eff: float,
        wavelength: float,
    ) -> np.ndarray:
        """Two-way linear gain at Doppler frequencies [Hz]."""
        return self.at_doppler_hz(f_hz, v_eff, wavelength) ** 2


# ─────────────────────────────────────────────────────────────────────────────
# AUX_CAL parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_aux_cal(
    aux_cal_safe: str,
    swath: str,
    polarisation: str,
) -> AzimuthAntennaPattern:
    """
    Parse the azimuth antenna pattern for one swath / polarisation from AUX_CAL.

    Parameters
    ----------
    aux_cal_safe : str   Path to the AUX_CAL .SAFE directory.
    swath        : str   e.g. 'IW1', 'IW2', 'IW3' (case-insensitive).
    polarisation : str   e.g. 'VV', 'VH' (case-insensitive).

    Returns
    -------
    AzimuthAntennaPattern

    Raises
    ------
    FileNotFoundError  if the AUX_CAL XML cannot be located.
    ValueError         if no matching swath/pol entry is found.
    """
    xml_files = glob.glob(os.path.join(aux_cal_safe, 'data', '*.xml'))
    if not xml_files:
        raise FileNotFoundError(f'No XML found in {aux_cal_safe}/data/')
    xml_path = xml_files[0]

    root = ET.parse(xml_path).getroot()
    sw  = swath.upper()
    pol = polarisation.upper()

    for cp in root.findall('.//calibrationParams'):
        if (cp.findtext('swath').upper() == sw
                and cp.findtext('polarisation').upper() == pol):
            aap  = cp.find('azimuthAntennaPattern')
            step = float(aap.findtext('azimuthAngleIncrement'))
            vals = np.array(
                [float(x) for x in aap.findtext('values').split()],
                dtype=np.float64,
            )
            n      = len(vals)
            angles = (np.arange(n) - n // 2) * step   # centred at 0°
            return AzimuthAntennaPattern(
                swath=sw,
                polarisation=pol,
                angle_step_deg=step,
                angles_deg=angles,
                gain_db=vals,
            )

    raise ValueError(
        f'AUX_CAL has no entry for swath={sw} polarisation={pol} in {xml_path}'
    )


# ─────────────────────────────────────────────────────────────────────────────
# POEORB / RESORB parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_poeorb(eof_path: str) -> list[tuple[datetime, list, list]]:
    """
    Parse a Sentinel-1 POEORB or RESORB .EOF file.

    Parameters
    ----------
    eof_path : str   Path to the .EOF file.

    Returns
    -------
    list of (utc_datetime, [x, y, z] m, [vx, vy, vz] m/s)
        State vectors in ECEF, ordered by time.
    """
    root = ET.parse(eof_path).getroot()
    records = []

    for osv in root.findall('.//OSV'):
        utc_raw = osv.findtext('UTC')
        # Format: 'UTC=2026-02-05T22:59:42.000000'
        utc_str = utc_raw.split('=', 1)[-1]
        t = datetime.strptime(utc_str, '%Y-%m-%dT%H:%M:%S.%f').replace(
            tzinfo=timezone.utc
        )
        x  = float(osv.findtext('X'))
        y  = float(osv.findtext('Y'))
        z  = float(osv.findtext('Z'))
        vx = float(osv.findtext('VX'))
        vy = float(osv.findtext('VY'))
        vz = float(osv.findtext('VZ'))
        records.append((t, [x, y, z], [vx, vy, vz]))

    return records


def apply_poeorb(annot, eof_path: str, margin_s: float = 60.0):
    """
    Replace the orbit state vectors in an S1Annotation with POEORB/RESORB data.

    Only state vectors within the scene time window (± margin_s) are kept so
    the annotation stays compact.  At least two state vectors must remain after
    filtering (raises ValueError otherwise — check that the POEORB file covers
    your acquisition).

    Parameters
    ----------
    annot     : S1Annotation  (from s1_io.parse_annotation)
    eof_path  : str   Path to the POEORB / RESORB .EOF file.
    margin_s  : float  Keep SVs this many seconds outside the scene window.

    Returns
    -------
    A shallow copy of *annot* with orbit_times / orbit_positions /
    orbit_velocities replaced by the POEORB data.
    """
    records = parse_poeorb(eof_path)

    t_start = annot.first_line_time
    t_end   = annot.bursts[-1].azimuth_time

    filtered = [
        (t, pos, vel) for t, pos, vel in records
        if (t - t_start).total_seconds() >= -margin_s
        and (t - t_end).total_seconds()   <=  margin_s
    ]

    if len(filtered) < 2:
        import warnings
        warnings.warn(
            f'POEORB {eof_path} covers only {len(filtered)} state vector(s) '
            f'in the scene window [{t_start}, {t_end}] ± {margin_s} s. '
            f'POEORB SVs span [{records[0][0]}, {records[-1][0]}]. '
            'Annotation orbit SVs are kept unchanged. '
            'Download the POEORB file whose validity window covers your acquisition.',
            UserWarning, stacklevel=2,
        )
        return annot

    new_annot = copy(annot)
    new_annot.orbit_times      = [r[0] for r in filtered]
    new_annot.orbit_positions  = [r[1] for r in filtered]
    new_annot.orbit_velocities = [r[2] for r in filtered]
    return new_annot
