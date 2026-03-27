"""
Ionospheric Doppler centroid correction for P-band SAR.

Reads IONEX v1.0 VTEC maps and computes the ionosphere-induced Doppler
centroid shift using the single-layer model (SLM).

Physics:
  Two-way ionospheric phase (SAR):
      phi_iono = -4*pi*K*STEC / (f_c * c)        [rad]
  Doppler shift (time derivative):
      f_iono   = -2*K*M / (f_c * c) * dVTEC/dt   [Hz]
  where dVTEC/dt = (azimuth gradient of VTEC) * v_IPP_az
"""

import numpy as np
from datetime import datetime, timezone
from ..io import parse_slc_par

K_IONO  = 40.3           # ionospheric constant  [m³/s²]
C_LIGHT = 299_792_458.0  # speed of light        [m/s]
TECU    = 1e16           # 1 TECU in electrons/m²


# ---------------------------------------------------------------------------
# IONEX parser
# ---------------------------------------------------------------------------

def parse_ionex(filepath: str) -> dict:
    """
    Parse an IONEX v1.0 file and return VTEC maps.

    Parameters
    ----------
    filepath : str
        Path to the .INX / .ION file.

    Returns
    -------
    dict with keys
        'epochs' : np.ndarray (N_epochs,)             Unix seconds UTC
        'lat'    : np.ndarray (N_lat,)                degrees (descending)
        'lon'    : np.ndarray (N_lon,)                degrees
        'vtec'   : np.ndarray (N_epochs, N_lat, N_lon) TECU
        'h_ion'  : float   single-layer height [km]
        'R_E'    : float   base radius [km]
    """
    exponent = -1
    h_ion = 450.0
    R_E   = 6371.0
    lat1 = lat2 = dlat = None
    lon1 = lon2 = dlon = None

    epochs    = []
    vtec_maps = []

    with open(filepath, 'r') as fh:
        lines = fh.readlines()

    def label(ln):
        return ln[60:].strip() if len(ln) > 60 else ''

    i = 0
    while i < len(lines):
        ln  = lines[i]
        lbl = label(ln)

        if 'EXPONENT' in lbl:
            exponent = int(ln[:6])

        elif 'BASE RADIUS' in lbl:
            R_E = float(ln[:8])

        elif 'HGT1 / HGT2 / DHGT' in lbl:
            h_ion = float(ln.split()[0])

        elif 'LAT1 / LAT2 / DLAT' in lbl:
            parts = ln[:60].split()
            lat1, lat2, dlat = float(parts[0]), float(parts[1]), float(parts[2])

        elif 'LON1 / LON2 / DLON' in lbl:
            parts = ln[:60].split()
            lon1, lon2, dlon = float(parts[0]), float(parts[1]), float(parts[2])

        elif 'END OF HEADER' in lbl:
            n_lat = round(abs(lat2 - lat1) / abs(dlat)) + 1
            n_lon = round(abs(lon2 - lon1) / abs(dlon)) + 1
            lat_arr = np.linspace(lat1, lat2, n_lat)
            lon_arr = np.linspace(lon1, lon2, n_lon)

        elif 'START OF TEC MAP' in lbl:
            i += 1
            # next line: EPOCH OF CURRENT MAP
            ep = lines[i][:60].split()
            yr, mo, dy = int(ep[0]), int(ep[1]), int(ep[2])
            hr, mn, sc = int(ep[3]), int(ep[4]), int(float(ep[5]))
            epoch_unix = datetime(yr, mo, dy, hr, mn, sc,
                                  tzinfo=timezone.utc).timestamp()
            epochs.append(epoch_unix)

            tec_map = np.full((len(lat_arr), len(lon_arr)), np.nan)
            i += 1

            while i < len(lines):
                lbl_inner = label(lines[i])
                if 'END OF TEC MAP' in lbl_inner:
                    break
                if 'LAT/LON1/LON2/DLON/H' in lbl_inner:
                    lat_val = float(lines[i][:8])
                    lat_idx = round((lat_val - lat1) / dlat)
                    i += 1
                    # read values across multiple lines until next label
                    vals = []
                    while i < len(lines):
                        inner_lbl = label(lines[i])
                        if 'LAT/LON1/LON2/DLON/H' in inner_lbl or \
                           'END OF TEC MAP' in inner_lbl:
                            break
                        vals.extend(lines[i].split())
                        i += 1
                    arr = np.array(vals, dtype=float)
                    arr[arr == 9999] = np.nan
                    arr *= 10.0 ** exponent        # convert to TECU
                    tec_map[lat_idx, :len(arr)] = arr
                    continue   # do NOT increment i (already at next LAT/END)
                i += 1

            vtec_maps.append(tec_map)

        i += 1

    return {
        'epochs': np.array(epochs),
        'lat':    lat_arr,
        'lon':    lon_arr,
        'vtec':   np.array(vtec_maps),
        'h_ion':  h_ion,
        'R_E':    R_E,
    }


# ---------------------------------------------------------------------------
# VTEC interpolation
# ---------------------------------------------------------------------------

def interpolate_vtec(ionex: dict, lat_deg: float, lon_deg: float,
                     unix_time: float) -> float:
    """
    Bilinear spatial + linear temporal interpolation of VTEC at (lat, lon, t).

    Parameters
    ----------
    ionex    : dict from parse_ionex()
    lat_deg  : float   geocentric latitude  [degrees]
    lon_deg  : float   longitude            [degrees]
    unix_time: float   UTC Unix timestamp

    Returns
    -------
    float  VTEC [TECU]
    """
    epochs = ionex['epochs']
    lats   = ionex['lat']
    lons   = ionex['lon']
    vtec   = ionex['vtec']

    # temporal interpolation index
    t_frac = np.interp(unix_time, epochs, np.arange(len(epochs)))
    t0 = int(np.clip(t_frac, 0, len(epochs) - 2))
    t1 = t0 + 1
    wt = t_frac - t0

    # lat index (lats may be descending → sort for interp)
    lat_frac = np.interp(lat_deg, lats[::-1], np.arange(len(lats))[::-1])
    j0 = int(np.clip(lat_frac, 0, len(lats) - 2))
    j1 = j0 + 1
    wj = lat_frac - j0

    # lon wraps −180…180; handle wrap-around
    lon_deg = ((lon_deg + 180) % 360) - 180
    lon_idx = np.interp(lon_deg, lons, np.arange(len(lons)))
    i0 = int(np.clip(lon_idx, 0, len(lons) - 2))
    i1 = i0 + 1
    wi = lon_idx - i0

    def get(ti, ji, ii):
        v = vtec[ti, ji, ii]
        return float(np.nanmean(vtec[ti])) if np.isnan(v) else v  # fallback

    def bilin(ti):
        return ((1 - wj) * ((1 - wi) * get(ti, j0, i0) + wi * get(ti, j0, i1)) +
                wj        * ((1 - wi) * get(ti, j1, i0) + wi * get(ti, j1, i1)))

    return (1 - wt) * bilin(t0) + wt * bilin(t1)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def compute_ipp(lat_tgt_deg: float, lon_tgt_deg: float,
                incidence_angle_deg: float, look_azimuth_deg: float,
                h_ion_km: float, R_E_km: float = 6371.0
                ) -> tuple[float, float]:
    """
    Compute the ionospheric pierce point (IPP) location.

    Uses the standard single-layer-model spherical geometry:
    the central angle between target and IPP is
        psi = theta_inc - arcsin(R_E / r_ion * sin(theta_inc))

    The IPP lies along the great-circle arc from the target toward
    the satellite sub-satellite point (look_azimuth_deg direction).

    Parameters
    ----------
    lat_tgt_deg, lon_tgt_deg : float   target geodetic lat/lon [deg]
    incidence_angle_deg      : float   local incidence angle [deg]
    look_azimuth_deg         : float   azimuth FROM target TOWARD satellite [deg]
    h_ion_km                 : float   single-layer height [km]
    R_E_km                   : float   Earth radius [km]

    Returns
    -------
    (lat_ipp_deg, lon_ipp_deg)
    """
    theta = np.radians(incidence_angle_deg)
    r_ion = R_E_km + h_ion_km

    # central angle between target and IPP
    psi = theta - np.arcsin(R_E_km / r_ion * np.sin(theta))  # radians

    # move psi radians from target in the look_azimuth direction
    lat0 = np.radians(lat_tgt_deg)
    lon0 = np.radians(lon_tgt_deg)
    az   = np.radians(look_azimuth_deg)

    lat_ipp = np.arcsin(np.sin(lat0) * np.cos(psi) +
                        np.cos(lat0) * np.sin(psi) * np.cos(az))
    dlon    = np.arctan2(np.sin(az) * np.sin(psi) * np.cos(lat0),
                         np.cos(psi) - np.sin(lat0) * np.sin(lat_ipp))
    lon_ipp = lon0 + dlon

    return np.degrees(lat_ipp), np.degrees(lon_ipp)


def mapping_function(incidence_angle_deg: float, R_E_km: float,
                     h_ion_km: float) -> float:
    """
    Single-layer model mapping function M = 1/cos(theta_ipp).

    Maps VTEC → STEC:  STEC = M * VTEC

    theta_ipp is computed as:
        sin(theta_ipp) = R_E / (R_E + h_ion) * sin(theta_inc)
    """
    theta    = np.radians(incidence_angle_deg)
    sin_ipp  = R_E_km / (R_E_km + h_ion_km) * np.sin(theta)
    cos_ipp  = np.sqrt(1.0 - sin_ipp ** 2)
    return 1.0 / cos_ipp


# ---------------------------------------------------------------------------
# Main correction function
# ---------------------------------------------------------------------------

def iono_doppler_correction(ionex_path: str, slc_par_path: str,
                            delta_az_m: float = 5e4) -> float:
    """
    Compute the ionosphere-induced Doppler centroid [Hz] for the scene.

    Uses the scene-center geometry and a spatial finite-difference to
    estimate the azimuth TEC gradient at the IPP.  The result is a
    scalar representative of the entire scene (GIM resolution ≫ scene
    extent, so spatial variation within the scene is negligible).

    Parameters
    ----------
    ionex_path    : str    path to IONEX .INX file
    slc_par_path  : str    path to GAMMA .slc.par file
    delta_az_m    : float  finite-difference step along azimuth [m], default 50 km

    Returns
    -------
    f_iono : float   ionospheric Doppler centroid [Hz]
             Positive → measured DC is higher than true (surface) DC.
             Subtract this from (DC_meas - DC_geom) to isolate f_surface.
    """
    par = parse_slc_par(slc_par_path)

    # ---- scene geometry from .par ----
    lat_c   = float(par['center_latitude'][0])
    lon_c   = float(par['center_longitude'][0])
    heading = float(par['heading'][0])
    theta_i = float(par['incidence_angle'][0])
    f_c     = float(par['radar_frequency'][0])

    # centre time (seconds of day → Unix)
    t_center   = float(par['center_time'][0])
    date_parts = par['date']
    yr, mo, dy = int(date_parts[0]), int(date_parts[1]), int(date_parts[2])
    day_start = datetime(yr, mo, dy, tzinfo=timezone.utc).timestamp()
    unix_center = day_start + t_center

    # ---- satellite velocity from state vectors ----
    n_sv = int(par['number_of_state_vectors'][0])
    t_sv0 = float(par['time_of_first_state_vector'][0])
    dt_sv = float(par['state_vector_interval'][0])

    vel_keys = [f'state_vector_velocity_{k}' for k in range(1, n_sv + 1)]
    vels = np.array([[float(par[k][j]) for j in range(3)]
                     for k in vel_keys if k in par])
    # interpolate to centre time
    sv_times  = t_sv0 + np.arange(len(vels)) * dt_sv
    v_interp  = np.array([np.interp(t_center, sv_times, vels[:, j])
                           for j in range(3)])
    v_sat_mag = float(np.linalg.norm(v_interp))

    # ---- IONEX ----
    ionex = parse_ionex(ionex_path)
    h_ion = ionex['h_ion']
    R_E   = ionex['R_E']

    # ---- IPP geometry ----
    # look azimuth: direction FROM target TOWARD satellite
    # For right-looking SAR: satellite is to the right of the heading direction
    # heading is satellite flight direction; right of heading → heading + 90°
    # from target, satellite is at heading + 90° - 180° = heading - 90°
    look_azimuth = (heading + 90.0) % 360.0   # satellite bearing from target

    lat_ipp, lon_ipp = compute_ipp(lat_c, lon_c, theta_i, look_azimuth,
                                   h_ion, R_E)

    M = mapping_function(theta_i, R_E, h_ion)

    # ---- IPP velocity in azimuth direction ----
    r_sat = float(par['sar_to_earth_center'][0]) / 1e3  # km
    r_ion = R_E + h_ion
    # IPP azimuth speed ≈ satellite ground-track speed scaled by layer height
    v_ground_az = v_sat_mag * R_E / r_sat
    v_ipp_az    = v_ground_az * r_ion / R_E              # m/s

    # ---- azimuth VTEC gradient at IPP (numerical, finite difference) ----
    # offset ±delta_az_m in the satellite heading direction
    deg_per_m   = 1.0 / 111_320.0                        # 1° ≈ 111.32 km
    delta_lat_d = delta_az_m * deg_per_m * np.cos(np.radians(heading))
    delta_lon_d = delta_az_m * deg_per_m * np.sin(np.radians(heading)) \
                  / np.cos(np.radians(lat_ipp))

    vtec_plus  = interpolate_vtec(ionex,
                                  lat_ipp + delta_lat_d,
                                  lon_ipp + delta_lon_d,
                                  unix_center)
    vtec_minus = interpolate_vtec(ionex,
                                  lat_ipp - delta_lat_d,
                                  lon_ipp - delta_lon_d,
                                  unix_center)

    # gradient [TECU/m], signed in azimuth direction (heading of satellite)
    dVTEC_ds_az = (vtec_plus - vtec_minus) / (2.0 * delta_az_m)  # TECU/m

    # ---- ionospheric Doppler ----
    # f_iono = -2 K M / (c f_c) * dVTEC/ds_az * v_ipp_az
    # VTEC gradient in TECU/m → convert to e/m³ using TECU = 1e16 e/m²
    dSTEC_dt = M * dVTEC_ds_az * TECU * v_ipp_az       # e/m²/s = e/m³·m/s

    f_iono = -2.0 * K_IONO / (C_LIGHT * f_c) * dSTEC_dt  # Hz

    return f_iono
