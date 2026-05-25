"""Load ERA5, OCN, and model fields and project them onto the SAR look direction."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from .ocn_product import load_ocn_safe

_OCN_FILL = -999.0
_SWATH_IDX = {"iw1": 0, "iw2": 1, "iw3": 2}


def _interp2d(lat_grid, lon_grid, values, query_lat, query_lon):
    """Bilinear interpolation of a regular (lat, lon) field onto scattered points."""
    interp = RegularGridInterpolator(
        (lat_grid, lon_grid),
        values,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    pts = np.column_stack([query_lat.ravel(), query_lon.ravel()])
    return interp(pts).reshape(query_lat.shape)


def _select_time(ds: xr.Dataset, acq_time: str, time_dim: str = "valid_time"):
    """Select the time step in *ds* closest to *acq_time* (ISO string)."""
    import pandas as pd

    target = np.datetime64(pd.Timestamp(acq_time).tz_localize(None))
    times = ds[time_dim].values.astype("datetime64[ns]")
    tidx = int(np.argmin(np.abs(times - target)))
    return ds.isel({time_dim: tidx})


@lru_cache(maxsize=8)
def _open_dataset_cached(path: str) -> xr.Dataset:
    """Reuse opened metocean datasets across per-burst processing."""
    return xr.open_dataset(path)


def load_era5_wind(path: str, acq_time: str) -> dict:
    """Load ERA5 10-m wind (u10, v10) at the hour closest to *acq_time*."""
    ds = _select_time(_open_dataset_cached(path), acq_time)
    lat = ds["latitude"].values.astype(np.float64)
    lon = ds["longitude"].values.astype(np.float64)
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        u10 = ds["u10"].values[::-1, :].astype(np.float64)
        v10 = ds["v10"].values[::-1, :].astype(np.float64)
    else:
        u10 = ds["u10"].values.astype(np.float64)
        v10 = ds["v10"].values.astype(np.float64)
    return {"lat": lat, "lon": lon, "u10": u10, "v10": v10, "time": str(ds["valid_time"].values)}


def load_era5_wave(path: str, acq_time: str) -> dict:
    """Load ERA5 surface Stokes drift (ust, vst) and wave params at *acq_time*."""
    ds = _select_time(_open_dataset_cached(path), acq_time)
    lat = ds["latitude"].values.astype(np.float64)
    lon = ds["longitude"].values.astype(np.float64)
    flip = lat[0] > lat[-1]
    if flip:
        lat = lat[::-1]

    def _get(name):
        values = ds[name].values.astype(np.float64)
        return values[::-1, :] if flip else values

    return {
        "lat": lat,
        "lon": lon,
        "ust": _get("ust"),
        "vst": _get("vst"),
        "swh": _get("swh"),
        "mwp": _get("mwp"),
        "mwd": _get("mwd"),
        "time": str(ds["valid_time"].values),
    }


def compute_stokes_radial(
    wave: dict,
    our_lat: np.ndarray,
    our_lon: np.ndarray,
    our_inc: np.ndarray,
    look_az_rad: float,
) -> np.ndarray:
    """Project ERA5 surface Stokes drift onto the radar slant-range direction."""
    ust = _interp2d(wave["lat"], wave["lon"], wave["ust"], our_lat, our_lon)
    vst = _interp2d(wave["lat"], wave["lon"], wave["vst"], our_lat, our_lon)
    inc_rad = np.deg2rad(our_inc.astype(np.float64))
    return ((ust * np.sin(look_az_rad) + vst * np.cos(look_az_rad)) * np.sin(inc_rad)).astype(np.float32)


def compute_wave_doppler_bias(
    wind: dict,
    our_lat: np.ndarray,
    our_lon: np.ndarray,
    our_inc: np.ndarray,
    look_az_rad: float,
) -> np.ndarray:
    """C-band VV empirical wave Doppler bias (Mouche et al. 2012, simplified)."""
    u10 = _interp2d(wind["lat"], wind["lon"], wind["u10"], our_lat, our_lon)
    v10 = _interp2d(wind["lat"], wind["lon"], wind["v10"], our_lat, our_lon)

    wind_speed = np.sqrt(u10**2 + v10**2)
    wind_dir_to = np.arctan2(u10, v10)
    delta_phi = wind_dir_to - look_az_rad

    scale = 0.025 + 0.001 * (our_inc.astype(np.float64) - 30.0)
    scale = np.clip(scale, 0.015, 0.05)
    return (scale * wind_speed * np.cos(delta_phi)).astype(np.float32)


def load_ocn_wave_velocity(ocn_safe: str, subswath: str, polarisation: str) -> dict:
    """Load OCN owiRadVel (wind-induced radial velocity) for one subswath.

    Returns a dict with ``lat``, ``lon``, ``rad_vel`` (m/s) — the OCN's
    calibrated wave-Doppler velocity, suitable as a drop-in replacement for
    :func:`compute_wave_doppler_bias`.
    """
    ocn = load_ocn_safe(ocn_safe, swath=subswath, polarisation=polarisation)
    ds = ocn["owi"]
    sw = _SWATH_IDX[subswath.lower()]
    if "owiSwath" in ds.dims:
        ds = ds.isel(owiSwath=sw)
    if "owiPolarisation" in ds.dims:
        ds = ds.isel(owiPolarisation=0)

    def _clean(name):
        arr = ds[name].values.astype(np.float64)
        arr[arr == _OCN_FILL] = np.nan
        return arr

    return {
        "lat":     _clean("owiLat"),
        "lon":     _clean("owiLon"),
        "rad_vel": _clean("owiRadVel"),
    }


def compute_wave_doppler_bias_ocn(
    owi: dict,
    our_lat: np.ndarray,
    our_lon: np.ndarray,
) -> np.ndarray:
    """Resample OCN ``owiRadVel`` onto the SAR retrieval grid."""
    return match_to_sar_grid(
        our_lat, our_lon, owi["lat"], owi["lon"], owi["rad_vel"],
    ).astype(np.float32)


def load_ocn_rvl(ocn_safe: str, subswath: str, polarisation: str) -> dict:
    """Load OCN RVL fields for one subswath, fill -> NaN."""
    ocn = load_ocn_safe(ocn_safe, swath=subswath, polarisation=polarisation)
    ds = ocn["rvl"]
    swath_index = _SWATH_IDX[subswath.lower()]
    if "rvlSwath" in ds.dims:
        ds = ds.isel(rvlSwath=swath_index)

    def _clean(name):
        arr = ds[name].values.astype(np.float64)
        arr[arr == _OCN_FILL] = np.nan
        return arr

    return {
        "lat": _clean("rvlLat"),
        "lon": _clean("rvlLon"),
        "rad_vel": _clean("rvlRadVel"),
        "dc_miss": _clean("rvlDcMiss"),
        "heading": _clean("rvlHeading"),
        "inc": _clean("rvlIncidenceAngle"),
    }


def match_to_sar_grid(
    our_lat: np.ndarray,
    our_lon: np.ndarray,
    ref_lat: np.ndarray,
    ref_lon: np.ndarray,
    ref_data: np.ndarray,
    max_dist_deg: float = 0.05,
) -> np.ndarray:
    """Nearest-neighbour resample of ref_data onto the SAR grid."""
    from scipy.spatial import KDTree

    our_pts = np.column_stack([our_lat.ravel(), our_lon.ravel()])
    ref_pts = np.column_stack([ref_lat.ravel(), ref_lon.ravel()])
    valid = np.isfinite(ref_pts).all(axis=1)
    tree = KDTree(ref_pts[valid])
    dists, idx = tree.query(our_pts, workers=-1)
    mapped = ref_data.ravel()[np.where(valid)[0][idx]].reshape(our_lat.shape).astype(np.float32)
    mapped[dists.reshape(our_lat.shape) > max_dist_deg] = np.nan
    return mapped


def load_glo12_current(path: str, acq_time: str) -> dict:
    """Load glo12 surface uo/vo at the hour closest to *acq_time*."""
    import pandas as pd

    ds = _open_dataset_cached(path).isel(depth=0)
    target = np.datetime64(pd.Timestamp(acq_time).tz_localize(None))
    times = ds["time"].values.astype("datetime64[ns]")
    tidx = int(np.argmin(np.abs(times - target)))
    ds = ds.isel(time=tidx)

    lat = ds["latitude"].values.astype(np.float64)
    lon = ds["longitude"].values.astype(np.float64)

    def _masked(name):
        values = ds[name].values.astype(np.float64)
        values[np.abs(values) > 1e10] = np.nan
        values[values < -100] = np.nan
        return values

    return {
        "lat": lat,
        "lon": lon,
        "uo": _masked("uo"),
        "vo": _masked("vo"),
        "time": str(ds["time"].values),
    }


def project_current_onto_look(
    model: dict,
    our_lat: np.ndarray,
    our_lon: np.ndarray,
    our_inc: np.ndarray,
    look_az_rad: float,
) -> np.ndarray:
    """Project glo12 (uo, vo) onto the radar slant-range direction."""
    uo = _interp2d(model["lat"], model["lon"], model["uo"], our_lat, our_lon)
    vo = _interp2d(model["lat"], model["lon"], model["vo"], our_lat, our_lon)
    inc_rad = np.deg2rad(our_inc.astype(np.float64))
    v_r = (uo * np.sin(look_az_rad) + vo * np.cos(look_az_rad)) * np.sin(inc_rad)
    return v_r.astype(np.float32)
