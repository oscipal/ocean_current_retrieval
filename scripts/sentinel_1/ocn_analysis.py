"""Derived analyses built on top of Sentinel-1 Level-2 OCN RVL and OWI fields."""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.interpolate import griddata

from .ocn_product import load_owi_from_ocn, load_rvl_from_ocn


def valid_rvl_mask(rvl: xr.Dataset) -> xr.DataArray:
    """
    Product-valid RVL mask.

    Local inspection for this repository indicates:
    - ``rvlMask == 0`` marks valid RVL cells
    - fill values are already converted to ``NaN`` by ``decode_cf=True``
    """
    if "rvlMask" not in rvl:
        raise KeyError("rvlMask is required for RVL validity masking")
    mask = (rvl["rvlMask"] == 0) & rvl["rvlRadVel"].notnull()
    mask.name = "rvlValid"
    mask.attrs["long_name"] = "Valid RVL cells"
    return mask


def look_azimuth(rvl: xr.Dataset, antenna_side: str = "right") -> xr.DataArray:
    """
    Radar look azimuth clockwise from north.

    Sentinel-1 is right-looking, so the default is ``heading + 90 deg``.
    """
    if "rvlHeading" not in rvl:
        raise KeyError("rvlHeading is required to derive radar look azimuth")

    side = antenna_side.lower()
    if side == "right":
        offset = 90.0
    elif side == "left":
        offset = -90.0
    else:
        raise ValueError("antenna_side must be 'right' or 'left'")

    look = (rvl["rvlHeading"] + offset) % 360.0
    look.name = "lookAzimuth"
    look.attrs["units"] = "degree"
    look.attrs["long_name"] = "Radar look azimuth clockwise from north"
    return look


def stokes_radial_velocity(rvl: xr.Dataset, antenna_side: str = "right") -> xr.DataArray:
    """Project ``rvlUssX`` / ``rvlUssY`` onto the radar look direction."""
    for name in ("rvlUssX", "rvlUssY", "rvlHeading"):
        if name not in rvl:
            raise KeyError(f"{name} is required for Stokes projection")

    angle = np.deg2rad(look_azimuth(rvl, antenna_side=antenna_side))
    radial = rvl["rvlUssX"] * np.sin(angle) + rvl["rvlUssY"] * np.cos(angle)
    radial.name = "rvlUssRadial"
    radial.attrs["units"] = "m s-1"
    radial.attrs["long_name"] = "Stokes drift radial velocity"
    return radial


def _regrid_owi_to_rvl(
    owi: xr.Dataset,
    rvl: xr.Dataset,
    variables: list[str],
) -> dict[str, np.ndarray]:
    """Interpolate OWI variables onto the RVL grid."""
    owi_lon = owi["owiLon"].values.ravel()
    owi_lat = owi["owiLat"].values.ravel()
    rvl_lon = rvl["rvlLon"].values
    rvl_lat = rvl["rvlLat"].values

    owi_points = np.column_stack([owi_lon, owi_lat])
    rvl_points = np.column_stack([rvl_lon.ravel(), rvl_lat.ravel()])

    result: dict[str, np.ndarray] = {}
    for var in variables:
        vals = owi[var].values.ravel().astype(float)
        valid = np.isfinite(vals)
        interpolated = griddata(owi_points[valid], vals[valid], rvl_points, method="linear")
        result[var] = interpolated.reshape(rvl_lon.shape)
    return result


def wind_drift_radial_velocity(
    rvl: xr.Dataset,
    owi: xr.Dataset,
    alpha: float = 0.016,
    antenna_side: str = "right",
) -> xr.DataArray:
    """
    Project wind-induced surface drift onto the radar look direction.

    The wind drift velocity is modelled as ``alpha * U_10`` directed along the
    wind. ``owiWindDirection`` is expected in oceanographic convention
    (direction the wind blows *toward*, clockwise from north).
    """
    for name in ("owiWindSpeed", "owiWindDirection", "owiLon", "owiLat"):
        if name not in owi:
            raise KeyError(f"{name} is required for wind drift correction")
    for name in ("rvlLon", "rvlLat"):
        if name not in rvl:
            raise KeyError(f"{name} is required for wind drift correction")

    regridded = _regrid_owi_to_rvl(owi, rvl, ["owiWindSpeed", "owiWindDirection"])

    look_deg = look_azimuth(rvl, antenna_side=antenna_side).values
    wind_dir_rad = np.deg2rad(regridded["owiWindDirection"])
    look_rad = np.deg2rad(look_deg)
    drift_values = alpha * regridded["owiWindSpeed"] * np.cos(wind_dir_rad - look_rad)

    ref = rvl["rvlRadVel"]
    drift = xr.DataArray(
        drift_values,
        dims=ref.dims,
        coords=ref.coords,
        attrs={
            "units": "m s-1",
            "long_name": "Wind-induced surface drift radial velocity",
            "wind_drift_factor": alpha,
            "wind_direction_convention": "oceanographic (toward), clockwise from north",
        },
    )
    drift.name = "windDriftRadial"
    return drift


def _compute_radial_current(
    rvl: xr.Dataset,
    owi: xr.Dataset | None = None,
    antenna_side: str = "right",
    wind_drift_alpha: float = 0.016,
) -> xr.Dataset:
    """Derive radial current from pre-loaded RVL and optionally OWI datasets."""
    keep = valid_rvl_mask(rvl)
    look = look_azimuth(rvl, antenna_side=antenna_side)
    stokes = stokes_radial_velocity(rvl, antenna_side=antenna_side)

    geophysical = rvl["rvlRadVel"] - stokes
    if owi is not None:
        wind_drift = wind_drift_radial_velocity(
            rvl, owi, alpha=wind_drift_alpha, antenna_side=antenna_side
        )
        geophysical = geophysical - wind_drift
    else:
        wind_drift = None

    current = geophysical.where(keep)
    current.name = "currentRadVel"
    current.attrs["units"] = "m s-1"
    current.attrs["long_name"] = "Radial velocity induced by ocean current"

    data_vars: dict[str, xr.DataArray] = {
        "currentRadVel": current,
        "rvlRadVel": rvl["rvlRadVel"],
        "rvlUssRadial": stokes,
        "rvlValid": keep,
        "rvlMask": rvl["rvlMask"],
        "lookAzimuth": look,
    }
    if wind_drift is not None:
        data_vars["windDriftRadial"] = wind_drift

    result = xr.Dataset(data_vars, attrs=dict(rvl.attrs))
    for name in (
        "rvlLon",
        "rvlLat",
        "rvlHeading",
        "rvlIncidenceAngle",
        "rvlRadVelStd",
        "rvlSnr",
        "rvlLandFlag",
        "rvlLandCoverage",
        "rvlConfDcObs",
        "rvlUssX",
        "rvlUssY",
        "rvlSwathNumber",
    ):
        if name in rvl:
            result[name] = rvl[name]

    retrieval = "currentRadVel = rvlRadVel - radial(rvlUssX, rvlUssY)"
    if wind_drift is not None:
        retrieval += f" - windDriftRadial (alpha={wind_drift_alpha})"
    result.attrs["retrieval"] = retrieval
    result.attrs["antenna_side"] = antenna_side
    return result


def retrieve_radial_current(
    safe_dir: str,
    swath: str | None = None,
    polarisation: str | None = None,
    antenna_side: str = "right",
    correct_wind_drift: bool = True,
    wind_drift_alpha: float = 0.016,
) -> xr.Dataset:
    """
    Retrieve the radial velocity induced by ocean current from an OCN SAFE.
    """
    rvl = load_rvl_from_ocn(safe_dir, swath=swath, polarisation=polarisation)
    owi = (
        load_owi_from_ocn(safe_dir, swath=swath, polarisation=polarisation)
        if correct_wind_drift
        else None
    )
    return _compute_radial_current(
        rvl, owi=owi, antenna_side=antenna_side, wind_drift_alpha=wind_drift_alpha
    )


def _decode_ocn_times(source_path: str, swath_index: int) -> np.ndarray:
    """
    Decode ``rvlZeroDopplerTime`` from the raw OCN NetCDF file.
    """
    import netCDF4
    from datetime import datetime, timezone

    with netCDF4.Dataset(source_path) as ds:
        raw = ds.variables["rvlZeroDopplerTime"][:]

    chars = raw[:, 0, :, swath_index]
    times = []
    for row in chars:
        filled = row.filled(b"-")
        value = b"".join(filled).decode("ascii").strip()
        try:
            t = datetime.strptime(value[:26], "%Y-%m-%d %H:%M:%S.%f").replace(
                tzinfo=timezone.utc
            )
            times.append(t.timestamp())
        except ValueError:
            times.append(float("nan"))
    return np.array(times)


def extract_mispointing_per_burst(
    rvl: xr.Dataset,
    annot,
) -> np.ndarray:
    """
    Extract one ``rvlDcMiss`` value per annotation burst by time-matching.
    """
    swath_value = rvl.attrs.get("selected_swath")
    swath_index = 0
    if isinstance(swath_value, str) and len(swath_value) >= 3 and swath_value[2:].isdigit():
        swath_index = int(swath_value[2:]) - 1
    source_path = rvl.attrs["source_path"]
    ocn_sec = _decode_ocn_times(source_path, swath_index)

    burst_sec = np.array([b.azimuth_time.timestamp() for b in annot.bursts])
    diffs = np.abs(ocn_sec[:, None] - burst_sec[None, :])
    burst_assign = np.argmin(diffs, axis=1)

    dc_miss = rvl["rvlDcMiss"].values
    return np.array([
        np.nanmean(dc_miss[burst_assign == j]) for j in range(len(annot.bursts))
    ])
