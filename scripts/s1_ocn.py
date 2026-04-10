"""
Sentinel-1 OCN radial-current retrieval from the delivered Level-2 product.

This module is intentionally narrow:

- it reads the actual OCN NetCDF measurement file inside a SAFE product
- it uses only the product variables that are present in that file
- it works on one swath at a time, defaulting to the first swath

The retrieval implemented here uses only information already present in the
OCN file:

    currentRadVel = rvlRadVel - radial(Stokes drift) - radial(wind drift)

Stokes drift is projected from ``rvlUssX`` / ``rvlUssY`` using the platform
heading.  Wind drift is estimated as ``alpha * owiWindSpeed`` projected onto
the look direction, where ``alpha`` (default 0.016) is the empirical wind
drift factor.  ``owiWindDirection`` follows the oceanographic convention
(direction the wind is blowing *toward*, clockwise from north).

This remains a radial current estimate, not a full 2-D current vector.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import griddata


def _find_combined_ocn_file(
    safe_dir: str,
    polarisation: str | None = None,
) -> Path:
    measurement_dir = Path(safe_dir) / "measurement"
    if not measurement_dir.is_dir():
        raise FileNotFoundError(f"Missing measurement directory in {safe_dir}")

    pol_token = polarisation.lower() if polarisation else None
    matches: list[Path] = []
    for path in sorted(measurement_dir.glob("*.nc")):
        name = path.name.lower()
        if "-ocn-" not in name:
            continue
        if pol_token and f"-{pol_token}-" not in name:
            continue
        matches.append(path)

    if not matches:
        raise FileNotFoundError(f"No combined OCN measurement file found in {measurement_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple combined OCN measurement files found in {measurement_dir}: {matches}")
    return matches[0]


def _select_swath(dataset: xr.Dataset, swath_index: int = 0) -> xr.Dataset:
    """
    Reduce the combined IW OCN file to one swath.

    Any variable carrying ``rvlSwath`` or ``owiSwath`` is sliced with the
    requested swath index.  Variables without a swath dimension are left as-is.
    """
    selection = {}
    if "rvlSwath" in dataset.dims:
        selection["rvlSwath"] = swath_index
    if "owiSwath" in dataset.dims:
        selection["owiSwath"] = swath_index

    reduced = dataset.isel(selection, drop=True) if selection else dataset
    reduced.attrs = dict(dataset.attrs)
    reduced.attrs["selected_swath"] = swath_index
    return reduced


def open_ocn(
    safe_dir: str,
    polarisation: str | None = None,
    swath_index: int = 0,
) -> xr.Dataset:
    """
    Open the combined OCN NetCDF file and select one swath.

    The file is opened with ``decode_cf=True`` so OCN fill values such as
    ``-999`` and ``-128`` are converted to ``NaN`` where appropriate.
    """
    path = _find_combined_ocn_file(safe_dir, polarisation=polarisation)
    dataset = xr.open_dataset(path, decode_cf=True).load()
    dataset.attrs["source_file"] = path.name
    dataset.attrs["source_path"] = str(path)
    return _select_swath(dataset, swath_index=swath_index)


def load_rvl(
    safe_dir: str,
    polarisation: str | None = None,
    swath_index: int = 0,
) -> xr.Dataset:
    """
    Load the RVL-relevant variables for a single swath.
    """
    dataset = open_ocn(safe_dir, polarisation=polarisation, swath_index=swath_index)
    wanted = [
        "rvlLon",
        "rvlLat",
        "rvlLandCoverage",
        "rvlLandFlag",
        "rvlIncidenceAngle",
        "rvlHeading",
        "rvlNrcs",
        "rvlDcObs",
        "rvlDcObsStd",
        "rvlRadVel",
        "rvlRadVelStd",
        "rvlDcGeo",
        "rvlDcMiss",
        "rvlConfDcObs",
        "rvlUssX",
        "rvlUssY",
        "rvlPitch",
        "rvlYaw",
        "rvlRoll",
        "rvlSnr",
        "rvlSweepAngle",
        "rvlSlantRangeTime",
        "rvlZeroDopplerTime",
        "rvlGroundRngSize",
        "rvlAziSize",
        "rvlMask",
        "rvlSwathNumber",
    ]
    available = [name for name in wanted if name in dataset]
    rvl = dataset[available]
    rvl.attrs = dict(dataset.attrs)
    return rvl


def load_owi(
    safe_dir: str,
    polarisation: str | None = None,
    swath_index: int = 0,
) -> xr.Dataset:
    """
    Load the OWI-relevant variables for a single swath.
    """
    dataset = open_ocn(safe_dir, polarisation=polarisation, swath_index=swath_index)
    wanted = [
        "owiLon",
        "owiLat",
        "owiIncidenceAngle",
        "owiElevationAngle",
        "owiWindSpeed",
        "owiWindDirection",
        "owiEcmwfWindSpeed",
        "owiEcmwfWindDirection",
        "owiNrcs",
        "owiNesz",
        "owiPBright",
        "owiNrcsCmod",
        "owiCalConstObsi",
        "owiCalConstInci",
        "owiInversionQuality",
        "owiMask",
        "owiHeading",
        "owiWindQuality",
        "owiNrcsNeszCorr",
        "owiRadVel",
        "owiSubswathCoverage",
        "owiPolarisationName",
    ]
    available = [name for name in wanted if name in dataset]
    owi = dataset[available]
    owi.attrs = dict(dataset.attrs)
    return owi


def valid_rvl_mask(rvl: xr.Dataset) -> xr.DataArray:
    """
    Product-valid RVL mask.

    The local data inspection for this repository shows:
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
    """
    Project ``rvlUssX`` / ``rvlUssY`` onto the radar look direction.
    """
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
    """
    Interpolate OWI variables onto the RVL grid.

    OWI and RVL use independent grids with different resolutions.  A scattered
    linear interpolation (scipy ``griddata``) is used because both grids are in
    swath geometry and their lat/lon arrays are 2-D.
    """
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
        interpolated = griddata(
            owi_points[valid], vals[valid], rvl_points, method="linear"
        )
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
    wind.  ``owiWindDirection`` is expected in oceanographic convention
    (direction the wind blows *toward*, clockwise from north).

    The radial component is:

        wind_drift_radial = alpha * U_10 * cos(theta_wind - theta_look)

    OWI variables are interpolated onto the RVL grid before projection.

    Parameters
    ----------
    alpha : float
        Empirical wind drift factor (dimensionless).  Typical range 0.013–0.017;
        default 0.016.
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

    # Recover a reference DataArray for dims/coords from rvlRadVel
    ref = rvl["rvlRadVel"]
    da = xr.DataArray(
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
    da.name = "windDriftRadial"
    return da


def _compute_radial_current(
    rvl: xr.Dataset,
    owi: xr.Dataset | None = None,
    antenna_side: str = "right",
    wind_drift_alpha: float = 0.016,
) -> xr.Dataset:
    """
    Core computation: derive radial current from pre-loaded RVL (and optionally OWI) datasets.

        currentRadVel = rvlRadVel - radial(Stokes drift) [- radial(wind drift)]

    masked to cells where ``rvlMask == 0``.  Wind drift correction is applied
    when ``owi`` is provided.
    """
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
    polarisation: str | None = None,
    swath_index: int = 0,
    antenna_side: str = "right",
    correct_wind_drift: bool = True,
    wind_drift_alpha: float = 0.016,
) -> xr.Dataset:
    """
    Retrieve the radial velocity induced by ocean current from the OCN product.

    The estimate uses only information already present in the OCN file:

        currentRadVel = rvlRadVel - radial(Stokes drift) [- radial(wind drift)]

    and is masked to cells where ``rvlMask == 0``.

    Parameters
    ----------
    correct_wind_drift : bool
        If True (default), subtract the wind-induced surface drift estimated
        from the OWI wind speed and direction.
    wind_drift_alpha : float
        Empirical wind drift factor (default 0.016).
    """
    rvl = load_rvl(safe_dir, polarisation=polarisation, swath_index=swath_index)
    owi = load_owi(safe_dir, polarisation=polarisation, swath_index=swath_index) if correct_wind_drift else None
    return _compute_radial_current(rvl, owi=owi, antenna_side=antenna_side, wind_drift_alpha=wind_drift_alpha)


def process_ocn(
    safe_dir: str,
    polarisation: str | None = None,
    swath_index: int = 0,
    antenna_side: str = "right",
    correct_wind_drift: bool = True,
    wind_drift_alpha: float = 0.016,
) -> dict[str, xr.Dataset]:
    """
    Convenience wrapper returning the opened OCN subsets and radial-current
    retrieval for one swath.
    """
    rvl = load_rvl(safe_dir, polarisation=polarisation, swath_index=swath_index)
    owi = load_owi(safe_dir, polarisation=polarisation, swath_index=swath_index)
    current = _compute_radial_current(
        rvl,
        owi=owi if correct_wind_drift else None,
        antenna_side=antenna_side,
        wind_drift_alpha=wind_drift_alpha,
    )
    return {
        "rvl": rvl,
        "owi": owi,
        "radial_current": current,
    }
