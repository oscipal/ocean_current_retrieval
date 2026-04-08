"""
Sentinel-1 OCN radial-current retrieval from the delivered Level-2 product.

This module is intentionally narrow:

- it reads the actual OCN NetCDF measurement file inside a SAFE product
- it uses only the product variables that are present in that file
- it works on one swath at a time, defaulting to the first swath

The retrieval implemented here is the strongest "current-only" estimate that
can be formed from the OCN product alone:

    radial_current = rvlRadVel - radial(Stokes drift)

where the Stokes drift is projected from ``rvlUssX`` and ``rvlUssY`` using the
platform heading.  No external wind or wave model is used beyond what is
already embedded in the OCN product.

This remains a radial current estimate, not a full 2-D current vector.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


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


def retrieve_radial_current(
    safe_dir: str,
    polarisation: str | None = None,
    swath_index: int = 0,
    antenna_side: str = "right",
) -> xr.Dataset:
    """
    Retrieve the radial velocity induced by ocean current from the OCN product.

    The estimate uses only information already present in the OCN file:

        currentRadVel = rvlRadVel - rvlUssRadial

    and is masked to cells where ``rvlMask == 0``.
    """
    rvl = load_rvl(
        safe_dir,
        polarisation=polarisation,
        swath_index=swath_index,
    )

    keep = valid_rvl_mask(rvl)
    look = look_azimuth(rvl, antenna_side=antenna_side)
    stokes = stokes_radial_velocity(rvl, antenna_side=antenna_side)

    current = (rvl["rvlRadVel"] - stokes).where(keep)
    current.name = "currentRadVel"
    current.attrs["units"] = "m s-1"
    current.attrs["long_name"] = "Radial velocity induced by ocean current"

    result = xr.Dataset(
        {
            "currentRadVel": current,
            "rvlRadVel": rvl["rvlRadVel"],
            "rvlUssRadial": stokes,
            "rvlValid": keep,
            "rvlMask": rvl["rvlMask"],
            "lookAzimuth": look,
        },
        attrs=dict(rvl.attrs),
    )

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

    result.attrs["retrieval"] = "currentRadVel = rvlRadVel - radial(rvlUssX, rvlUssY)"
    result.attrs["antenna_side"] = antenna_side
    return result


def process_ocn(
    safe_dir: str,
    polarisation: str | None = None,
    swath_index: int = 0,
    antenna_side: str = "right",
) -> dict[str, xr.Dataset]:
    """
    Convenience wrapper returning the opened OCN subsets and radial-current
    retrieval for one swath.
    """
    rvl = load_rvl(safe_dir, polarisation=polarisation, swath_index=swath_index)
    owi = load_owi(safe_dir, polarisation=polarisation, swath_index=swath_index)
    current = retrieve_radial_current(
        safe_dir,
        polarisation=polarisation,
        swath_index=swath_index,
        antenna_side=antenna_side,
    )
    return {
        "rvl": rvl,
        "owi": owi,
        "radial_current": current,
    }
