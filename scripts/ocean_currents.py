"""
Ocean current data loading and export utilities.

Handles Copernicus Marine Service NetCDF products and GeoTIFF export.
"""

import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_origin


def load_currents(
    nc_path: str,
    lon_slice: tuple[float, float],
    lat_slice: tuple[float, float],
    time: str,
) -> xr.Dataset:
    """
    Load ocean current data from a Copernicus Marine NetCDF file.

    Parameters
    ----------
    nc_path : str
        Path to the NetCDF file.
    lon_slice : (west, east)
        Longitude bounds.
    lat_slice : (south, north)
        Latitude bounds.
    time : str
        ISO8601 timestamp to select (nearest match).

    Returns
    -------
    xr.Dataset with variables uo, vo (and optionally utide, vtide, vsdx, vsdy).
    """
    return xr.open_dataset(nc_path).sel(
        longitude=slice(*lon_slice),
        latitude=slice(*lat_slice),
        time=time,
    ).load()


def save_currents_geotiff(ds: xr.Dataset, out_path: str, variables: list[str] = None) -> None:
    """
    Export ocean current velocity components from an xr.Dataset to a GeoTIFF.

    Each variable becomes one band in the output file.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with latitude/longitude coordinates (from load_currents).
    out_path : str
        Output GeoTIFF path.
    variables : list of str, optional
        Variable names to export. Defaults to ['uo', 'vo'].
    """
    if variables is None:
        variables = ['uo', 'vo']

    data = ds[variables].to_array().values.squeeze()
    if data.ndim == 2:
        data = data[np.newaxis]  # ensure (bands, height, width)

    transform = from_origin(
        west=ds.longitude.min().item(),
        north=ds.latitude.max().item(),
        xsize=(ds.longitude[1] - ds.longitude[0]).item(),
        ysize=(ds.latitude[1] - ds.latitude[0]).item(),
    )

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


def load_currents_geotiff(tif_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load uo and vo current components from a GeoTIFF written by save_currents_geotiff.

    Returns
    -------
    uo, vo : np.ndarray
        East and north velocity components [m/s], NaN where fill value (-1000).
    """
    with rasterio.open(tif_path) as src:
        uo = src.read(1).astype(float)
        vo = src.read(2).astype(float)

    uo[uo == -1000] = np.nan
    vo[vo == -1000] = np.nan
    return uo, vo
