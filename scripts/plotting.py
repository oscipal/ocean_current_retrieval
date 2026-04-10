"""
Plotting utilities for Doppler centroid, DCA maps, and OCN radial current.
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.interpolate import NearestNDInterpolator


def plot_dc_map(img: np.ndarray, ax: plt.Axes, title: str, cmap: str = 'viridis_r') -> None:
    """
    Plot a Doppler centroid or DCA map with robust color limits and summary stats.

    Parameters
    ----------
    img : np.ndarray
        2D array of Doppler centroid values [Hz].
    ax : matplotlib.axes.Axes
        Axes to plot on.
    title : str
        Plot title.
    cmap : str
        Colormap name.
    """
    vmin = np.round(np.nanquantile(img, 0.05))
    vmax = np.round(np.nanquantile(img, 0.95))
    im = ax.imshow(img, aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap)

    median = np.nanmedian(img)
    mad = np.nanmedian(np.abs(img - median))
    ax.set_ylabel('azimuth')
    ax.set_xlabel(
        f'range\n'
        f'median: {median:.2f} Hz,  MAD: {mad:.2f} Hz\n'
        f'mean: {np.nanmean(img):.2f} Hz,  std: {np.nanstd(img):.2f} Hz'
    )
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='DC [Hz]')


def _fill_coord_nans(arr: np.ndarray) -> np.ndarray:
    """Fill NaN values in a 2D coordinate array using nearest-neighbour interpolation."""
    mask = np.isfinite(arr)
    if mask.all():
        return arr
    rows, cols = np.indices(arr.shape)
    interp = NearestNDInterpolator(
        np.column_stack([rows[mask], cols[mask]]), arr[mask]
    )
    filled = arr.copy()
    filled[~mask] = interp(np.column_stack([rows[~mask], cols[~mask]]))
    return filled


def plot_radial_current(
    result: xr.Dataset,
    ax: plt.Axes | None = None,
    step: int = 5,
    scale: float = 20.0,
    arrow_width: float = 0.003,
    arrow_headwidth: float = 3.0,
    cmap: str = "RdBu_r",
) -> plt.Axes:
    """
    Plot the radial ocean current estimate as a quiver map on a lat/lon grid.

    Arrows point in the radar look direction (or opposite for negative values)
    with length proportional to ``currentRadVel``.  A background colormap shows
    the signed radial speed.

    Parameters
    ----------
    result : xr.Dataset
        Output of ``retrieve_radial_current`` or ``process_ocn["radial_current"]``.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.  A new figure is created if not provided.
    step : int
        Subsampling step for quiver arrows (default 5 — one arrow every 5 cells).
    scale : float
        Quiver scale parameter passed to ``ax.quiver``.  Lower = longer arrows.
        Default 20.
    arrow_width : float
        Shaft width as a fraction of plot width (default 0.003).  Increase for
        thicker arrows.
    arrow_headwidth : float
        Head width in units of shaft width (default 3).  Increase for larger
        arrowheads.
    cmap : str
        Colormap for the background speed field (default ``RdBu_r``).

    Returns
    -------
    matplotlib.axes.Axes
    """
    lon = result["rvlLon"].values
    lat = result["rvlLat"].values
    vel = result["currentRadVel"].values
    look = result["lookAzimuth"].values

    # Decompose radial velocity into east / north components along the look direction
    look_rad = np.deg2rad(look)
    u = vel * np.sin(look_rad)   # east
    v = vel * np.cos(look_rad)   # north

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    # Background: pcolormesh on the actual 2-D swath grid.
    # Coordinate arrays must be NaN-free; fill with nearest-neighbour before plotting.
    clim = np.nanquantile(np.abs(vel[np.isfinite(vel)]), 0.98) if np.any(np.isfinite(vel)) else 1.0
    lon_filled = _fill_coord_nans(lon)
    lat_filled = _fill_coord_nans(lat)
    pcm = ax.pcolormesh(
        lon_filled, lat_filled, vel,
        cmap=cmap, vmin=-clim, vmax=clim,
        shading="auto", rasterized=True,
    )
    plt.colorbar(pcm, ax=ax, label="Radial current velocity [m s$^{-1}$]", pad=0.02)

    # Arrows: subsampled quiver on the same lon/lat grid
    sl = (slice(None, None, step), slice(None, None, step))
    lon_q, lat_q = lon[sl], lat[sl]
    u_q, v_q = u[sl], v[sl]
    valid_q = np.isfinite(lon_q) & np.isfinite(lat_q) & np.isfinite(u_q) & np.isfinite(v_q)
    ax.quiver(
        lon_q[valid_q], lat_q[valid_q],
        u_q[valid_q], v_q[valid_q],
        scale=scale,
        scale_units="inches",
        width=arrow_width,
        headwidth=arrow_headwidth,
        color="k",
        alpha=0.7,
    )

    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Latitude [°]")
    ax.set_title(
        f"Radial ocean current — look azimuth ~{np.nanmedian(look):.0f}°\n"
        f"({result.attrs.get('retrieval', '')})"
    )
    ax.set_aspect("equal")

    return ax


def plot_spectrum(freqs: np.ndarray, spectrum: np.ndarray, ax: plt.Axes, **kwargs) -> None:
    """
    Plot a range-averaged azimuth power spectrum.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency axis [Hz].
    spectrum : np.ndarray
        1D power spectrum array.
    ax : matplotlib.axes.Axes
        Axes to plot on.
    """
    ax.plot(freqs, spectrum, **kwargs)
    ax.set_xlabel('Doppler Frequency [Hz]')
    ax.set_ylabel('Power')
