"""
Plotting utilities for Doppler centroid and DCA maps.
"""

import numpy as np
import matplotlib.pyplot as plt


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
