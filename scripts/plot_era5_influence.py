#!/usr/bin/env python3
"""
Plot ERA5 wind and wave corrections stored in run_all_bursts() results.

Notebook usage
--------------
    from plot_era5_influence import plot_wind_influence, plot_wave_influence

    results = run_all_bursts(...)

    plot_wind_influence(results)   # v_wave  — Mouche 2012 wave Doppler bias
    plot_wave_influence(results)   # v_stokes — Stokes drift radial projection
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rvl_pipeline import merge_burst_grids


def plot_wind_influence(
    results: list,
    resolution_deg: float = 0.01,
    vmax: float | None = None,
    out_path: str | None = None,
) -> plt.Figure:
    """
    Map of the wave Doppler bias correction (v_wave) across all bursts.

    v_wave is the Mouche 2012 empirical correction that accounts for the
    apparent Doppler shift caused by wind-driven waves at the sea surface.
    Large values indicate bursts where wind conditions strongly bias the
    raw SAR radial velocity.

    Parameters
    ----------
    results        : list of dicts from run_all_bursts()
    resolution_deg : regrid resolution in degrees (default 0.01 ≈ 1 km)
    vmax           : colour scale limit [m/s] (auto if None)
    out_path       : save figure to this path if given
    """
    grid_lat, grid_lon, v_wave = merge_burst_grids(
        results, variable='v_wave', resolution_deg=resolution_deg
    )

    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(v_wave[np.isfinite(v_wave)]), 98)) or 0.3

    extent = [grid_lon.min(), grid_lon.max(), grid_lat.min(), grid_lat.max()]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(v_wave, extent=extent, origin='lower',
                   cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    plt.colorbar(im, ax=ax, label='v_wave [m/s]', fraction=0.046, pad=0.04)
    ax.set_title('Wind influence — wave Doppler bias (v_wave)\n'
                 'Mouche 2012:  a(θ)·U₁₀·cos(wind_dir − look_az)',
                 fontsize=11)
    ax.set_xlabel('Longitude [°]')
    ax.set_ylabel('Latitude [°]')

    # Per-burst mean labels
    for r in results:
        mean_val = float(np.nanmean(r['v_wave']))
        ax.text(
            float(np.nanmean(r['lon'])), float(np.nanmean(r['lat'])),
            f"b{r['burst_idx']}\n{mean_val:+.3f} m/s",
            ha='center', va='center', fontsize=8, color='k',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6, ec='none'),
        )

    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved → {out_path}')

    return fig


def plot_wave_influence(
    results: list,
    resolution_deg: float = 0.01,
    vmax: float | None = None,
    out_path: str | None = None,
) -> plt.Figure:
    """
    Map of the Stokes drift correction (v_stokes) across all bursts.

    v_stokes is the surface Stokes drift projected onto the radar look
    direction.  Large values indicate bursts where swell or wind-sea
    significantly biases the SAR radial velocity estimate.

    Parameters
    ----------
    results        : list of dicts from run_all_bursts()
    resolution_deg : regrid resolution in degrees (default 0.01 ≈ 1 km)
    vmax           : colour scale limit [m/s] (auto if None)
    out_path       : save figure to this path if given
    """
    grid_lat, grid_lon, v_stokes = merge_burst_grids(
        results, variable='v_stokes', resolution_deg=resolution_deg
    )

    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(v_stokes[np.isfinite(v_stokes)]), 98)) or 0.1

    extent = [grid_lon.min(), grid_lon.max(), grid_lat.min(), grid_lat.max()]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(v_stokes, extent=extent, origin='lower',
                   cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    plt.colorbar(im, ax=ax, label='v_stokes [m/s]', fraction=0.046, pad=0.04)
    ax.set_title('Wave influence — Stokes drift radial projection (v_stokes)\n'
                 '(ust·sin(ψ) + vst·cos(ψ))·sin(θ)',
                 fontsize=11)
    ax.set_xlabel('Longitude [°]')
    ax.set_ylabel('Latitude [°]')

    # Per-burst mean labels
    for r in results:
        mean_val = float(np.nanmean(r['v_stokes']))
        ax.text(
            float(np.nanmean(r['lon'])), float(np.nanmean(r['lat'])),
            f"b{r['burst_idx']}\n{mean_val:+.3f} m/s",
            ha='center', va='center', fontsize=8, color='k',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6, ec='none'),
        )

    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved → {out_path}')

    return fig
