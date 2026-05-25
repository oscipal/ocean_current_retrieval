"""Burst-by-burst comparison of SLC-derived vs OCN raw Doppler centroid."""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.interpolate import griddata

from .ocn_analysis import _decode_ocn_times
from .burst_pipeline import compute_rvl_burst
from .safe_io import find_safe_files, parse_annotation


# ─────────────────────────────────────────────────────────────────────────────
# OCN loading helpers
# ─────────────────────────────────────────────────────────────────────────────

_RVL_SKIP = {"rvlZeroDopplerTime", "rvlSwathNumber"}
_RVL_FILL = -999.0


def load_ocn_rvl_swath(ocn_nc_path: str, swath_idx: int) -> xr.Dataset:
    """
    Load RVL variables from the combined OCN NetCDF for one swath.

    Parameters
    ----------
    ocn_nc_path : str
        Path to the combined OCN NetCDF file (``s1a-iw-ocn-*.nc``).
    swath_idx : int
        0-based swath index (0 = IW1, 1 = IW2, 2 = IW3).

    Returns
    -------
    xr.Dataset
        2-D RVL dataset (rvlAzSize × rvlRaSize) with float fill values
        replaced by NaN.
    """
    ds = xr.open_dataset(ocn_nc_path, decode_cf=False).load()
    rvl_vars = [
        v for v in ds.data_vars
        if v.lower().startswith("rvl")
        and v not in _RVL_SKIP
        and "rvlSwath" in ds[v].dims
    ]
    extracted = ds[rvl_vars].isel(rvlSwath=swath_idx)

    # Replace fill values with NaN for all float arrays
    for var in extracted.data_vars:
        da = extracted[var]
        if np.issubdtype(da.dtype, np.floating):
            extracted[var] = da.where(da != _RVL_FILL)

    extracted.attrs.update(ds.attrs)
    extracted.attrs["swath_index"] = swath_idx
    extracted.attrs["source_path"] = ocn_nc_path
    return extracted


# ─────────────────────────────────────────────────────────────────────────────
# Burst assignment
# ─────────────────────────────────────────────────────────────────────────────

def assign_burst_indices(ocn_nc_path: str, annot, swath_idx: int) -> np.ndarray:
    """
    Assign each OCN azimuth row to the closest annotation burst (0-based index).

    Parameters
    ----------
    ocn_nc_path : str
        Path to the combined OCN NetCDF file.
    annot : S1Annotation
        Parsed annotation for the matching subswath.
    swath_idx : int
        0-based swath index.

    Returns
    -------
    np.ndarray
        Integer array of shape (rvlAzSize,) with burst indices.
    """
    ocn_sec = _decode_ocn_times(ocn_nc_path, swath_idx)
    burst_sec = np.array([b.azimuth_time.timestamp() for b in annot.bursts])
    diffs = np.abs(ocn_sec[:, None] - burst_sec[None, :])
    return np.argmin(diffs, axis=1)


def extract_ocn_burst_data(
    ocn_ds: xr.Dataset,
    burst_assign: np.ndarray,
    burst_idx: int,
) -> xr.Dataset:
    """
    Slice OCN dataset rows belonging to one burst.

    Parameters
    ----------
    ocn_ds : xr.Dataset
        Full-swath OCN RVL dataset (from :func:`load_ocn_rvl_swath`).
    burst_assign : np.ndarray
        Per-row burst assignment (from :func:`assign_burst_indices`).
    burst_idx : int
        0-based burst index to extract.

    Returns
    -------
    xr.Dataset
        OCN rows for the requested burst.
    """
    row_indices = np.where(burst_assign == burst_idx)[0]
    if row_indices.size == 0:
        raise ValueError(f"No OCN rows assigned to burst {burst_idx}")
    return ocn_ds.isel(rvlAzSize=row_indices)


# ─────────────────────────────────────────────────────────────────────────────
# Spatial regridding
# ─────────────────────────────────────────────────────────────────────────────

def regrid_to_ocn(our_ds: xr.Dataset, ocn_burst: xr.Dataset) -> np.ndarray:
    """
    Interpolate the SLC-derived ``doppler_obs`` onto the OCN lat/lon grid.

    Parameters
    ----------
    our_ds : xr.Dataset
        Single-burst output from :func:`~burst_pipeline.compute_rvl_burst`.
    ocn_burst : xr.Dataset
        OCN rows for the same burst (from :func:`extract_ocn_burst_data`).

    Returns
    -------
    np.ndarray
        ``doppler_obs`` values on the OCN grid; NaN where extrapolation
        is required.
    """
    src_lat = our_ds["latitude"].values.ravel()
    src_lon = our_ds["longitude"].values.ravel()
    src_val = our_ds["doppler_obs"].values.ravel()

    dst_lat = ocn_burst["rvlLat"].values.ravel()
    dst_lon = ocn_burst["rvlLon"].values.ravel()

    valid = np.isfinite(src_val)
    interp = griddata(
        np.column_stack([src_lat[valid], src_lon[valid]]),
        src_val[valid],
        np.column_stack([dst_lat, dst_lon]),
        method="linear",
    )
    return interp.reshape(ocn_burst["rvlLat"].shape)


# ─────────────────────────────────────────────────────────────────────────────
# Main comparison entry point
# ─────────────────────────────────────────────────────────────────────────────

def compare_burst_doppler(
    slc_safe: str,
    ocn_nc_path: str,
    subswath: str,
    burst_idx: int,
    polarisation: str = "vv",
    block_az: int = 256,
    block_rg: int = 512,
    stride_az: int = 128,
    stride_rg: int = 256,
    deramp_method: str = "current",
    poeorb_path: str | None = None,
    aux_cal_path: str | None = None,
) -> dict:
    """
    Compare SLC-derived raw Doppler vs OCN ``rvlDcObs`` for one burst.

    Parameters
    ----------
    slc_safe : str
        Path to the Level-1 IW SLC SAFE directory.
    ocn_nc_path : str
        Path to the combined Level-2 OCN NetCDF file (``s1a-iw-ocn-*.nc``).
    subswath : str
        ``'iw1'``, ``'iw2'``, or ``'iw3'``.
    burst_idx : int
        0-based burst index within the subswath.
    polarisation : str
        ``'vv'`` (default) or ``'vh'``.
    block_az, block_rg : int
        Block dimensions for the CDE estimator.
    stride_az, stride_rg : int
        Block stride for the CDE estimator.
    deramp_method : str
        Deramp method passed to :func:`~burst_pipeline.compute_rvl_burst`.
    poeorb_path, aux_cal_path : str or None
        Optional precision orbit / AUX_CAL paths.

    Returns
    -------
    dict with keys:

    ``'our'``
        :class:`xr.Dataset` from :func:`~burst_pipeline.compute_rvl_burst`.
        Contains ``doppler_obs`` (raw, pre-subtraction) among other fields.
    ``'ocn'``
        :class:`xr.Dataset` — OCN rows for this burst (``rvlDcObs``,
        ``rvlDcGeo``, ``rvlLat``, ``rvlLon``, etc.).
    ``'our_regrid'``
        ``doppler_obs`` interpolated onto the OCN lat/lon grid.
    ``'diff'``
        ``our_regrid - rvlDcObs`` [Hz].
    ``'burst_assign'``
        Full-swath array mapping each OCN row to its burst index.
    """
    swath_idx = int(subswath[-1]) - 1

    files = find_safe_files(slc_safe, subswath, polarisation)
    annot = parse_annotation(files["annotation"])

    our_ds = compute_rvl_burst(
        slc_safe, subswath, burst_idx, polarisation,
        block_az=block_az, block_rg=block_rg,
        stride_az=stride_az, stride_rg=stride_rg,
        deramp_method=deramp_method,
        poeorb_path=poeorb_path,
        aux_cal_path=aux_cal_path,
    )

    ocn_ds = load_ocn_rvl_swath(ocn_nc_path, swath_idx)
    burst_assign = assign_burst_indices(ocn_nc_path, annot, swath_idx)
    ocn_burst = extract_ocn_burst_data(ocn_ds, burst_assign, burst_idx)

    our_regrid = regrid_to_ocn(our_ds, ocn_burst)
    ocn_dc = ocn_burst["rvlDcObs"].values.astype(float)
    diff = our_regrid - ocn_dc

    return {
        "our": our_ds,
        "ocn": ocn_burst,
        "our_regrid": our_regrid,
        "diff": diff,
        "burst_assign": burst_assign,
    }


def compare_all_bursts(
    slc_safe: str,
    ocn_nc_path: str,
    subswath: str,
    polarisation: str = "vv",
    block_az: int = 256,
    block_rg: int = 512,
    stride_az: int = 128,
    stride_rg: int = 256,
    deramp_method: str = "current",
    poeorb_path: str | None = None,
    aux_cal_path: str | None = None,
) -> list[dict]:
    """
    Run :func:`compare_burst_doppler` for every burst in *subswath*.

    Returns a list of comparison dicts in burst order.
    """
    files = find_safe_files(slc_safe, subswath, polarisation)
    annot = parse_annotation(files["annotation"])
    n_bursts = len(annot.bursts)

    return [
        compare_burst_doppler(
            slc_safe, ocn_nc_path, subswath, bi, polarisation,
            block_az=block_az, block_rg=block_rg,
            stride_az=stride_az, stride_rg=stride_rg,
            deramp_method=deramp_method,
            poeorb_path=poeorb_path,
            aux_cal_path=aux_cal_path,
        )
        for bi in range(n_bursts)
    ]


def plot_rvl_comparison(
    results: list[dict],
    overlap: str = "average",
    resolution_deg: float = 0.01,
    vmax: float | None = None,
    out_path: str | None = None,
) -> None:
    """
    Plot our raw Doppler (f_dc) and OCN rvlDcObs (f_dc_ocn) side by side
    with a scatter, given the output of run_all_bursts.

    Mirrors the layout of plots.plot_comparison: two image panels on a shared
    colour scale followed by a scatter with 1:1 line and aggregate statistics.

    Parameters
    ----------
    results : list[dict]
        Output of :func:`~pipeline.run_all_bursts`.
    overlap : str
        How overlapping burst regions are combined when merging (``'average'``
        or ``'first'``).
    resolution_deg : float
        Regular grid resolution in degrees.
    vmax : float or None
        Colour scale limit [Hz].  Auto-scaled to the 98th percentile if None.
    out_path : str or None
        Save figure to this path when provided.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from .grid_merge import compute_stats, merge_burst_grids

    grid_lat, grid_lon, our_grid = merge_burst_grids(
        results, variable="f_dc", overlap=overlap, resolution_deg=resolution_deg,
    )
    _, _, ocn_grid = merge_burst_grids(
        results, variable="f_dc_ocn", overlap=overlap, resolution_deg=resolution_deg,
    )
    extent = [grid_lon.min(), grid_lon.max(), grid_lat.min(), grid_lat.max()]

    if vmax is None:
        finite = np.concatenate([
            our_grid[np.isfinite(our_grid)],
            ocn_grid[np.isfinite(ocn_grid)],
        ])
        vmax = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 50.0

    bias, rmse, r = compute_stats(our_grid, ocn_grid)

    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)

    ax0 = fig.add_subplot(gs[0])
    im0 = ax0.imshow(
        our_grid, extent=extent, origin="lower",
        cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
    )
    plt.colorbar(im0, ax=ax0, label="Hz", fraction=0.046, pad=0.04)
    ax0.set_title("Our f_dc [Hz]")
    ax0.set_xlabel("Longitude [°]")
    ax0.set_ylabel("Latitude [°]")

    ax1 = fig.add_subplot(gs[1])
    im1 = ax1.imshow(
        ocn_grid, extent=extent, origin="lower",
        cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
    )
    plt.colorbar(im1, ax=ax1, label="Hz", fraction=0.046, pad=0.04)
    ax1.set_title("OCN rvlDcObs [Hz]")
    ax1.set_xlabel("Longitude [°]")
    ax1.set_ylabel("Latitude [°]")

    ax2 = fig.add_subplot(gs[2])
    finite = np.isfinite(our_grid) & np.isfinite(ocn_grid)
    our_v = our_grid[finite]
    ocn_v = ocn_grid[finite]
    ax2.scatter(ocn_v, our_v, s=2, alpha=0.3, rasterized=True)
    lim = vmax * 1.05
    ax2.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_aspect("equal")
    ax2.set_xlabel("OCN rvlDcObs [Hz]")
    ax2.set_ylabel("Our f_dc [Hz]")
    ax2.set_title(f"Our vs OCN\nbias={bias:+.2f}  RMSE={rmse:.2f}  r={r:.4f}")

    plt.suptitle("Raw Doppler comparison — Our vs OCN", fontsize=12)
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


def compare_fdc_shape(
    result: dict,
    out_path: str | None = None,
) -> dict:
    """
    Compare the range profile of our f_dc against OCN rvlDcObs for one burst.

    Collapses the azimuth axis with a NaN-tolerant mean, then asks: does our
    profile (mean-subtracted) follow the same shape as OCN's, ignoring scale
    and offset?  Reports Pearson r, the linear-fit scale and offset that map
    our profile onto OCN, and the residual RMSE after that fit.

    Parameters
    ----------
    result : dict
        One entry of the list returned by :func:`pipeline.run_all_bursts`.
        Must contain ``f_dc`` and ``f_dc_ocn`` (both 2-D, az × rg).
    out_path : str or None
        Save the figure to this path if provided.

    Returns
    -------
    dict with keys ``r``, ``scale``, ``offset``, ``rmse_after_fit``,
    ``our_range_profile``, ``ocn_range_profile``, ``burst_idx``.
    """
    import matplotlib.pyplot as plt

    our  = np.asarray(result["f_dc"], dtype=float)
    ocn  = np.asarray(result["f_dc_ocn"], dtype=float)
    bidx = result.get("burst_idx", -1)

    our_profile = np.nanmean(our, axis=0)
    ocn_profile = np.nanmean(ocn, axis=0)

    mask = np.isfinite(our_profile) & np.isfinite(ocn_profile)
    if mask.sum() < 3:
        raise ValueError(
            f"compare_fdc_shape: burst {bidx} has fewer than 3 finite range "
            f"columns (our={np.isfinite(our_profile).sum()}, "
            f"ocn={np.isfinite(ocn_profile).sum()})"
        )

    x = our_profile[mask]
    y = ocn_profile[mask]
    scale, offset = np.polyfit(x, y, 1)
    y_pred = scale * x + offset
    rmse_after_fit = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    r = float(np.corrcoef(x, y)[0, 1])

    rg_idx = np.arange(our_profile.size)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax0 = axes[0]
    ax0b = ax0.twinx()
    l0a, = ax0.plot(rg_idx, our_profile, color="tab:blue", lw=1.5, label="our f_dc")
    l0b, = ax0b.plot(rg_idx, ocn_profile, color="tab:red", lw=1.5, label="OCN rvlDcObs")
    ax0.set_xlabel("Range block index")
    ax0.set_ylabel("our f_dc [Hz]", color="tab:blue")
    ax0b.set_ylabel("OCN rvlDcObs [Hz]", color="tab:red")
    ax0.tick_params(axis="y", labelcolor="tab:blue")
    ax0b.tick_params(axis="y", labelcolor="tab:red")
    ax0.set_title("Raw range profiles (twin y-axes)")
    ax0.legend([l0a, l0b], [l0a.get_label(), l0b.get_label()], loc="best")

    ax1 = axes[1]
    ax1.plot(rg_idx, our_profile - np.nanmean(our_profile), color="tab:blue", lw=1.5, label="our − mean")
    ax1.plot(rg_idx, ocn_profile - np.nanmean(ocn_profile), color="tab:red", lw=1.5, label="OCN − mean")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_xlabel("Range block index")
    ax1.set_ylabel("Doppler [Hz]")
    ax1.set_title(f"Mean-subtracted shapes (r = {r:+.3f})")
    ax1.legend(loc="best")

    ax2 = axes[2]
    fitted = scale * our_profile + offset
    ax2.plot(rg_idx, fitted, color="tab:blue", lw=1.5,
             label=f"scale·our + offset")
    ax2.plot(rg_idx, ocn_profile, color="tab:red", lw=1.5, label="OCN rvlDcObs")
    ax2.set_xlabel("Range block index")
    ax2.set_ylabel("Doppler [Hz]")
    ax2.set_title(
        f"After fit: scale={scale:+.3f}  offset={offset:+.2f} Hz  "
        f"rmse={rmse_after_fit:.2f} Hz"
    )
    ax2.legend(loc="best")

    fig.suptitle(f"f_dc shape comparison — burst {bidx}", fontsize=12)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(
        f"burst {bidx}:  r={r:+.3f}  scale={scale:+.3f}  "
        f"offset={offset:+.2f} Hz  rmse_after_fit={rmse_after_fit:.2f} Hz"
    )

    return {
        "burst_idx": bidx,
        "r": r,
        "scale": float(scale),
        "offset": float(offset),
        "rmse_after_fit": rmse_after_fit,
        "our_range_profile": our_profile,
        "ocn_range_profile": ocn_profile,
    }


def diagnose_burst_geo(
    results: list[dict],
    burst_idx: int,
    neighbour_idx: int | None = None,
    out_path: str | None = None,
) -> dict:
    """
    Investigate whether a burst's f_dc_ocn was silently pulled from a
    neighbouring burst by ``match_to_sar_grid`` (KDTree nearest neighbour
    with a 0.05° / ~5.5 km cutoff).

    For ``burst_idx`` (and optionally a neighbour to compare against):
      - Print lat/lon span, NaN fraction in f_dc_ocn, f_dc and f_dc_ocn ranges.
      - Plot the SAR footprint (lat/lon scatter coloured by f_dc) next to the
        f_dc_ocn it received.  If the f_dc_ocn pattern of the burst under test
        matches its neighbour rather than its own f_dc, the KDTree is reaching
        into the wrong burst.

    Parameters
    ----------
    results : list[dict]
        Output of :func:`pipeline.run_all_bursts`.
    burst_idx : int
        Burst to investigate.
    neighbour_idx : int or None
        Adjacent burst to compare against.  Defaults to ``burst_idx - 1``
        (or ``burst_idx + 1`` when ``burst_idx`` is 0).
    """
    import matplotlib.pyplot as plt

    if neighbour_idx is None:
        neighbour_idx = burst_idx - 1 if burst_idx > 0 else burst_idx + 1

    def _summarise(r):
        lat, lon = np.asarray(r["lat"]), np.asarray(r["lon"])
        fdc = np.asarray(r["f_dc"], dtype=float)
        focn = np.asarray(r["f_dc_ocn"], dtype=float)
        return {
            "burst_idx":     r.get("burst_idx", -1),
            "lat_min":       float(np.nanmin(lat)),
            "lat_max":       float(np.nanmax(lat)),
            "lon_min":       float(np.nanmin(lon)),
            "lon_max":       float(np.nanmax(lon)),
            "fdc_nanfrac":   float(np.isnan(fdc).mean()),
            "focn_nanfrac":  float(np.isnan(focn).mean()),
            "fdc_min":       float(np.nanmin(fdc)),
            "fdc_max":       float(np.nanmax(fdc)),
            "focn_min":      float(np.nanmin(focn)),
            "focn_max":      float(np.nanmax(focn)),
            "lat": lat, "lon": lon, "f_dc": fdc, "f_dc_ocn": focn,
        }

    a = _summarise(results[burst_idx])
    b = _summarise(results[neighbour_idx])

    def _row(s, tag):
        print(
            f"{tag} burst {s['burst_idx']:2d}: "
            f"lat[{s['lat_min']:+.3f}, {s['lat_max']:+.3f}]  "
            f"lon[{s['lon_min']:+.3f}, {s['lon_max']:+.3f}]  "
            f"f_dc[{s['fdc_min']:+6.1f}, {s['fdc_max']:+6.1f}]Hz  "
            f"f_dc_ocn[{s['focn_min']:+6.1f}, {s['focn_max']:+6.1f}]Hz  "
            f"NaN%(focn)={100*s['focn_nanfrac']:.1f}"
        )

    print("─" * 110)
    _row(a, "TEST    ")
    _row(b, "NEIGHBOR")
    print(
        f"lat overlap: [{max(a['lat_min'], b['lat_min']):+.3f}, "
        f"{min(a['lat_max'], b['lat_max']):+.3f}]   "
        f"(disjoint if min > max)"
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col, s, label in [(0, a, f"burst {a['burst_idx']} (test)"),
                          (1, b, f"burst {b['burst_idx']} (neighbour)")]:
        vmax = max(
            np.nanmax(np.abs(s["f_dc"])) if np.isfinite(s["f_dc"]).any() else 1.0,
            np.nanmax(np.abs(s["f_dc_ocn"])) if np.isfinite(s["f_dc_ocn"]).any() else 1.0,
        )
        sc0 = axes[0, col].scatter(
            s["lon"].ravel(), s["lat"].ravel(),
            c=s["f_dc"].ravel(), cmap="RdBu_r",
            vmin=-vmax, vmax=vmax, s=8,
        )
        axes[0, col].set_title(f"our f_dc — {label}")
        axes[0, col].set_xlabel("Longitude [°]")
        axes[0, col].set_ylabel("Latitude [°]")
        plt.colorbar(sc0, ax=axes[0, col], label="Hz")

        sc1 = axes[1, col].scatter(
            s["lon"].ravel(), s["lat"].ravel(),
            c=s["f_dc_ocn"].ravel(), cmap="RdBu_r",
            vmin=-vmax, vmax=vmax, s=8,
        )
        axes[1, col].set_title(f"f_dc_ocn — {label}")
        axes[1, col].set_xlabel("Longitude [°]")
        axes[1, col].set_ylabel("Latitude [°]")
        plt.colorbar(sc1, ax=axes[1, col], label="Hz")

    fig.suptitle(
        f"Burst geo / OCN-match diagnostic — burst {burst_idx} vs {neighbour_idx}",
        fontsize=12,
    )
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()

    return {"test": a, "neighbour": b}


def diagnose_burst_2d(
    results: list[dict],
    burst_idx: int,
    vmax: float | None = None,
    out_path: str | None = None,
) -> dict:
    """
    Detailed 2-D side-by-side of our f_dc vs OCN f_dc_ocn for one burst.

    Produces a 2 × 2 figure:
      [0,0] our f_dc                   (lat/lon scatter, shared colour scale)
      [0,1] OCN f_dc_ocn               (same scale)
      [1,0] residual (our − OCN)       (diverging scale)
      [1,1] SNR                        (Greys)

    Also prints quantitative stats: 2-D r, mean / std / RMSE of the residual,
    and where the largest |residual| pixels are located.
    """
    import matplotlib.pyplot as plt

    r = results[burst_idx]
    lat = np.asarray(r["lat"], dtype=float)
    lon = np.asarray(r["lon"], dtype=float)
    fdc = np.asarray(r["f_dc"], dtype=float)
    focn = np.asarray(r["f_dc_ocn"], dtype=float)
    snr = np.asarray(r.get("snr", np.full_like(fdc, np.nan)), dtype=float)

    diff = fdc - focn

    mask = np.isfinite(fdc) & np.isfinite(focn)
    n_valid = int(mask.sum())
    if n_valid > 1:
        r2d = float(np.corrcoef(fdc[mask], focn[mask])[0, 1])
        bias = float(np.mean(diff[mask]))
        std_diff = float(np.std(diff[mask]))
        rmse = float(np.sqrt(np.mean(diff[mask] ** 2)))
    else:
        r2d = bias = std_diff = rmse = float("nan")

    if vmax is None:
        finite = np.concatenate([fdc[np.isfinite(fdc)], focn[np.isfinite(focn)]])
        vmax = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 50.0
    vdiff = float(np.nanpercentile(np.abs(diff[mask]), 98)) if mask.any() else vmax

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    for ax, data, title, vlim, cmap in [
        (axes[0, 0], fdc,  f"our f_dc — burst {burst_idx}",        vmax,  "RdBu_r"),
        (axes[0, 1], focn, f"OCN f_dc_ocn — burst {burst_idx}",    vmax,  "RdBu_r"),
        (axes[1, 0], diff, f"residual (our − OCN)",                vdiff, "RdBu_r"),
        (axes[1, 1], snr,  f"SNR",                                  None,  "Greys"),
    ]:
        if vlim is None:
            sc = ax.scatter(lon.ravel(), lat.ravel(), c=data.ravel(),
                            cmap=cmap, s=10)
        else:
            sc = ax.scatter(lon.ravel(), lat.ravel(), c=data.ravel(),
                            cmap=cmap, vmin=-vlim, vmax=vlim, s=10)
        ax.set_title(title)
        ax.set_xlabel("Longitude [°]")
        ax.set_ylabel("Latitude [°]")
        plt.colorbar(sc, ax=ax,
                     label="Hz" if cmap == "RdBu_r" else "SNR")

    fig.suptitle(
        f"Burst {burst_idx} 2-D comparison — "
        f"r={r2d:+.3f}  bias={bias:+.2f} Hz  std={std_diff:.2f} Hz  "
        f"RMSE={rmse:.2f} Hz  (n={n_valid})",
        fontsize=12,
    )
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(
        f"burst {burst_idx}: 2-D r={r2d:+.3f}  bias={bias:+.2f} Hz  "
        f"std={std_diff:.2f} Hz  RMSE={rmse:.2f} Hz  n={n_valid}"
    )

    abs_diff = np.where(mask, np.abs(diff), -np.inf)
    if mask.any():
        k = min(5, n_valid)
        flat_idx = np.argpartition(abs_diff.ravel(), -k)[-k:]
        flat_idx = flat_idx[np.argsort(-abs_diff.ravel()[flat_idx])]
        rows, cols = np.unravel_index(flat_idx, fdc.shape)
        print(f"  Top {k} largest residuals:")
        for rr, cc in zip(rows, cols):
            print(
                f"    az={rr:3d} rg={cc:3d}  "
                f"lat={lat[rr, cc]:+.3f} lon={lon[rr, cc]:+.3f}  "
                f"our={fdc[rr, cc]:+6.1f}  ocn={focn[rr, cc]:+6.1f}  "
                f"diff={diff[rr, cc]:+6.1f} Hz"
            )

    return {
        "burst_idx": burst_idx,
        "r_2d": r2d,
        "bias": bias,
        "std": std_diff,
        "rmse": rmse,
        "n_valid": n_valid,
    }


def burst_summary_stats(comparisons: list[dict]) -> dict:
    """
    Compute per-burst summary statistics from a list of comparison dicts.

    Returns
    -------
    dict with arrays indexed by burst number:

    ``'bias_hz'``
        Mean difference (our − OCN) [Hz] per burst.
    ``'std_hz'``
        Std of difference [Hz] per burst.
    ``'rmse_hz'``
        RMSE [Hz] per burst.
    ``'our_mean_hz'``
        Mean of our raw Doppler per burst.
    ``'ocn_mean_hz'``
        Mean of OCN ``rvlDcObs`` per burst.
    ``'n_valid'``
        Number of grid cells where both estimates are finite.
    """
    n = len(comparisons)
    bias = np.full(n, np.nan)
    std = np.full(n, np.nan)
    rmse = np.full(n, np.nan)
    our_mean = np.full(n, np.nan)
    ocn_mean = np.full(n, np.nan)
    n_valid = np.zeros(n, dtype=int)

    for i, cmp in enumerate(comparisons):
        diff = cmp["diff"].ravel()
        our = cmp["our_regrid"].ravel()
        ocn = cmp["ocn"]["rvlDcObs"].values.ravel().astype(float)
        mask = np.isfinite(diff) & np.isfinite(ocn)
        if mask.sum() == 0:
            continue
        d = diff[mask]
        bias[i] = d.mean()
        std[i] = d.std()
        rmse[i] = np.sqrt((d ** 2).mean())
        our_mean[i] = our[mask].mean()
        ocn_mean[i] = ocn[mask].mean()
        n_valid[i] = mask.sum()

    return {
        "bias_hz": bias,
        "std_hz": std,
        "rmse_hz": rmse,
        "our_mean_hz": our_mean,
        "ocn_mean_hz": ocn_mean,
        "n_valid": n_valid,
    }
