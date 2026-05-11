"""Plot merged Sentinel-1 retrievals and model comparisons on common grids."""

from __future__ import annotations

import numpy as np

from .grid_merge import compute_stats, merge_burst_grids, merge_model_grid


def plot_comparison(
    results: list[dict],
    overlap: str = "average",
    resolution_deg: float = 0.01,
    vmax: float | None = None,
    out_path: str | None = None,
    variable: str = "v_current_ocn",
) -> None:
    """Plot merged SAR current, merged model current, and comparison scatter."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    grid_lat, grid_lon, merged_sar = merge_burst_grids(
        results, overlap=overlap, resolution_deg=resolution_deg, variable=variable
    )
    _, _, merged_model = merge_model_grid(results, resolution_deg=resolution_deg)
    extent = [grid_lon.min(), grid_lon.max(), grid_lat.min(), grid_lat.max()]

    if vmax is None:
        finite = np.concatenate([
            merged_sar[np.isfinite(merged_sar)],
            merged_model[np.isfinite(merged_model)],
        ])
        vmax = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 0.5

    total_bias, total_rmse, total_r = compute_stats(merged_sar, merged_model)

    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)

    ax0 = fig.add_subplot(gs[0])
    im0 = ax0.imshow(merged_sar, extent=extent, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im0, ax=ax0, label="m/s", fraction=0.046, pad=0.04)
    ax0.set_title(f"SAR ocean current ({overlap})\nbias={total_bias:+.3f}  RMSE={total_rmse:.3f}  r={total_r:.3f}")
    ax0.set_xlabel("Longitude [°]")
    ax0.set_ylabel("Latitude [°]")

    ax1 = fig.add_subplot(gs[1])
    im1 = ax1.imshow(merged_model, extent=extent, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im1, ax=ax1, label="m/s", fraction=0.046, pad=0.04)
    ax1.set_title("GLO12 model current")
    ax1.set_xlabel("Longitude [°]")
    ax1.set_ylabel("Latitude [°]")

    ax2 = fig.add_subplot(gs[2])
    finite = np.isfinite(merged_sar) & np.isfinite(merged_model)
    sar_f = merged_sar[finite]
    model_f = merged_model[finite]
    ax2.scatter(model_f, sar_f, s=2, alpha=0.3, rasterized=True)
    lim = vmax * 1.05
    ax2.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_aspect("equal")
    ax2.set_xlabel("GLO12 [m/s]")
    ax2.set_ylabel("SAR [m/s]")
    ax2.set_title(f"SAR vs GLO12 (total)\nbias={total_bias:+.3f}  RMSE={total_rmse:.3f}  r={total_r:.3f}")

    plt.suptitle("Ocean current radial velocity — SAR vs GLO12", fontsize=12)
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f'\n{"Burst":>6}  {"bias [m/s]":>10}  {"RMSE [m/s]":>10}  {"r":>6}')
    print("─" * 40)
    for result in results:
        print(
            f'  {result["burst_idx"]:>4}  {result["bias_vs_glo12"]:>+10.4f}  '
            f'{result["rmse_vs_glo12"]:>10.4f}  {result["r_vs_glo12"]:>6.4f}'
        )
    print("─" * 40)
    print(f'{"total":>6}  {total_bias:>+10.4f}  {total_rmse:>10.4f}  {total_r:>6.4f}')
