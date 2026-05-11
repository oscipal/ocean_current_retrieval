"""Diagnostic plots and checks for intermediate Sentinel-1 RVL pipeline outputs."""

from __future__ import annotations

import numpy as np

from scripts.sentinel_1.aux_files import apply_poeorb, parse_aux_cal, parse_aux_ins
from scripts.sentinel_1.safe_io import _nearest_estimate, find_safe_files, parse_annotation, read_slc_burst
from scripts.sentinel_1.rvl import (
    _fm_rate_at_burst,
    _interpolate_orbit,
    _steering_doppler_rate,
    apply_burst_valid_sample_mask,
    compute_gamma_ambiguity,
    correlation_to_doppler,
    deramp_burst,
    estimate_correlation_grid,
    mispointing_doppler_from_yaw,
)


def plot_pipeline_steps(
    slc_safe: str,
    subswath: str,
    poeorb_path: str,
    aux_cal_path: str,
    polarisation: str = "vv",
    block_az: int = 256,
    block_rg: int = 512,
    stride_az: int = 128,
    stride_rg: int = 256,
    burst_indices: list[int] | None = None,
    out_path: str | None = None,
) -> None:
    """Plot stitched per-burst pipeline intermediates for diagnostics."""
    import matplotlib.pyplot as plt

    files = find_safe_files(slc_safe, subswath, polarisation)
    annot = apply_poeorb(parse_annotation(files["annotation"]), poeorb_path)
    aap = parse_aux_cal(aux_cal_path, subswath, polarisation)
    if burst_indices is None:
        burst_indices = list(range(len(annot.bursts)))

    amp_raw_list, amp_drp_list, snr_list, fdc_list, n_rows_list = [], [], [], [], []
    for burst_idx in burst_indices:
        burst = annot.bursts[burst_idx]
        raw = read_slc_burst(files["measurement"], annot, burst_idx)
        deramped = deramp_burst(raw, annot, burst_idx)
        valid_mask = apply_burst_valid_sample_mask(deramped, burst)
        p0, p1, az_centers, rg_centers = estimate_correlation_grid(
            deramped, block_az, block_rg, stride_az, stride_rg, valid_mask=valid_mask
        )
        _, vel = _interpolate_orbit(annot, burst.azimuth_time)
        gamma_amb = compute_gamma_ambiguity(aap, annot.radar_prf, float(np.linalg.norm(vel)), annot.wavelength)
        f_dc, _, snr = correlation_to_doppler(p0, p1, annot.prf, annot.wavelength, gamma_amb=gamma_amb)

        ix = np.ix_(az_centers, rg_centers)
        amp_raw_list.append((20.0 * np.log10(np.abs(raw[ix]) + 1e-10)).astype(np.float32))
        amp_drp_list.append((20.0 * np.log10(np.abs(deramped[ix]) + 1e-10)).astype(np.float32))
        snr_list.append(snr.astype(np.float32))
        fdc_list.append(f_dc.astype(np.float32))
        n_rows_list.append(len(az_centers))

    def _stack(arrays):
        n_cols = arrays[0].shape[1]
        sep = np.full((1, n_cols), np.nan, dtype=np.float32)
        parts = []
        for i, arr in enumerate(arrays):
            parts.append(arr)
            if i < len(arrays) - 1:
                parts.append(sep)
        return np.vstack(parts)

    stacked = {
        "amp_raw": _stack(amp_raw_list),
        "amp_drp": _stack(amp_drp_list),
        "snr": _stack(snr_list),
        "fdc": _stack(fdc_list),
    }

    def _clim(arr):
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return 0.0, 1.0
        return float(np.percentile(finite, 2)), float(np.percentile(finite, 98))

    amp_vmin, amp_vmax = _clim(stacked["amp_raw"])
    snr_vmin, snr_vmax = _clim(stacked["snr"])
    fdc_finite = stacked["fdc"][np.isfinite(stacked["fdc"])]
    fdc_lim = float(np.percentile(np.abs(fdc_finite), 98)) if fdc_finite.size else 1.0

    titles = ["Raw amplitude [dB]", "Deramped amplitude [dB]", "Block SNR", "f_dc [Hz] (raw)"]
    keys = ["amp_raw", "amp_drp", "snr", "fdc"]
    cmaps = ["gray", "gray", "plasma", "RdBu_r"]
    vlims = [(amp_vmin, amp_vmax), (amp_vmin, amp_vmax), (snr_vmin, snr_vmax), (-fdc_lim, fdc_lim)]

    fig, axes = plt.subplots(1, 4, figsize=(22, max(6, len(burst_indices) * 1.2)))
    for ax, title, key, cmap, (vmin, vmax) in zip(axes, titles, keys, cmaps, vlims):
        im = ax.imshow(stacked[key], cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Range block")
        ax.set_ylabel("Azimuth block")
        row = 0
        for i, (burst_idx, n_rows) in enumerate(zip(burst_indices, n_rows_list)):
            ax.text(-1, row + n_rows / 2.0, f"B{burst_idx}", va="center", ha="right", fontsize=7, clip_on=False)
            row += n_rows
            if i < len(burst_indices) - 1:
                ax.axhline(row - 0.5, color="yellow", lw=0.8, ls="--", alpha=0.8)
                row += 1

    plt.suptitle(f"{subswath.upper()} — pipeline diagnostic steps", fontsize=11)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


def diagnose_burst_doppler(
    slc_safe: str,
    subswath: str,
    poeorb_path: str,
    aux_cal_path: str,
    polarisation: str = "vv",
    block_az: int = 256,
    block_rg: int = 512,
    stride_az: int = 128,
    stride_rg: int = 256,
    burst_indices: list[int] | None = None,
) -> None:
    """Print a per-burst diagnostic table for deramping and block Doppler."""
    files = find_safe_files(slc_safe, subswath, polarisation)
    annot = apply_poeorb(parse_annotation(files["annotation"]), poeorb_path)
    aap = parse_aux_cal(aux_cal_path, subswath, polarisation)
    if burst_indices is None:
        burst_indices = list(range(len(annot.bursts)))
    n_rg_mid = annot.n_samples // 2

    hdr = (
        f'{"burst":>5}  {"k_a [Hz/s]":>12}  {"k_psi [Hz/s]":>13}  {"k_s [Hz/s]":>11}  '
        f'{"n_valid":>7}  {"fdc_top [Hz]":>12}  {"fdc_bot [Hz]":>12}  {"top-bot [Hz]":>12}'
    )
    print(hdr)
    print("─" * len(hdr))

    for burst_idx in burst_indices:
        burst = annot.bursts[burst_idx]
        k_a_vec = _fm_rate_at_burst(annot, burst_idx)
        k_a = float(k_a_vec[n_rg_mid])
        k_psi = float(_steering_doppler_rate(annot, burst_idx))
        k_s = float(-k_a * k_psi / (k_a - k_psi))
        n_valid = int(np.count_nonzero(burst.first_valid_sample != -1))

        _, vel = _interpolate_orbit(annot, burst.azimuth_time)
        gamma_amb = compute_gamma_ambiguity(aap, annot.radar_prf, float(np.linalg.norm(vel)), annot.wavelength)
        raw = read_slc_burst(files["measurement"], annot, burst_idx)
        deramped = deramp_burst(raw, annot, burst_idx)
        valid_mask = apply_burst_valid_sample_mask(deramped, burst)
        p0, p1, _, _ = estimate_correlation_grid(
            deramped, block_az, block_rg, stride_az, stride_rg, valid_mask=valid_mask
        )
        f_dc, _, _ = correlation_to_doppler(p0, p1, annot.prf, annot.wavelength, gamma_amb=gamma_amb)

        half = max(1, f_dc.shape[0] // 2)
        fdc_top = float(np.nanmedian(f_dc[:half, :]))
        fdc_bot = float(np.nanmedian(f_dc[half:, :]))
        diff = fdc_top - fdc_bot
        print(
            f"{burst_idx:>5}  {k_a:>12.2f}  {k_psi:>13.2f}  {k_s:>11.2f}  {n_valid:>7}  "
            f"{fdc_top:>12.2f}  {fdc_bot:>12.2f}  {diff:>+12.2f}"
        )


def plot_dc_estimates(
    slc_safe: str,
    subswath: str,
    polarisation: str = "vv",
    out_path: str | None = None,
) -> None:
    """Plot stacked-burst annotation DC polynomials without SLC processing."""
    import matplotlib.pyplot as plt

    files = find_safe_files(slc_safe, subswath, polarisation)
    annot = parse_annotation(files["annotation"])
    tau = annot.slant_range_time_start + np.arange(annot.n_samples) / annot.range_sampling_rate

    rg_step = max(1, annot.n_samples // 500)
    tau_ds = tau[::rg_step]
    rows_per_burst = 10
    geom_imgs, data_imgs, diff_imgs = [], [], []
    sep = np.full((1, len(tau_ds)), np.nan, dtype=np.float32)

    hdr = f'{"burst":>5}  {"dc_idx":>6}  {"f_geom_mid [Hz]":>16}  {"f_data_mid [Hz]":>16}  {"diff_mid [Hz]":>14}'
    print(hdr)
    print("─" * len(hdr))

    for burst_idx, burst in enumerate(annot.bursts):
        dc = _nearest_estimate(annot.dc_estimates, burst.azimuth_time)
        dc_idx = annot.dc_estimates.index(dc)
        dt = tau - dc.t0
        f_geom = sum(c * dt**k for k, c in enumerate(dc.geometry_poly))
        f_data = sum(c * dt**k for k, c in enumerate(dc.data_poly))

        f_geom_ds = f_geom[::rg_step].astype(np.float32)
        f_data_ds = f_data[::rg_step].astype(np.float32)
        geom_imgs.append(np.tile(f_geom_ds, (rows_per_burst, 1)))
        data_imgs.append(np.tile(f_data_ds, (rows_per_burst, 1)))
        diff_imgs.append(np.tile((f_data_ds - f_geom_ds), (rows_per_burst, 1)))
        if burst_idx < len(annot.bursts) - 1:
            geom_imgs.append(sep)
            data_imgs.append(sep)
            diff_imgs.append(sep)

        mid = annot.n_samples // 2
        print(f"{burst_idx:>5}  {dc_idx:>6}  {float(f_geom[mid]):>16.2f}  {float(f_data[mid]):>16.2f}  {float(f_data[mid] - f_geom[mid]):>+14.2f}")

    arrays = [np.vstack(geom_imgs), np.vstack(data_imgs), np.vstack(diff_imgs)]
    titles = ["f_geom — orbit predicted", "f_data — ESA measured", "f_data − f_geom"]
    fig, axes = plt.subplots(1, 3, figsize=(16, max(6, len(annot.bursts))), sharex=True, sharey=True)
    for ax, arr, title in zip(axes, arrays, titles):
        finite = arr[np.isfinite(arr)]
        vmax = float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0
        im = ax.imshow(arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("range sample")
        plt.colorbar(im, ax=ax, label="Hz", shrink=0.6)
    axes[0].set_ylabel("azimuth line (bursts stacked)")
    fig.suptitle(f"{subswath.upper()} {polarisation.upper()} — annotation DC polynomials")
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


def diagnose_mispointing_aux_ins(
    slc_safe: str,
    subswath: str,
    aux_ins_path: str,
    poeorb_path: str | None = None,
    polarisation: str = "vv",
    out_path: str | None = None,
) -> None:
    """Compare per-burst attitude-derived mispointing Doppler against annotation DC."""
    import matplotlib.pyplot as plt

    files = find_safe_files(slc_safe, subswath, polarisation)
    annot_orig = parse_annotation(files["annotation"])
    annot = apply_poeorb(annot_orig, poeorb_path) if poeorb_path else annot_orig
    ins = parse_aux_ins(aux_ins_path)
    print(f"AUX_INS  referenceAntennaAngle = {ins.reference_antenna_angle_deg:.2f}°\n")

    n_rg = annot.n_samples
    tau = annot.slant_range_time_start + np.arange(n_rg) / annot.range_sampling_rate
    mid = n_rg // 2

    f_miss_arr, f_data_arr, f_geom_arr = [], [], []
    for burst_idx, burst in enumerate(annot.bursts):
        dc = _nearest_estimate(annot.dc_estimates, burst.azimuth_time)
        dt_mid = tau[mid] - dc.t0
        f_geom = float(sum(c * dt_mid**k for k, c in enumerate(dc.geometry_poly)))
        f_data = float(sum(c * dt_mid**k for k, c in enumerate(dc.data_poly)))
        f_miss = mispointing_doppler_from_yaw(annot, burst_idx)
        f_miss_arr.append(f_miss)
        f_data_arr.append(f_data)
        f_geom_arr.append(f_geom)

    f_miss_arr = np.array(f_miss_arr)
    f_data_arr = np.array(f_data_arr)
    f_geom_arr = np.array(f_geom_arr)
    f_dca_arr = f_data_arr - f_geom_arr
    residual = f_dca_arr - f_miss_arr

    hdr = f'{"burst":>5}  {"f_miss [Hz]":>12}  {"f_data [Hz]":>12}  {"f_geom [Hz]":>12}  {"f_dca [Hz]":>11}  {"residual [Hz]":>14}'
    print(hdr)
    print("─" * len(hdr))
    for i in range(len(annot.bursts)):
        print(f"{i:>5}  {f_miss_arr[i]:>12.2f}  {f_data_arr[i]:>12.2f}  {f_geom_arr[i]:>12.2f}  {f_dca_arr[i]:>11.2f}  {residual[i]:>+14.2f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    bursts = np.arange(len(annot.bursts))
    ax1.plot(bursts, f_data_arr, "o-", label="f_data (annotation)")
    ax1.plot(bursts, f_geom_arr, "s--", label="f_geom (orbit)")
    ax1.plot(bursts, f_miss_arr, "^-", label="f_miss (AUX_INS attitude)")
    ax1.set_xlabel("Burst index")
    ax1.set_ylabel("Doppler [Hz]")
    ax1.set_title("Doppler components per burst")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.bar(bursts - 0.2, f_dca_arr, width=0.4, label="f_dca = f_data − f_geom", alpha=0.7)
    ax2.bar(bursts + 0.2, f_miss_arr, width=0.4, label="f_miss (AUX_INS)", alpha=0.7)
    ax2.set_xlabel("Burst index")
    ax2.set_ylabel("Doppler [Hz]")
    ax2.set_title("f_dca vs mispointing Doppler")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"{subswath.upper()} {polarisation.upper()} — mispointing diagnostic (AUX_INS)")
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
