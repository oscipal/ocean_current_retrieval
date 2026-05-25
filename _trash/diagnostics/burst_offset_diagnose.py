"""
Diagnose burst-to-burst Doppler offset.

Outputs per burst:
  - which DC record is selected and the time gap
  - mean f_dc (observed from deramped data)
  - mean f_geom (geometry polynomial)
  - mean f_dca = f_dc - f_geom
  - same for esa_eq1 deramp
  - with esa_eq1, also shows the eta_ref range-variation (near vs far range)

Run:
    python -m scripts.diagnostics.burst_offset_diagnose <slc_safe> [--subswath iw1] [--pol vv]
"""
from __future__ import annotations

import argparse
import numpy as np

from scripts.sentinel_1.safe_io import (
    find_safe_files,
    parse_annotation,
    read_slc_burst,
    slant_range_time_vector,
    _nearest_estimate,
)
from scripts.sentinel_1.rvl import (
    apply_burst_valid_sample_mask,
    correlation_to_doppler,
    deramp_burst,
    estimate_correlation_grid,
    _geom_doppler_annotation,
    _fm_rate_at_burst,
    _burst_mid_time,
)


def _nanmean(a: np.ndarray) -> float:
    v = a[np.isfinite(a)]
    return float(np.mean(v)) if v.size else float('nan')


def run(
    slc_safe: str,
    subswath: str,
    polarisation: str = 'vv',
    block_az: int = 256,
    block_rg: int = 512,
    stride_az: int = 128,
    stride_rg: int = 256,
) -> None:
    files = find_safe_files(slc_safe, subswath, polarisation)
    annot = parse_annotation(files['annotation'])

    n_bursts = len(annot.bursts)
    n_dc = len(annot.dc_estimates)
    n_fm = len(annot.azimuth_fm_rates)

    inter_burst_s = (
        (annot.bursts[1].azimuth_time - annot.bursts[0].azimuth_time).total_seconds()
        if n_bursts >= 2 else float('nan')
    )

    print(f"Subswath   : {subswath.upper()}")
    print(f"Bursts     : {n_bursts}")
    print(f"DC records : {n_dc}")
    print(f"FM records : {n_fm}")
    print(f"Inter-burst interval: {inter_burst_s:.4f} s")
    print(f"Half interval: {inter_burst_s/2:.4f} s (records within this are 'correct')")
    print()

    tau = slant_range_time_vector(annot)

    lpb = annot.lines_per_burst
    ati = annot.azimuth_time_interval
    inter_burst_lines = round(inter_burst_s / ati)
    overlap_lines = lpb - inter_burst_lines

    print(f"Lines per burst  : {lpb}")
    print(f"Inter-burst lines: {inter_burst_lines}  (= {inter_burst_s:.4f} s / {ati:.6f} s)")
    print(f"Overlap lines    : {overlap_lines}")
    print()

    # ── Header — Table 1: record selection + burst-mean Doppler ───────────────
    hdr1 = (
        "burst | dc_rec | dc_gap_s | fm_rec | fm_gap_s |"
        " f_dc_cur | f_geom | f_dca_cur |"
        " f_dc_esa | f_dca_esa"
    )
    print(hdr1)
    print('-' * len(hdr1))

    # ── Header — Table 2: range-profile Δf_dc (near − far) ───────────────────
    hdr2 = (
        "\nburst |  cur_near |  cur_mid  |  cur_far  | cur(near-far) "
        "| esa_near |  esa_mid  |  esa_far  | esa(near-far)"
    )

    rows_t1 = []
    rows_t2 = []
    rows_t3 = []

    # Keep previous burst's deramped arrays for overlap comparison
    prev_cur: np.ndarray | None = None
    prev_esa: np.ndarray | None = None

    for j, burst in enumerate(annot.bursts):
        raw = read_slc_burst(files['measurement'], annot, j)

        # ── DC / FM record selection ───────────────────────────────────────────
        diffs_dc = [abs((dc.azimuth_time - burst.azimuth_time).total_seconds())
                    for dc in annot.dc_estimates]
        best_dc = int(np.argmin(diffs_dc))
        dc_gap  = diffs_dc[best_dc]

        diffs_fm = [abs((fm.azimuth_time - burst.azimuth_time).total_seconds())
                    for fm in annot.azimuth_fm_rates]
        best_fm = int(np.argmin(diffs_fm))
        fm_gap  = diffs_fm[best_fm]

        dc_warn = " *" if dc_gap > inter_burst_s / 2 else ""
        fm_warn = " *" if fm_gap > inter_burst_s / 2 else ""

        # ── Current deramp ────────────────────────────────────────────────────
        deramped_cur = deramp_burst(raw, annot, j, deramp_method='current')
        apply_burst_valid_sample_mask(deramped_cur, burst)
        p0, p1, _, rg_c = estimate_correlation_grid(
            deramped_cur, block_az, block_rg, stride_az, stride_rg,
        )
        f_dc_cur, _, _ = correlation_to_doppler(p0, p1, annot.prf, annot.wavelength)
        rg_c_arr = np.asarray(rg_c, dtype=int)
        f_geom = _geom_doppler_annotation(annot, j, rg_c_arr)

        mean_f_dc_cur  = _nanmean(f_dc_cur)
        mean_f_geom    = _nanmean(f_geom)
        mean_f_dca_cur = mean_f_dc_cur - mean_f_geom

        # ── ESA eq1 deramp ────────────────────────────────────────────────────
        deramped_esa = deramp_burst(raw, annot, j, deramp_method='esa_eq1')
        apply_burst_valid_sample_mask(deramped_esa, burst)
        p0e, p1e, _, _ = estimate_correlation_grid(
            deramped_esa, block_az, block_rg, stride_az, stride_rg,
        )
        f_dc_esa, _, _ = correlation_to_doppler(p0e, p1e, annot.prf, annot.wavelength)

        mean_f_dc_esa  = _nanmean(f_dc_esa)
        mean_f_dca_esa = mean_f_dc_esa - mean_f_geom

        rows_t1.append(
            f"  {j:2d}  |  {best_dc:2d}{dc_warn:<2s}  | {dc_gap:7.4f}  "
            f"|  {best_fm:2d}{fm_warn:<2s}  | {fm_gap:7.4f}  "
            f"|  {mean_f_dc_cur:8.2f}  | {mean_f_geom:8.2f} | {mean_f_dca_cur:+8.2f}  "
            f"|  {mean_f_dc_esa:8.2f}  | {mean_f_dca_esa:+8.2f}"
        )

        # ── Range-profile: f_dc in near / mid / far thirds ────────────────────
        # Split the range estimation grid into thirds to expose the eta_ref slope.
        n_rg_cells = f_dc_cur.shape[1]
        t3 = n_rg_cells // 3
        if t3 < 1:
            rows_t2.append(f"  {j:2d}  | (too few range cells)")
        else:
            def _third_mean(arr2d, lo, hi):
                return _nanmean(arr2d[:, lo:hi])

            cur_near = _third_mean(f_dc_cur, 0,      t3)
            cur_mid  = _third_mean(f_dc_cur, t3,   2*t3)
            cur_far  = _third_mean(f_dc_cur, 2*t3, n_rg_cells)
            esa_near = _third_mean(f_dc_esa, 0,      t3)
            esa_mid  = _third_mean(f_dc_esa, t3,   2*t3)
            esa_far  = _third_mean(f_dc_esa, 2*t3, n_rg_cells)

            rows_t2.append(
                f"  {j:2d}  | {cur_near:+8.2f}  | {cur_mid:+8.2f}  | {cur_far:+8.2f}  "
                f"| {cur_near - cur_far:+10.2f}   "
                f"| {esa_near:+7.2f}  | {esa_mid:+8.2f}  | {esa_far:+8.2f}  "
                f"| {esa_near - esa_far:+9.2f}"
            )

        # ── Table 3: overlap-zone comparison ─────────────────────────────────
        # The last `overlap_lines` of burst j and the first `overlap_lines` of
        # burst j+1 image the same ground patch (~2.76 s apart).  If f_dc
        # differs significantly the per-burst offset is a processing artefact.
        if prev_cur is not None and overlap_lines > 0:
            ov = min(overlap_lines, deramped_cur.shape[0], prev_cur.shape[0])
            if ov >= 8:
                baz_ov = min(64, ov // 2)
                saz_ov = max(1, baz_ov // 2)

                # Last ov lines of previous burst (j-1)
                seg_prev_cur = prev_cur[-ov:, :]
                seg_prev_esa = prev_esa[-ov:, :]
                # First ov lines of current burst (j)
                seg_cur_cur  = deramped_cur[:ov, :]
                seg_cur_esa  = deramped_esa[:ov, :]

                def _ov_fdc(seg):
                    p0s, p1s, _, _ = estimate_correlation_grid(
                        seg, baz_ov, block_rg, saz_ov, stride_rg,
                    )
                    fdc, _, _ = correlation_to_doppler(p0s, p1s, annot.prf, annot.wavelength)
                    return _nanmean(fdc)

                fj_cur  = _ov_fdc(seg_prev_cur)
                fj1_cur = _ov_fdc(seg_cur_cur)
                fj_esa  = _ov_fdc(seg_prev_esa)
                fj1_esa = _ov_fdc(seg_cur_esa)

                rows_t3.append(
                    f" {j-1:2d}-{j:2d} |  {fj_cur:+8.2f}  |  {fj1_cur:+8.2f}  "
                    f"| {fj1_cur - fj_cur:+8.2f}   "
                    f"|  {fj_esa:+8.2f}  |  {fj1_esa:+8.2f}  "
                    f"| {fj1_esa - fj_esa:+8.2f}"
                )
            else:
                rows_t3.append(f" {j-1:2d}-{j:2d} | (overlap too small: {ov} lines)")

        prev_cur = deramped_cur
        prev_esa = deramped_esa

    for r in rows_t1:
        print(r)

    print(hdr2)
    print('-' * (len(hdr2) - 1))
    for r in rows_t2:
        print(r)

    if rows_t3:
        hdr3 = (
            "\npair  |  j_end(cur) | j+1_start(cur) |  Δf_cur   "
            "|  j_end(esa) | j+1_start(esa) |  Δf_esa"
        )
        print(hdr3)
        print('-' * (len(hdr3) - 1))
        for r in rows_t3:
            print(r)

    print()
    print("* = record selected from different burst (|Δt| > half inter-burst interval)")
    print("Table 2: 'near-far' = near-range f_dc minus far-range f_dc within each burst.")
    print("  With 'current' deramp this should be large (eta_ref slope ≈ 2×k_t×eta_ref_near).")
    print("  With 'esa_eq1'  this should be near zero (range bias corrected).")
    print("Table 3: overlap-zone comparison. Same ground, same time (~2.76 s apart).")
    print("  |Δf| < 2 Hz → burst variation is real ocean signal.")
    print("  |Δf| > 5 Hz → uniform per-burst offset is a PROCESSING ARTEFACT.")


def main() -> None:
    p = argparse.ArgumentParser(
        description='Diagnose burst-to-burst Doppler offset source.',
    )
    p.add_argument('slc_safe')
    p.add_argument('--subswath', default='iw1')
    p.add_argument('--pol', default='vv')
    p.add_argument('--block-az',  type=int, default=256)
    p.add_argument('--block-rg',  type=int, default=512)
    p.add_argument('--stride-az', type=int, default=128)
    p.add_argument('--stride-rg', type=int, default=256)
    args = p.parse_args()
    run(
        slc_safe=args.slc_safe,
        subswath=args.subswath,
        polarisation=args.pol,
        block_az=args.block_az,
        block_rg=args.block_rg,
        stride_az=args.stride_az,
        stride_rg=args.stride_rg,
    )


if __name__ == '__main__':
    main()
