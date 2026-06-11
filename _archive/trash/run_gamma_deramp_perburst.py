def run_gamma_deramp_perburst(
    slc_safe:         str,
    subswath:         str,
    gamma_deramp_slc: str,
    gamma_deramp_par: str,
    poeorb_path:      str,
    aux_cal_path:     str,
    ocn_safe:         str,
    era5_wind:        str,
    era5_wave:        str,
    glo12:            str,
    polarisation:     str = 'vv',
    block_az:         int = 256,
    block_rg:         int = 512,
    stride_az:        int = 128,
    stride_rg:        int = 256,
    estimator:        str = 'fft',
    do_descallop:     bool = True,
    burst_indices:    list[int] | None = None,
) -> list[dict]:
    """
    Per-burst RVL pipeline on GAMMA-deramped data — mosaic (stitch) last.

    Reads the GAMMA ``SLC_deramp_ScanSAR`` burst-format output, slices out each
    burst, and runs :func:`run_pipeline` on it with ``deramped_override`` set —
    so the deramping is GAMMA's, but the Doppler estimation, geometry
    subtraction and all downstream corrections are ours.  No estimation block
    crosses a burst boundary; the per-burst result dicts are returned as a list
    so the bursts are only mosaicked (stitched) at the end by
    :func:`merge_burst_grids` / :func:`plot_comparison`.

    Parameters
    ----------
    gamma_deramp_slc, gamma_deramp_par : str
        GAMMA deramped burst-format SLC and its ``.par`` (``*.deramp.slc`` /
        ``*.deramp.slc.par``).
    estimator : 'fft' or 'cde'
        Our Doppler estimator forwarded to :func:`run_pipeline`.
    burst_indices : list of int or None
        Which bursts to process; all if None.

    Returns
    -------
    list[dict]
        One :func:`run_pipeline` result dict per burst.
    """
    from ..gamma_io import read_slc as _read_slc

    files = find_safe_files(slc_safe, subswath, polarisation)
    annot = parse_annotation(files['annotation'])
    n_bursts = len(annot.bursts)
    lpb = annot.lines_per_burst

    if burst_indices is None:
        burst_indices = list(range(n_bursts))

    print(f'gamma_deramp_perburst: reading GAMMA-deramped {subswath.upper()} '
          f'from {gamma_deramp_slc} …')
    full = _read_slc(gamma_deramp_slc, gamma_deramp_par).astype(np.complex64)
    expected = n_bursts * lpb
    if full.shape[0] != expected:
        raise ValueError(
            f'GAMMA deramp SLC has {full.shape[0]} azimuth lines, expected '
            f'{expected} = {n_bursts} bursts x {lpb} lines/burst'
        )

    results = []
    for bidx in burst_indices:
        print(f'\n─── Burst {bidx} / {n_bursts - 1} ───────────────────────────')
        burst_arr = full[bidx * lpb:(bidx + 1) * lpb, :].copy()
        results.append(run_pipeline(
            slc_safe=slc_safe, subswath=subswath, burst_idx=bidx,
            poeorb_path=poeorb_path, aux_cal_path=aux_cal_path,
            ocn_safe=ocn_safe, era5_wind=era5_wind, era5_wave=era5_wave,
            glo12=glo12, polarisation=polarisation,
            block_az=block_az, block_rg=block_rg,
            stride_az=stride_az, stride_rg=stride_rg,
            do_descallop=do_descallop, estimator=estimator,
            deramped_override=burst_arr,
        ))

    del full
    return results


