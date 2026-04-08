"""
Sentinel-1 Ocean Radial Velocity Layer (RVL) from SLC bursts.

Implements the ESA OCN RVL algorithm described in:
    Engen & Johnsen, "Sentinel-1 Doppler and Ocean Radial Velocity Algorithm
    Definition", DI-MPC-RVL-0534, NORCE/ESA, 2024.

Pipeline (Section 5 of the document):

  Step I   : Deramp and merge TOPS bursts into a continuous complex image
             (Section 5.4).  The TOPS antenna sweep imposes a frequency–time
             tilt on each burst; deramping removes it so the azimuth spectrum
             is centred at zero for a stationary target.

  Step II  : Estimate azimuth lag-0 and lag-1 correlation coefficients (p0,
             p1) per estimation block (Section 5.5).  The phase of p1 encodes
             the Doppler centroid of the block.

  Step III : Convert the Doppler centroid to radial velocity (Section 5.7).
             The antenna-model sideband correction (Section 5.6) is omitted
             because it requires ESA auxiliary files (LOP_PAT, LOP_COE,
             AUX_ECE) not present in the standard .SAFE product.  Over
             homogeneous ocean the simplified estimator is unbiased; near
             sharp backscatter gradients a small range-gradient bias may appear.

  Step IV  : Descallop burst-periodic artefacts from the Doppler grid
             (Appendix B / Section 5.8).

  Step V   : Geolocate the output grid (Section 5.10).
"""

import numpy as np
import xarray as xr
from scipy.interpolate import RectBivariateSpline

from .s1_io import (
    S1Annotation,
    find_safe_files,
    parse_annotation,
    read_slc_burst,
    slant_range_time_vector,
    _nearest_estimate,
)
from .io import read_slc, parse_slc_par

C_LIGHT = 299_792_458.0    # m/s


# ─────────────────────────────────────────────────────────────────────────────
# Deramp parameters
# ─────────────────────────────────────────────────────────────────────────────

def _fm_rate_at_burst(annot: S1Annotation, burst_idx: int) -> np.ndarray:
    """
    Azimuth FM rate k_a(τ_range) [Hz/s] for one burst.

    Uses the nearest azimuthFmRate record to the burst mid-time.  k_a is
    negative for typical side-looking geometry (≈ −2200 Hz/s for IW).

    Returns
    -------
    np.ndarray, shape (n_samples,)
    """
    burst = annot.bursts[burst_idx]
    afr   = _nearest_estimate(annot.azimuth_fm_rates, burst.azimuth_time)
    tau   = slant_range_time_vector(annot)          # (n_samples,)
    dt    = tau - afr.t0
    return sum(c * dt**k for k, c in enumerate(afr.poly))  # Hz/s


def _steering_doppler_rate(annot: S1Annotation) -> float:
    """
    Doppler rate k_psi [Hz/s] induced by the TOPS antenna steering.

    k_psi = (2 f_0 / c) * v_eff * |ψ̇|

    where v_eff ≈ azimuth_pixel_spacing × PRF is the effective ground speed
    and ψ̇ is the physical steering rate in rad/s.  k_psi > 0 for forward
    steering (IW ≈ 7 500 Hz/s).
    """
    v_eff    = annot.azimuth_pixel_spacing * annot.prf           # m/s
    psi_dot  = abs(annot.azimuth_steering_rate) * np.pi / 180    # rad/s
    return 2.0 * annot.radar_frequency / C_LIGHT * v_eff * psi_dot


def _deramp_rate(annot: S1Annotation, burst_idx: int) -> np.ndarray:
    """
    Per-range-sample deramp chirp rate k_s(τ) [Hz/s].

    From Miranda (2017) TOPS SLC deramping definition (eq. 2 in the RVL doc):

        k_s = −k_a · k_psi / (k_a − k_psi)

    k_a < 0, k_psi > 0  →  k_s < 0  (downchirp removes the TOPS upswing).

    Returns
    -------
    np.ndarray, shape (n_samples,)
    """
    k_a   = _fm_rate_at_burst(annot, burst_idx)    # < 0
    k_psi = _steering_doppler_rate(annot)           # > 0
    return -k_a * k_psi / (k_a - k_psi)            # < 0, varies with range


# ─────────────────────────────────────────────────────────────────────────────
# Step I — Deramp + window + merge  (Section 5.4)
# ─────────────────────────────────────────────────────────────────────────────

def deramp_burst(burst: np.ndarray, annot: S1Annotation, burst_idx: int) -> np.ndarray:
    """
    Apply the TOPS deramp chirp to one complex burst (eq. 1–3 in the doc).

        I_cbd(t_az, τ) = c1(t_az, τ) · I_cb(t_az, τ)
        c1(t_az, τ)   = exp(iπ k_s(τ) t_az²)

    where t_az is azimuth time measured from the burst centre [s].

    Parameters
    ----------
    burst     : complex64, shape (linesPerBurst, samplesPerBurst)
    annot     : S1Annotation
    burst_idx : int

    Returns
    -------
    complex64, same shape
    """
    lpb = annot.lines_per_burst
    k_s = _deramp_rate(annot, burst_idx)                               # (n_samples,)
    t_az = (np.arange(lpb) - lpb / 2.0) * annot.azimuth_time_interval # (lpb,) [s]

    # Broadcast to (lpb, n_samples)
    chirp = np.exp(1j * np.pi * k_s[np.newaxis, :] * t_az[:, np.newaxis] ** 2)
    return (burst * chirp).astype(np.complex64)


def _burst_hanning(lpb: int) -> np.ndarray:
    """
    Hanning window in azimuth time, normalised to approximate the partition-
    of-unity condition Σ_j w²(τ − τ̄_j) = 1 at the burst-overlap region
    (eq. 5 in the doc).

    For IW's ≈ 30 % burst overlap, dividing by sqrt(mean(w²)) over the
    overlap region gives a close-enough approximation for Doppler estimation.
    """
    w = np.hanning(lpb).astype(np.float32)
    # Normalise so that two windows overlapping at 50 % square-sum to unity.
    w /= np.sqrt(np.mean(w ** 2))
    return w


def merge_bursts(annot: S1Annotation, measurement_path: str) -> np.ndarray:
    """
    Read, deramp, window and coherently merge all TOPS bursts (eq. 6).

    Each burst is multiplied by a Hanning window before being added to the
    output array; overlapping regions from adjacent bursts blend smoothly.

    Returns
    -------
    I_c : complex64, shape (n_bursts × linesPerBurst, samplesPerBurst)
    """
    n_bursts = len(annot.bursts)
    lpb      = annot.lines_per_burst
    n_rg     = annot.samples_per_burst
    w        = _burst_hanning(lpb)

    I_c = np.zeros((n_bursts * lpb, n_rg), dtype=np.complex64)

    for j in range(n_bursts):
        raw      = read_slc_burst(measurement_path, annot, j)
        deramped = deramp_burst(raw, annot, j)
        I_c[j * lpb: (j + 1) * lpb, :] += (deramped * w[:, np.newaxis]).astype(np.complex64)

    return I_c


# ─────────────────────────────────────────────────────────────────────────────
# Step II — Azimuth correlation coefficients  (Section 5.5)
# ─────────────────────────────────────────────────────────────────────────────

def _block_p0_p1(block: np.ndarray) -> tuple[float, complex]:
    """
    Lag-0 and lag-1 azimuth correlation coefficients for one tile.

    The lag-1 estimator is the time-domain equivalent of the first inverse
    Fourier coefficient of the azimuth power spectrum (eq. 10, 13):

        p0 = E[|I|²]
        p1 = E[I(τ) I*(τ−1)]   (lag−1 in azimuth samples)

    The phase of p1 gives the mean Doppler shift:
        f_dc = PRF · arg(p1) / (2π)
    """
    p0 = float(np.mean(np.abs(block) ** 2))
    p1 = complex(np.mean(block[1:, :] * np.conj(block[:-1, :])))
    return p0, p1


def estimate_correlation_grid(
    I_c: np.ndarray,
    block_az: int = 256,
    block_rg: int = 512,
    stride_az: int = 128,
    stride_rg: int = 256,
    min_valid_fraction: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Slide an estimation window over the merged image and compute (p0, p1).

    Blocks where fewer than *min_valid_fraction* of pixels are non-zero
    (i.e. mostly invalid SLC samples) are set to NaN.

    Parameters
    ----------
    I_c                 : complex64, shape (n_az, n_rg)
    block_az, block_rg  : estimation block dimensions [samples]
    stride_az, stride_rg: block stride [samples]
    min_valid_fraction  : skip blocks with too many zero pixels

    Returns
    -------
    p0_grid  : float32,   shape (n_out_az, n_out_rg)
    p1_grid  : complex64, shape (n_out_az, n_out_rg)
    az_centers : int array, azimuth pixel centres in I_c
    rg_centers : int array, range pixel centres in I_c
    """
    n_az, n_rg = I_c.shape

    az_starts = np.arange(0, n_az - block_az + 1, stride_az)
    rg_starts = np.arange(0, n_rg - block_rg + 1, stride_rg)

    n_out_az = len(az_starts)
    n_out_rg = len(rg_starts)

    p0_grid = np.full((n_out_az, n_out_rg), np.nan, dtype=np.float32)
    p1_grid = np.zeros((n_out_az, n_out_rg), dtype=np.complex64)

    for i, az0 in enumerate(az_starts):
        for j, rg0 in enumerate(rg_starts):
            block = I_c[az0: az0 + block_az, rg0: rg0 + block_rg]

            valid_frac = np.count_nonzero(block) / block.size
            if valid_frac < min_valid_fraction:
                continue

            p0, p1 = _block_p0_p1(block)
            p0_grid[i, j] = p0
            p1_grid[i, j] = p1

    az_centers = az_starts + block_az // 2
    rg_centers = rg_starts + block_rg // 2

    return p0_grid, p1_grid, az_centers, rg_centers


# ─────────────────────────────────────────────────────────────────────────────
# Step III — Parameter estimation  (Section 5.7)
# ─────────────────────────────────────────────────────────────────────────────

def correlation_to_doppler(
    p0: np.ndarray,
    p1: np.ndarray,
    prf: float,
    wavelength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert correlation coefficients to Doppler frequency and radial velocity.

    Doppler centroid (eq. 20, 27 simplified — no geometric or mispointing
    corrections since both are set to zero per Section 4):

        f_dc = PRF · arg(p1) / (2π)
        v_r  = (λ / 2) · f_dc

    SNR from the lag-1 coherence magnitude ρ = |p1| / p0:

        SNR = ρ / (1 − ρ)

    This exploits the fact that noise is spatially uncorrelated (noise
    contributes to p0 but not to p1), so |p1| ≈ signal power.

    Parameters
    ----------
    p0, p1     : estimation grids
    prf        : PRF [Hz]
    wavelength : radar wavelength [m]

    Returns
    -------
    f_dc : float32 [Hz]
    v_r  : float32 [m/s]
    snr  : float32
    """
    f_dc = (prf / (2.0 * np.pi) * np.angle(p1)).astype(np.float32)
    v_r  = (wavelength / 2.0 * f_dc).astype(np.float32)

    with np.errstate(invalid='ignore', divide='ignore'):
        rho = np.abs(p1) / np.where(p0 > 0, p0, np.nan)
        snr = (rho / np.where(rho < 1.0, 1.0 - rho, np.nan)).astype(np.float32)

    return f_dc, v_r, snr


# ─────────────────────────────────────────────────────────────────────────────
# Step IV — Descalloping  (Section 5.8, Appendix B)
# ─────────────────────────────────────────────────────────────────────────────

def descallop(
    f_dc: np.ndarray,
    snr: np.ndarray,
    burst_period_rows: float,
    n_harmonics: int = 6,
    beta: float = 2.0,
) -> np.ndarray:
    """
    Remove burst-periodic scalloping from the Doppler grid (Appendix B).

    The scalloping arises because the input SL2 / SLC may not have been
    processed at full bandwidth.  It appears as an azimuth-periodic oscillation
    with period equal to the burst length.

    Algorithm (5 steps from Appendix B):

    1. Compute SNR-weighted mean Doppler profile along azimuth; subtract mean.
    2. FFT the profile.
    3. Detect the scalloping fundamental frequency and n_harmonics harmonics.
    4. Build a sinusoidal correction profile (amplitude scaled by beta = 2.0).
    5. Subtract from every range column of the Doppler grid.

    Parameters
    ----------
    f_dc             : float32 (n_az, n_rg)  Doppler centroid grid [Hz]
    snr              : float32 (n_az, n_rg)  SNR weights (NaN → zero weight)
    burst_period_rows: float  burst period in output-grid rows
    n_harmonics      : int    number of harmonics to remove (default 6)
    beta             : float  amplitude scale factor (default 2.0, per doc)

    Returns
    -------
    float32 (n_az, n_rg)  descalloped Doppler grid [Hz]
    """
    n_az, n_rg = f_dc.shape

    # Step 1: SNR-weighted mean azimuth profile, mean removed
    weights = np.where(np.isfinite(snr) & (snr > 0), snr, 0.0)
    w_sum   = np.maximum(weights.sum(axis=1), 1e-9)[:, np.newaxis]
    profile = np.nansum(np.where(np.isfinite(f_dc), f_dc * weights, 0.0), axis=1) / w_sum[:, 0]
    profile -= np.nanmean(profile)

    # Step 2: FFT
    F    = np.fft.fft(profile)
    freq = np.fft.fftfreq(n_az)                      # cycles per output row

    f0 = 1.0 / burst_period_rows                     # fundamental scalloping freq

    # Steps 3 + 4: detect each harmonic and build correction
    correction = np.zeros(n_az, dtype=np.float64)
    y = np.arange(n_az)

    for n in range(1, n_harmonics + 1):
        fn = n * f0
        # Search in a ±10 % window around the expected harmonic
        mask = (np.abs(freq - fn) < 0.1 * fn) | (np.abs(freq + fn) < 0.1 * fn)
        if not mask.any():
            # Fallback: nearest bin
            idx = int(np.argmin(np.abs(freq - fn)))
        else:
            idx = int(np.argmax(np.abs(F) * mask))

        an   = 2.0 * np.abs(F[idx]) / n_az           # one-sided amplitude [Hz]
        phi  = np.angle(F[idx])
        fn_actual = freq[idx]

        correction += beta * an * np.cos(2.0 * np.pi * fn_actual * y + phi)

    # Step 5: subtract from every range column
    return (f_dc - correction[:, np.newaxis]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Step V — Geolocation  (Section 5.10)
# ─────────────────────────────────────────────────────────────────────────────

def _geolocate_grid(
    annot: S1Annotation,
    az_centers: np.ndarray,
    rg_centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bilinearly interpolate lat, lon, incidence angle at block-centre positions.

    Parameters
    ----------
    az_centers : (n_az_out,)  azimuth pixel indices in the merged image
    rg_centers : (n_rg_out,)  range pixel indices

    Returns
    -------
    lat, lon, inc_angle : each float32, shape (n_az_out, n_rg_out)
    """
    g      = annot.geoloc_grid
    result = []

    for field in ('latitude', 'longitude', 'incidence_angle'):
        spline = RectBivariateSpline(
            g['line'].astype(float),
            g['pixel'].astype(float),
            g[field].astype(float),
            kx=1, ky=1,
        )
        result.append(
            spline(az_centers.astype(float), rg_centers.astype(float)).astype(np.float32)
        )

    return result[0], result[1], result[2]


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def compute_rvl(
    safe_dir: str,
    subswath: str,
    polarisation: str = 'vv',
    block_az: int = 256,
    block_rg: int = 512,
    stride_az: int = 128,
    stride_rg: int = 256,
    do_descallop: bool = True,
) -> xr.Dataset:
    """
    Compute the Radial Velocity Layer (RVL) from Sentinel-1 IW SLC bursts.

    Follows the ESA OCN RVL algorithm (Engen & Johnsen, DI-MPC-RVL-0534).
    Each burst is processed independently so no estimation window ever crosses
    a burst boundary (which would mix samples deramped with different chirp
    rates and produce stripe artefacts).

    Parameters
    ----------
    safe_dir      : str   Path to the .SAFE directory.
    subswath      : str   'iw1', 'iw2', or 'iw3'.
    polarisation  : str   'vv' (default) or 'vh'.
    block_az      : int   Estimation block height in azimuth samples.
    block_rg      : int   Estimation block width in range samples.
    stride_az     : int   Block stride in azimuth.
    stride_rg     : int   Block stride in range.
    do_descallop  : bool  Apply burst-periodic scalloping correction.

    Returns
    -------
    xr.Dataset with variables:
        doppler_hz  [Hz]    Estimated Doppler centroid per block
        radial_vel  [m/s]   Radial surface velocity
        snr         [-]     Signal-to-noise ratio estimate
    Coordinates:
        latitude, longitude, incidence_angle  (block-centre positions)
        az_pixel, rg_pixel                   (pixel indices in merged image)
    """
    files = find_safe_files(safe_dir, subswath, polarisation)
    annot = parse_annotation(files['annotation'])

    # Process each burst independently.  Deramping removes the TOPS azimuth
    # ramp per-burst; estimation within each burst only avoids cross-burst
    # correlation bias.
    p0_list, p1_list, az_centers_list = [], [], []
    rg_centers = None
    f_dc_list  = []

    for j in range(len(annot.bursts)):
        # Step I — read and deramp this burst (Section 5.4)
        raw      = read_slc_burst(files['measurement'], annot, j)
        deramped = deramp_burst(raw, annot, j)

        # Step II — azimuth correlation within this burst only (Section 5.5)
        p0_j, p1_j, az_loc, rg_centers_j = estimate_correlation_grid(
            deramped, block_az, block_rg, stride_az, stride_rg,
        )
        if rg_centers is None:
            rg_centers = rg_centers_j

        # Step III — Doppler centroid then subtract per-burst geometry.
        f_dc_j, _, _ = correlation_to_doppler(p0_j, p1_j, annot.prf, annot.wavelength)
        f_geom_j = _geom_doppler_annotation(annot, j, rg_centers_j).astype(np.float32)
        f_dc_j  -= f_geom_j[np.newaxis, :]

        p0_list.append(p0_j)
        p1_list.append(p1_j)
        f_dc_list.append(f_dc_j)
        az_centers_list.append(az_loc + j * annot.lines_per_burst)

    p0         = np.concatenate(p0_list,        axis=0)
    p1         = np.concatenate(p1_list,        axis=0)
    f_dc       = np.concatenate(f_dc_list,      axis=0)
    az_centers = np.concatenate(az_centers_list, axis=0)

    _, _, snr = correlation_to_doppler(p0, p1, annot.prf, annot.wavelength)
    v_r = (annot.wavelength / 2.0 * f_dc).astype(np.float32)

    # Step IV
    if do_descallop:
        burst_period_rows = annot.lines_per_burst / stride_az
        f_dc = descallop(f_dc, snr, burst_period_rows)
        v_r  = (annot.wavelength / 2.0 * f_dc).astype(np.float32)

    # Step V
    lat, lon, inc = _geolocate_grid(annot, az_centers, rg_centers)

    dims = ('az_cell', 'rg_cell')

    return xr.Dataset(
        {
            'doppler_hz': xr.DataArray(
                f_dc, dims=dims,
                attrs={'long_name': 'Estimated Doppler centroid', 'units': 'Hz'},
            ),
            'radial_vel': xr.DataArray(
                v_r, dims=dims,
                attrs={'long_name': 'Radial velocity', 'units': 'm s-1'},
            ),
            'snr': xr.DataArray(
                snr, dims=dims,
                attrs={'long_name': 'Signal-to-noise ratio estimate'},
            ),
        },
        coords={
            'latitude':        (dims, lat),
            'longitude':       (dims, lon),
            'incidence_angle': (dims, inc),
            'az_pixel':        ('az_cell', az_centers),
            'rg_pixel':        ('rg_cell', rg_centers),
        },
        attrs={
            'subswath':            annot.subswath,
            'polarisation':        annot.polarisation,
            'radar_frequency_hz':  annot.radar_frequency,
            'wavelength_m':        annot.wavelength,
            'prf_hz':              annot.prf,
            'block_az_samples':    block_az,
            'block_rg_samples':    block_rg,
            'stride_az':           stride_az,
            'stride_rg':           stride_rg,
            'sideband_correction': 'not applied (no antenna aux files)',
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# GAMMA-format entry point (pre-deramped per-burst .slc files)
# ─────────────────────────────────────────────────────────────────────────────

def _geom_doppler_annotation(
    annot: S1Annotation,
    burst_idx: int,
    rg_centers: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the annotation geometry Doppler polynomial at range pixel centres.

    Uses the DcEstimate nearest to the burst azimuth time.  The polynomial
    variable is slant-range time [s] relative to the reference t0.

    Returns
    -------
    np.ndarray, shape (len(rg_centers),) — geometric Doppler [Hz]
    """
    burst = annot.bursts[burst_idx]
    dc    = _nearest_estimate(annot.dc_estimates, burst.azimuth_time)
    tau   = annot.slant_range_time_start + rg_centers / annot.range_sampling_rate
    dt    = tau - dc.t0
    return sum(c * dt ** k for k, c in enumerate(dc.geometry_poly))


def _geom_doppler_at_pixels(par_path: str, rg_centers: np.ndarray) -> np.ndarray:
    """
    Evaluate the GAMMA static geometric Doppler polynomial at range pixel centres.

    After TOPS deramping, the azimuth-varying ramp is gone, but a residual
    range-dependent Doppler (due to satellite velocity + look angle) remains,
    encoded in ``doppler_polynomial`` of the GAMMA .slc.par.  The polynomial
    variable is slant-range offset from near range in **metres**.

    Returns
    -------
    np.ndarray, shape (len(rg_centers),)  — geometric Doppler [Hz]
    """
    par    = parse_slc_par(par_path)
    dr     = float(par['range_pixel_spacing'][0])
    coeffs = np.array(par['doppler_polynomial'][:4], dtype=np.float64)
    rg_m   = rg_centers.astype(np.float64) * dr
    return sum(c * rg_m ** k for k, c in enumerate(coeffs))


def load_gamma_bursts(slc_paths: list[str], slc_par_paths: list[str]) -> np.ndarray:
    """
    Read and concatenate GAMMA-format per-burst SLC files into one image.

    GAMMA's ``ScanSAR_burst_copy`` produces big-endian float32 complex binary
    files that are **already deramped** (the TOPS phase ramp is removed during
    focusing).  No further deramping is needed.

    Each burst is read via ``io.read_slc`` and stacked in azimuth order.

    Parameters
    ----------
    slc_paths     : list of str   Paths to per-burst .slc binary files.
    slc_par_paths : list of str   Corresponding .slc.par metadata files.

    Returns
    -------
    complex64, shape (n_bursts × lines_per_burst, range_samples)
    """
    bursts = [read_slc(slc, par).astype(np.complex64)
              for slc, par in zip(slc_paths, slc_par_paths)]
    return np.concatenate(bursts, axis=0)


def compute_rvl_gamma(
    slc_paths: list[str],
    slc_par_paths: list[str],
    annotation_xml: str,
    block_az: int = 256,
    block_rg: int = 512,
    stride_az: int = 128,
    stride_rg: int = 256,
    do_descallop: bool = True,
) -> xr.Dataset:
    """
    Compute the Radial Velocity Layer from GAMMA-produced per-burst SLC files.

    Each burst is processed independently so no estimation window ever crosses
    a burst boundary (which would mix phase discontinuities from separate GAMMA
    focusing runs and produce stripe artefacts).  The per-burst grids are
    concatenated before descalloping and geolocation.

    Parameters
    ----------
    slc_paths     : list of str
        Per-burst .slc binary files in azimuth order (burst 1 first).
        Typically ``iw1_vv_burst1.slc``, …, ``iw1_vv_burst9.slc``.
    slc_par_paths : list of str
        Corresponding GAMMA .slc.par metadata files.
    annotation_xml : str
        Path to the .SAFE annotation XML for the same subswath/polarisation
        (needed for PRF, wavelength, geolocation grid).
        E.g. ``…/annotation/s1a-iw1-slc-vv-…-004.xml``.
    block_az, block_rg : int
        Estimation block dimensions [samples].
    stride_az, stride_rg : int
        Block stride [samples].
    do_descallop : bool
        Apply burst-periodic scalloping correction (default True).

    Returns
    -------
    xr.Dataset with variables ``doppler_hz``, ``radial_vel``, ``snr`` and
    coordinates ``latitude``, ``longitude``, ``incidence_angle``,
    ``az_pixel``, ``rg_pixel``.
    """
    annot = parse_annotation(annotation_xml)

    # Step I — read pre-deramped GAMMA bursts and process each independently.
    # Processing per-burst avoids estimation windows that straddle burst
    # boundaries, which would produce periodic stripe artefacts.
    p0_list, p1_list, az_centers_list = [], [], []
    rg_centers = None

    f_dc_list = []

    for j, (slc_path, par_path) in enumerate(zip(slc_paths, slc_par_paths)):
        burst = read_slc(slc_path, par_path).astype(np.complex64)
        lpb   = burst.shape[0]

        # Step II — correlation coefficients within this burst
        p0_j, p1_j, az_loc, rg_centers_j = estimate_correlation_grid(
            burst, block_az, block_rg, stride_az, stride_rg,
        )

        if rg_centers is None:
            rg_centers = rg_centers_j   # same for every burst

        # Step III (per burst) — raw Doppler then subtract static geometric term.
        # GAMMA deramping removes the TOPS azimuth ramp but leaves a residual
        # range-dependent geometric Doppler encoded in doppler_polynomial (.slc.par).
        f_dc_j, _, snr_j = correlation_to_doppler(p0_j, p1_j, annot.prf, annot.wavelength)
        f_geom_j = _geom_doppler_at_pixels(par_path, rg_centers_j).astype(np.float32)
        f_dca_j  = f_dc_j - f_geom_j[np.newaxis, :]   # DCA: ocean + noise only

        p0_list.append(p0_j)
        p1_list.append(p1_j)
        f_dc_list.append(f_dca_j)
        az_centers_list.append(az_loc + j * lpb)

    p0         = np.concatenate(p0_list,        axis=0)
    p1         = np.concatenate(p1_list,        axis=0)
    f_dc       = np.concatenate(f_dc_list,      axis=0)
    az_centers = np.concatenate(az_centers_list, axis=0)

    # SNR from the concatenated correlation grids (uses raw p0/p1 coherence)
    _, _, snr = correlation_to_doppler(p0, p1, annot.prf, annot.wavelength)
    v_r = (annot.wavelength / 2.0 * f_dc).astype(np.float32)

    # Step IV — descalloping
    if do_descallop:
        burst_period_rows = annot.lines_per_burst / stride_az
        f_dc = descallop(f_dc, snr, burst_period_rows)
        v_r  = (annot.wavelength / 2.0 * f_dc).astype(np.float32)

    # Step V — geolocation
    lat, lon, inc = _geolocate_grid(annot, az_centers, rg_centers)

    dims = ('az_cell', 'rg_cell')

    return xr.Dataset(
        {
            'doppler_hz': xr.DataArray(
                f_dc, dims=dims,
                attrs={'long_name': 'Estimated Doppler centroid', 'units': 'Hz'},
            ),
            'radial_vel': xr.DataArray(
                v_r, dims=dims,
                attrs={'long_name': 'Radial velocity', 'units': 'm s-1'},
            ),
            'snr': xr.DataArray(
                snr, dims=dims,
                attrs={'long_name': 'Signal-to-noise ratio estimate'},
            ),
        },
        coords={
            'latitude':        (dims, lat),
            'longitude':       (dims, lon),
            'incidence_angle': (dims, inc),
            'az_pixel':        ('az_cell', az_centers),
            'rg_pixel':        ('rg_cell', rg_centers),
        },
        attrs={
            'radar_frequency_hz':  annot.radar_frequency,
            'wavelength_m':        annot.wavelength,
            'prf_hz':              annot.prf,
            'block_az_samples':    block_az,
            'block_rg_samples':    block_rg,
            'stride_az':           stride_az,
            'stride_rg':           stride_rg,
            'sideband_correction': 'not applied (no antenna aux files)',
            'input_format':        'GAMMA per-burst SLC (pre-deramped)',
        },
    )
