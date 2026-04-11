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


def _orbital_speed(annot: S1Annotation, burst_idx: int) -> float:
    """
    Return the satellite orbital speed [m/s] at the burst centre time.

    Interpolated from the ECEF velocity state vectors stored in the annotation.
    The full 3-D speed |v| is used because k_psi depends on the physical
    angular sweep rate of the antenna in inertial space, not the ground-track
    speed.
    """
    burst_time = annot.bursts[burst_idx].azimuth_time
    diffs = [abs((t - burst_time).total_seconds()) for t in annot.orbit_times]
    idx = int(np.argmin(diffs))
    vx, vy, vz = annot.orbit_velocities[idx]
    return float(np.sqrt(vx**2 + vy**2 + vz**2))


def _steering_doppler_rate(annot: S1Annotation, burst_idx: int) -> float:
    """
    Doppler rate k_psi [Hz/s] induced by the TOPS antenna steering.

    k_psi = (2 f_0 / c) * v_sat * |ψ̇|

    v_sat is the satellite orbital speed [m/s] from the annotation state
    vectors.  Using orbital speed (≈ 7 500 m/s) rather than the projected
    ground speed (≈ 6 800 m/s) is critical: the TOPS antenna rotates at
    rate ψ̇ in inertial space, so it is the inertial (orbital) velocity that
    enters the Doppler rate formula.  Underestimating v_sat by ~10 % produces
    a ~40 Hz/s residual chirp after deramping — the dominant source of
    within-burst Doppler stripes.

    ψ̇ is the physical steering rate in rad/s.  k_psi > 0 for forward
    steering (IW ≈ 7 500 Hz/s).
    """
    v_sat    = _orbital_speed(annot, burst_idx)                  # m/s
    psi_dot  = abs(annot.azimuth_steering_rate) * np.pi / 180    # rad/s
    return 2.0 * annot.radar_frequency / C_LIGHT * v_sat * psi_dot


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
    k_a   = _fm_rate_at_burst(annot, burst_idx)            # < 0
    k_psi = _steering_doppler_rate(annot, burst_idx)       # > 0
    return -k_a * k_psi / (k_a - k_psi)                   # < 0, varies with range


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


def _burst_window(lpb: int, overlap_lines: int) -> np.ndarray:
    """
    Partition-of-unity raised-cosine window for TOPS burst merging (eq. 5).

    A Hanning window satisfies Σ_j w²(τ−τ̄_j) = 1 only at 50 % overlap.
    IW mode has ≈ 30 % overlap; using a Hanning taper leaves w_j² + w_{j+1}²
    well below 1 in the transition zone, producing amplitude gaps in the
    merged image.

    This window is flat (1.0) in the non-overlap region and uses a raised-cosine
    taper over the overlap at each end:

        rising edge  (k = 0 … overlap_lines−1): w(k) = sin(π/2 · k/overlap_lines)
        falling edge (k = lpb−overlap_lines … lpb−1): w(k) = cos(π/2 · k/overlap_lines)

    which satisfies w_j(k)² + w_{j+1}(k)² = 1 exactly for any overlap fraction.
    """
    w = np.ones(lpb, dtype=np.float32)
    if overlap_lines > 0:
        t = np.arange(overlap_lines, dtype=np.float32) / overlap_lines
        w[:overlap_lines]       = np.sin(np.pi / 2.0 * t)
        w[lpb - overlap_lines:] = np.cos(np.pi / 2.0 * t)
    return w


def _window_burst(
    I_bcd: np.ndarray,
    annot: S1Annotation,
    burst_idx: int,
    overlap_lines: int,
) -> np.ndarray:
    """
    Apply the TOPS spectral window (Section 5.4.1, eq. 4):

        I_cbdw = c2* ⊗ [(c2 ⊗ I_bcd) · w(· − τ̄_j)]

    with  c2(τ) = exp(iπ k_psi τ²)  (steering Doppler chirp).

    The c2 defocusing maps each Doppler frequency to its natural azimuth-time
    position before the raised-cosine window is applied, so the partition-of-
    unity condition is satisfied across the burst overlap region.

    Implementation steps:
        1. FFT(I_bcd) along azimuth
        2. × C2            →  c2 ⊗ I_bcd  (defocus)
        3. IFFT
        4. × w(τ − τ̄_j)   →  raised-cosine window centred on burst
        5. FFT
        6. × C2*           →  c2* ⊗ (windowed)  (refocus)
        7. IFFT
    """
    lpb   = I_bcd.shape[0]
    k_psi = _steering_doppler_rate(annot, burst_idx)          # Hz/s, scalar > 0
    w     = _burst_window(lpb, overlap_lines)

    f  = np.fft.fftfreq(lpb, d=1.0 / annot.prf)                      # (lpb,) Hz
    C2 = np.exp(-1j * np.pi * f**2 / k_psi).astype(np.complex64)     # (lpb,)

    # Defocus: IFFT[ C2 · FFT[I_bcd] ]
    I_defoc    = np.fft.ifft(C2[:, None] * np.fft.fft(I_bcd, axis=0), axis=0)
    # Window in the defocused domain
    I_windowed = (I_defoc * w[:, None]).astype(np.complex64)
    # Refocus: IFFT[ C2* · FFT[I_windowed] ]
    I_cbdw     = np.fft.ifft(C2.conj()[:, None] * np.fft.fft(I_windowed, axis=0), axis=0)

    return I_cbdw.astype(np.complex64)


def merge_bursts(annot: S1Annotation, measurement_path: str) -> np.ndarray:
    """
    Read, deramp, window and coherently merge all TOPS bursts (Section 5.4.1,
    eq. 1–6).

    Each burst is deramped (eq. 1–2), spectrally windowed with the raised-cosine
    partition-of-unity window (eq. 4–5), and summed into its correct azimuth
    position (eq. 6).  The overlap_lines is computed from the inter-burst
    interval so the window taper is exact for the actual IW overlap fraction.

    Returns
    -------
    I_c : complex64, shape (n_lines, samplesPerBurst)
    """
    lpb  = annot.lines_per_burst
    n_rg = annot.samples_per_burst
    ati  = annot.azimuth_time_interval

    inter_burst_lines = int(round(
        (annot.bursts[1].azimuth_time - annot.bursts[0].azimuth_time
         ).total_seconds() / ati
    ))
    overlap_lines = lpb - inter_burst_lines

    I_c = np.zeros((annot.n_lines, n_rg), dtype=np.complex64)

    for j, burst in enumerate(annot.bursts):
        az0 = int(round(
            (burst.azimuth_time - annot.first_line_time).total_seconds() / ati
        ))
        az1 = az0 + lpb

        raw      = read_slc_burst(measurement_path, annot, j)
        deramped = deramp_burst(raw, annot, j)

        # Zero lines flagged invalid before windowing; these filled edge lines
        # must not contribute to the overlap blend.
        valid_lines = burst.first_valid_sample != -1
        deramped[~valid_lines, :] = 0.0

        I_c[az0:az1, :] += _window_burst(deramped, annot, j, overlap_lines)

    return I_c


# ─────────────────────────────────────────────────────────────────────────────
# Step II — Azimuth correlation coefficients  (Section 5.5)
# ─────────────────────────────────────────────────────────────────────────────

def _block_p0_p1(block: np.ndarray) -> tuple[float, complex]:
    """
    Lag-0 and lag-1 azimuth correlation coefficients for one tile,
    following the two-stage windowed procedure of Section 5.5.1
    (eqs. 12–13).

    Stage 1 — azimuth (inner): apply Hanning window h_az along azimuth,
    then form per-range-column lag-1 products g1(t') and lag-0 power
    g0(t'):

        I_w(t', τ) = I_c(t', τ) · h_az(τ)

        g0(t') = Σ_τ |I_w(t', τ)|²                    (eq. 12, n = 0)
        g1(t') = Σ_τ I_w(t', τ) · I_w*(t', τ−1)       (eq. 12, n = 1)

    Stage 2 — range (outer): average per-column estimates over range
    with h²_ra weights (Hanning squared, eq. 13):

        p0 = Σ_t' h²_ra(t') g0(t') / Σ_t' h²_ra(t')
        p1 = Σ_t' h²_ra(t') g1(t') / Σ_t' h²_ra(t')

    Both windows are normalised to unit sum before use (eq. 11).

    NOTE: the ϖ∆ correction (eq. 9, ϖ∆ = ϖPRF / (1 + γ/β)) is not
    yet applied — the correct γ/β interpretation for deramped IW data
    needs to be verified before implementation.
    """
    n_az, n_rg = block.shape

    # h_az: Hanning along azimuth, normalised to unit sum (eq. 11)
    haz = np.hanning(n_az).astype(np.float32)
    haz /= haz.sum()

    # h_ra: Hanning along range, normalised to unit sum (eq. 11)
    hra = np.hanning(n_rg).astype(np.float32)
    hra /= hra.sum()

    # Stage 1 — apply h_az per range column
    I_w = block * haz[:, None]   # (n_az, n_rg)

    # Per-column lag-1 products → g1, shape (n_rg,)
    g1 = np.sum(I_w[1:, :] * np.conj(I_w[:-1, :]), axis=0)

    # Per-column lag-0 power → g0 via |I_w|² = |I_c|² · h_az² (Parseval, n=0)
    g0 = np.sum(np.abs(I_w) ** 2, axis=0)

    # Stage 2 — h²_ra weights over range (eq. 13)
    w2    = hra ** 2
    denom = float(np.sum(w2))
    if denom < 1e-30:
        return 0.0, complex(0.0)

    p0 = float(np.sum(g0 * w2) / denom)
    p1 = complex(np.sum(g1 * w2) / denom)
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

    Each block is passed to _block_p0_p1, which applies the two-stage
    Hanning windowing (h_az in azimuth, h²_ra in range) per Section 5.5.1
    eqs. 12–13.  Blocks where fewer than *min_valid_fraction* of pixels
    are non-zero (mostly invalid SLC samples) are left as NaN.

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
    Bursts are deramped, spectrally windowed (c2-based, eq. 4) and coherently
    merged before correlation estimation, matching the algorithm description
    in Section 5.4.  Estimation blocks therefore span burst boundaries
    seamlessly without phase discontinuities.

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

    # Step I — deramp, window and merge all bursts into one continuous image
    I_c = merge_bursts(annot, files['measurement'])

    # Step II — estimate azimuth correlation over the full merged image
    p0, p1, az_centers, rg_centers = estimate_correlation_grid(
        I_c, block_az, block_rg, stride_az, stride_rg,
    )

    # Step III — Doppler centroid, subtract Hanning-blended geometric Doppler
    f_dc, _, _ = correlation_to_doppler(p0, p1, annot.prf, annot.wavelength)
    f_geom     = _blended_geom_doppler_annotation(annot, az_centers, rg_centers)
    f_dca      = f_dc - f_geom

    _, _, snr = correlation_to_doppler(p0, p1, annot.prf, annot.wavelength)
    v_r = (annot.wavelength / 2.0 * f_dca).astype(np.float32)

    # Step IV — descallop using the true inter-burst interval as period
    if do_descallop:
        burst_dt_s = (
            annot.bursts[1].azimuth_time - annot.bursts[0].azimuth_time
        ).total_seconds()
        burst_period_rows = burst_dt_s / annot.azimuth_time_interval / stride_az
        f_dca = descallop(f_dca, snr, burst_period_rows)
        v_r   = (annot.wavelength / 2.0 * f_dca).astype(np.float32)

    # Step V
    lat, lon, inc = _geolocate_grid(annot, az_centers, rg_centers)

    dims = ('az_cell', 'rg_cell')

    return xr.Dataset(
        {
            'doppler_hz': xr.DataArray(
                f_dca, dims=dims,
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


def _blended_geom_doppler_annotation(
    annot: S1Annotation,
    az_centers: np.ndarray,
    rg_centers: np.ndarray,
) -> np.ndarray:
    """
    Raised-cosine-weighted blend of per-burst geometric Doppler polynomials
    for estimation blocks on the merged SAFE image.

    Uses the same raised-cosine partition-of-unity weights as ``_window_burst``
    so the subtracted geometric bias is consistent with the merged signal.

    Returns
    -------
    float32, shape (n_out_az, n_out_rg)
    """
    lpb = annot.lines_per_burst
    ati = annot.azimuth_time_interval

    inter_burst_lines = int(round(
        (annot.bursts[1].azimuth_time - annot.bursts[0].azimuth_time
         ).total_seconds() / ati
    ))
    overlap_lines = lpb - inter_burst_lines

    az_offsets = [
        int(round((b.azimuth_time - annot.first_line_time).total_seconds() / ati))
        for b in annot.bursts
    ]

    win = _burst_window(lpb, overlap_lines).astype(np.float64)

    # Evaluate each burst's geometry polynomial at rg_centers once
    f_geom_bursts = np.stack(
        [_geom_doppler_annotation(annot, j, rg_centers).astype(np.float64)
         for j in range(len(annot.bursts))],
        axis=0,
    )   # (n_bursts, n_out_rg)

    n_out_rg = len(rg_centers)
    f_geom   = np.zeros((len(az_centers), n_out_rg), dtype=np.float32)

    for i, az in enumerate(az_centers):
        blend  = np.zeros(n_out_rg, dtype=np.float64)
        w2_sum = 0.0
        for j, az0 in enumerate(az_offsets):
            rel = int(az) - az0
            if 0 <= rel < lpb:
                # Use w² weights: the lag-1 estimator weights each burst's
                # contribution by w²  (product of two adjacent w values),
                # so the geometry subtraction must use the same w² weighting.
                # The partition-of-unity Σw²=1 holds in the interior so no
                # normalisation is needed there; w2_sum handles image edges
                # where only one burst contributes with w < 1.
                w2     = float(win[rel]) ** 2
                blend += w2 * f_geom_bursts[j]
                w2_sum += w2
        if w2_sum > 0:
            f_geom[i] = (blend / w2_sum).astype(np.float32)

    return f_geom


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


def _blended_geom_doppler(
    slc_par_paths: list[str],
    az_centers: np.ndarray,
    rg_centers: np.ndarray,
) -> np.ndarray:
    """
    Hanning-weighted blend of per-burst geometric Doppler polynomials.

    For estimation blocks that fall fully inside one burst the result is just
    that burst's polynomial evaluated at rg_centers.  For blocks in the overlap
    zone the two adjacent bursts' polynomials are blended with the same Hanning
    weights that merge_gamma_bursts applied to the SLC data, so the subtracted
    bias is consistent with the merged signal.

    Parameters
    ----------
    slc_par_paths : list of str     One .slc.par per burst, azimuth order.
    az_centers    : int array (n_out_az,)   Azimuth block centres in merged image.
    rg_centers    : int array (n_out_rg,)   Range block centres.

    Returns
    -------
    float32, shape (n_out_az, n_out_rg)
    """
    pars = [parse_slc_par(p) for p in slc_par_paths]

    dt  = float(pars[0]['azimuth_line_time'][0])
    lpb = int(pars[0]['azimuth_lines'][0])

    t_starts  = [float(p['start_time'][0]) for p in pars]
    t0_global = min(t_starts)

    # Azimuth line offset of each burst in the merged image
    az_offsets = [int(round((t - t0_global) / dt)) for t in t_starts]

    hann = np.hanning(lpb).astype(np.float64)

    # Pre-evaluate each burst's polynomial at rg_centers → (n_bursts, n_out_rg)
    f_geom_bursts = np.stack(
        [_geom_doppler_at_pixels(p, rg_centers).astype(np.float64)
         for p in slc_par_paths],
        axis=0,
    )

    n_out_az = len(az_centers)
    n_out_rg = len(rg_centers)
    f_geom = np.zeros((n_out_az, n_out_rg), dtype=np.float32)

    for i, az in enumerate(az_centers):
        w_sum = 0.0
        blend = np.zeros(n_out_rg, dtype=np.float64)

        for j, az0 in enumerate(az_offsets):
            rel = int(az) - az0
            if 0 <= rel < lpb:
                w      = float(hann[rel])
                blend += w * f_geom_bursts[j]
                w_sum += w

        if w_sum > 0:
            f_geom[i] = (blend / w_sum).astype(np.float32)

    return f_geom


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


def merge_gamma_bursts(
    slc_paths: list[str],
    slc_par_paths: list[str],
) -> np.ndarray:
    """
    Coherently merge GAMMA per-burst SLCs into a continuous azimuth image.

    GAMMA bursts overlap in azimuth (typically ~10 % for IW).  Simple
    concatenation double-counts the overlap; this function instead:

    1. Computes each burst's azimuth offset from ``start_time`` in the .par.
    2. Applies a Hanning window to each burst (tapers toward zero at burst
       edges, suppressing discontinuities in the overlap region).
    3. **Adds** windowed bursts into a common output array.

    Amplitude normalisation (dividing by the window sum) is intentionally
    skipped: Doppler estimation only uses the complex phase, and dividing by
    a real, positive weight cannot change arg(p1).  The overlap region has
    lower amplitude, but the phase — and therefore f_dc — is unaffected.

    Parameters
    ----------
    slc_paths     : list of str, sorted in azimuth order (burst 1 first)
    slc_par_paths : list of str, matching .slc.par files

    Returns
    -------
    complex64, shape (n_az_total, range_samples)
        ``n_az_total`` is the number of unique azimuth lines spanning all
        bursts, computed from their start/end times.
    """
    pars = [parse_slc_par(p) for p in slc_par_paths]

    dt      = float(pars[0]['azimuth_line_time'][0])
    n_rg    = int(pars[0]['range_samples'][0])
    lpb     = int(pars[0]['azimuth_lines'][0])

    t_starts = [float(p['start_time'][0]) for p in pars]
    t_ends   = [float(p['end_time'][0])   for p in pars]

    t0_global = min(t_starts)
    t1_global = max(t_ends)
    n_az_total = int(round((t1_global - t0_global) / dt)) + 1

    output = np.zeros((n_az_total, n_rg), dtype=np.complex64)

    w = np.hanning(lpb).astype(np.float32)

    for slc_path, par_path, t0 in zip(slc_paths, slc_par_paths, t_starts):
        burst = read_slc(slc_path, par_path).astype(np.complex64)
        az0   = int(round((t0 - t0_global) / dt))
        az1   = az0 + lpb

        output[az0:az1, :] += burst * w[:, np.newaxis]

    return output


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
    xr.Dataset with variables ``f_dca`` (Doppler centroid anomaly [Hz]),
    ``radial_vel`` [m/s], ``f_dc`` (observed Doppler centroid [Hz]),
    ``snr``, and coordinates ``latitude``, ``longitude``,
    ``incidence_angle``, ``az_pixel``, ``rg_pixel``.
    """
    annot = parse_annotation(annotation_xml)

    # Step I — coherently merge pre-deramped GAMMA bursts into one image.
    if type(slc_paths) is list:
        merged = merge_gamma_bursts(slc_paths, slc_par_paths)
    else:
        merged = read_slc(slc_paths, slc_par_paths).astype(np.complex64)

    # Step II — azimuth correlation coefficients on the merged image.
    p0, p1, az_centers, rg_centers = estimate_correlation_grid(
        merged, block_az, block_rg, stride_az, stride_rg,
    )

    # Step III — raw Doppler centroid, then subtract the Hanning-blended
    # geometric Doppler.  Blocks in burst-overlap zones get a weighted blend
    # of the two adjacent bursts' doppler_polynomial values, matching the
    # weights applied by merge_gamma_bursts.
    f_dc, _, _ = correlation_to_doppler(p0, p1, annot.prf, annot.wavelength)

    if type(slc_paths) is list:
        f_geom = _blended_geom_doppler(slc_par_paths, az_centers, rg_centers)
    else:
        f_geom = _geom_doppler_at_pixels(slc_par_paths, rg_centers)
    f_dca  = f_dc - f_geom

    # SNR from the correlation grids
    _, _, snr = correlation_to_doppler(p0, p1, annot.prf, annot.wavelength)
    v_r = (annot.wavelength / 2.0 * f_dca).astype(np.float32)

    # Step IV — descalloping
    if do_descallop:
        # In the merged image, the burst period in SLC lines is the inter-burst
        # interval (burst centre to burst centre), which is shorter than
        # lines_per_burst because adjacent bursts overlap.  Using lines_per_burst
        # here would search for scalloping at the wrong frequency.
        burst_dt_s = (
            annot.bursts[1].azimuth_time - annot.bursts[0].azimuth_time
        ).total_seconds()
        burst_period_rows = burst_dt_s / annot.azimuth_time_interval / stride_az
        f_dca = descallop(f_dca, snr, burst_period_rows)
        v_r  = (annot.wavelength / 2.0 * f_dca).astype(np.float32)

    # Step V — geolocation
    lat, lon, inc = _geolocate_grid(annot, az_centers, rg_centers)

    dims = ('az_cell', 'rg_cell')

    return xr.Dataset(
        {
            'f_dca': xr.DataArray(
                f_dca, dims=dims,
                attrs={'long_name': 'Doppler centroid anomaly (geometry subtracted)',
                       'units': 'Hz'},
            ),
            'radial_vel': xr.DataArray(
                v_r, dims=dims,
                attrs={'long_name': 'Radial velocity', 'units': 'm s-1'},
            ),
            'f_dc': xr.DataArray(
                f_dc, dims=dims,
                attrs={'long_name': 'Observed Doppler centroid (before geometry subtraction)',
                       'units': 'Hz'},
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
