"""
Robust calibration bias estimation from a static reference scene.

Uses sigma-clipped, SNR-weighted mean of the DC residual map to derive
a calibration offset and its uncertainty. Intended for use with a desert
or other stationary scene where the true surface Doppler is zero.
"""

import numpy as np
from ..doppler import spectral_snr


def compute_calibration_bias(
    dc_residual: np.ndarray,
    spectrum: np.ndarray,
    n_sigma: float = 3.0,
) -> tuple[float, float, float, np.ndarray]:
    """
    Compute a robust calibration bias from a static-scene DC residual map.

    Parameters
    ----------
    dc_residual : np.ndarray, shape (n_az*stride_az, n_rg*stride_rg)
        DC_meas - DC_geom - f_iono  [Hz], as returned by the correction chain.
    spectrum    : np.ndarray of objects, shape (n_az, n_rg)
        Per-window power spectra from fft_doppler().
    n_sigma     : float
        Clipping threshold in robust sigma units (default 3.0).

    Returns
    -------
    bias        : float   calibration offset to subtract from ocean scenes [Hz]
    std_err     : float   standard error on the bias estimate [Hz]
    sigma_scene : float   spatial variability / detection noise floor [Hz]
    mask        : np.ndarray bool, shape (n_az, n_rg), True = window kept
    """
    n_az, n_rg = spectrum.shape

    # --- SNR per window ---
    snr_map = np.zeros((n_az, n_rg))
    for j in range(n_az):
        for i in range(n_rg):
            snr_map[j, i], _ = spectral_snr(spectrum[j, i])

    # --- collapse tiled DC image to window grid (centre pixel of each window) ---
    step_az = dc_residual.shape[0] // n_az
    step_rg = dc_residual.shape[1] // n_rg
    dc_windows = dc_residual[::step_az, ::step_rg][:n_az, :n_rg]

    flat_dc  = dc_windows.ravel()
    flat_snr = snr_map.ravel()

    # --- sigma clipping ---
    median    = np.median(flat_dc)
    mad       = np.median(np.abs(flat_dc - median))
    sigma_rob = 1.4826 * mad          # MAD → equivalent Gaussian sigma

    mask_1d = np.abs(flat_dc - median) < n_sigma * sigma_rob
    dc_clip  = flat_dc[mask_1d]
    w        = flat_snr[mask_1d]

    # --- SNR-weighted mean ---
    W    = w.sum()
    bias = np.sum(w * dc_clip) / W

    # weighted standard deviation (spatial variability / noise floor)
    sigma_scene = np.sqrt(np.sum(w * (dc_clip - bias) ** 2) / W)

    # standard error on the mean
    # azimuth windows overlap by 50% (STRIDE/WIN = 0.5) → halve effective N
    n_eff   = mask_1d.sum() / 2
    std_err = sigma_scene / np.sqrt(n_eff)

    return bias, std_err, sigma_scene, mask_1d.reshape(n_az, n_rg)
