"""
Doppler centroid estimation from GAMMA RS SLC data.

Two estimators are provided:
  - fft_doppler : FFT-based weighted centroid (slower, returns full spectrum)
  - cde_doppler : Complex Doppler Estimator / lag-1 autocorrelation (fast)
"""

import numpy as np
from .io import read_slc, parse_slc_par


def fft_doppler(
    slc_path: str,
    slc_par_path: str,
    win_az: int,
    win_rg: int,
    stride_az: int,
    stride_rg: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate Doppler centroids using the FFT weighted-centroid method.

    For each azimuth×range window the azimuth FFT is computed, range pixels
    are averaged (increasing SNR), and the centroid of the power spectrum is
    returned.

    Parameters
    ----------
    slc_path, slc_par_path : str
        Paths to the GAMMA RS SLC and .par files.
    win_az, win_rg : int
        Window size in azimuth and range pixels.
    stride_az, stride_rg : int
        Step size between windows in azimuth and range.

    Returns
    -------
    doppler_img : np.ndarray, shape (n_az*stride_az, n_rg*stride_rg)
        Doppler centroid frequency [Hz] tiled to pixel resolution.
    freqs : np.ndarray, shape (win_az,)
        Frequency axis of the power spectrum [Hz].
    spectrum : np.ndarray of objects, shape (n_az, n_rg)
        Per-window range-averaged power spectrum P [shape (win_az,)].
    """
    slc = read_slc(slc_path, slc_par_path)
    dt_a = float(parse_slc_par(slc_par_path)['azimuth_line_time'][0])

    n_az = (slc.shape[0] - win_az) // stride_az + 1
    n_rg = (slc.shape[1] - win_rg) // stride_rg + 1

    freqs = np.fft.fftshift(np.fft.fftfreq(win_az, d=dt_a))
    doppler_img = np.zeros((n_az * stride_az, n_rg * stride_rg))
    spectrum = np.empty((n_az, n_rg), dtype=object)

    for j in range(n_az):
        az_0 = j * stride_az
        for i in range(n_rg):
            rg_0 = i * stride_rg

            patch = slc[az_0:az_0 + win_az, rg_0:rg_0 + win_rg]
            S = np.fft.fftshift(np.fft.fft(patch, axis=0), axes=0)
            P = np.sum(np.abs(S) ** 2, axis=1)

            f_dc = np.sum(freqs * P) / np.sum(P)
            spectrum[j, i] = P
            doppler_img[j * stride_az:(j + 1) * stride_az,
                        i * stride_rg:(i + 1) * stride_rg] = f_dc

    return doppler_img, freqs, spectrum


def cde_doppler(
    slc_path: str,
    slc_par_path: str,
    win_az: int,
    win_rg: int,
    stride_az: int,
    stride_rg: int,
) -> np.ndarray:
    """
    Estimate Doppler centroids using the Complex Doppler Estimator (CDE),
    also known as the lag-1 autocorrelation method (Madsen 1989).

    Faster than FFT but does not return the power spectrum.

    Parameters
    ----------
    slc_path, slc_par_path : str
        Paths to the GAMMA RS SLC and .par files.
    win_az, win_rg : int
        Window size in azimuth and range pixels.
    stride_az, stride_rg : int
        Step size between windows.

    Returns
    -------
    doppler_img : np.ndarray, shape (n_az*stride_az, n_rg*stride_rg)
        Doppler centroid frequency [Hz] tiled to pixel resolution.
    """
    slc = read_slc(slc_path, slc_par_path)
    dt_a = float(parse_slc_par(slc_par_path)['azimuth_line_time'][0])
    F_az = 1.0 / dt_a

    n_az = (slc.shape[0] - win_az) // stride_az + 1
    n_rg = (slc.shape[1] - win_rg) // stride_rg + 1
    doppler_img = np.zeros((n_az * stride_az, n_rg * stride_rg))

    for i in range(n_az):
        a0 = i * stride_az
        for j in range(n_rg):
            r0 = j * stride_rg

            patch = slc[a0:a0 + win_az, r0:r0 + win_rg]
            R1 = np.mean(patch[1:, :] * np.conj(patch[:-1, :]))
            f_dc = (F_az / (2.0 * np.pi)) * np.angle(R1)

            doppler_img[i * stride_az:(i + 1) * stride_az,
                        j * stride_rg:(j + 1) * stride_rg] = f_dc

    return doppler_img


def spectral_snr(P: np.ndarray, noise_fraction: float = 0.1) -> tuple[float, float]:
    """
    Estimate spectral SNR from a 1D azimuth power spectrum.

    The noise floor is estimated from the lowest-power bins (far from the
    Doppler peak). SNR = (peak - noise) / noise.

    Parameters
    ----------
    P : np.ndarray
        Range-averaged azimuth power spectrum.
    noise_fraction : float
        Fraction of lowest-power bins used to estimate the noise floor.

    Returns
    -------
    snr : float
        Spectral SNR (linear).
    noise_floor : float
        Estimated noise floor power.
    """
    n_noise = max(1, int(noise_fraction * len(P)))
    noise_floor = np.mean(np.sort(P)[:n_noise])
    snr = (np.max(P) - noise_floor) / noise_floor
    return snr, noise_floor


def dc_precision(snr: float, prf: float, win_az: int, win_rg: int) -> float:
    """
    Theoretical standard deviation of the FFT DC centroid estimate [Hz].

    Derived from the weighted-centroid Cramér–Rao bound:
        sigma_dc = delta_f / sqrt(2 * snr * win_rg)
    where delta_f = prf / win_az is the frequency resolution.

    Parameters
    ----------
    snr : float
        Spectral SNR (linear), from spectral_snr().
    prf : float
        Pulse repetition frequency [Hz].
    win_az, win_rg : int
        Azimuth and range window sizes used during estimation.

    Returns
    -------
    float
        Standard deviation of the DC estimate [Hz].
    """
    delta_f = prf / win_az
    return delta_f / np.sqrt(2 * snr * win_rg)
