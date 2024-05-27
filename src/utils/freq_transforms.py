import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import pywt


def wavelet_transform_truncate_reconstruct(y, n, wavelet="db4", mode="symmetric"):
    cA, cD = pywt.dwt(y, "bior6.8", "per")
    # print(cA.shape, cD.shape)
    # truncate coefficients
    if n > cA.shape[0]:
        n = cA.shape[0]
    cA_truncated = np.zeros_like(cA)
    # keep n equidistant coefficients from cA, i.e. 0, 0 + cA.shape[0] / n, 0 + 2 * cA.shape[0] / n, ...
    cA_truncated[:: cA.shape[0] // n] = cA[:: cA.shape[0] // n]
    cD_truncated = np.zeros_like(cD)
    cD_truncated[:: cD.shape[0] // n] = cD[:: cD.shape[0] // n]

    y_reconstructed = pywt.idwt(cA, cD, "bior6.8", "per")
    y_reconstructed_truncated = pywt.idwt(cA_truncated, cD_truncated, "bior6.8", "per")
    # make reconstructed signal same length as original signal
    y_reconstructed_truncated = y_reconstructed_truncated[: y.shape[0]]
    return (cA_truncated, cD_truncated), y_reconstructed_truncated


def fourier_transform_truncate_reconstruct(
    y, n, regularize=True, regularize_threshold=1e-0
):
    """
    Keep the first n//2 coeff from the fft of y, and the last n//2 coeff from the fft of y
    Remove the middle frequency components
    ::param y: signal to be truncated
    ::param n: number of coefficients to keep
    """
    y_fft = fft(y)
    # print(f"coefficients shape {y_fft.shape}")
    # print(f"coefficients {y_fft}")
    y_fft_truncated = np.zeros(y_fft.shape, dtype=complex)
    # keep the lowest and highest freqs
    # y_fft_truncated[: n // 2] = y_fft[: n // 2]
    # y_fft_truncated[-n // 2 :] = y_fft[-n // 2 :]
    # just keep the lowest freqs
    y_fft_truncated[:n] = y_fft[:n]
    # print(f"truncated coefficients {y_fft_truncated}")

    # regularize the reconstructed signal by setting zero all small coefficients
    if regularize:
        y_fft_truncated[np.abs(y_fft_truncated) < regularize_threshold] = 0

    y_reconstructed = ifft(y_fft_truncated)  # complex
    # cast to real
    y_reconstructed = np.real(y_reconstructed)

    return y_fft_truncated, y_reconstructed
