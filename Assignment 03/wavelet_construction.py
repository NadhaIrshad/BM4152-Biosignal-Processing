import numpy as np
from scipy.signal import fftconvolve


fs = 250
N = 3000
t = np.arange(-N, N + 1) / fs

# 0.01, 0.11, ..., 1.91.
scales = 0.01 + 0.1 * np.arange(20)


def mexican_hat(t, scale, translation=0.0):
    """Return the normalized Mexican-hat daughter wavelet."""
    # TODO: implement psi_{s,tau}(t), including the 1/sqrt(scale) factor.
    raise NotImplementedError