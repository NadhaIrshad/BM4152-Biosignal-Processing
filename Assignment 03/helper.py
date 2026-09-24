import numpy as np

def awgn(x, snr_db, rng):
    'Returns a copy of x with additive white Gaussian noise (AWGN) at the specified SNR in dB.'
    signal_power = np.mean(np.asarray(x, dtype=float) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    return x + rng.normal(0.0, np.sqrt(noise_power), size=np.shape(x))