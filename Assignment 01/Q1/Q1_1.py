import numpy as np
from scipy.signal import lfilter, freqz, periodogram
import matplotlib.pyplot as plt

fs = 500  # Hz

# 1.1.1 Preliminaries

# (1) Load ECG_template.npz
data = np.load("ECG_template.npz")
print("Keys in ECG_template.npz:", data.files)
ECG_template = data[data.files[0]].squeeze()  

t = np.arange(len(ECG_template)) / fs

# (2) Plot the loaded signal with adjusted time scale 
plt.figure(figsize=(12, 4))
plt.plot(t, ECG_template, color="k", linewidth=0.8)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.title("ECG_template")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_1_1_1_ECG_template.png", dpi=150)
plt.show()

# (3) Add white Gaussian noise scaled to a target SNR of 5 dB 
def add_awgn(signal, snr_db, seed=None):
    """Add zero-mean white Gaussian noise scaled to achieve the target SNR (dB),
    measured from the signal's own power."""
    rng = np.random.default_rng(seed)
    sig_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = sig_power / snr_linear
    noise = rng.normal(0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise

nECG = add_awgn(ECG_template, snr_db=5, seed=0)

plt.figure(figsize=(12, 4))
plt.plot(t, nECG, color="steelblue", linewidth=0.7)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.title("nECG: ECG_template + 5 dB Gaussian white noise")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_1_1_1_nECG.png", dpi=150)
plt.show()

# (4) PSD estimate of nECG
f_psd, Pxx = periodogram(nECG, fs=fs, window="hann", nfft=2048)

plt.figure(figsize=(10, 5))
plt.semilogy(f_psd, Pxx, color="darkblue")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.title("Power spectral density of nECG")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_1_1_1_nECG_psd.png", dpi=150)
plt.show()

# 1.1.2 MA(3) filter - custom implementation (no built-in filtering function)

def ma_filter_custom(x, N):
    """
    Custom moving-average filter implementing y(n) = (1/N) * sum_{k=0}^{N-1} x(n-k)
    directly from Equation (1), without using any built-in filtering function.
    Causal filter: samples before the start of the signal are treated as 0
    (equivalent to zero-padding the input).
    """
    x_padded = np.concatenate([np.zeros(N - 1), x])
    y = np.zeros_like(x)
    for n in range(len(x)):
        y[n] = np.mean(x_padded[n:n + N])
    return y

ma3ECG_1 = ma_filter_custom(nECG, N=3)

# Group delay of an MA(N) filter
N_ma3 = 3
group_delay_ma3 = (N_ma3 - 1) / 2
print(f"MA(3) group delay = {group_delay_ma3} samples "
      f"({group_delay_ma3 / fs * 1000:.2f} ms)")

def compensate_delay(y, delay_samples):
    """Shift a filtered signal back by its (integer) group delay."""
    delay_samples = int(round(delay_samples))
    y_comp = np.zeros_like(y)
    if delay_samples > 0:
        y_comp[:len(y) - delay_samples] = y[delay_samples:]
    else:
        y_comp = y.copy()
    return y_comp

ma3ECG_1_comp = compensate_delay(ma3ECG_1, group_delay_ma3)

# Plot delay-compensated ma3ECG_1 vs ECG_template vs nECG
plt.figure(figsize=(12, 5))
plt.plot(t, nECG, color="lightgray", linewidth=0.7, label="nECG")
plt.plot(t, ECG_template, color="k", linewidth=1.2, label="ECG_template")
plt.plot(t, ma3ECG_1_comp, color="tab:red", linewidth=1.2,
          label="ma3ECG_1 (custom MA(3), delay-compensated)")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.title("MA(3) filtering (custom implementation) vs. original and noisy signal")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_1_1_2_ma3_custom_comparison.png", dpi=150)
plt.show()

# Overlapping PSDs of ma3ECG_1 and nECG
f_ma3, Pxx_ma3 = periodogram(ma3ECG_1, fs=fs, window="hann", nfft=2048)

plt.figure(figsize=(10, 5))
plt.semilogy(f_psd, Pxx, color="lightgray", label="nECG")
plt.semilogy(f_ma3, Pxx_ma3, color="tab:red", label="ma3ECG_1 (MA(3) filtered)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.title("PSD comparison: nECG vs. MA(3)-filtered signal")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_1_1_2_ma3_psd_comparison.png", dpi=150)
plt.show()

# 1.1.3 MA(3) filter - built-in function (scipy.signal.lfilter)

b_ma3 = np.ones(3) / 3
a_ma3 = [1.0]

ma3ECG_2 = lfilter(b_ma3, a_ma3, nECG)
ma3ECG_2_comp = compensate_delay(ma3ECG_2, group_delay_ma3)

plt.figure(figsize=(12, 5))
plt.plot(t, nECG, color="lightgray", linewidth=0.7, label="nECG")
plt.plot(t, ECG_template, color="k", linewidth=1.2, label="ECG_template")
plt.plot(t, ma3ECG_2_comp, color="tab:green", linewidth=1.2,
          label="ma3ECG_2 (lfilter MA(3), delay-compensated)")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.title("MA(3) filtering (scipy.signal.lfilter) vs. original and noisy signal")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_1_1_3_ma3_lfilter_comparison.png", dpi=150)
plt.show()

# Magnitude/phase response (freqz) and pole-zero plot (numpy.roots)
w_ma3, h_ma3 = freqz(b_ma3, a_ma3, worN=2048, fs=fs)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
axes[0].plot(w_ma3, np.abs(h_ma3))
axes[0].set_title("MA(3) magnitude response")
axes[0].set_xlabel("Frequency (Hz)"); axes[0].set_ylabel("|H|")
axes[0].grid(alpha=0.3)

axes[1].plot(w_ma3, np.unwrap(np.angle(h_ma3)))
axes[1].set_title("MA(3) phase response")
axes[1].set_xlabel("Frequency (Hz)"); axes[1].set_ylabel("Phase (rad)")
axes[1].grid(alpha=0.3)

zeros_ma3 = np.roots(b_ma3)
poles_ma3 = np.roots(a_ma3) if len(a_ma3) > 1 else np.array([])
theta = np.linspace(0, 2 * np.pi, 200)
axes[2].plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.7)  # unit circle
axes[2].scatter(zeros_ma3.real, zeros_ma3.imag, marker="o", s=80,
                  facecolors="none", edgecolors="b", label="Zeros")
if len(poles_ma3) > 0:
    axes[2].scatter(poles_ma3.real, poles_ma3.imag, marker="x", s=80,
                      color="r", label="Poles")
else:
    axes[2].scatter(0, 0, marker="x", s=80, color="r", label="Poles (at origin)")
axes[2].set_title("MA(3) pole-zero plot")
axes[2].set_xlabel("Real"); axes[2].set_ylabel("Imaginary")
axes[2].axis("equal")
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("fig_1_1_3_ma3_freqz_polezero.png", dpi=150)
plt.show()

# 1.1.4 MA(10) filter - built-in function

b_ma10 = np.ones(10) / 10
a_ma10 = [1.0]
group_delay_ma10 = (10 - 1) / 2

w_ma10, h_ma10 = freqz(b_ma10, a_ma10, worN=2048, fs=fs)
zeros_ma10 = np.roots(b_ma10)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
axes[0].plot(w_ma3, np.abs(h_ma3), label="MA(3)")
axes[0].plot(w_ma10, np.abs(h_ma10), label="MA(10)")
axes[0].set_title("Magnitude response: MA(3) vs MA(10)")
axes[0].set_xlabel("Frequency (Hz)"); axes[0].set_ylabel("|H|")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(w_ma3, np.unwrap(np.angle(h_ma3)), label="MA(3)")
axes[1].plot(w_ma10, np.unwrap(np.angle(h_ma10)), label="MA(10)")
axes[1].set_title("Phase response: MA(3) vs MA(10)")
axes[1].set_xlabel("Frequency (Hz)"); axes[1].set_ylabel("Phase (rad)")
axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.7)
axes[2].scatter(zeros_ma3.real, zeros_ma3.imag, marker="o", s=60,
                  facecolors="none", edgecolors="b", label="MA(3) zeros")
axes[2].scatter(zeros_ma10.real, zeros_ma10.imag, marker="o", s=60,
                  facecolors="none", edgecolors="tab:orange", label="MA(10) zeros")
axes[2].scatter(0, 0, marker="x", s=80, color="r", label="Poles (at origin)")
axes[2].set_title("Pole-zero plot: MA(3) vs MA(10)")
axes[2].set_xlabel("Real"); axes[2].set_ylabel("Imaginary")
axes[2].axis("equal")
axes[2].legend(fontsize=8)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("fig_1_1_4_MA3_vs_MA10_freqz_polezero.png", dpi=150)
plt.show()

ma10ECG = lfilter(b_ma10, a_ma10, nECG)
ma10ECG_comp = compensate_delay(ma10ECG, group_delay_ma10)

plt.figure(figsize=(12, 5))
plt.plot(t, nECG, color="lightgray", linewidth=0.6, label="nECG")
plt.plot(t, ECG_template, color="k", linewidth=1.2, label="ECG_template")
plt.plot(t, ma3ECG_2_comp, color="tab:green", linewidth=1, label="ma3ECG_2 (MA(3))")
plt.plot(t, ma10ECG_comp, color="tab:purple", linewidth=1.2, label="ma10ECG (MA(10))")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.title("Comparison: nECG, ECG_template, MA(3) and MA(10) filtered signals")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_1_1_4_MA3_vs_MA10_timedomain.png", dpi=150)
plt.show()

# 1.1.5 Optimum MA(N) filter order

def mse(reference, estimate):
    """Mean-squared error between two equal-length signals."""
    return np.mean((reference - estimate) ** 2)

N_range = np.arange(1, 101)   # test MA(N) for N = 1..40
mse_values = np.zeros_like(N_range, dtype=float)

for i, N in enumerate(N_range):
    b_N = np.ones(N) / N
    y_N = lfilter(b_N, [1.0], nECG)
    y_N_comp = compensate_delay(y_N, (N - 1) / 2)
    mse_values[i] = mse(ECG_template, y_N_comp)

optimum_N = N_range[np.argmin(mse_values)]
print(f"Optimum MA(N) filter order: N = {optimum_N} "
      f"(MSE = {mse_values.min():.6f})")

plt.figure(figsize=(9, 5))
plt.plot(N_range, mse_values, marker="o", markersize=3)
plt.axvline(optimum_N, color="red", linestyle="--",
             label=f"Optimum N = {optimum_N}")
plt.xlabel("MA filter order, N")
plt.ylabel("MSE")
plt.title("MSE vs. MA(N) filter order")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_1_1_5_MA_MSE_vs_N.png", dpi=150)
plt.show()
