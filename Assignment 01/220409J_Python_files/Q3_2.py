import numpy as np
from scipy.signal import (firwin, freqz, kaiserord, lfilter, periodogram,
                           get_window)
import matplotlib.pyplot as plt

fs = 500  # Hz
nyq = fs / 2

# 3.2 (1) Load ECG_with_noise, plot time domain + PSD

data = np.load("ECG_with_noise.npz")
print("Keys in ECG_with_noise.npz:", data.files)
ecg_noisy = data[data.files[0]].squeeze()  

t = np.arange(len(ecg_noisy)) / fs

plt.figure(figsize=(12, 4))
plt.plot(t, ecg_noisy, color="k", linewidth=0.6)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.title("ECG_with_noise - time domain")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_3_2_1_time_domain.png", dpi=150)
plt.show()

f_psd, Pxx = periodogram(ecg_noisy, fs=fs, window="hann", nfft=4096)

plt.figure(figsize=(10, 5))
plt.semilogy(f_psd, Pxx, color="darkblue")
plt.axvspan(0.5, 40, color="green", alpha=0.15, label="Typical ECG bandwidth (0.5-40 Hz)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.title("Power spectral density of ECG_with_noise")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_3_2_1_psd.png", dpi=150)
plt.show()

# 3.2 (2)-(3) Filter parameter choices and Kaiser beta / M calculation


ripple_db = 40      # desired stopband attenuation / passband ripple, in dB
                

# --- Highpass filter: remove baseline wander --------------------------------
hp_fstop = 3.0       # Hz - stopband edge (attenuate below this)
hp_fpass = 5.0       # Hz - passband edge (pass above this)
hp_width = hp_fpass - hp_fstop            # transition width, Hz
hp_M, hp_beta = kaiserord(ripple_db, hp_width / nyq)
hp_M |= 1            # force odd length -> Type I linear-phase FIR (symmetric, odd length)
print(f"Highpass filter: M = {hp_M}, beta = {hp_beta:.3f}")

hp_taps = firwin(hp_M, hp_fpass / nyq, window=("kaiser", hp_beta), pass_zero=False)

# --- Lowpass filter: remove high-frequency/EMG noise ------------------------
lp_fpass = 125.0     # Hz - passband edge (pass below this)
lp_fstop = 135.0     # Hz - stopband edge (attenuate above this)
lp_width = lp_fstop - lp_fpass
lp_M, lp_beta = kaiserord(ripple_db, lp_width / nyq)
lp_M |= 1
print(f"Lowpass filter: M = {lp_M}, beta = {lp_beta:.3f}")

lp_taps = firwin(lp_M, lp_fpass / nyq, window=("kaiser", lp_beta), pass_zero=True)

# --- Notch/comb filter: remove 50 Hz powerline interference + harmonics -----
# fstop1, fstop2, fstop3 correspond to the 50 Hz fundamental and its first
# two harmonics (100 Hz, 150 Hz), which are common sources of narrowband
# interference. Adjust these to match spikes actually visible in your PSD.
notch_freqs = [50.0, 100.0, 150.0]   # fstop1, fstop2, fstop3 (Hz)
notch_halfwidth = 2.0                 # Hz, width of each rejected band (+/-)
notch_ripple_db = 40
notch_width_hz = 1.0                  # transition width either side of each notch
notch_M, notch_beta = kaiserord(notch_ripple_db, notch_width_hz / nyq)
notch_M |= 1
print(f"Notch/comb filter: M = {notch_M}, beta = {notch_beta:.3f}")

# Build one combined multiband stopband filter rejecting all three bands.
bands = []
for f0 in notch_freqs:
    bands.extend([(f0 - notch_halfwidth) / nyq, (f0 + notch_halfwidth) / nyq])
notch_taps = firwin(notch_M, bands, window=("kaiser", notch_beta), pass_zero=True)

# ==========================================================================
# 3.2 (4) Visualise windows, magnitude responses, phase responses
# ==========================================================================
filters = [("Highpass", hp_taps, hp_M, hp_beta),
           ("Lowpass", lp_taps, lp_M, lp_beta),
           ("Notch/comb", notch_taps, notch_M, notch_beta)]

fig, axes = plt.subplots(3, 3, figsize=(15, 11))
for row, (name, taps, M, beta) in enumerate(filters):
    win = get_window(("kaiser", beta), M)
    axes[row, 0].plot(np.arange(M), win)
    axes[row, 0].set_title(f"{name} Kaiser window (M={M}, beta={beta:.2f})")
    axes[row, 0].grid(alpha=0.3)

    w, h = freqz(taps, worN=4096, fs=fs)
    axes[row, 1].plot(w, 20 * np.log10(np.abs(h) + 1e-12))
    axes[row, 1].set_title(f"{name} magnitude response")
    axes[row, 1].set_xlabel("Frequency (Hz)")
    axes[row, 1].set_ylabel("dB")
    axes[row, 1].grid(alpha=0.3)

    axes[row, 2].plot(w, np.unwrap(np.angle(h)))
    axes[row, 2].set_title(f"{name} phase response")
    axes[row, 2].set_xlabel("Frequency (Hz)")
    axes[row, 2].set_ylabel("Phase (rad)")
    axes[row, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("fig_3_2_4_filter_windows_and_responses.png", dpi=150)
plt.show()

# ==========================================================================
# 3.2 (5) Apply filters sequentially with group delay compensation
# ==========================================================================
def apply_fir_with_delay_compensation(x, taps):
    """Apply a symmetric linear-phase FIR filter and shift out the constant
    group delay of (len(taps)-1)/2 samples introduced by lfilter."""
    y = lfilter(taps, 1.0, x)
    delay = (len(taps) - 1) // 2
    y_compensated = np.zeros_like(y)
    y_compensated[:len(y) - delay] = y[delay:]
    return y_compensated

stage1_hp   = apply_fir_with_delay_compensation(ecg_noisy, hp_taps)
stage2_lp   = apply_fir_with_delay_compensation(stage1_hp, lp_taps)
stage3_comb = apply_fir_with_delay_compensation(stage2_lp, notch_taps)
final_filtered = stage3_comb

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
axes[0].plot(t, ecg_noisy, linewidth=0.6, color="k")
axes[0].set_title("Original ECG_with_noise")
axes[1].plot(t, stage1_hp, linewidth=0.6, color="tab:blue")
axes[1].set_title("After highpass (baseline wander removed)")
axes[2].plot(t, stage2_lp, linewidth=0.6, color="tab:green")
axes[2].set_title("After highpass + lowpass")
axes[3].plot(t, stage3_comb, linewidth=0.6, color="tab:red")
axes[3].set_title("After highpass + lowpass + notch/comb (final)")
axes[3].set_xlabel("Time (s)")
for ax in axes:
    ax.set_ylabel("mV")
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_3_2_5_stagewise_filtering.png", dpi=150)
plt.show()

# ==========================================================================
# 3.2 (5) Zoomed-in comparison of filtering stages
# ==========================================================================

zoom_start = 10
zoom_end = 11.5

zoom_idx = (t >= zoom_start) & (t <= zoom_end)

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

# --------------------------------------------------------------------------
# 1. Original noisy ECG
# --------------------------------------------------------------------------
axes[0].plot(
    t[zoom_idx], ecg_noisy[zoom_idx],
    color="k", linewidth=0.8,
    label="Original ECG with noise"
)
axes[0].set_title("Original ECG with noise")

# --------------------------------------------------------------------------
# 2. Noisy ECG + highpass filtered signal
# --------------------------------------------------------------------------
axes[1].plot(
    t[zoom_idx], ecg_noisy[zoom_idx],
    color="k", linewidth=0.6, alpha=0.5,
    label="Original ECG with noise"
)
axes[1].plot(
    t[zoom_idx], stage1_hp[zoom_idx],
    color="tab:blue", linewidth=1.0,
    label="After highpass"
)
axes[1].set_title("Highpass Filtering")

# --------------------------------------------------------------------------
# 3. Noisy ECG + highpass + lowpass filtered signal
# --------------------------------------------------------------------------
axes[2].plot(
    t[zoom_idx], ecg_noisy[zoom_idx],
    color="k", linewidth=0.6, alpha=0.5,
    label="Original ECG with noise"
)
axes[2].plot(
    t[zoom_idx], stage2_lp[zoom_idx],
    color="tab:green", linewidth=1.0,
    label="After highpass + lowpass"
)
axes[2].set_title("Highpass + Lowpass Filtering")

# --------------------------------------------------------------------------
# 4. Noisy ECG + final filtered signal
# --------------------------------------------------------------------------
axes[3].plot(
    t[zoom_idx], ecg_noisy[zoom_idx],
    color="k", linewidth=0.6, alpha=0.5,
    label="Original ECG with noise"
)
axes[3].plot(
    t[zoom_idx], stage3_comb[zoom_idx],
    color="tab:red", linewidth=1.0,
    label="Final: highpass + lowpass + notch/comb"
)
axes[3].set_title("Highpass + Lowpass + Notch/Comb Filtering")

# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------
for ax in axes:
    ax.set_ylabel("mV")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

axes[3].set_xlabel("Time (s)")

plt.xlim(zoom_start, zoom_end)
plt.tight_layout()

plt.savefig(
    "fig_3_2_5_stagewise_filtering_zoomed.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

# ==========================================================================
# 3.2 (6) Combined magnitude response + final PSD
# ==========================================================================
combined_taps = np.convolve(np.convolve(hp_taps, lp_taps), notch_taps)
w_comb, h_comb = freqz(combined_taps, worN=4096, fs=fs)

plt.figure(figsize=(10, 5))
plt.plot(w_comb, 20 * np.log10(np.abs(h_comb) + 1e-12), color="purple")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Combined magnitude response: highpass + lowpass + notch/comb (FIR)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_3_2_6_combined_magnitude_response.png", dpi=150)
plt.show()

f_final, Pxx_final = periodogram(final_filtered, fs=fs, window="hann", nfft=4096)

plt.figure(figsize=(10, 5))
plt.semilogy(f_psd, Pxx, color="lightgray", label="Original ECG_with_noise")
plt.semilogy(f_final, Pxx_final, color="darkred", label="Final filtered ECG")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.title("PSD comparison: before vs. after FIR filtering")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_3_2_6_psd_comparison.png", dpi=150)
plt.show()
