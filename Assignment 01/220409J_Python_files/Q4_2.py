import numpy as np
from scipy.signal import (firwin, kaiserord, butter, iircomb,
                           lfilter, filtfilt, periodogram, find_peaks)
import matplotlib.pyplot as plt

fs = 500
nyq = fs / 2

data = np.load("ECG_with_noise.npz")
ecg_noisy = data[data.files[0]].squeeze()
t = np.arange(len(ecg_noisy)) / fs


# Recreate FIR filters (Section 3.2) and filter the signal, with group-delay
# compensation, exactly as in Q3_2_Kaiser_FIR_design.py

ripple_db = 40

hp_M, hp_beta = kaiserord(ripple_db, (5 - 3) / nyq); hp_M |= 1
hp_taps = firwin(hp_M, 5 / nyq, window=("kaiser", hp_beta), pass_zero=False)

lp_M, lp_beta = kaiserord(ripple_db, (135 - 125) / nyq); lp_M |= 1
lp_taps = firwin(lp_M, 125 / nyq, window=("kaiser", lp_beta), pass_zero=True)

notch_M, notch_beta = kaiserord(ripple_db, 1 / nyq); notch_M |= 1
bands = []
for f0 in [50, 100, 150]:
    bands.extend([(f0 - 2) / nyq, (f0 + 2) / nyq])
notch_taps = firwin(notch_M, bands, window=("kaiser", notch_beta), pass_zero=True)

def apply_fir_with_delay_compensation(x, taps):
    y = lfilter(taps, 1.0, x)
    delay = (len(taps) - 1) // 2
    y_comp = np.zeros_like(y)
    y_comp[:len(y) - delay] = y[delay:]
    return y_comp

fir_stage1 = apply_fir_with_delay_compensation(ecg_noisy, hp_taps)
fir_stage2 = apply_fir_with_delay_compensation(fir_stage1, lp_taps)
fir_filtered = apply_fir_with_delay_compensation(fir_stage2, notch_taps)

# ==========================================================================
# Recreate IIR filters (Section 4.1) - fixed modest order, matching only the
# cutoff frequency (not the full ripple/attenuation spec - see Section 4.1
# discussion on numerical stability of high-order direct-form IIR filters).
# ==========================================================================
IIR_ORDER = 4

hp_b, hp_a = butter(IIR_ORDER, 5, btype="high", fs=fs)
lp_b, lp_a = butter(IIR_ORDER, 125, btype="low", fs=fs)

comb_w0, comb_bw = 50.0, 4.0
comb_b, comb_a = iircomb(comb_w0, comb_w0 / comb_bw, ftype="notch", fs=fs)

# ==========================================================================
# 4.2 (1) Forward filtering only (lfilter) - cascade hp -> lp -> comb
# ==========================================================================
iir_fwd = lfilter(hp_b, hp_a, ecg_noisy)
iir_fwd = lfilter(lp_b, lp_a, iir_fwd)
iir_fwd = lfilter(comb_b, comb_a, iir_fwd)

# ==========================================================================
# 4.2 (2) Forward-backward filtering (filtfilt) - cascade hp -> lp -> comb
# ==========================================================================
iir_fwdback = filtfilt(hp_b, hp_a, ecg_noisy)
iir_fwdback = filtfilt(lp_b, lp_a, iir_fwdback)
iir_fwdback = filtfilt(comb_b, comb_a, iir_fwdback)

# ==========================================================================
# 4.2 (3) Overlapping time-domain comparison, zoomed to ~2 pulses
# ==========================================================================
# Automatically find a clean 2-beat window using peaks in the FIR-filtered
# signal (most reliable reference) rather than a manually guessed index.
min_beat_spacing = int(0.4 * fs)          # ~0.4 s minimum RR interval
peaks, _ = find_peaks(fir_filtered, height=np.std(fir_filtered) * 2,
                        distance=min_beat_spacing)

if len(peaks) >= 3:
    zoom_start = max(peaks[1] - min_beat_spacing // 2, 0)
    zoom_end = min(peaks[2] + min_beat_spacing // 2, len(fir_filtered))
else:
    # fallback: just show a fixed 2-second window
    zoom_start, zoom_end = 0, int(2 * fs)

sl = slice(zoom_start, zoom_end)

plt.figure(figsize=(12, 5))
plt.plot(t[sl], fir_filtered[sl], label="FIR filtered", linewidth=1.2)
plt.plot(t[sl], iir_fwd[sl], label="IIR forward only (lfilter)", linewidth=1.2)
plt.plot(t[sl], iir_fwdback[sl], label="IIR forward-backward (filtfilt)", linewidth=1.2)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.title("Comparison of FIR, IIR forward, and IIR forward-backward filtered ECG (zoomed)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_4_2_3_time_domain_comparison.png", dpi=150)
plt.show()

# ==========================================================================
# 4.2 (4) Overlapping PSD comparison
# ==========================================================================
f_fir, Pxx_fir = periodogram(fir_filtered, fs=fs, window="hann", nfft=4096)
f_fwd, Pxx_fwd = periodogram(iir_fwd, fs=fs, window="hann", nfft=4096)
f_fwdback, Pxx_fwdback = periodogram(iir_fwdback, fs=fs, window="hann", nfft=4096)

plt.figure(figsize=(11, 5))
plt.semilogy(f_fir, Pxx_fir, label="FIR filtered", alpha=0.8)
plt.semilogy(f_fwd, Pxx_fwd, label="IIR forward only", alpha=0.8)
plt.semilogy(f_fwdback, Pxx_fwdback, label="IIR forward-backward", alpha=0.8)
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.title("PSD comparison: FIR vs. IIR forward vs. IIR forward-backward")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_4_2_4_psd_comparison.png", dpi=150)
plt.show()
