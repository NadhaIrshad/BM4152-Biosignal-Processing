import numpy as np
import matplotlib.pyplot as plt

fs = 128  # Hz

# 2.2.1 Viewing the signal and addition of Gaussian white noise

# --- (1) Load ECG_rec.npz -------------------------------------------------
data = np.load("ECG_rec.npz")
print("Keys in ECG_rec.npz:", data.files)
ECG_rec = data[data.files[0]].squeeze()       

t = np.arange(len(ECG_rec)) / fs

# --- (2) Plot the raw ECG recording ---------------------------------------
plt.figure(figsize=(12, 4))
plt.plot(t, ECG_rec, color="k", linewidth=0.8)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.title("ECG signal")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_2_2_1_raw_ECG.png", dpi=150)
plt.show()

# --- (2b) Zoomed-in view --------------------------

zoom_end_sec = 6
zoom_samples = int(zoom_end_sec * fs)
 
plt.figure(figsize=(12, 4))
plt.plot(t[:zoom_samples], ECG_rec[:zoom_samples], color="k", linewidth=1.0,
          marker=".", markersize=3)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.title(f"ECG signal - zoomed view (first {zoom_end_sec:.0f} s)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_2_2_1_raw_ECG_zoomed.png", dpi=150)
plt.show()

# --- (3) Extract a single PQRST waveform as ECG_template -------------------

start_idx = 100     
end_idx   = 220      
ECG_template = ECG_rec[start_idx:end_idx]

plt.figure(figsize=(6, 4))
plt.plot(np.arange(len(ECG_template)) / fs, ECG_template)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.title("Extracted ECG_template (single PQRST)")
plt.tight_layout()
plt.savefig("fig_2_2_1_ECG_template.png", dpi=150)
plt.show()

# --- (4) Add 5 dB Gaussian white noise -> nECG -----------------------------
def add_awgn(signal, snr_db):
    """Add zero-mean white Gaussian noise scaled to achieve the target SNR (dB)."""
    sig_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = sig_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise

np.random.seed(0)   # for reproducibility
nECG = add_awgn(ECG_rec, snr_db=5)

plt.figure(figsize=(12, 4))
plt.plot(t, nECG, color="steelblue", linewidth=0.6)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.title("nECG: ECG_rec + 5 dB Gaussian white noise")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_2_2_1_nECG.png", dpi=150)
plt.show()

# ==========================================================================
# 2.2.2 Segmenting ECG into separate epochs and ensemble averaging
# ==========================================================================

# --- (1) Normalised cross-correlation between ECG_template and nECG --------
templ = ECG_template - np.mean(ECG_template)
sig   = nECG - np.mean(nECG)

cross_corr = np.correlate(sig, templ, mode="full")
# Normalise so the maximum possible value is 1
norm_factor = np.sqrt(np.sum(templ ** 2) * np.sum(sig ** 2))
norm_xcorr = cross_corr / norm_factor

lags = np.arange(-len(templ) + 1, len(sig))
lag_time = lags / fs

# --- (2) Plot normalised cross-correlation vs. time-converted lag axis -----
plt.figure(figsize=(12, 4))
plt.plot(lag_time, norm_xcorr, color="darkgreen", linewidth=0.6)
plt.xlabel("Lag (s)")
plt.ylabel("Normalised cross-correlation")
plt.title("Normalised cross-correlation: ECG_template vs nECG")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_2_2_2_xcorr.png", dpi=150)
plt.show()

# --- (3) Segment ECG pulses using a cross-correlation threshold ------------
# Use scipy.signal.find_peaks instead of a manual/hardcoded threshold: with
# noisy data the correlation peak height is data-dependent, so a fixed
# absolute threshold (e.g. 0.5) can easily match zero peaks. Instead, set
# the threshold relative to the observed maximum correlation.
from scipy.signal import find_peaks

print(f"Max normalised cross-correlation value: {norm_xcorr.max():.3f}")

relative_threshold = 0.4                       # fraction of the peak value
xcorr_threshold = relative_threshold * norm_xcorr.max()
min_spacing = int(0.4 * fs)                     # ~0.4 s minimum RR interval, adjust as needed

peak_indices, _ = find_peaks(norm_xcorr, height=xcorr_threshold, distance=min_spacing)

if len(peak_indices) == 0:
    raise RuntimeError(
        "No peaks found above threshold - lower relative_threshold and/or "
        "check the normalised cross-correlation plot to pick a sensible value."
    )
print(f"Threshold used: {xcorr_threshold:.3f}  ->  {len(peak_indices)} peaks found")

# Convert cross-correlation index back to the corresponding start sample in nECG
# (mode="full" cross-correlation: signal index = peak_index - (len(templ)-1))
pulse_starts = peak_indices - (len(templ) - 1)
pulse_len = len(templ)

pulses = []
for s in pulse_starts:
    e = s + pulse_len
    if s >= 0 and e <= len(nECG):
        pulses.append(nECG[s:e])
pulses = np.array(pulses)
print(f"Number of ECG pulses segmented: {pulses.shape[0]}")

# --- (3b) Plot 5 segmented pulses to visually check segmentation/alignment -
n_show = min(5, pulses.shape[0])
show_idx = np.linspace(0, pulses.shape[0] - 1, n_show, dtype=int)  # spread across the recording
t_pulse_check = np.arange(pulse_len) / fs
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
# (a) Overlaid - shows how well individual epochs line up with each other
for i in show_idx:
    axes[0].plot(t_pulse_check, pulses[i], alpha=0.7, linewidth=0.9, label=f"Pulse {i}")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Voltage (mV)")
axes[0].set_title("5 segmented pulses (overlaid)")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)
 
# (b) Stacked - easier to see each pulse's own noise/shape individually
offset = 1.5
for n, i in enumerate(show_idx):
    axes[1].plot(t_pulse_check, pulses[i] + n * offset, linewidth=0.9)
    axes[1].text(t_pulse_check[-1], n * offset, f"  #{i}", va="center", fontsize=8)
axes[1].set_xlabel("Time (s)")
axes[1].set_yticks([])
axes[1].set_title("5 segmented pulses (stacked)")
axes[1].grid(alpha=0.3)
 
plt.tight_layout()
plt.savefig("fig_2_2_2_five_segments.png", dpi=150)
plt.show()

# --- (4) SNR improvement as more pulses are included in the ensemble avg ---
def progressive_ensemble_mse(pulses, template):
    """Progressive MSE, same construction as Section 2.1.2 (Eq. 2 & 3)."""
    M, N = pulses.shape
    mse = np.zeros(M)
    running_sum = np.zeros(N)
    for k in range(1, M + 1):
        running_sum += pulses[k - 1]
        yhat_k = running_sum / k
        mse[k - 1] = np.sqrt(np.sum((template - yhat_k) ** 2) / N)
    return mse

final_ensemble_avg = np.mean(pulses, axis=0)
mse_k_ecg = progressive_ensemble_mse(pulses, final_ensemble_avg)

plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, len(mse_k_ecg) + 1), mse_k_ecg, color="purple")
plt.xlabel("Number of ECG pulses averaged, k")
plt.ylabel("MSE$_k$")
plt.title("SNR improvement: progressive MSE vs. number of pulses averaged")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_2_2_2_mse_vs_k.png", dpi=150)
plt.show()

# --- (5) Compare a single noisy pulse vs two ensemble-averaged pulses ------
# "Two arbitrarily-selected ensemble-averaged pulses" = ensemble averages
# computed using two different arbitrary subsets/counts of pulses.
avg_10  = np.mean(pulses[:10], axis=0)
avg_all = np.mean(pulses, axis=0)
t_pulse = np.arange(pulse_len) / fs

plt.figure(figsize=(8, 5))
plt.plot(t_pulse, pulses[0], label="Single noisy ECG pulse", alpha=0.6)
plt.plot(t_pulse, avg_10, label="Ensemble avg (first 10 pulses)", linewidth=2)
plt.plot(t_pulse, avg_all, label="Ensemble avg (all pulses)", linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.title("Noisy pulse vs. ensemble-averaged pulses")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_2_2_2_pulse_comparison.png", dpi=150)
plt.show()

#Restuls obtained 
#Max normalised cross-correlation value: 0.113
#Threshold used: 0.045  ->  79 peaks found
# Number of ECG pulses segmented: 79