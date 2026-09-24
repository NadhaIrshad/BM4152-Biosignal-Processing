import numpy as np
import matplotlib.pyplot as plt


# 2.1.1 (1) Load ABR_rec.npz
data = np.load("ABR_rec.npz")
print("Keys in ABR_rec.npz:", data.files)  

if len(data.files) == 1:
    arr = data[data.files[0]]
    stimulus = arr[:, 0]
    abr_eeg  = arr[:, 1]
else:
    stimulus = data[data.files[0]]
    abr_eeg  = data[data.files[1]]

fs = 40_000                       # Hz
t = np.arange(len(abr_eeg)) / fs  # seconds


downsample = 20
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    t[::downsample],
    abr_eeg[::downsample],
    color="orangered",
    linewidth=0.5,
    label="ABR + EEG",
)
ax.plot(
    t[::downsample],
    stimulus[::downsample],
    color="steelblue",
    linewidth=0.5,
    label="Stimulus",
)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title("ABR + EEG Signal and Stimulus Train (Raw - Downsampled)")
ax.grid(alpha=0.3)
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("fig_2_1_1_raw_signals_overview.png", dpi=150)
plt.show()


# --------------------------------------------------------------------------
# 2.1.1 (3) Threshold-based stimulus detection using np.where
# --------------------------------------------------------------------------
threshold = 50
above_thresh = np.where(stimulus > threshold)[0]

# --------------------------------------------------------------------------
# 2.1.1 (4) Extract the leading edge (onset) of each detected pulse
# --------------------------------------------------------------------------
leading_edges = above_thresh[np.where(np.diff(above_thresh) > 1)[0] + 1]
print(f"Number of stimuli detected: {len(leading_edges)}")

# --------------------------------------------------------------------------
# 2.1.1 (5) Window ABR epochs: -2 ms to +10 ms  =>  -80 to +399 samples @40kHz
# --------------------------------------------------------------------------
pre_samples  = 80    # -2 ms
post_samples = 400   # +10 ms  (so epoch length = 480 samples)
epoch_len = pre_samples + post_samples

epochs = []
for edge in leading_edges:
    start = edge - pre_samples
    end   = edge + post_samples
    if start >= 0 and end <= len(abr_eeg):        # discard incomplete epochs at the edges
        epochs.append(abr_eeg[start:end])
epochs = np.array(epochs)          # shape: (num_epochs, epoch_len)
print(f"Number of usable epochs: {epochs.shape[0]}")

# --------------------------------------------------------------------------
# 2.1.1 (6) Ensemble average of all epochs
# --------------------------------------------------------------------------
ensmbl_avg = np.mean(epochs, axis=0)

# --------------------------------------------------------------------------
# 2.1.1 (7) Plot the ensemble-averaged ABR waveform (cf. Figure 1)
# --------------------------------------------------------------------------
t_epoch = np.linspace(-2, 10, epoch_len)   # ms

plt.figure(figsize=(8, 5))
plt.plot(t_epoch, ensmbl_avg, color="k")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (µV)")
plt.title(f"Synchronised averaged ABR from {epochs.shape[0]} epochs")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_2_1_1_ensemble_avg_ABR.png", dpi=150)
plt.show()

# ==========================================================================
# 2.1.2 Improvement of the SNR
# ==========================================================================
# MSE_k = sqrt( sum_n (y(n) - yhat_k(n))^2 / N ),  k = 1..M
# y(n)      = final ensemble average (ensmbl_avg), used as the "template"
# yhat_k(n) = progressive average of the first k epochs
# --------------------------------------------------------------------------

def progressive_mse(epochs, template):
    """
    epochs   : (M, N) array - each row is a single epoch
    template : (N,)   array - reference/template waveform (final ensemble avg)
    returns  : (M,) array of MSE_k values, k = 1..M
    """
    M, N = epochs.shape
    mse = np.zeros(M)
    running_sum = np.zeros(N)
    for k in range(1, M + 1):
        running_sum += epochs[k - 1]
        yhat_k = running_sum / k
        mse[k - 1] = np.sqrt(np.sum((template - yhat_k) ** 2) / N)
    return mse

mse_k = progressive_mse(epochs, ensmbl_avg)

plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, len(mse_k) + 1), mse_k, color="darkred")
plt.xlabel("Number of epochs averaged, k")
plt.ylabel("MSE$_k$")
plt.title("Progressive MSE vs. number of epochs in the ensemble average")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_2_1_2_mse_vs_k.png", dpi=150)
plt.show()
