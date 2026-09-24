import numpy as np
from scipy.signal import (butter, iircomb, freqz, group_delay,
                           tf2sos, sosfreqz)
import matplotlib.pyplot as plt

fs = 500
nyq = fs / 2

data = np.load("ECG_with_noise.npz")
print("Keys in ECG_with_noise.npz:", data.files)
ecg_noisy = data[data.files[0]].squeeze()

gpass = 1     # dB - max passband ripple (used only for discussion/reference)
gstop = 40    # dB - min stopband attenuation (used only for discussion/reference)

IIR_ORDER = 4


# 4.1 (1)-(2) Butterworth lowpass filter (same cutoff as FIR lowpass, Sec 3.2)

lp_cutoff = 125          # Hz - same passband edge used for the FIR lowpass
lp_N = IIR_ORDER
lp_b, lp_a = butter(lp_N, lp_cutoff, btype="low", fs=fs)
print(f"Butterworth lowpass: order N = {lp_N}, cutoff = {lp_cutoff} Hz")
print("Butterworth lowpass coefficients:")
print("b =", lp_b)
print("a =", lp_a)

w_lp, h_lp = freqz(lp_b, lp_a, worN=4096, fs=fs)
w_gd, gd_lp = group_delay((lp_b, lp_a), w=4096, fs=fs)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].plot(w_lp, 20 * np.log10(np.abs(h_lp) + 1e-12))
axes[0].set_title(f"Butterworth LP magnitude response (N={lp_N})")
axes[0].set_xlabel("Frequency (Hz)"); axes[0].set_ylabel("dB")
axes[0].grid(alpha=0.3)

axes[1].plot(w_lp, np.unwrap(np.angle(h_lp)))
axes[1].set_title("Phase response")
axes[1].set_xlabel("Frequency (Hz)"); axes[1].set_ylabel("Phase (rad)")
axes[1].grid(alpha=0.3)

axes[2].plot(w_gd, gd_lp)
axes[2].set_title("Group delay")
axes[2].set_xlabel("Frequency (Hz)"); axes[2].set_ylabel("Samples")
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("fig_4_1_lowpass_response.png", dpi=150)
plt.show()

# 4.1 (3a) Butterworth highpass filter (same cutoff as FIR highpass, Sec 3.2)
# ==========================================================================
hp_cutoff = 5             # Hz - same passband edge used for the FIR highpass
hp_N = IIR_ORDER
hp_b, hp_a = butter(hp_N, hp_cutoff, btype="high", fs=fs)
print(f"Butterworth highpass: order N = {hp_N}, cutoff = {hp_cutoff} Hz")
print("Butterworth highpass coefficients:")
print("b =", hp_b)
print("a =", hp_a)

w_hp, h_hp = freqz(hp_b, hp_a, worN=4096, fs=fs)
w_gd_hp, gd_hp = group_delay((hp_b, hp_a), w=4096, fs=fs)


# 4.1 (3b) IIR comb (notch) filter using scipy.signal.iircomb

comb_w0 = 50.0          # Hz - fundamental powerline frequency
comb_bw = 4.0           # Hz - desired notch bandwidth (~ +/-2 Hz, matching Sec 3.2)
comb_Q = comb_w0 / comb_bw
comb_b, comb_a = iircomb(comb_w0, comb_Q, ftype="notch", fs=fs)
print(f"IIR comb filter: w0 = {comb_w0} Hz, Q = {comb_Q:.2f}")
print("IIR comb coefficients:")
print("b =", comb_b)
print("a =", comb_a)

w_comb, h_comb = freqz(comb_b, comb_a, worN=4096, fs=fs)
w_gd_comb, gd_comb = group_delay((comb_b, comb_a), w=4096, fs=fs)

fig, axes = plt.subplots(2, 3, figsize=(16, 8))

axes[0, 0].plot(w_hp, 20 * np.log10(np.abs(h_hp) + 1e-12), color="tab:blue")
axes[0, 0].set_title(f"Butterworth HP magnitude response (N={hp_N})")
axes[0, 0].set_xlabel("Frequency (Hz)"); axes[0, 0].set_ylabel("dB")
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(w_hp, np.unwrap(np.angle(h_hp)), color="tab:blue")
axes[0, 1].set_title("HP phase response")
axes[0, 1].set_xlabel("Frequency (Hz)"); axes[0, 1].set_ylabel("Phase (rad)")
axes[0, 1].grid(alpha=0.3)

axes[0, 2].plot(w_gd_hp, gd_hp, color="tab:blue")
axes[0, 2].set_title("HP group delay")
axes[0, 2].set_xlabel("Frequency (Hz)"); axes[0, 2].set_ylabel("Samples")
# The highpass magnitude response passes through ~0 right at DC, making the
# phase (and hence group delay) numerically singular there - clip the
# y-axis so this single-point singularity doesn't hide the actual passband
# group delay behaviour.
axes[0, 2].set_ylim(np.percentile(gd_hp, 1) - 2, np.percentile(gd_hp, 99) + 2)
axes[0, 2].grid(alpha=0.3)

axes[1, 0].plot(w_comb, 20 * np.log10(np.abs(h_comb) + 1e-12), color="tab:red")
axes[1, 0].set_title(f"IIR comb magnitude response (w0={comb_w0} Hz, Q={comb_Q:.1f})")
axes[1, 0].set_xlabel("Frequency (Hz)"); axes[1, 0].set_ylabel("dB")
axes[1, 0].grid(alpha=0.3)

axes[1, 1].plot(w_comb, np.unwrap(np.angle(h_comb)), color="tab:red")
axes[1, 1].set_title("Comb phase response")
axes[1, 1].set_xlabel("Frequency (Hz)"); axes[1, 1].set_ylabel("Phase (rad)")
axes[1, 1].grid(alpha=0.3)

axes[1, 2].plot(w_gd_comb, gd_comb, color="tab:red")
axes[1, 2].set_title("Comb group delay")
axes[1, 2].set_xlabel("Frequency (Hz)"); axes[1, 2].set_ylabel("Samples")
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("fig_4_1_hp_and_comb_response.png", dpi=150)
plt.show()

# ==========================================================================
# 4.1 (4) Combined magnitude response - compare against the FIR filter (3.2)
# ==========================================================================
# Cascade the three IIR filters: overall transfer function is the product of
# the individual transfer functions, i.e. numerator/denominator polynomials
# convolved together.
comb_num = np.convolve(np.convolve(hp_b, lp_b), comb_b)
comb_den = np.convolve(np.convolve(hp_a, lp_a), comb_a)
w_iir_comb, h_iir_comb = freqz(comb_num, comb_den, worN=4096, fs=fs)

# Recreate the FIR combined filter from Section 3.2 for a direct comparison
from scipy.signal import firwin, kaiserord
ripple_db = 40
hp_fir_M, hp_fir_beta = kaiserord(ripple_db, (5 - 3) / nyq); hp_fir_M |= 1
hp_fir_taps = firwin(hp_fir_M, 5 / nyq, window=("kaiser", hp_fir_beta), pass_zero=False)
lp_fir_M, lp_fir_beta = kaiserord(ripple_db, (135 - 125) / nyq); lp_fir_M |= 1
lp_fir_taps = firwin(lp_fir_M, 125 / nyq, window=("kaiser", lp_fir_beta), pass_zero=True)
notch_fir_M, notch_fir_beta = kaiserord(ripple_db, 1 / nyq); notch_fir_M |= 1
bands = []
for f0 in [50, 100, 150]:
    bands.extend([(f0 - 2) / nyq, (f0 + 2) / nyq])
notch_fir_taps = firwin(notch_fir_M, bands, window=("kaiser", notch_fir_beta), pass_zero=True)
fir_combined = np.convolve(np.convolve(hp_fir_taps, lp_fir_taps), notch_fir_taps)
w_fir, h_fir = freqz(fir_combined, worN=4096, fs=fs)

plt.figure(figsize=(11, 5))
plt.plot(w_fir, 20 * np.log10(np.abs(h_fir) + 1e-12), label="FIR (Kaiser window, Sec. 3.2)",
          color="purple", alpha=0.8)
plt.plot(w_iir_comb, 20 * np.log10(np.abs(h_iir_comb) + 1e-12),
          label="IIR (Butterworth + comb, Sec. 4.1)", color="darkorange", alpha=0.8)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Combined magnitude response: FIR (Kaiser) vs. IIR (Butterworth+comb)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_4_1_4_FIR_vs_IIR_combined_response.png", dpi=150)
plt.show()
