import numpy as np
from scipy.signal import firwin, freqz, get_window
import matplotlib.pyplot as plt

# 3.1 (1) Rectangular window - effect of window length M

Ms = [5, 50, 100]
colors = ['tab:blue', 'tab:orange', 'tab:green']

plt.figure(figsize=(10, 5))

for M, color in zip(Ms, colors):
    # FIR filter using a rectangular window
    h = firwin(
        numtaps=M,
        cutoff=0.2,
        window='boxcar'
    )

    n = np.arange(M)

    markerline, stemlines, baseline = plt.stem(
        n, h,
        label=f'M = {M}'
    )

    plt.setp(markerline, color=color)
    plt.setp(stemlines, color=color)
    plt.setp(baseline, color='black')

plt.xlabel('Sample index, n')
plt.ylabel('Amplitude')
plt.title('Impulse Response of FIR Filters with Rectangular Window')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_3_1_1_rect_impulse.png", dpi=150)
plt.show()

# 3.1 (2) Lowpass FIR filter (wc = 0.4*pi) - compare M = 5, 50, 100

wc = 0.4 * np.pi                 # normalised cutoff (rad/sample)
fc_normalised = wc / np.pi       # firwin uses cutoff as a fraction of Nyquist (0 to 1)

fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

for M in Ms:
    taps = firwin(M, fc_normalised, window="boxcar")
    w, h = freqz(taps, worN=2048)
    ax_mag.plot(w / np.pi, np.abs(h), label=f"M = {M}")
    ax_phase.plot(w / np.pi, np.unwrap(np.angle(h)), label=f"M = {M}")

ax_mag.set_ylabel("|H(e^{j$\\omega$})|  (linear)")
ax_mag.set_title(f"Lowpass FIR (rectangular window), $\\omega_c$ = 0.4$\\pi$: magnitude response")
ax_mag.legend()
ax_mag.grid(alpha=0.3)

ax_phase.set_xlabel("Normalised frequency ($\\times \\pi$ rad/sample)")
ax_phase.set_ylabel("Phase (rad)")
ax_phase.set_title("Phase response")
ax_phase.legend()
ax_phase.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("fig_3_1_2_rect_lowpass_response.png", dpi=150)
plt.show()

# 3.1 (3) Rectangular vs Hanning vs Hamming vs Blackman, M = 50

M = 50
window_names = ["boxcar", "hann", "hamming", "blackman"]
window_labels = ["Rectangular", "Hanning", "Hamming", "Blackman"]

# --- (a) Morphology of the windows -----------------------------------------
plt.figure(figsize=(9, 5))
for name, label in zip(window_names, window_labels):
    w = get_window(name, M)
    plt.plot(np.arange(M), w, label=label, marker=".", markersize=3)
plt.xlabel("n (samples)")
plt.ylabel("Amplitude")
plt.title(f"Window morphology, M = {M}")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_3_1_3_window_morphology.png", dpi=150)
plt.show()

# --- (b) Magnitude response (linear scale) and (c) (log scale) -------------
fig, (ax_lin, ax_log) = plt.subplots(2, 1, figsize=(9, 8))
for name, label in zip(window_names, window_labels):
    taps = firwin(M, fc_normalised, window=name)
    w, h = freqz(taps, worN=2048)
    ax_lin.plot(w / np.pi, np.abs(h), label=label)
    ax_log.plot(w / np.pi, 20 * np.log10(np.abs(h) + 1e-12), label=label)

ax_lin.set_ylabel("|H(e^{j$\\omega$})|  (linear)")
ax_lin.set_title(f"Magnitude response comparison, M = {M} (linear scale)")
ax_lin.legend()
ax_lin.grid(alpha=0.3)

ax_log.set_xlabel("Normalised frequency ($\\times \\pi$ rad/sample)")
ax_log.set_ylabel("Magnitude (dB)")
ax_log.set_title("Magnitude response comparison (log scale)")
ax_log.set_ylim([-120, 5])
ax_log.legend()
ax_log.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("fig_3_1_3_window_magnitude_responses.png", dpi=150)
plt.show()

# --- (d) Phase response -----------------------------------------------------
plt.figure(figsize=(9, 5))
for name, label in zip(window_names, window_labels):
    taps = firwin(M, fc_normalised, window=name)
    w, h = freqz(taps, worN=2048)
    plt.plot(w / np.pi, np.unwrap(np.angle(h)), label=label)
plt.xlabel("Normalised frequency ($\\times \\pi$ rad/sample)")
plt.ylabel("Phase (rad)")
plt.title(f"Phase response comparison, M = {M}")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_3_1_3_window_phase_responses.png", dpi=150)
plt.show()

