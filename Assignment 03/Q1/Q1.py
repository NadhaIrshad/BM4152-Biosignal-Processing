import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

# Run from the script's own folder so figures are saved next to it
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# =====================================================================
# 1.2 WAVELET PROPERTIES
# =====================================================================
#
# 1.2(i)   g(t) = 1/sqrt(2*pi) * exp(-t^2/2)            (mu = 0, sigma = 1)
#          g'(t)  = -t * g(t)
#          g''(t) = (t^2 - 1) * g(t)
#          m(t) = -g''(t) = (1 - t^2) * exp(-t^2/2) / sqrt(2*pi)
#
# 1.2(ii)  E = integral m^2(t) dt
#            = 1/(2*pi) * integral (1 - 2t^2 + t^4) exp(-t^2) dt
#            = 1/(2*pi) * (sqrt(pi) - sqrt(pi) + 3*sqrt(pi)/4)
#            = 3 / (8*sqrt(pi))
#          Normalizing factor c = 1/sqrt(E) = sqrt(8*sqrt(pi)/3)
#
# 1.2(iii) psi(t) = c * m(t) = 2 / (sqrt(3) * pi^(1/4)) * (1 - t^2) * exp(-t^2/2)
#          psi_{s,tau}(t) = 1/sqrt(s) * psi((t - tau)/s)
# =====================================================================

fs = 250
N = 3000
t = np.arange(-N, N + 1) / fs
dt = 1 / fs

# 0.01, 0.11, ..., 1.91.
scales = 0.01 + 0.1 * np.arange(20)


def mexican_hat(t, scale, translation=0.0):
    """Return the normalized Mexican-hat daughter wavelet."""
    u = (t - translation) / scale
    c = 2.0 / (np.sqrt(3.0) * np.pi ** 0.25)
    return c * (1.0 - u ** 2) * np.exp(-0.5 * u ** 2) / np.sqrt(scale)


E_m = 3 / (8 * np.sqrt(np.pi))
print("1.2(ii)")
print(f"Energy of m(t)          = 3/(8*sqrt(pi)) = {E_m:.6f}")
print(f"Normalizing factor      = {1 / np.sqrt(E_m):.6f}")
print(f"psi(t) leading constant = 2/(sqrt(3)*pi^0.25) = "
      f"{2 / (np.sqrt(3) * np.pi ** 0.25):.6f}")

# ---------------------------------------------------------------------
# 1.2(iv) Time-domain daughter wavelets
# ---------------------------------------------------------------------

wavelets = np.array([mexican_hat(t, s) for s in scales])

fig, axs = plt.subplots(5, 4, figsize=(14, 11))

for ax, s, w in zip(axs.ravel(), scales, wavelets):
    ax.plot(t, w)
    ax.set_xlim(-6 * s - 0.05, 6 * s + 0.05)     # zoom on the support of each wavelet
    ax.set_title(f"s = {s:.2f}", fontsize=10)
    ax.grid(alpha=0.3)

for ax in axs[-1, :]:
    ax.set_xlabel("Time (s)")
for ax in axs[:, 0]:
    ax.set_ylabel("Amplitude")

fig.suptitle("1.2(iv): Mexican-hat daughter wavelets (individually zoomed)")
plt.tight_layout()
plt.savefig("fig1_2_iv_daughter_wavelets.png", dpi=150)
plt.show()

# All daughter wavelets on a common time axis
plt.figure(figsize=(11, 5))

for s, w in zip(scales, wavelets):
    plt.plot(t, w, label=f"s={s:.2f}")

plt.xlim(-8, 8)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("1.2(iv): Mexican-hat daughter wavelets (common axis)")
plt.legend(ncol=4, fontsize=7)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig1_2_iv_daughter_wavelets_overlay.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------
# 1.2(v) Zero mean, unity energy, compact support
# ---------------------------------------------------------------------

print("\n1.2(v): Wavelet properties")
print(f"{'scale':>6} {'mean':>12} {'energy':>10} {'support (s)':>12}")

means = []
energies = []
supports = []

for s, w in zip(scales, wavelets):

    mean = np.sum(w) * dt                 # numerical integral of psi(t)
    energy = np.sum(w ** 2) * dt          # numerical integral of psi^2(t)

    # Effective support: region where |psi| exceeds 0.1% of its peak
    significant = t[np.abs(w) > 1e-3 * np.max(np.abs(w))]
    support = significant[-1] - significant[0]

    means.append(mean)
    energies.append(energy)
    supports.append(support)

    print(f"{s:6.2f} {mean:12.3e} {energy:10.6f} {support:12.3f}")

fig, axs = plt.subplots(1, 3, figsize=(14, 4))

axs[0].stem(scales, means)
axs[0].set_title("Mean")
axs[0].set_ylabel(r"$\int \psi_s(t)\,dt$")

axs[1].stem(scales, energies)
axs[1].set_title("Energy")
axs[1].set_ylabel(r"$\int \psi_s^2(t)\,dt$")
axs[1].set_ylim(0, 1.2)

axs[2].stem(scales, supports)
axs[2].set_title("Effective support width")
axs[2].set_ylabel("Width (s)")

for ax in axs:
    ax.set_xlabel("Scale s")
    ax.grid(alpha=0.3)

fig.suptitle("1.2(v): Properties of the daughter wavelets")
plt.tight_layout()
plt.savefig("fig1_2_v_wavelet_properties.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------
# 1.2(vi) Spectra of daughter wavelets
# ---------------------------------------------------------------------

NFFT = 2 ** 15
freq = np.fft.rfftfreq(NFFT, d=dt)

plt.figure(figsize=(11, 5))

print("\n1.2(vi): Spectral peaks")
print(f"{'scale':>6} {'peak (Hz)':>10} {'theory (Hz)':>12}")

for s, w in zip(scales, wavelets):

    spectrum = np.abs(np.fft.rfft(w, n=NFFT)) * dt

    plt.plot(freq, spectrum, label=f"s={s:.2f}")

    # Theoretical centre frequency of the Mexican hat: f = sqrt(2)/(2*pi*s)
    print(f"{s:6.2f} {freq[np.argmax(spectrum)]:10.3f} "
          f"{np.sqrt(2) / (2 * np.pi * s):12.3f}")

plt.xlim(0, 5)
plt.xlabel("Frequency (Hz)")
plt.ylabel(r"$|\Psi_s(f)|$")
plt.title("1.2(vi): Spectra of Mexican-hat daughter wavelets")
plt.legend(ncol=4, fontsize=7)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig1_2_vi_wavelet_spectra.png", dpi=150)
plt.show()

# Same spectra on a log-frequency axis (shows the constant-Q behaviour)
plt.figure(figsize=(11, 5))

for s, w in zip(scales, wavelets):
    spectrum = np.abs(np.fft.rfft(w, n=NFFT)) * dt
    plt.semilogx(freq[1:], spectrum[1:], label=f"s={s:.2f}")

plt.xlabel("Frequency (Hz)")
plt.ylabel(r"$|\Psi_s(f)|$")
plt.title("1.2(vi): Spectra of daughter wavelets (log-frequency axis)")
plt.legend(ncol=4, fontsize=7)
plt.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig("fig1_2_vi_wavelet_spectra_log.png", dpi=150)
plt.show()


# =====================================================================
# 1.3 CONTINUOUS WAVELET DECOMPOSITION
# =====================================================================

# ---------------------------------------------------------------------
# 1.3(i) Test waveform
# ---------------------------------------------------------------------

n = np.arange(1, 3 * N)

x = np.where(
    n < 3 * N / 2,
    np.sin(0.5 * np.pi * n / fs),      # 0.25 Hz
    np.sin(1.5 * np.pi * n / fs)       # 0.75 Hz
)

plt.figure(figsize=(11, 4))
plt.plot(n, x)
plt.axvline(3 * N / 2, color="k", linestyle="--", label="Frequency switch")
plt.xlabel("Sample index n")
plt.ylabel("Amplitude")
plt.title("1.3(i): x[n]")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig1_3_i_signal.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------
# 1.3(ii) CWT by convolution with the scaled wavelets
# ---------------------------------------------------------------------

cwt_scales = 0.01 * np.arange(1, 201)

coefficients = np.zeros((len(cwt_scales), len(x)))

for i, s in enumerate(cwt_scales):

    # The Mexican hat is even, so convolution equals correlation.
    # dt approximates the integral in the CWT definition.
    coefficients[i, :] = fftconvolve(
        x,
        mexican_hat(t, s),
        mode="same"
    ) * dt

# Coefficients for all scales on one figure (each line is one row of the
# coefficient matrix, i.e. one convolution output, coloured by its scale)
fig, axs = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                        gridspec_kw={"height_ratios": [1, 2.5]},
                        constrained_layout=True)

axs[0].plot(n, x, color="k")
axs[0].set_title("x[n]")

cmap = plt.get_cmap("viridis")
norm = plt.Normalize(cwt_scales[0], cwt_scales[-1])

for s, row in zip(cwt_scales, coefficients):
    axs[1].plot(n, row, color=cmap(norm(s)), linewidth=0.6)

axs[1].set_title(r"$W(s, \tau)$ for all 200 scales")
axs[1].set_xlabel("Sample index n (translation)")

for ax in axs:
    ax.axvline(3 * N / 2, color="k", linestyle="--", linewidth=1)
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)

fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs[1], label="Scale s")
fig.suptitle("1.3(ii): CWT coefficients at all scales")
plt.savefig("fig1_3_ii_cwt_all_scales.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------
# 1.3(iii) Scalogram
# ---------------------------------------------------------------------

plt.figure(figsize=(11, 6))

pcm = plt.pcolormesh(
    n,
    cwt_scales,
    coefficients,
    cmap="jet",
    shading="auto"
)

plt.colorbar(pcm, label="CWT coefficient")
plt.xlabel("Sample index n")
plt.ylabel("Scale")
plt.title("1.3(iii): Spectrogram from the CWT")
plt.tight_layout()
plt.savefig("fig1_3_iii_cwt_spectrogram.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------
# 1.3(iv) Scale of maximum response in each half
# ---------------------------------------------------------------------

half = len(x) // 2
margin = 1000                           # skip edge / transition regions

rms_first = np.sqrt(np.mean(coefficients[:, margin:half - margin] ** 2, axis=1))
rms_second = np.sqrt(np.mean(coefficients[:, half + margin:-margin] ** 2, axis=1))

s_first = cwt_scales[np.argmax(rms_first)]
s_second = cwt_scales[np.argmax(rms_second)]

print("\n1.3(iv): Dominant scales")
print(f"First half  (0.25 Hz): scale = {s_first:.2f}, "
      f"pseudo-frequency = {np.sqrt(2) / (2 * np.pi * s_first):.3f} Hz")
print(f"Second half (0.75 Hz): scale = {s_second:.2f}, "
      f"pseudo-frequency = {np.sqrt(2) / (2 * np.pi * s_second):.3f} Hz")

plt.figure(figsize=(8, 4))
plt.plot(cwt_scales, rms_first, label=f"First half (peak s = {s_first:.2f})")
plt.plot(cwt_scales, rms_second, label=f"Second half (peak s = {s_second:.2f})")
plt.xlabel("Scale")
plt.ylabel("RMS of CWT coefficients")
plt.title("1.3(iv): Coefficient strength vs scale")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig1_3_iv_rms_vs_scale.png", dpi=150)
plt.show()

print("All figures have been saved.")
