import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pywt                                  # pip install PyWavelets

# Run from the script's own folder so helper.py, ECGsig.npz and figures are found
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from helper import awgn

# 'db9' on 1024 samples exceeds PyWavelets' recommended maximum level (5);
# the assignment asks for 10 levels, so the boundary-effect warning is silenced.
warnings.filterwarnings("ignore", category=UserWarning, module="pywt")

rng = np.random.default_rng(0)

WAVELETS = ["db9", "haar"]
LEVEL = 10

# 'periodization' gives a non-redundant, orthogonal transform: exactly N
# coefficients for N samples, and coefficient energy equals signal energy.
MODE = "periodization"


# =====================================================================
# 2.2 APPLYING DWT WITH PYWAVELETS
# =====================================================================

# ---------------------------------------------------------------------
# 2.2(i) Signal construction
# ---------------------------------------------------------------------

fs = 512
n = np.arange(1024)

x1 = np.where(
    n < 512,
    2 * np.sin(20 * np.pi * n / fs) + np.sin(80 * np.pi * n / fs),
    0.5 * np.sin(40 * np.pi * n / fs) + np.sin(60 * np.pi * n / fs)
)

x2 = np.zeros(len(n))
x2[(n >= 0) & (n < 64)] = 1
x2[(n >= 192) & (n < 256)] = 2
x2[(n >= 256) & (n < 512)] = -1
x2[(n >= 512) & (n < 704)] = 3
x2[(n >= 704) & (n < 960)] = 1

SNR_dB = 10

y1 = awgn(x1, SNR_dB, rng)
y2 = awgn(x2, SNR_dB, rng)

signals = {
    "x1": (x1, y1),
    "x2": (x2, y2)
}

fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

for ax, (name, (x, y)) in zip(axs, signals.items()):
    ax.plot(n, y, label=rf"$y_{name[1]}[n]$ (noisy, {SNR_dB} dB)", alpha=0.6)
    ax.plot(n, x, label=rf"$x_{name[1]}[n]$ (clean)", linewidth=1.8)
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

axs[1].set_xlabel("Sample index n")
fig.suptitle("2.2(i): Clean and noisy signals")
plt.tight_layout()
plt.savefig("fig2_2_i_signals.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------
# 2.2(ii) Wavelet and scaling functions
# ---------------------------------------------------------------------

fig, axs = plt.subplots(2, 2, figsize=(11, 7))

for row, wname in zip(axs, WAVELETS):

    phi, psi, xw = pywt.Wavelet(wname).wavefun(level=10)

    row[0].plot(xw, phi)
    row[0].set_title(rf"{wname}: scaling function $\phi(t)$")

    row[1].plot(xw, psi, color="tab:orange")
    row[1].set_title(rf"{wname}: wavelet function $\psi(t)$")

    for ax in row:
        ax.set_xlabel("t")
        ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("fig2_2_ii_wavefun.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------
# 2.2(iii) 10-level decomposition
# ---------------------------------------------------------------------

def decompose(y, wname, level=LEVEL):
    """Return [cA_L, cD_L, ..., cD_1]."""
    return pywt.wavedec(y, wname, mode=MODE, level=level)


coeffs = {}

print("2.2(iii): Coefficient lengths [cA10, cD10, ..., cD1]")

for name, (x, y) in signals.items():
    for wname in WAVELETS:
        coeffs[name, wname] = decompose(y, wname)
        print(f"  y{name[1]}, {wname:>4}: {[len(c) for c in coeffs[name, wname]]}")

# Plot the decomposition coefficients
for (name, wname), c in coeffs.items():

    labels = [f"cA{LEVEL}"] + [f"cD{i}" for i in range(LEVEL, 0, -1)]

    fig, axs = plt.subplots(len(c), 1, figsize=(10, 13))

    for ax, ci, lab in zip(axs, c, labels):
        ax.stem(ci, markerfmt=" ", basefmt="k-")
        ax.set_ylabel(lab, rotation=0, labelpad=20)
        ax.grid(alpha=0.3)

    axs[-1].set_xlabel("Coefficient index")
    fig.suptitle(f"2.2(iii): {LEVEL}-level DWT coefficients of y{name[1]} ({wname})")
    plt.tight_layout()
    plt.savefig(f"fig2_2_iii_coeffs_y{name[1]}_{wname}.png", dpi=150)
    plt.show()

# ---------------------------------------------------------------------
# 2.2(iv) Reconstruct A10, D10, ..., D1 and verify y = A10 + sum(Di)
# ---------------------------------------------------------------------

def reconstruct_components(c, wname):
    """
    Reconstruct each sub-band separately with the inverse DWT.

    For every sub-band, all other coefficient arrays are set to zero
    before calling waverec, so the result is the contribution of that
    sub-band alone, at full signal length.
    """
    components = []

    for i in range(len(c)):
        only_i = [ci if j == i else np.zeros_like(ci) for j, ci in enumerate(c)]
        components.append(pywt.waverec(only_i, wname, mode=MODE))

    return components        # [A_L, D_L, ..., D_1]


print("\n2.2(iv): Verification of y = A10 + sum(Di)")

for (name, wname), c in coeffs.items():

    y = signals[name][1]

    components = reconstruct_components(c, wname)
    y_rec = np.sum(components, axis=0)[:len(y)]

    E_y = np.sum(y ** 2)
    E_rec = np.sum(y_rec ** 2)
    E_err = np.sum((y - y_rec) ** 2)

    print(f"  y{name[1]}, {wname:>4}: E(y) = {E_y:.6f}, "
          f"E(reconstructed) = {E_rec:.6f}, "
          f"E(y - reconstructed) = {E_err:.3e}")

    labels = [f"A{LEVEL}"] + [f"D{i}" for i in range(LEVEL, 0, -1)]

    fig, axs = plt.subplots(len(components) + 1, 1, figsize=(10, 14), sharex=True)

    axs[0].plot(n, y, label="y")
    axs[0].plot(n, y_rec, "--", label=r"$A_{10}+\sum D_i$")
    axs[0].legend(loc="upper right", fontsize=7, ncol=2)
    axs[0].set_ylabel("y", rotation=0, labelpad=20)

    for ax, comp, lab in zip(axs[1:], components, labels):
        ax.plot(n, comp[:len(y)])
        ax.set_ylabel(lab, rotation=0, labelpad=20)

    for ax in axs:
        ax.grid(alpha=0.3)

    axs[-1].set_xlabel("Sample index n")
    fig.suptitle(f"2.2(iv): Reconstructed sub-bands of y{name[1]} ({wname})")
    plt.tight_layout()
    plt.savefig(f"fig2_2_iv_components_y{name[1]}_{wname}.png", dpi=150)
    plt.show()


# =====================================================================
# 2.3 SIGNAL DENOISING WITH DWT
# =====================================================================

# Thresholds selected by observation of the sorted-coefficient stem plots:
# the value at the "knee", below which the coefficients form the flat noise floor.
THRESHOLDS = {
    ("x1", "db9"): 1.0,          # >>> ADJUST after viewing the stem plots
    ("x1", "haar"): 1.0,
    ("x2", "db9"): 1.5,
    ("x2", "haar"): 2.0
}

rmse = {}
denoised = {}

print("\n2.3: Denoising")

for (name, wname), c in coeffs.items():

    x, y = signals[name]
    threshold = THRESHOLDS[name, wname]

    # -----------------------------------------------------------------
    # 2.3(i) Sorted coefficient magnitudes
    # -----------------------------------------------------------------

    flat, slices = pywt.coeffs_to_array(c)
    sorted_mag = np.sort(np.abs(flat))[::-1]

    kept = np.sum(np.abs(flat) > threshold)

    plt.figure(figsize=(11, 4))
    plt.stem(sorted_mag, markerfmt=" ", basefmt="k-")
    plt.axhline(threshold, color="r", linestyle="--",
                label=f"Threshold = {threshold} ({kept} coefficients kept)")
    plt.xlabel("Coefficient rank")
    plt.ylabel("|coefficient|")
    plt.title(f"2.3(i): Sorted DWT coefficient magnitudes of y{name[1]} ({wname})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"fig2_3_i_sorted_coeffs_y{name[1]}_{wname}.png", dpi=150)
    plt.show()

    # -----------------------------------------------------------------
    # 2.3(ii) Suppress small coefficients (hard threshold) and reconstruct
    # -----------------------------------------------------------------

    flat_thr = np.where(np.abs(flat) > threshold, flat, 0.0)

    c_thr = pywt.array_to_coeffs(flat_thr, slices, output_format="wavedec")
    x_hat = pywt.waverec(c_thr, wname, mode=MODE)[:len(y)]

    # -----------------------------------------------------------------
    # 2.3(iii) RMSE against the clean signal
    # -----------------------------------------------------------------

    rmse[name, wname] = np.sqrt(np.mean((x - x_hat) ** 2))
    denoised[name, wname] = x_hat

    rmse_noisy = np.sqrt(np.mean((x - y) ** 2))

    print(f"  x{name[1]}, {wname:>4}: threshold = {threshold}, "
          f"kept {kept}/{len(flat)} coefficients, "
          f"RMSE noisy = {rmse_noisy:.4f}, RMSE denoised = {rmse[name, wname]:.4f}")

    plt.figure(figsize=(11, 4))
    plt.plot(n, y, color="0.75", label="Noisy")
    plt.plot(n, x, label="Original", linewidth=1.8)
    plt.plot(n, x_hat, label=f"Denoised (RMSE = {rmse[name, wname]:.4f})", linewidth=1.3)
    plt.xlabel("Sample index n")
    plt.ylabel("Amplitude")
    plt.title(f"2.3(iii): Original vs denoised x{name[1]} ({wname})")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"fig2_3_iii_denoised_x{name[1]}_{wname}.png", dpi=150)
    plt.show()

# ---------------------------------------------------------------------
# 2.3(v) db9 vs Haar comparison
# ---------------------------------------------------------------------

fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

for ax, name in zip(axs, signals):
    ax.plot(n, signals[name][0], "k", linewidth=1.8, label="Original")
    for wname in WAVELETS:
        ax.plot(n, denoised[name, wname],
                label=f"{wname} (RMSE = {rmse[name, wname]:.4f})", alpha=0.85)
    ax.set_ylabel("Amplitude")
    ax.set_title(f"x{name[1]}")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

axs[1].set_xlabel("Sample index n")
fig.suptitle("2.3(v): Denoising with db9 vs Haar")
plt.tight_layout()
plt.savefig("fig2_3_v_db9_vs_haar.png", dpi=150)
plt.show()

print("\n2.3(v): RMSE summary")
print(f"{'signal':>8} {'db9':>10} {'haar':>10}")
for name in signals:
    print(f"{name:>8} {rmse[name, 'db9']:10.4f} {rmse[name, 'haar']:10.4f}")


# =====================================================================
# 2.4 SIGNAL COMPRESSION WITH DWT
# =====================================================================

data = np.load("ECGsig.npz")

print("\nKeys found in ECGsig.npz:")
print(data.files)

ecg = data[data.files[0]].astype(float).flatten()

fs_ecg = 257
t_ecg = np.arange(len(ecg)) / fs_ecg

ENERGY_FRACTION = 0.99

compressed = {}

print("\n2.4: Compression")

for wname in WAVELETS:

    # -----------------------------------------------------------------
    # 2.4(i) Discrete wavelet coefficients
    # -----------------------------------------------------------------

    level = pywt.dwt_max_level(len(ecg), pywt.Wavelet(wname).dec_len)

    c = pywt.wavedec(ecg, wname, mode=MODE, level=level)
    flat, slices = pywt.coeffs_to_array(c)

    # -----------------------------------------------------------------
    # 2.4(ii) Number of coefficients holding 99% of the energy
    # -----------------------------------------------------------------

    order = np.argsort(np.abs(flat))[::-1]
    cumulative = np.cumsum(flat[order] ** 2) / np.sum(flat ** 2)

    K = int(np.searchsorted(cumulative, ENERGY_FRACTION) + 1)

    # -----------------------------------------------------------------
    # 2.4(iii) Keep only the K largest coefficients and reconstruct
    # -----------------------------------------------------------------

    flat_c = np.zeros_like(flat)
    flat_c[order[:K]] = flat[order[:K]]

    c_c = pywt.array_to_coeffs(flat_c, slices, output_format="wavedec")
    ecg_hat = pywt.waverec(c_c, wname, mode=MODE)[:len(ecg)]

    ratio = len(ecg) / K
    err = np.sqrt(np.mean((ecg - ecg_hat) ** 2))
    prd = 100 * np.sqrt(np.sum((ecg - ecg_hat) ** 2) / np.sum(ecg ** 2))

    compressed[wname] = (ecg_hat, K, ratio, err)

    print(f"  {wname:>4}: level = {level}, total coefficients = {len(flat)}, "
          f"K(99%) = {K}, compression ratio = {ratio:.2f}:1, "
          f"RMSE = {err:.5f}, PRD = {prd:.2f}%")

    fig, axs = plt.subplots(1, 2, figsize=(13, 4))

    axs[0].stem(np.abs(flat[order]), markerfmt=" ", basefmt="k-")
    axs[0].axvline(K, color="r", linestyle="--", label=f"K = {K}")
    axs[0].set_xlabel("Coefficient rank")
    axs[0].set_ylabel("|coefficient|")
    axs[0].set_title(f"Sorted coefficient magnitudes ({wname})")

    axs[1].plot(np.arange(1, len(flat) + 1), 100 * cumulative)
    axs[1].axhline(100 * ENERGY_FRACTION, color="k", linestyle=":")
    axs[1].axvline(K, color="r", linestyle="--", label=f"K = {K}")
    axs[1].set_xscale("log")
    axs[1].set_xlabel("Number of coefficients retained")
    axs[1].set_ylabel("Cumulative energy (%)")
    axs[1].set_title(f"Cumulative energy ({wname})")

    for ax in axs:
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"fig2_4_ii_energy_{wname}.png", dpi=150)
    plt.show()

# Original vs compressed ECG
ZOOM_START = 0.0       # seconds     >>> ADJUST
ZOOM_END = 3.0

for xlim, tag in [((t_ecg[0], t_ecg[-1]), "full"), ((ZOOM_START, ZOOM_END), "zoomed")]:

    fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    for ax, wname in zip(axs, WAVELETS):
        ecg_hat, K, ratio, err = compressed[wname]
        ax.plot(t_ecg, ecg, label="Original aVR", linewidth=1.8)
        ax.plot(t_ecg, ecg_hat, label="Reconstructed", linewidth=1.2)
        ax.set_title(f"{wname}: K = {K}, compression ratio = {ratio:.2f}:1, "
                     f"RMSE = {err:.5f}")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

    axs[1].set_xlim(*xlim)
    axs[1].set_xlabel("Time (s)")
    fig.suptitle(f"2.4(iii): ECG compression with 99% energy ({tag})")
    plt.tight_layout()
    plt.savefig(f"fig2_4_iii_compressed_ecg_{tag}.png", dpi=150)
    plt.show()

print("\nDone.")
