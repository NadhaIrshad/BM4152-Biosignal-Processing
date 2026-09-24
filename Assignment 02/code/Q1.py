import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import lfilter, periodogram
from scipy.linalg import toeplitz
from scipy.signal import freqz


np.random.seed(0)


# =====================================================================
# 1.1 DATA CONSTRUCTION
# =====================================================================

data = np.load("idealECG.npz")

print("Keys found in idealECG.npz:")
print(data.files)

IDEAL_KEY = data.files[0]

yi_full = data[IDEAL_KEY].astype(float).flatten()

fs = 500.0
N = len(yi_full)

n = np.arange(N)
t = n / fs

# ---------------------------------------------------------------------
# Noise construction
# ---------------------------------------------------------------------

SNR_dB = 10.0

sig_power = np.mean(yi_full ** 2)

noise_power = sig_power / (10 ** (SNR_dB / 10))

eta_wg = np.sqrt(noise_power) * np.random.randn(N)

eta_50 = 0.2 * np.sin(2 * np.pi * 50 * t)

eta = eta_wg + eta_50

x_full = yi_full + eta

#zoomed in signal plot
ZOOM_START = 0.0       
ZOOM_END = 1.0         

plt.figure(figsize=(11, 5))

plt.plot(
    t,
    yi_full,
    label=r"$y_i(n)$"
)

plt.plot(
    t,
    x_full,
    label=r"$x(n)$",
    alpha=0.7
)

plt.xlim(
    ZOOM_START,
    ZOOM_END
)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.title(
    "1.1: ECG for Manual Segment Selection" 
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "fig1_1_zoomed_signals.png",
    dpi=150
)

plt.show()


# =====================================================================
# MANUAL SEGMENT SELECTION
# =====================================================================

BEAT_START = 220       
BEAT_END = 320         

NOISE_START = 320      
NOISE_END = 360 


# ---------------------------------------------------------------------
# Extract desired ECG beat
# ---------------------------------------------------------------------

y_beat_p1 = yi_full[
    BEAT_START:BEAT_END
]

# ---------------------------------------------------------------------
# Extract noise reference from x(n)
# ---------------------------------------------------------------------

noise_base = x_full[
    NOISE_START:NOISE_END
]

# ---------------------------------------------------------------------
# Noise segment
# ---------------------------------------------------------------------

x_beat = x_full[
    BEAT_START:BEAT_END
]

# =====================================================================
# PLOT SELECTED ECG BEAT AND NOISE REFERENCE
# =====================================================================
plt.figure(figsize=(11, 5))

plt.plot(
    t,
    yi_full,
    label=r"$y_i(n)$",
    linewidth=1.2
)

plt.axvspan(
    BEAT_START / fs,
    BEAT_END / fs,
    alpha=0.25,
    label="Selected ECG beat"
)

plt.xlim(
    max(0, (BEAT_START - 100) / fs),
    min(N / fs, (BEAT_END + 100) / fs)
)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.title(
    "Selected ECG Beat from $y_i(n)$"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "fig1_1_selected_ecg_beat.png",
    dpi=150
)

plt.show()

plt.figure(figsize=(11, 5))

plt.plot(
    t,
    x_full,
    label=r"$x(n)$",
    linewidth=1.2
)

plt.axvspan(
    NOISE_START / fs,
    NOISE_END / fs,
    alpha=0.25,
    label="Selected noise reference"
)

plt.xlim(
    max(0, (NOISE_START - 100) / fs),
    min(N / fs, (NOISE_END + 100) / fs)
)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.title(
    "Selected Noise Reference from $x(n)$"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "fig1_1_selected_noise_reference.png",
    dpi=150
)

plt.show()

# =====================================================================
# WIENER FILTER FUNCTIONS
# =====================================================================

def acf(seg, M):
    """
    Biased autocorrelation estimate.

    Returns:
        r[0], r[1], ..., r[M-1]
    """

    seg = np.asarray(seg)

    if M > len(seg):
        raise ValueError(
            "M cannot be greater than the segment length."
        )

    r = np.correlate(
        seg,
        seg,
        mode="full"
    ) / len(seg)

    mid = len(r) // 2

    return r[mid:mid + M]


def wiener_filter_design(
    desired_segment,
    noise_segment,
    M
):
    """
    Design a discrete-time Wiener FIR filter.

        w = (Phi_Y + Phi_N)^(-1) r_Y

    M = number of filter coefficients.

    FIR filter order = M - 1.
    """

    if M > len(desired_segment):
        raise ValueError(
            "M is larger than the desired segment length."
        )

    if M > len(noise_segment):
        raise ValueError(
            "M is larger than the noise-reference segment length."
        )


    # ---------------------------------------------------------------
    # Autocorrelation estimates
    # ---------------------------------------------------------------

    r_y = acf(
        desired_segment,
        M
    )

    r_n = acf(
        noise_segment,
        M
    )


    # ---------------------------------------------------------------
    # Toeplitz autocorrelation matrices
    # ---------------------------------------------------------------

    Phi_Y = toeplitz(r_y)

    Phi_N = toeplitz(r_n)


    # ---------------------------------------------------------------
    # Wiener-Hopf equation
    #
    # w = (Phi_Y + Phi_N)^(-1) r_Y
    # ---------------------------------------------------------------

    theta_Yy = r_y

    w = np.linalg.solve(
        Phi_Y + Phi_N,
        theta_Yy
    )

    return w


def apply_wiener_filter(w, x):
    """
    Apply FIR Wiener filter to input x.
    """

    return lfilter(
        w,
        [1.0],
        x
    )


def calculate_mse(
    y_reference,
    y_estimated,
    M=None,
    skip_transient=True
):
    """
    Calculate MSE.

    If skip_transient=True, the first M-1 samples are
    ignored to reduce the effect of FIR startup transient.
    """

    if skip_transient and M is not None:

        if len(y_reference) > M - 1:

            y_reference = y_reference[M - 1:]

            y_estimated = y_estimated[M - 1:]


    return np.mean(
        (y_reference - y_estimated) ** 2
    )


# =====================================================================
# 1.2 PART 1
# =====================================================================

print("PART 1 - TIME-DOMAIN WIENER FILTER")

# ---------------------------------------------------------------------
# 1.2(a) Arbitrary filter order
# ---------------------------------------------------------------------

ARBITRARY_M = 10     #selected arbitrary filter order


w0_arbitrary_p1 = wiener_filter_design(
    y_beat_p1,
    noise_base,
    ARBITRARY_M
)

print("\n1.2(a)")

print(
    f"Arbitrary number of coefficients M = "
    f"{ARBITRARY_M}"
)

print(
    f"FIR filter order = "
    f"{ARBITRARY_M - 1}"
)

print("\nPart 1 arbitrary Wiener coefficients:")

print(
    np.array2string(
        w0_arbitrary_p1,
        precision=5,
        separator=", "
    )
)

# ---------------------------------------------------------------------
# 1.2(c) FILTERED SIGNAL USING ARBITRARY FILTER
# ---------------------------------------------------------------------

yhat_arbitrary_p1 = apply_wiener_filter(
    w0_arbitrary_p1,
    x_beat
)

mse_arbitrary_p1 = calculate_mse(
    y_beat_p1,
    yhat_arbitrary_p1,
    M=ARBITRARY_M,
    skip_transient=False
)

print("\n1.2(c)")

print(
    f"Arbitrary-M filtered beat MSE = "
    f"{mse_arbitrary_p1:.6f}"
)

# ---------------------------------------------------------------------
# Plot noisy ECG vs filtered ECG
# ---------------------------------------------------------------------

time_beat = np.arange(
    len(y_beat_p1)
) / fs

plt.figure(figsize=(10, 4))

plt.plot(
    time_beat,
    y_beat_p1,
    label=r"Ideal $y_i(n)$",
    linewidth=2
)

plt.plot(
    time_beat,
    x_beat,
    label=r"Noisy $x(n)$",
    alpha=0.6
)

plt.plot(
    time_beat,
    yhat_arbitrary_p1,
    label=rf"Filtered $\hat y(n)$, M={ARBITRARY_M}",
    linewidth=2
)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.title(
    f"Part 1: Noisy vs Filtered ECG "
    f"(Arbitrary M={ARBITRARY_M})"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "fig1_2c_part1_arbitrary_filter.png",
    dpi=150
)

plt.show()

# =====================================================================
# 1.2(b) FIND OPTIMUM M - PART 1
# =====================================================================

print("\n1.2(b): Finding optimum M")

# IMPORTANT:
# M must not exceed the length of either reference segment.

MAX_M = min(
    40,
    len(y_beat_p1),
    len(noise_base)
)

M_values = np.arange(
    2,
    MAX_M + 1
)

mse_values_p1 = []


for M in M_values:

    w = wiener_filter_design(
        y_beat_p1,
        noise_base,
        M
    )

    yhat = apply_wiener_filter(
        w,
        x_beat
    )

    mse = calculate_mse(
        y_beat_p1,
        yhat,
        M=M,
        skip_transient=True
    )

    mse_values_p1.append(
        mse
    )


mse_values_p1 = np.array(
    mse_values_p1
)


best_index = np.argmin(
    mse_values_p1
)

M_p1 = M_values[best_index]


w0_p1 = wiener_filter_design(
    y_beat_p1,
    noise_base,
    M_p1
)


print(
    f"Optimum number of coefficients M = "
    f"{M_p1}"
)

print(
    f"Optimum FIR filter order = "
    f"{M_p1 - 1}"
)

print(
    f"Minimum beat MSE = "
    f"{mse_values_p1[best_index]:.6f}"
)


# ---------------------------------------------------------------------
# MSE vs M
# ---------------------------------------------------------------------

plt.figure(figsize=(7, 4))

plt.plot(
    M_values,
    mse_values_p1,
    marker="o",
    markersize=3
)

plt.axvline(
    M_p1,
    linestyle="--",
    label=f"Optimum M = {M_p1}"
)

plt.xlabel(
    "Number of filter coefficients, M"
)

plt.ylabel("MSE")

plt.title(
    "Part 1: MSE vs Filter Length"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "fig1_2b_part1_M_selection.png",
    dpi=150
)

plt.show()

#magnitude response of the optimum filter

def plot_magnitude_response(w, fs, M, part):
    """
    Plot magnitude response of the Wiener FIR filter.
    """

    frequency, response = freqz(
        w,
        worN=2048,
        fs=fs
    )

    magnitude = np.abs(response)

    plt.figure(figsize=(9, 4))

    plt.plot(
        frequency,
        magnitude
    )

    plt.xlim(0, 150)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel(r"$|H(f)|$")

    plt.title(
        f"{part}: Magnitude Response of Wiener Filter (M={M})"
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        f"fig1_2b_{part}_magnitude_response.png",
        dpi=150
    )

    plt.show()

# ---------------------------------------------------------------------
# Magnitude response of optimum Wiener filter
# ---------------------------------------------------------------------

plot_magnitude_response(
    w0_p1,
    fs,
    M_p1,
    "Part1"
)


# =====================================================================
# 1.2(c) FILTERED SIGNAL USING OPTIMUM FILTER - PART 1
# =====================================================================

yhat_p1_beat = apply_wiener_filter(
    w0_p1,
    x_beat
)

mse_p1 = calculate_mse(
    y_beat_p1,
    yhat_p1_beat,
    M=M_p1,
    skip_transient=False
)

print("\n1.2(c): Optimum filter")

print(
    f"Part 1 optimum MSE = "
    f"{mse_p1:.6f}"
)


plt.figure(figsize=(10, 4))

plt.plot(
    time_beat,
    y_beat_p1,
    label=r"Ideal $y_i(n)$",
    linewidth=2
)

plt.plot(
    time_beat,
    x_beat,
    label=r"Noisy $x(n)$",
    alpha=0.6
)

plt.plot(
    time_beat,
    yhat_p1_beat,
    label=rf"Filtered $\hat y(n)$, M={M_p1}",
    linewidth=2
)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.title(
    f"Part 1: Wiener Filtered ECG "
    f"(Optimum M={M_p1})"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "fig1_2c_part1_optimum_filtered.png",
    dpi=150
)

plt.show()

# =====================================================================
# 1.2(d) SPECTRAL CALCULATION - PART 1
# =====================================================================

print("\n1.2(d): Spectral calculation - Part 1")

# ---------------------------------------------------------------------
# Periodogram-based PSD calculation
# ---------------------------------------------------------------------

def calculate_spectrum(signal, fs):

    """
    Calculate one-sided Power Spectral Density (PSD)
    using a periodogram.

    'boxcar' corresponds to a rectangular window,
    equivalent to MATLAB rectwin(L).

    scaling='density' gives units of
    amplitude^2 / Hz.
    """

    frequency, psd = periodogram(
        signal,
        fs=fs,
        window="boxcar",
        scaling="density"
    )

    return frequency, psd

# ---------------------------------------------------------------------
# Apply optimum Wiener filter to FULL noisy ECG
# ---------------------------------------------------------------------

yhat_full_p1 = lfilter(
    w0_p1,
    [1.0],
    x_full
)

# ---------------------------------------------------------------------
# Calculate PSDs
# ---------------------------------------------------------------------

freq_p1, psd_yi_p1 = calculate_spectrum(
    yi_full,
    fs
)

_, psd_eta_p1 = calculate_spectrum(
    eta,
    fs
)

_, psd_x_p1 = calculate_spectrum(
    x_full,
    fs
)

_, psd_yhat_p1 = calculate_spectrum(
    yhat_full_p1,
    fs
)

# ---------------------------------------------------------------------
# Check 50-Hz frequency component
# ---------------------------------------------------------------------

print("\nSpectral component around 50 Hz:")

idx_50 = np.argmin(
    np.abs(freq_p1 - 50)
)

print(
    f"Frequency bin = "
    f"{freq_p1[idx_50]:.3f} Hz"
)

print(
    f"Ideal ECG PSD at 50 Hz : "
    f"{psd_yi_p1[idx_50]:.4e}"
)

print(
    f"Noise PSD at 50 Hz     : "
    f"{psd_eta_p1[idx_50]:.4e}"
)

print(
    f"Noisy ECG PSD at 50 Hz : "
    f"{psd_x_p1[idx_50]:.4e}"
)

print(
    f"Filtered PSD at 50 Hz  : "
    f"{psd_yhat_p1[idx_50]:.4e}"
)

# ---------------------------------------------------------------------
# PSD comparison
# ---------------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    freq_p1,
    10 * np.log10(psd_yi_p1 + 1e-12),
    label="Ideal ECG"
)

plt.plot(
    freq_p1,
    10 * np.log10(psd_eta_p1 + 1e-12),
    label="Noise"
)

plt.plot(
    freq_p1,
    10 * np.log10(psd_x_p1 + 1e-12),
    label="Noisy ECG"
)

plt.plot(
    freq_p1,
    10 * np.log10(psd_yhat_p1 + 1e-12),
    label="Filtered ECG"
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power/frequency (dB/Hz)")
plt.title("Power Spectral Density")

plt.xlim(0, 150)
plt.ylim(-100, 0)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "fig1_2d_part1_psd.png",
    dpi=150
)

plt.show()

# =====================================================================
# 1.3 FREQUENCY-DOMAIN WIENER FILTER - PART 1
# =====================================================================
print(
    "1.3 PART 1 - FREQUENCY-DOMAIN WIENER FILTER"
)

def frequency_domain_wiener(
    desired_segment,
    noise_segment,
    x_input,
    yi_reference,
    label
):

    L = len(x_input)


    # ---------------------------------------------------------------
    # Power spectra
    # ---------------------------------------------------------------

    Y = np.fft.fft(
        desired_segment,
        n=L
    )

    N_noise = np.fft.fft(
        noise_segment,
        n=L
    )

    X = np.fft.fft(
        x_input
    )


    S_YY = np.abs(Y) ** 2

    S_NN = np.abs(N_noise) ** 2

    S_XX = np.abs(X) ** 2


    # ---------------------------------------------------------------
    # Wiener frequency response
    #
    # W(f) = SYY(f) / [SYY(f) + SNN(f)]
    # ---------------------------------------------------------------

    W = S_YY / (
        S_YY + S_NN + 1e-12
    )


    # ---------------------------------------------------------------
    # Filter signal in frequency domain
    # ---------------------------------------------------------------

    Yhat = W * X

    yhat = np.real(
        np.fft.ifft(Yhat)
    )


    # ---------------------------------------------------------------
    # Assignment equation
    #
    # S_YhatYhat(f) = W(f) S_XX(f)
    # ---------------------------------------------------------------

    S_YhatYhat = W * S_XX


    mse = np.mean(
        (yhat - yi_reference) ** 2
    )


    # ---------------------------------------------------------------
    # Frequency axis
    # ---------------------------------------------------------------

    frequency = np.fft.fftfreq(
        L,
        d=1 / fs
    )

    positive = frequency >= 0


    # ---------------------------------------------------------------
    # Wiener gain plot
    # ---------------------------------------------------------------

    plt.figure(figsize=(9, 4))

    plt.plot(
        frequency[positive],
        W[positive]
    )

    plt.xlim(0, 150)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("W(f)")

    plt.title(
        f"{label}: Frequency-Domain Wiener Filter"
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        f"fig1_3_{label}_Wiener_gain.png",
        dpi=150
    )

    plt.show()


    # ---------------------------------------------------------------
    # Power spectrum calculation
    # ---------------------------------------------------------------

    plt.figure(figsize=(9, 4))

    plt.plot(
        frequency[positive],
        S_XX[positive],
        label=r"$S_{XX}(f)$"
    )

    plt.plot(
        frequency[positive],
        S_YhatYhat[positive],
        label=r"$S_{\hat Y\hat Y}(f)=W(f)S_{XX}(f)$",
        linewidth=2
    )

    plt.xlim(0, 150)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")

    plt.title(
        f"{label}: Input and Filtered Power Spectrum"
    )

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        f"fig1_3_{label}_power_spectrum.png",
        dpi=150
    )

    plt.show()


    return yhat, mse, W
# =====================================================================
# 1.3 FREQUENCY-DOMAIN WIENER FILTER - PART 1
# =====================================================================

print("1.3 PART 1 - FREQUENCY-DOMAIN WIENER FILTER")

def frequency_domain_wiener(
    desired_segment,
    noise_segment,
    x_input,
    yi_reference,
    label
):

    # ---------------------------------------------------------------
    # Length of input signal
    # ---------------------------------------------------------------

    L = len(x_input)

    # ---------------------------------------------------------------
    # Fourier transforms
    # ---------------------------------------------------------------

    Y = np.fft.fft(
        desired_segment,
        n=L
    )

    N_noise = np.fft.fft(
        noise_segment,
        n=L
    )

    X = np.fft.fft(x_input)

    # ---------------------------------------------------------------
    # Power spectra
    #
    # SZZ(f) = |F{z(n)}|^2
    # ---------------------------------------------------------------

    S_YY = np.abs(Y) ** 2
    S_NN = np.abs(N_noise) ** 2

    # ---------------------------------------------------------------
    # Frequency-domain Wiener filter
    #
    # W(f) = SYY(f) / [SYY(f) + SNN(f)]
    # ---------------------------------------------------------------

    W = S_YY / (
        S_YY + S_NN + 1e-12
    )

    # ---------------------------------------------------------------
    # Apply Wiener filter in frequency domain
    #
    # S_YhatYhat(f) = W(f) S_XX(f)
    #
    # Equivalent filtering operation:
    # Yhat(f) = W(f) X(f)
    # ---------------------------------------------------------------

    Yhat = W * X

    # Convert back to time domain
    yhat = np.real(
        np.fft.ifft(Yhat)
    )

    # ---------------------------------------------------------------
    # Mean squared error with respect to ideal ECG
    # ---------------------------------------------------------------

    mse = np.mean(
        (yhat - yi_reference) ** 2
    )

    # ---------------------------------------------------------------
    # Plot ideal ECG and frequency-domain Wiener filtered ECG
    # ---------------------------------------------------------------

    t_segment = np.arange(L) / fs

    plt.figure(figsize=(11, 5))

    plt.plot(
        t_segment,
        yi_reference,
        label=r"Ideal ECG $y_i(n)$",
        linewidth=1.5
    )

    plt.plot(
        t_segment,
        yhat,
        label=r"Filtered ECG $\hat{y}(n)$",
        linewidth=1.2
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.title(
        f"{label}: Frequency-Domain Wiener Filter"
    )

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        f"fig1_3_{label}_filtered_signal.png",
        dpi=150
    )

    plt.show()

    return yhat, mse

# =====================================================================
# APPLY TO PART 1
# =====================================================================

yhat_fd_p1, mse_fd_p1 = frequency_domain_wiener(
    y_beat_p1,
    noise_base,
    x_beat,
    y_beat_p1,
    "Part1"
)

print(
    f"Part 1 frequency-domain MSE = "
    f"{mse_fd_p1:.6f}"
)

# =====================================================================
# PART 2
# =====================================================================

print(
    "PART 2 - CONSTRUCTED LINEAR MODEL"
)

# ---------------------------------------------------------------------
# Part 2 desired signal
# ---------------------------------------------------------------------

def piecewise_ecg_template(length):

    """
    Manually constructed piecewise-linear ECG-like signal.

    It has:
        P wave
        QRS complex
        T wave
    """

    control_points = [

        (0.00, 0.00),
        (0.15, 0.00),

        (0.20, 0.30),
        (0.30, 0.30),

        (0.35, 0.00),

        (0.42, -0.20),
        (0.46, 1.10),
        (0.50, -0.25),

        (0.55, 0.00),
        (0.65, 0.00),

        (0.72, 0.55),
        (0.85, 0.55),

        (0.90, 0.00),
        (1.00, 0.00)
    ]

    xs = np.array(
        [p[0] for p in control_points]
    ) * (length - 1)

    ys = np.array(
        [p[1] for p in control_points]
    )

    return np.interp(
        np.arange(length),
        xs,
        ys
    )


y_beat_p2 = piecewise_ecg_template(
    len(y_beat_p1)
)

# ---------------------------------------------------------------------
# Plot Part 2 desired signal
# ---------------------------------------------------------------------

plt.figure(figsize=(9, 4))

plt.plot(
    time_beat,
    y_beat_p2
)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.title(
    "Part 2: Constructed Linear ECG Model"
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "fig1_2_part2_desired_signal.png",
    dpi=150
)

plt.show()

# =====================================================================
# 1.2(a) PART 2 - ARBITRARY M
# =====================================================================

ARBITRARY_M_P2 = 10

w0_arbitrary_p2 = wiener_filter_design(
    y_beat_p2,
    noise_base,
    ARBITRARY_M_P2
)


print("\n1.2(a)")

print(
    f"Part 2 arbitrary M = "
    f"{ARBITRARY_M_P2}"
)

print(
    f"Part 2 FIR order = "
    f"{ARBITRARY_M_P2 - 1}"
)

# ---------------------------------------------------------------------
# 1.2(c) Filter using arbitrary M
# ---------------------------------------------------------------------

yhat_arbitrary_p2 = apply_wiener_filter(
    w0_arbitrary_p2,
    x_beat
)

mse_arbitrary_p2 = calculate_mse(
    y_beat_p1,
    yhat_arbitrary_p2,
    M=ARBITRARY_M_P2,
    skip_transient=False
)

print(
    f"Part 2 arbitrary-M MSE = "
    f"{mse_arbitrary_p2:.6f}"
)


# ---------------------------------------------------------------------
# Noisy vs filtered
# ---------------------------------------------------------------------

plt.figure(figsize=(10, 4))

plt.plot(
    time_beat,
    y_beat_p1,
    label=r"Ideal ECG $y_i(n)$",
    linewidth=2
)

plt.plot(
    time_beat,
    x_beat,
    label=r"Noisy $x(n)$",
    alpha=0.6
)

plt.plot(
    time_beat,
    yhat_arbitrary_p2,
    label=rf"Filtered $\hat y(n)$, M={ARBITRARY_M_P2}",
    linewidth=2
)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.title(
    f"Part 2: Noisy vs Filtered ECG "
    f"(Arbitrary M={ARBITRARY_M_P2})"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "fig1_2c_part2_arbitrary_filter.png",
    dpi=150
)

plt.show()

# =====================================================================
# 1.2(b) FIND OPTIMUM M - PART 2
# =====================================================================

print("\n1.2(b): Finding optimum M")


MAX_M_P2 = min(
    40,
    len(y_beat_p2),
    len(noise_base)
)

M_values_p2 = np.arange(
    2,
    MAX_M_P2 + 1
)

mse_values_p2 = []


for M in M_values_p2:

    w = wiener_filter_design(
        y_beat_p2,
        noise_base,
        M
    )

    yhat = apply_wiener_filter(
        w,
        x_beat
    )

    mse = calculate_mse(
        y_beat_p1,
        yhat,
        M=M,
        skip_transient=True
    )

    mse_values_p2.append(
        mse
    )


mse_values_p2 = np.array(
    mse_values_p2
)


best_index_p2 = np.argmin(
    mse_values_p2
)

M_p2 = M_values_p2[
    best_index_p2
]


w0_p2 = wiener_filter_design(
    y_beat_p2,
    noise_base,
    M_p2
)


print(
    f"Part 2 optimum M = "
    f"{M_p2}"
)

print(
    f"Part 2 FIR order = "
    f"{M_p2 - 1}"
)

print(
    f"Minimum MSE = "
    f"{mse_values_p2[best_index_p2]:.6f}"
)

# ---------------------------------------------------------------------
# MSE vs M
# ---------------------------------------------------------------------

plt.figure(figsize=(7, 4))

plt.plot(
    M_values_p2,
    mse_values_p2,
    marker="o",
    markersize=3
)

plt.axvline(
    M_p2,
    linestyle="--",
    label=f"Optimum M = {M_p2}"
)

plt.xlabel(
    "Number of filter coefficients, M"
)

plt.ylabel("MSE")

plt.title(
    "Part 2: MSE vs Filter Length"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "fig1_2b_part2_M_selection.png",
    dpi=150
)

plt.show()

# ---------------------------------------------------------------------
# Magnitude response of optimum Wiener filter
# ---------------------------------------------------------------------

plot_magnitude_response(
    w0_p2,
    fs,
    M_p2,
    "Part2"
)

# =====================================================================
# 1.2(c) FILTERED SIGNAL USING OPTIMUM FILTER - PART 2
# =====================================================================

yhat_p2_beat = apply_wiener_filter(
    w0_p2,
    x_beat
)

mse_p2 = calculate_mse(
    y_beat_p1,
    yhat_p2_beat,
    M=M_p2,
    skip_transient=False
)

print(
    "\n1.2(c): Optimum filter"
)

print(
    f"Part 2 optimum MSE = "
    f"{mse_p2:.6f}"
)


plt.figure(figsize=(10, 4))

plt.plot(
    time_beat,
    y_beat_p1,
    label=r"Ideal $y_i(n)$",
    linewidth=2
)

plt.plot(
    time_beat,
    x_beat,
    label=r"Noisy $x(n)$",
    alpha=0.6
)

plt.plot(
    time_beat,
    yhat_p2_beat,
    label=rf"Filtered $\hat y(n)$, M={M_p2}",
    linewidth=2
)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.title(
    f"Part 2: Wiener Filtered ECG "
    f"(Optimum M={M_p2})"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "fig1_2c_part2_optimum_filtered.png",
    dpi=150
)

plt.show()


# =====================================================================
# 1.2(d) SPECTRAL CALCULATION - PART 2
# =====================================================================

print(
    "\n1.2(d): Spectral calculation - Part 2"
)


# ---------------------------------------------------------------------
# Apply optimum Part 2 Wiener filter to FULL noisy ECG
# ---------------------------------------------------------------------

yhat_full_p2 = lfilter(
    w0_p2,
    [1.0],
    x_full
)

# ---------------------------------------------------------------------
# Calculate PSDs
# ---------------------------------------------------------------------

freq_p2, psd_yi_p2 = calculate_spectrum(
    yi_full,
    fs
)

_, psd_eta_p2 = calculate_spectrum(
    eta,
    fs
)

_, psd_x_p2 = calculate_spectrum(
    x_full,
    fs
)

_, psd_yhat_p2 = calculate_spectrum(
    yhat_full_p2,
    fs
)


# ---------------------------------------------------------------------
# Check 50-Hz component
# ---------------------------------------------------------------------

print(
    "\nSpectral component around 50 Hz:"
)

idx_50_p2 = np.argmin(
    np.abs(freq_p2 - 50)
)

print(
    f"Frequency bin = "
    f"{freq_p2[idx_50_p2]:.3f} Hz"
)

print(
    f"Ideal ECG PSD at 50 Hz : "
    f"{psd_yi_p2[idx_50_p2]:.4e}"
)

print(
    f"Noise PSD at 50 Hz     : "
    f"{psd_eta_p2[idx_50_p2]:.4e}"
)

print(
    f"Noisy ECG PSD at 50 Hz : "
    f"{psd_x_p2[idx_50_p2]:.4e}"
)

print(
    f"Filtered PSD at 50 Hz  : "
    f"{psd_yhat_p2[idx_50_p2]:.4e}"
)

# ---------------------------------------------------------------------
# PSD comparison
# ---------------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    freq_p2,
    10 * np.log10(psd_yi_p2 + 1e-12),
    label="Ideal ECG"
)

plt.plot(
    freq_p2,
    10 * np.log10(psd_eta_p2 + 1e-12),
    label="Noise"
)

plt.plot(
    freq_p2,
    10 * np.log10(psd_x_p2 + 1e-12),
    label="Noisy ECG"
)

plt.plot(
    freq_p2,
    10 * np.log10(psd_yhat_p2 + 1e-12),
    label="Filtered ECG"
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power/frequency (dB/Hz)")
plt.title("Power Spectral Density - Part 2")

plt.xlim(0, 150)
plt.ylim(-100, 0)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "fig1_2d_part2_psd.png",
    dpi=150
)

plt.show()


# =====================================================================
# 1.3 FREQUENCY-DOMAIN WIENER FILTER - PART 2
# =====================================================================
print(
    "1.3 PART 2 - FREQUENCY-DOMAIN WIENER FILTER"
)

yhat_fd_p2, mse_fd_p2 = frequency_domain_wiener(
    y_beat_p2,
    noise_base,
    x_beat,
    y_beat_p1,
    "Part2"
)

print(
    f"Part 2 frequency-domain MSE = "
    f"{mse_fd_p2:.6f}"
)

# 1.4 EFFECT OF NON-STATIONARY NOISE
# =====================================================================

print(
    "1.4 NON-STATIONARY NOISE"
)


eta_wg_2 = np.sqrt(
    noise_power
) * np.random.randn(N)


eta_switch = np.where(
    n < N / 2,
    0.2 * np.sin(2 * np.pi * 50 * t),
    0.2 * np.sin(2 * np.pi * 100 * t)
)


eta_nonstationary = (
    eta_wg_2 +
    eta_switch
)


x_nonstationary = (
    yi_full +
    eta_nonstationary
)


# IMPORTANT:
# Use the Part 1 Wiener filter without recalculating weights.

y_hat_ns = lfilter(
    w0_p1,
    [1.0],
    x_nonstationary
)

# =====================================================================
# PLOT 1: WHOLE SIGNAL
# =====================================================================

plt.figure(figsize=(11, 5))

plt.plot(
    t,
    yi_full,
    label=r"Ideal $y_i(n)$",
    linewidth=1.5
)

plt.plot(
    t,
    x_nonstationary,
    label=r"Non-stationary noisy $x(n)$",
    alpha=0.5
)

plt.plot(
    t,
    y_hat_ns,
    label=r"Filtered $\hat{y}(n)$",
    linewidth=1.5
)

switch_time = N / (2 * fs)

plt.axvline(
    switch_time,
    linestyle="--",
    label="50 Hz → 100 Hz switch"
)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.title("Wiener Filter with Non-Stationary Noise")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "fig1_4_nonstationary_full.png",
    dpi=150
)

plt.show()

# =====================================================================
# PLOT 2: ZOOM AROUND THE NOISE FREQUENCY SWITCH
# =====================================================================

zoom_width = 0.2       # seconds on each side of T/2

plt.figure(figsize=(11, 5))

plt.plot(
    t,
    yi_full,
    label=r"Ideal $y_i(n)$",
    linewidth=1.5
)

plt.plot(
    t,
    x_nonstationary,
    label=r"Non-stationary noisy $x(n)$",
    alpha=0.6
)

plt.plot(
    t,
    y_hat_ns,
    label=r"Filtered $\hat{y}(n)$",
    linewidth=1.5
)

plt.axvline(
    switch_time,
    linestyle="--",
    label="50 Hz → 100 Hz switch"
)

plt.xlim(
    switch_time - zoom_width,
    switch_time + zoom_width
)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.title("Noise Frequency Switch")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "fig1_4_nonstationary_zoom.png",
    dpi=150
)

plt.show()

print("All figures have been saved.")