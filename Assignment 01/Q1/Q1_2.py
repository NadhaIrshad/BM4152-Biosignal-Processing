import numpy as np
from scipy.signal import savgol_filter, lfilter
import matplotlib.pyplot as plt

fs = 500  # Hz

# Recreate ECG_template and nECG (same procedure/seed as Section 1.1)

data = np.load("ECG_template.npz")
ECG_template = data[data.files[0]].squeeze()
t = np.arange(len(ECG_template)) / fs

def add_awgn(signal, snr_db, seed=None):
    rng = np.random.default_rng(seed)
    sig_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = sig_power / snr_linear
    noise = rng.normal(0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise

nECG = add_awgn(ECG_template, snr_db=5, seed=0)

def mse(reference, estimate):
    return np.mean((reference - estimate) ** 2)

# 1.2.1 Application of the SG(3, 11) filter

polyorder_311 = 3
L_311 = 11                      
Lprime_311 = 2 * L_311 + 1     
sg311ECG = savgol_filter(nECG, window_length=Lprime_311, polyorder=polyorder_311)
print(f"SG(3,11): N={polyorder_311}, L={L_311}  ->  actual window_length L' = {Lprime_311}")

plt.figure(figsize=(12, 5))
plt.plot(t, nECG, color="lightgray", linewidth=0.7, label="nECG")
plt.plot(t, ECG_template, color="k", linewidth=1.2, label="ECG_template")
plt.plot(t, sg311ECG, color="tab:orange", linewidth=1.2, label="sg311ECG (SG(3,11))")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.title("Savitzky-Golay SG(3,11) filtering vs. original and noisy signal")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_1_2_1_SG311_comparison.png", dpi=150)
plt.show()

# 1.2.2 Optimum SG(N, L) filter parameters

polyorders = np.arange(2, 8)                  # N: polynomial order
L_values = np.arange(2, 26)                   # L: half-window parameter

mse_grid = np.full((len(polyorders), len(L_values)), np.nan)

for i, N in enumerate(polyorders):
    for j, L in enumerate(L_values):
        Lprime = 2 * L + 1                     # actual window_length
        if N > Lprime - 1:
            continue   # invalid combination - skip (left as NaN)
        y = savgol_filter(nECG, window_length=Lprime, polyorder=N)
        mse_grid[i, j] = mse(ECG_template, y)

# Find the optimum (ignoring NaN entries for invalid parameter combinations)
min_idx = np.nanargmin(mse_grid)
min_i, min_j = np.unravel_index(min_idx, mse_grid.shape)
optimum_N = polyorders[min_i]
optimum_L = L_values[min_j]
optimum_Lprime = 2 * optimum_L + 1
optimum_mse = mse_grid[min_i, min_j]
print(f"Optimum SG filter: N (polyorder) = {optimum_N}, "
      f"L = {optimum_L}  (L' = {optimum_Lprime}), MSE = {optimum_mse:.6f}")

# --- Visualise the MSE surface using pcolormesh ----------------------------
plt.figure(figsize=(9, 6))
mesh = plt.pcolormesh(L_values, polyorders, mse_grid, shading="auto",
                        cmap="viridis")
plt.colorbar(mesh, label="MSE")
plt.scatter([optimum_L], [optimum_N], color="red", marker="*", s=200,
             label=f"Optimum: N={optimum_N}, L={optimum_L}")
plt.xlabel("L (half-window parameter, L' = 2L+1)")
plt.ylabel("Polynomial order, N")
plt.title("MSE across SG(N, L) filter parameters")
plt.legend()
plt.tight_layout()
plt.savefig("fig_1_2_2_SG_MSE_surface.png", dpi=150)
plt.show()

# 3D surface view for additional clarity 
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (enables 3D projection)

L_grid, N_grid = np.meshgrid(L_values, polyorders)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(L_grid, N_grid, mse_grid, cmap="viridis", edgecolor="none")
ax.set_xlabel("L (half-window parameter)")
ax.set_ylabel("Polynomial order, N")
ax.set_zlabel("MSE")
ax.set_title("MSE surface across SG(N, L) filter parameters")
fig.colorbar(surf, shrink=0.6, label="MSE")
plt.tight_layout()
plt.savefig("fig_1_2_2_SG_MSE_surface_3D.png", dpi=150)
plt.show()

# Compare ECG_template, sg311ECG, and the optimum SG filter
sg_optimum_ECG = savgol_filter(nECG, window_length=optimum_Lprime, polyorder=optimum_N)

plt.figure(figsize=(12, 5))
plt.plot(t, ECG_template, color="k", linewidth=1.2, label="ECG_template")
plt.plot(t, sg311ECG, color="tab:orange", linewidth=1, label="sg311ECG (SG(3,11))")
plt.plot(t, sg_optimum_ECG, color="tab:green", linewidth=1.2,
          label=f"Optimum SG({optimum_N},{optimum_L})")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.title("Comparison: ECG_template vs. SG(3,11) vs. optimum SG filter")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_1_2_2_SG311_vs_optimum.png", dpi=150)
plt.show()

# Compare optimum MA(N) and optimum SG(N,L) filters
# Recompute the optimum MA(N) filter here (same procedure as Section 1.1.5)
def compensate_delay(y, delay_samples):
    delay_samples = int(round(delay_samples))
    y_comp = np.zeros_like(y)
    if delay_samples > 0:
        y_comp[:len(y) - delay_samples] = y[delay_samples:]
    else:
        y_comp = y.copy()
    return y_comp

N_range = np.arange(1, 41)
ma_mse_values = np.zeros_like(N_range, dtype=float)
for i, N in enumerate(N_range):
    b_N = np.ones(N) / N
    y_N = lfilter(b_N, [1.0], nECG)
    y_N_comp = compensate_delay(y_N, (N - 1) / 2)
    ma_mse_values[i] = mse(ECG_template, y_N_comp)
optimum_MA_N = N_range[np.argmin(ma_mse_values)]
optimum_MA_mse = ma_mse_values.min()

b_opt = np.ones(optimum_MA_N) / optimum_MA_N
ma_optimum_ECG = compensate_delay(lfilter(b_opt, [1.0], nECG), (optimum_MA_N - 1) / 2)

print(f"\nComparison of optimum filters:")
print(f"  MA({optimum_MA_N}):        MSE = {optimum_MA_mse:.6f}, "
      f"{optimum_MA_N} coefficients, group delay = {(optimum_MA_N-1)/2} samples")
print(f"  SG({optimum_N},{optimum_L}): MSE = {optimum_mse:.6f}, "
      f"{optimum_Lprime} coefficients (L'=2L+1, least-squares fit), zero delay")

plt.figure(figsize=(12, 5))
plt.plot(t, ECG_template, color="k", linewidth=1.4, label="ECG_template")
plt.plot(t, ma_optimum_ECG, color="tab:purple", linewidth=1.1,
          label=f"Optimum MA({optimum_MA_N})")
plt.plot(t, sg_optimum_ECG, color="tab:green", linewidth=1.1,
          label=f"Optimum SG({optimum_N},{optimum_L})")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.title("Optimum MA(N) vs. optimum SG(N,L) filtered ECG")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_1_2_2_MA_vs_SG_optimum.png", dpi=150)
plt.show()
