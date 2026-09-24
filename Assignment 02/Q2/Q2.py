import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sawtooth

rng_global_seed = 0

# 2.1  DATA CONSTRUCTION
fs = 500.0
N = 5000
n = np.arange(N)
t = n / fs

SAWTOOTH_FREQ = 1.0   # cycles per second, chosen so a handful of periods are visible over the 10 s record (matches Fig.2)
yi_saw = sawtooth(2 * np.pi * SAWTOOTH_FREQ * t, width=0.5)

def build_anc_signals(yi, fs, a=0.5, phi1=np.pi / 4, phi2=np.pi / 3,
                       SNR_dB=10.0, seed=0):
    """
    Build the primary input x(n) and reference input r(n) used by the
    adaptive noise canceller, for any desired signal yi(n).

    eta(n) is the SAME non-stationary noise as Section 1.4: white
    Gaussian noise (fixed SNR relative to yi) plus a sinusoidal
    interferer that is 50 Hz for the first half of the record and
    100 Hz for the second half.

    r(n) = a*(eta_wg(n) + sin(2*pi*50*n/fs + phi1) + sin(2*pi*100*n/fs + phi2))
    reuses the SAME white-noise realisation as eta(n) (so it is
    genuinely correlated with the noise in x, as an ANC reference must
    be) but carries BOTH the 50 Hz and 100 Hz tones at all times, with
    arbitrary gain 'a' and phase offsets phi1/phi2 -- unlike eta(n),
    which only contains ONE of those tones at any given time. This
    mismatch is intentional: it is exactly what makes the adaptive
    filter's ability to TRACK a changing correlation (as opposed to the
    fixed Wiener filter of Q1) meaningful to test.
    """
    rng = np.random.default_rng(seed)
    N_ = len(yi)
    n_ = np.arange(N_)
    t_ = n_ / fs
    T_ = N_

    sig_power = np.mean(yi ** 2)
    noise_power = sig_power / (10 ** (SNR_dB / 10))
    eta_wg = np.sqrt(noise_power) * rng.standard_normal(N_)

    eta_ns = np.where(n_ < T_ / 2,
                       0.2 * np.sin(2 * np.pi * 50 * t_),
                       0.2 * np.sin(2 * np.pi * 100 * t_))
    eta = eta_wg + eta_ns
    x = yi + eta

    r = a * (eta_wg + np.sin(2 * np.pi * 50 * t_ + phi1)
             + np.sin(2 * np.pi * 100 * t_ + phi2))
    return x, r, eta, t_

x_saw, r_saw, eta_saw, t_saw = build_anc_signals(yi_saw, fs, seed=rng_global_seed)

# ===================================================================
# ADAPTIVE FILTER IMPLEMENTATIONS
# ===================================================================
def run_lms(x, r, mu, M):
    """
    LMS adaptive noise canceller.

        R(n)  = [r(n), r(n-1), ..., r(n-M+1)]^T
        e(n)  = x(n) - w(n)^T R(n)                (output = y_hat(n))
        w(n+1)= w(n) + 2*mu*e(n)*R(n)

    Returns e (the output/estimate of y_i(n)) and the final weights.
    """
    N_ = len(x)
    w = np.zeros(M)
    R_buf = np.zeros(M)
    e = np.zeros(N_)
    for i in range(N_):
        R_buf = np.roll(R_buf, 1)
        R_buf[0] = r[i]
        e[i] = x[i] - w @ R_buf
        w = w + 2 * mu * e[i] * R_buf
    return e, w

def run_rls(x, r, M, lam=0.99, delta=1e-2):
    """
    RLS adaptive noise canceller, following the assignment's equations
    exactly:
        k(n)  = (P(n-1) r(n)/lam) / (1 + r^T(n) P(n-1) r(n)/lam)
        P(n)  = P(n-1)/lam - k(n) r^T(n) P(n-1) /lam
        alpha_pri(n) = x(n) - w(n-1)^T r(n)        (a priori error, used to update w)
        w(n)  = w(n-1) + k(n) alpha_pri(n)
        alpha_post(n) = x(n) - w(n)^T r(n)         (a posteriori error = output y_hat(n))

    Initialisation: P(0) = delta^-1 I, w(0) = 0.
    """
    N_ = len(x)
    w = np.zeros(M)
    P = np.eye(M) / delta
    R_buf = np.zeros(M)
    alpha_post = np.zeros(N_)
    for i in range(N_):
        R_buf = np.roll(R_buf, 1)
        R_buf[0] = r[i]

        Pr = P @ R_buf
        denom = 1.0 + (R_buf @ Pr) / lam
        k = (Pr / lam) / denom

        alpha_pri = x[i] - w @ R_buf
        w = w + k * alpha_pri
        P = P / lam - np.outer(k, R_buf @ P) / lam

        alpha_post[i] = x[i] - w @ R_buf
    return alpha_post, w

def mse_vs_ref(yhat, yi_ref):
    """MSE over the entire signal."""
    L = min(len(yhat), len(yi_ref))
    return np.mean((yhat[:L] - yi_ref[:L]) ** 2)


# 2.2  LMS METHOD
MU_DEFAULT = 0.01  
M_DEFAULT  = 15      

e_lms, w_lms = run_lms(x_saw, r_saw, MU_DEFAULT, M_DEFAULT)
mse_lms = mse_vs_ref(e_lms, yi_saw)
print(f"[2.2(b)] LMS: mu={MU_DEFAULT}, M={M_DEFAULT} taps, "
      f"MSE = {mse_lms:.5f}")

fig, axs = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
axs[0].plot(t_saw, yi_saw); axs[0].set_title(r"$y_i(n)$")
axs[1].plot(t_saw, x_saw);  axs[1].set_title(r"$x(n)$")
axs[2].plot(t_saw, e_lms);  axs[2].set_title(r"$e(n) = \hat{y}(n)$ (LMS)")
axs[3].plot(t_saw, np.abs(yi_saw - e_lms), color="tab:red")
axs[3].set_title(r"Absolute error $|y_i(n) - \hat{y}(n)|$")
axs[3].set_xlabel("Time (s)")
for a_ in axs:
    a_.axvline(t_saw[-1] / 2, color="k", ls="--", lw=1)
    a_.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("fig2_2_lms_result.png", dpi=150); plt.show()


# 2.2(c) LMS: PCOLORMESH (optimum search)

#MU_SWEEP = [0.001, 0.003, 0.01, 0.03, 0.05, 0.08, 0.12, 0.15]
#ORDER_SWEEP = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]

MU_SWEEP = np.linspace(0.001, 0.05, 20)
ORDER_SWEEP = range(1, 31)

mse_lms_grid = np.zeros((len(MU_SWEEP), len(ORDER_SWEEP)))

for i, mu in enumerate(MU_SWEEP):
    for j, M in enumerate(ORDER_SWEEP):
        e_i, _ = run_lms(x_saw, r_saw, mu, M)
        mse_lms_grid[i, j] = mse_vs_ref(e_i, yi_saw)

fig, ax = plt.subplots(figsize=(9, 6))
finite_vals = mse_lms_grid[np.isfinite(mse_lms_grid)]
vmax_lms = np.percentile(finite_vals, 95) if finite_vals.size else None
pcm = ax.pcolormesh(MU_SWEEP, ORDER_SWEEP, mse_lms_grid.T,
                     cmap="viridis", shading="auto", vmax=vmax_lms)
fig.colorbar(pcm, ax=ax, label="MSE (clipped at 95th pct for contrast)")
ax.set_xlabel(r"Step size $\mu$")
ax.set_ylabel("Number of taps M")
ax.set_title("LMS: MSE heatmap (pcolormesh)")

# find the optimum (mu, M) that minimises MSE on the grid ---
i_opt, j_opt = np.unravel_index(np.nanargmin(mse_lms_grid), mse_lms_grid.shape)
MU_OPT = MU_SWEEP[i_opt]
M_OPT_LMS = ORDER_SWEEP[j_opt]
mse_opt_lms = mse_lms_grid[i_opt, j_opt]
print(f"[2.2(c)] Optimum LMS on grid: mu={MU_OPT}, M={M_OPT_LMS}, "
      f"MSE={mse_opt_lms:.5f}")

ax.plot(MU_OPT, M_OPT_LMS, marker="*", color="red", markersize=20,
         markeredgecolor="k",
         label=f"Optimum ($\\mu$={MU_OPT}, M={M_OPT_LMS})")
ax.legend()
plt.tight_layout()
plt.savefig("fig2_2c_lms_pcolormesh.png", dpi=150)
plt.show()

# rerun LMS at the optimum (mu, M) and plot its absolute error 
e_lms_opt, w_lms_opt = run_lms(x_saw, r_saw, MU_OPT, M_OPT_LMS)

fig, axs = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
axs[0].plot(t_saw, yi_saw); axs[0].set_title(r"$y_i(n)$")
axs[1].plot(t_saw, x_saw);  axs[1].set_title(r"$x(n)$")
axs[2].plot(t_saw, e_lms_opt)
axs[2].set_title(rf"$e(n)=\hat{{y}}(n)$ (LMS, optimum $\mu$={MU_OPT}, M={M_OPT_LMS})")
axs[3].plot(t_saw, np.abs(yi_saw - e_lms_opt), color="tab:red")
axs[3].set_title(rf"Absolute error $|y_i(n)-\hat{{y}}(n)|$ "
                  rf"(optimum LMS, MSE={mse_opt_lms:.5f})")
axs[3].set_xlabel("Time (s)")
for a_ in axs:
    a_.axvline(t_saw[-1] / 2, color="k", ls="--", lw=1)
    a_.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig2_2c_lms_optimum_result.png", dpi=150)
plt.show()

# ===================================================================
# 2.3  RLS METHOD
# ===================================================================
LAM_DEFAULT   = 0.995
DELTA_DEFAULT = 1e-2   

e_rls, w_rls = run_rls(x_saw, r_saw, M_DEFAULT, LAM_DEFAULT, DELTA_DEFAULT)
mse_rls = mse_vs_ref(e_rls, yi_saw)
print(f"[2.3(b)] RLS: lambda={LAM_DEFAULT}, M={M_DEFAULT} taps, "
      f"MSE = {mse_rls:.5f}")
print(f"[2.3(b)] LMS MSE={mse_lms:.5f} vs RLS MSE={mse_rls:.5f}")

# ---- 2.3(b) compare LMS vs RLS, plot similar to Figure 2 ------------------
fig, axs = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
axs[0].plot(t_saw, yi_saw); axs[0].set_title(r"$y_i(n)$")
axs[1].plot(t_saw, x_saw);  axs[1].set_title(r"$x(n)$")
axs[2].plot(t_saw, e_lms, label=f"LMS (MSE={mse_lms:.4f})")
axs[2].plot(t_saw, e_rls, label=f"RLS (MSE={mse_rls:.4f})", alpha=0.8)
axs[2].set_title(r"$e(n) = \hat{y}(n)$: LMS vs RLS"); axs[2].legend(fontsize=8)
axs[3].plot(t_saw, np.abs(yi_saw - e_lms), label="LMS", alpha=0.8)
axs[3].plot(t_saw, np.abs(yi_saw - e_rls), label="RLS", alpha=0.8)
axs[3].set_title(r"Absolute error $|y_i(n)-\hat{y}(n)|$"); axs[3].legend(fontsize=8)
axs[3].set_xlabel("Time (s)")
for a_ in axs:
    a_.axvline(t_saw[-1] / 2, color="k", ls="--", lw=1)
    a_.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("fig2_3_lms_vs_rls.png", dpi=150); plt.show()

# ===================================================================
# 2.3(c) RLS: PCOLORMESH (optimum search)
# ===================================================================

#LAM_SWEEP = [0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0]
LAM_SWEEP = np.linspace(0.90, 1.0, 21)

mse_rls_grid = np.zeros((len(LAM_SWEEP), len(ORDER_SWEEP)))

for i, lam in enumerate(LAM_SWEEP):
    for j, M in enumerate(ORDER_SWEEP):
        e_i, _ = run_rls(
            x_saw,
            r_saw,
            M,
            lam,
            DELTA_DEFAULT
        )

        mse_rls_grid[i, j] = mse_vs_ref(e_i, yi_saw)

fig, ax = plt.subplots(figsize=(9, 6))
finite_vals = mse_rls_grid[np.isfinite(mse_rls_grid)]
vmax_rls = np.percentile(finite_vals, 95) if finite_vals.size else None
pcm = ax.pcolormesh(LAM_SWEEP, ORDER_SWEEP, mse_rls_grid.T,
                     cmap="viridis", shading="auto", vmax=vmax_rls)
fig.colorbar(pcm, ax=ax, label="MSE (clipped at 95th pct for contrast)")
ax.set_xlabel(r"Forgetting factor $\lambda$")
ax.set_ylabel("Number of taps M")
ax.set_title("RLS: MSE heatmap (pcolormesh)")

# --- find the optimum (lambda, M) that minimises MSE on the grid ---
i_opt, j_opt = np.unravel_index(np.nanargmin(mse_rls_grid), mse_rls_grid.shape)
LAM_OPT = LAM_SWEEP[i_opt]
M_OPT_RLS = ORDER_SWEEP[j_opt]
mse_opt_rls = mse_rls_grid[i_opt, j_opt]
print(f"[2.3(c)] Optimum RLS on grid: lambda={LAM_OPT}, M={M_OPT_RLS}, "
      f"MSE={mse_opt_rls:.5f}")

ax.plot(LAM_OPT, M_OPT_RLS, marker="*", color="red", markersize=20,
         markeredgecolor="k",
         label=f"Optimum ($\\lambda$={LAM_OPT}, M={M_OPT_RLS})")
ax.legend()
plt.tight_layout()
plt.savefig("fig2_3c_rls_pcolormesh.png", dpi=150)
plt.show()

# ---- rerun RLS at the optimum (lambda, M) and plot its absolute error ----
e_rls_opt, w_rls_opt = run_rls(x_saw, r_saw, M_OPT_RLS, LAM_OPT, DELTA_DEFAULT)

fig, axs = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
axs[0].plot(t_saw, yi_saw); axs[0].set_title(r"$y_i(n)$")
axs[1].plot(t_saw, x_saw);  axs[1].set_title(r"$x(n)$")
axs[2].plot(t_saw, e_rls_opt)
axs[2].set_title(rf"$e(n)=\hat{{y}}(n)$ (RLS, optimum $\lambda$={LAM_OPT}, M={M_OPT_RLS})")
axs[3].plot(t_saw, np.abs(yi_saw - e_rls_opt), color="tab:red")
axs[3].set_title(rf"Absolute error $|y_i(n)-\hat{{y}}(n)|$ "
                  rf"(optimum RLS, MSE={mse_opt_rls:.5f})")
axs[3].set_xlabel("Time (s)")
for a_ in axs:
    a_.axvline(t_saw[-1] / 2, color="k", ls="--", lw=1)
    a_.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig2_3c_rls_optimum_result.png", dpi=150)
plt.show()

# ===================================================================
# 2.3(d) Re-test LMS and RLS using yi(n) = idealECG data
# ===================================================================
try:
    data = np.load("idealECG.npz")
    IDEAL_KEY = data.files[0]            
    yi_ecg = data[IDEAL_KEY].astype(float).flatten()

    x_ecg, r_ecg, eta_ecg, t_ecg = build_anc_signals(yi_ecg, fs, seed=rng_global_seed)

    e_lms_ecg, _ = run_lms(x_ecg, r_ecg, MU_DEFAULT, M_DEFAULT)
    e_rls_ecg, _ = run_rls(x_ecg, r_ecg, M_DEFAULT, LAM_DEFAULT, DELTA_DEFAULT)

    e_lms_ecg, _ = run_lms(x_ecg, r_ecg, MU_OPT, M_OPT_LMS ) #use the optmial values 
    e_rls_ecg, _ = run_rls(x_ecg, r_ecg,M_OPT_RLS, LAM_OPT, DELTA_DEFAULT)

    mse_lms_ecg = mse_vs_ref(e_lms_ecg, yi_ecg)
    mse_rls_ecg = mse_vs_ref(e_rls_ecg, yi_ecg)
    print(f"\n[2.3(d)] On idealECG data: LMS MSE={mse_lms_ecg:.5f}, "
          f"RLS MSE={mse_rls_ecg:.5f}")

    fig, axs = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    axs[0].plot(t_ecg, yi_ecg); axs[0].set_title(r"$y_i(n)$ (idealECG)")
    axs[1].plot(t_ecg, x_ecg);  axs[1].set_title(r"$x(n)$")
    axs[2].plot(t_ecg, e_lms_ecg, label=f"LMS (MSE={mse_lms_ecg:.4f})")
    axs[2].plot(t_ecg, e_rls_ecg, label=f"RLS (MSE={mse_rls_ecg:.4f})", alpha=0.8)
    axs[2].set_title(r"$e(n) = \hat{y}(n)$: LMS vs RLS on ECG"); axs[2].legend(fontsize=8)
    axs[3].plot(t_ecg, np.abs(yi_ecg - e_lms_ecg), label="LMS", alpha=0.8)
    axs[3].plot(t_ecg, np.abs(yi_ecg - e_rls_ecg), label="RLS", alpha=0.8)
    axs[3].set_title(r"Absolute error $|y_i(n)-\hat{y}(n)|$"); axs[3].legend(fontsize=8)
    axs[3].set_xlabel("Time (s)")
    for a_ in axs:
        a_.axvline(t_ecg[-1] / 2, color="k", ls="--", lw=1)
        a_.grid(alpha=0.3)
    ZOOM_SECONDS_ECG = 3.0                                  # >>> ADJUST
    plt.xlim(0, min(ZOOM_SECONDS_ECG, t_ecg[-1]))
    plt.tight_layout(); plt.savefig("fig2_3d_ecg_lms_vs_rls.png", dpi=150); plt.show()

except FileNotFoundError:
    print("\n[2.3(d)] SKIPPED: idealECG.npz not found in the working directory. ")

print("\nDone.")