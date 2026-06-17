from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

DATA_FILE = Path(__file__)).with_name("data.csv")
FIG_DIR = Path(__file__)).with_name("figures")

REF_CALG = 7.5      # alginate concentration (wt %) used to calibrate q
REF_CTONCF = 3.0    # TONCF stock concentration (wt %) of the reference set
SCALE = 3.0         # maps TONCF mixing fraction f -> scaled coordinate x = SCALE * f
PEAK_WEIGHT = 500.0 # weight of the peak self-consistency penalty during calibration
SAVE_FIGURES = True
SHOW_FIGURES = True

CA = 2.0             # Ca2+ crosslinker concentration, wt %
CT_REF = 3.0         # reference TONCF stock concentration, wt %

R_GAS = 8.314        # J / (mol K)
T = 293              # K
NION_TONCF = 1.5     # TONCF carboxylate density, mmol/g
NION_ALG = 2.4       # alginate carboxylate density, mmol/g
A_ION = 0.55         # ionic-affinity coefficient

AFFINITY = np.exp(-A_ION / CA)                            # ionic affinity factor
B1 = NION_ALG * 1e-3 * R_GAS * T * 10 / 2 * AFFINITY      # alginate reinforcement amplitude
B3 = NION_TONCF * 1e-3 * R_GAS * T * 10 / 2 * AFFINITY    # TONCF amplitude = slope of B vs alginate
BETA = NION_TONCF / (NION_ALG * CT_REF * AFFINITY ** 2)   # network-disruption rate
ALPHA = NION_TONCF / NION_ALG                            # ionic site ratio (kept for reference)

A2_RATIO = 2.0       # second power-law amplitude:  A2 = A2_RATIO * A1
PC1 = 0.02           # percolation onset (also motivated by excluded-volume theory)
PC2 = 1.918          # second, geometric threshold (inflection of the falling branch)
MU = 1.63            # agglomeration (Gaussian) centre = dip position
T1 = 0.526           # reinforcement power-law exponent
T2 = 1.437           # structural-reorganisation power-law exponent
XV1, XV2 = 0.975, 2.09                # the two peak positions of the reference curve
SIGMA = ((XV2 - XV1) / 10) ** 0.5     # agglomeration (Gaussian) width

# C_sat is COMPUTED below (not fixed). This is only the iteration seed.
C_SAT = 20.0

def activation(x):
    return 1.0 / (1.0 + np.exp(-(x - PC1) / 0.08))

def gaussian(x):
    return np.exp(-(x - MU) ** 2 / (2 * SIGMA ** 2))

def model(x, A1, A2, B, y0):
    reinforcement = A1 * np.maximum(x - PC1, 0.0) ** T1
    reorganisation = -A2 * np.maximum(x - PC2, 0.0) ** T2
    agglomeration = -B * gaussian(x) * activation(x)
    return y0 + reinforcement + reorganisation + agglomeration

def analytical_peak(A1, A2):
    x = PC2 + 0.5
    for _ in range(40):
        x = max(x, PC2 + 1e-9)
        dx1 = max(x - PC1, 1e-9)
        dx2 = max(x - PC2, 1e-9)
        first = A1 * T1 * dx1 ** (T1 - 1) - A2 * T2 * dx2 ** (T2 - 1)
        second = A1 * T1 * (T1 - 1) * dx1 ** (T1 - 2) - A2 * T2 * (T2 - 1) * dx2 ** (T2 - 2)
        x -= first / (second + 1e-9)
    return x

def g_prime(x):
    return -(x - MU) / SIGMA ** 2 * gaussian(x)

def g_second(x):
    return ((x - MU) ** 2 - SIGMA ** 2) / SIGMA ** 4 * gaussian(x)

def y_second(x, A1, A2, B):
    dx1 = np.maximum(x - PC1, 1e-9)
    dx2 = np.maximum(x - PC2, 1e-9)
    return (A1 * T1 * (T1 - 1) * dx1 ** (T1 - 2)
            - A2 * T2 * (T2 - 1) * dx2 ** (T2 - 2)
            - B * g_second(x))

def csat(ctoncf):
    return C_SAT * min(ctoncf, CT_REF) / CT_REF

def disruption(ctoncf):
    return np.exp(-BETA * max(ctoncf - CT_REF, 0.0))

def amplitudes(calg, ctoncf, q):
    d_fac = disruption(ctoncf)
    A1 = B1 * (calg + q * NION_TONCF * ctoncf / NION_ALG) * d_fac
    A2 = A2_RATIO * A1
    csat_eff = csat(ctoncf)                       # alginate conc. where B vanishes
    B = max(B3 * (csat_eff - calg), 0.0)          # B = b3 * (C_sat - C_Alg), >= 0
    return A1, A2, B, csat_eff, d_fac

def load_data(path=DATA_FILE):
    df = pd.read_csv(path)
    base = df[np.isclose(df["f"], 0.0)]                       # fibre-free moduli
    coeffs = np.polyfit(base["Calg_pct"], base["E_kPa"], 1)   # linear baseline
    return df, lambda calg: float(np.polyval(coeffs, calg))

def conditions(df):
    for (ctoncf, calg), sub in df.groupby(["Ctoncf_pct", "Calg_pct"], sort=True):
        yield ctoncf, calg, sub.sort_values("f")

def calibrate_q(ref, y0_of_calg):
    x_ref = ref["f"].values * SCALE
    E_ref = ref["E_kPa"].values
    y0_ref = y0_of_calg(REF_CALG)

    def residuals(params):
        (q,) = params
        A1, A2, B, _, _ = amplitudes(REF_CALG, REF_CTONCF, q)
        res = list(model(x_ref, A1, A2, B, y0_ref) - E_ref)

        # peak self-consistency penalty: X_num ~= X_ana + k * B
        xs = np.linspace(0, 2.8, 4000)
        x_num = xs[np.argmax(model(xs, A1, A2, B, y0_ref))]
        x_ana = analytical_peak(A1, A2)
        k = g_prime(x_num) / (y_second(x_num, A1, A2, B) + 1e-9)
        res.append((x_num - (x_ana + k * B)) * PEAK_WEIGHT)
        return np.array(res)

    # deterministic multi-start over a small grid of initial guesses
    best = None
    for q0 in np.linspace(1.0, 10.0, 6):
        r = least_squares(residuals, [q0], bounds=([0.0], [20.0]), max_nfev=40000)
        if best is None or r.cost < best.cost:
            best = r
    return float(best.x[0]), float(best.cost)


def agglomeration_amplitudes(df, q, y0_of_calg):
    w = gaussian(MU) * activation(MU)
    out = []
    for ctoncf, calg, sub in conditions(df):
        if not np.isclose(ctoncf, REF_CTONCF):
            continue
        A1 = B1 * (calg + q * NION_TONCF * REF_CTONCF / NION_ALG) * disruption(REF_CTONCF)
        A2 = A2_RATIO * A1
        reinf_only = y0_of_calg(calg) + A1 * max(MU - PC1, 0.0) ** T1 - A2 * max(MU - PC2, 0.0) ** T2
        e_at_mu = np.interp(MU, sub.sort_values("f")["f"].values * SCALE,
                            sub.sort_values("f")["E_kPa"].values)
        out.append((calg, (reinf_only - e_at_mu) / w))
    return out

def csat_from_amplitudes(df, q, y0_of_calg):
    return float(np.mean([calg + B / B3 for calg, B in agglomeration_amplitudes(df, q, y0_of_calg)]))

def calibrate(df, ref, y0_of_calg, n_iter=12):
    global C_SAT
    q = cost = None
    for _ in range(n_iter):
        q, cost = calibrate_q(ref, y0_of_calg)
        C_SAT = csat_from_amplitudes(df, q, y0_of_calg)
    return q, cost, C_SAT

def goodness_of_fit(sub, q, y0_of_calg):
    ctoncf = sub["Ctoncf_pct"].iloc[0]
    calg = sub["Calg_pct"].iloc[0]
    A1, A2, B, _, _ = amplitudes(calg, ctoncf, q)
    pred = model(sub["f"].values * SCALE, A1, A2, B, y0_of_calg(calg))
    meas = sub["E_kPa"].values
    ss_res = np.sum((meas - pred) ** 2)
    ss_tot = np.sum((meas - meas.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = np.sqrt(np.mean((meas - pred) ** 2))
    return r2, rmse

def smooth_curve(calg, ctoncf, q, y0_of_calg, n=400):
    A1, A2, B, csat_eff, d_fac = amplitudes(calg, ctoncf, q)
    xs = np.linspace(0, 2.8, n)
    return xs / SCALE, model(xs, A1, A2, B, y0_of_calg(calg)), (A1, A2, B, csat_eff, d_fac)

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

def _save(fig, name):
    if SAVE_FIGURES:
        FIG_DIR.mkdir(exist_ok=True)
        fig.savefig(FIG_DIR / name, dpi=300, bbox_inches="tight")

def fig_calibration_and_prediction(df, q, y0_of_calg):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    series = [(c, calg, sub) for c, calg, sub in conditions(df) if np.isclose(c, REF_CTONCF)]
    for i, (ctoncf, calg, sub) in enumerate(series):
        col = COLORS[i % len(COLORS)]
        is_ref = np.isclose(calg, REF_CALG)
        f, E, _ = smooth_curve(calg, ctoncf, q, y0_of_calg)
        ax.scatter(sub["f"], sub["E_kPa"], color=col, s=28, zorder=5)
        ax.plot(f, E, color=col, lw=2.6 if is_ref else 1.6,
                label=f"{calg:g} % alginate" + (" (calibration)" if is_ref else " (prediction)"))
    ax.set_xlabel("TONCF mixing fraction $f$")
    ax.set_ylabel("Young's modulus $E$ [kPa]")
    ax.set_title(f"Calibration on {REF_CALG:g} % alginate, prediction of the rest  ($q = {q:.3f}$)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, "fig1_alginate_series.png")

def fig_stock_prediction(df, q, y0_of_calg):
    sub = next((s for c, _, s in conditions(df) if c > REF_CTONCF), None)
    if sub is None:
        return
    ctoncf = sub["Ctoncf_pct"].iloc[0]
    calg = sub["Calg_pct"].iloc[0]
    f, E, _ = smooth_curve(calg, ctoncf, q, y0_of_calg)
    A1, A2, B, _, _ = amplitudes(calg, ctoncf, q)
    pred_pts = model(sub["f"].values * SCALE, A1, A2, B, y0_of_calg(calg))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(f, E, "k-", lw=1.5, alpha=0.5, label="model")
    ax.scatter(sub["f"], sub["E_kPa"], color="#d62728", s=80, zorder=5, label="measured")
    ax.scatter(sub["f"], pred_pts, color="black", marker="D", s=70, zorder=5, label="predicted")
    ax.set_xlabel("TONCF mixing fraction $f$")
    ax.set_ylabel("Young's modulus $E$ [kPa]")
    ax.set_title(f"Prediction at {ctoncf:g} % TONCF stock  (Calg = {calg:g} %, no re-fitting)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, "fig2_stock_prediction.png")

def fig_disruption():
    ct = np.linspace(2.5, 6, 200)
    d = [disruption(c) for c in ct]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ct, d, "k-", lw=2, label=fr"$d = \exp(-{BETA:.3f}\,(C_t - {CT_REF:g}))$")
    ax.axvline(CT_REF, color="gray", ls=":", lw=1.5, label=f"$C_{{t,ref}}$ = {CT_REF:g} %")
    for c, (dx, dy) in zip([3.0, 4.0], [(3.55, 0.30), (5.10, 0.30)]):
        val = disruption(c)
        ax.scatter([c], [val], s=80, zorder=5)
        ax.annotate(f"$C_t$ = {c:g} %\n$d$ = {val:.3f}  ({(1 - val) * 100:.1f} % reduction)",
                    xy=(c, val), xytext=(dx, dy), fontsize=9, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1))
    ax.set_xlabel("TONCF stock concentration $C_t$ [%]")
    ax.set_ylabel("disruption factor $d$")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, "fig3_disruption_factor.png")

def fig_agglomeration_vs_alginate(df, q, y0_of_calg):
    pts = agglomeration_amplitudes(df, q, y0_of_calg)          # (calg, B) computed from dips
    csat_val = csat_from_amplitudes(df, q, y0_of_calg)
    c = np.linspace(0, csat_val * 1.15, 300)

    fig, ax = plt.subplots(figsize=(7, 5))
    cbar = np.mean([cc for cc, _ in pts]); bbar = np.mean([bb for _, bb in pts])
    ax.plot(c, np.maximum(bbar - B3 * (c - cbar), 0.0), "k-", lw=2,
            label=r"slope $-b_3$ (theory), $B = b_3\,(C_{sat}-C_{Alg})$")
    for i, (calg, B) in enumerate(pts):
        ax.scatter([calg], [B], color=COLORS[i % len(COLORS)], s=70, zorder=5,
                   label=f"{calg:g} % alginate (computed $B$)")
    ax.axvline(csat_val, color="red", ls="--", lw=1.5, label=fr"$C_{{sat}}$ = {csat_val:.1f} % (computed)")
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("alginate concentration $C_{Alg}$ [%]")
    ax.set_ylabel("agglomeration amplitude $B$ [kPa]")
    ax.set_title(f"$C_{{sat}}$ from the falling $B$: dip vanishes at $C_{{Alg}} \\approx$ {csat_val:.1f} %")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, "fig4_agglomeration_vs_alginate.png")

def fig_peak_consistency(q, y0_of_calg):
    csat_val = csat(CT_REF)
    c = np.linspace(5, csat_val * 1.2, 200)
    x_ana, x_num, x_pred = [], [], []
    for ci in c:
        A1 = B1 * (ci + q * NION_TONCF * CT_REF / NION_ALG)
        A2 = A2_RATIO * A1
        B = max(B3 * (csat_val - ci), 0.0)
        xs = np.linspace(0, 2.8, 8000)
        xn = xs[np.argmax(model(xs, A1, A2, B, 0))]
        xa = analytical_peak(A1, A2)
        k = g_prime(xn) / (y_second(xn, A1, A2, B) + 1e-9)
        x_num.append(xn / SCALE)
        x_ana.append(xa / SCALE)
        x_pred.append((xa + k * B) / SCALE)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(c, x_num, color="#1f77b4", lw=2, label="$X_{NUM}$")
    ax1.plot(c, x_pred, color="#2ca02c", lw=2, ls="--", label="$X_{ANA} + k\\,B$")
    ax1.plot(c, x_ana, color="#d62728", lw=2, ls=":", label="$X_{ANA}$")
    ax1.axvline(csat_val, color="gray", ls="--", lw=1, alpha=0.6, label=f"$C_{{sat}}$ = {csat_val:.1f} %")
    ax1.set_ylabel("peak position (TONCF fraction)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax2.plot(c, np.array(x_num) - np.array(x_pred), color="#ff7f0e", lw=1.5)
    ax2.axhline(0, color="gray", ls="--", lw=0.8)
    ax2.axvline(csat_val, color="gray", ls="--", lw=1, alpha=0.6)
    ax2.set_xlabel("alginate concentration $C_{Alg}$ [%]")
    ax2.set_ylabel("residual")
    ax2.grid(alpha=0.3)
    _save(fig, "fig5_peak_consistency.png")

def main():
    df, y0_of_calg = load_data()

    ref = df[np.isclose(df["Ctoncf_pct"], REF_CTONCF) & np.isclose(df["Calg_pct"], REF_CALG)]
    if ref.empty:
        raise SystemExit(f"Reference set ({REF_CALG} % alginate, {REF_CTONCF} % stock) not found in data.")

    q, cost, c_sat = calibrate(df, ref, y0_of_calg)

    print("(A) Fixed from theory / ionic-crosslinking framework")
    print(f"    affinity = {AFFINITY:.4f}   b1 = {B1:.4f} kPa/%   b3 = {B3:.4f} kPa/%   beta = {BETA:.4f} %^-1")
    print("(B) Fixed from the geometry of the reference curve")
    print(f"    Pc1 = {PC1}   Pc2 = {PC2}   mu = {MU}   t1 = {T1}   t2 = {T2}   sigma = {SIGMA:.4f}")
    print(f"(C) Single FITTED parameter, calibrated on the {REF_CALG:g} % alginate curve only")
    print(f"    q = {q:.4f}   (least-squares cost = {cost:.3f})")
    print("    --> q is the ONLY parameter passed to least_squares.")
    print("(D) Computed (NOT fitted) from the falling agglomeration amplitude + theory slope b3")
    print(f"    C_sat = C_Alg + B/b3, averaged over compositions = {c_sat:.2f} %")

    r2_ref, rmse_ref = goodness_of_fit(ref, q, y0_of_calg)
    print(f"\nreference fit:  R2 = {r2_ref:.4f}   RMSE = {rmse_ref:.2f} kPa")

    for ctoncf, calg, sub in conditions(df):
        if np.isclose(ctoncf, REF_CTONCF) and np.isclose(calg, REF_CALG):
            continue
        r2, rmse = goodness_of_fit(sub, q, y0_of_calg)
        kind = "alginate" if np.isclose(ctoncf, REF_CTONCF) else "TONCF stock"
        print(f"  {calg:>5g} % alg, {ctoncf:g} % stock ({kind:11s}):  R2 = {r2:6.3f}   RMSE = {rmse:6.2f} kPa")

    print(f"\nDip-vanishing alginate concentration (computed): C_sat = {c_sat:.1f} %")

    fig_calibration_and_prediction(df, q, y0_of_calg)
    fig_stock_prediction(df, q, y0_of_calg)
    fig_disruption()
    fig_agglomeration_vs_alginate(df, q, y0_of_calg)
    fig_peak_consistency(q, y0_of_calg)

    if SHOW_FIGURES:
        plt.show()


if __name__ == "__main__":
    main()
