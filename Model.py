import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
DATA_FILE = None
MAIN_SHEET = "5-10% Alg"
CA_SHEET = "TONCF_Ca2+ Variations"

REF_CALG = 7.5        # alginate stock concentration of the reference series (% w/v)
REF_CTONCF = 3.0      # TONCF stock concentration of the alginate series (% w/v)
CA = 2.0              # CaCl2 concentration of the cross-linking bath for the modelled series (% w/v, anhydrous CaCl2)
C_REF = 3.0           # TONCF stock concentration up to which the fibres contribute independently (% w/v)
TRAIN = [5.0, 7.5, 10.0]          # predefined training series (alginate stock, % w/v)

R_GAS = 8.314         # J / (mol K)
T = 293               # K
N_TONCF = 1.5         # TONCF carboxylate density (mmol/g)
N_ALG = 2.4           # alginate carboxylate density (mmol/g)
A_ION = 0.55          # ionic-affinity coefficient (%)
ALPHA = N_TONCF / N_ALG

AFF = np.exp(-A_ION / CA)                              # Eq. (8)
B1 = N_ALG * 1e-3 * R_GAS * T * 10 / 2 * AFF            # Eq. (10): rho_x R T per % w/v alginate, kPa
B2 = N_TONCF * 1e-3 * R_GAS * T * 10 / 2 * AFF          # Eq. (21): same scale for TONCF, kPa
BETA = N_TONCF / (N_ALG * C_REF * AFF ** 2)             # disruption rate, (% w/v)^-1

A2_RATIO = 2.0        # Eq. (12)

# Percolation onset from excluded-volume theory for cylindrical particles, Eqs. (5)-(7)
R_FIB = 15.0          # TONCF fibril radius (nm), supplier specification
L_FIB = 8000.0        # TONCF fibril length (nm), supplier specification
V_FIB = np.pi * R_FIB ** 2 * L_FIB + 4.0 / 3.0 * np.pi * R_FIB ** 3                                   # Eq. (6)
V_EX = 32.0 / 3.0 * np.pi * R_FIB ** 3 * (1 + 0.75 * L_FIB / R_FIB + 3.0 / 32.0 * (L_FIB / R_FIB) ** 2)  # Eq. (7)
RHO_TONCF = 1.5       # TONCF (cellulose) density, g/cm3
PHI_C = V_FIB / V_EX                  # Eq. (5): critical volume fraction, 1.85e-3
PC1 = 100.0 * RHO_TONCF * PHI_C       # critical concentration of dispersed TONCF, % (w/v), on the x axis: 0.28
PC2 = 1.918           # second threshold (inflection of the falling branch)
MU = 1.63             # centre of the agglomeration Gaussian
T1 = T2 = None        # exponents of the rising and falling terms, set by fit_exponents()
XV1, XV2 = 0.975, 2.09                    # peak positions of the reference curve
OMEGA = ((XV2 - XV1) / 10) ** 0.5         # Eq. (18)
SIG_WIDTH = 0.08                          # Eq. (19)

# set by calibrate()
PSI = PSI_A = None
C_SAT = 20.0          # iteration seed only
FIT = {}

def _norm(name):
    return re.sub(r"\s+", "", str(name)).lower()


def _find_sheet(names, wanted):
    for n in names:
        if _norm(n) == _norm(wanted):
            return n
    for n in names:
        if _norm(n).endswith(_norm(wanted)):
            return n
    return None


def find_data_file(path=None):
    for p in (path, DATA_FILE):
        if p:
            p = Path(p).expanduser()
            if not p.is_file():
                raise FileNotFoundError(f"Excel workbook not found: {p}")
            return p
    found = []
    for p in sorted(HERE.glob("*.xls*")):
        if p.name.startswith("~$"):                      # lock file of a workbook open in Excel
            continue
        try:
            with pd.ExcelFile(p) as xl:
                if _find_sheet(xl.sheet_names, MAIN_SHEET):
                    found.append(p)
        except Exception:
            pass
    if not found:
        raise FileNotFoundError(f'No Excel workbook with the sheet "{MAIN_SHEET}" in {HERE}. '
                                f"Set DATA_FILE at the top of Model.py or give the path on the command line.")
    if len(found) > 1:
        print(f"Several workbooks found in {HERE}; using {found[0].name}")
    return found[0]


def _numbers(col):
    if not pd.api.types.is_numeric_dtype(col):
        col = col.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(col, errors="coerce")


def _long_format(sheet, cols):
    reps = [c for c in sheet.columns if re.match(r"^\s*n\s*=", str(c), re.IGNORECASE)]
    if not reps:
        raise ValueError('No specimen columns found (headers "n = 1", "n = 2", ...).')
    sheet = sheet[cols + reps].copy()
    for c in cols + reps:
        sheet[c] = _numbers(sheet[c])
    long = sheet.melt(id_vars=cols, value_vars=reps, var_name="specimen", value_name="E_Pa").dropna()
    long["E_kPa"] = long["E_Pa"] / 1000.0
    return long.drop(columns="E_Pa")


def load_measurements(path=None):
    path = find_data_file(path)
    with pd.ExcelFile(path) as xl:
        main_name, ca_name = _find_sheet(xl.sheet_names, MAIN_SHEET), _find_sheet(xl.sheet_names, CA_SHEET)
        if main_name is None:
            raise ValueError(f'Sheet "{MAIN_SHEET}" not found in {path.name} (sheets: {xl.sheet_names})')
        main = pd.read_excel(xl, sheet_name=main_name)
        ca = pd.read_excel(xl, sheet_name=ca_name) if ca_name else None
    print(f"Data: {path.name if path.resolve().parent == HERE else path}")

    main = main.rename(columns={main.columns[0]: "Calg", main.columns[1]: "f"})
    main["Ctoncf"], main["Ca"] = REF_CTONCF, CA
    parts = [_long_format(main, ["Calg", "Ctoncf", "Ca", "f"])]

    if ca is not None:
        ca = ca.rename(columns={ca.columns[0]: "Calg", ca.columns[1]: "Ctoncf",
                                ca.columns[2]: "Ca", ca.columns[3]: "f"})
        parts.append(_long_format(ca, ["Calg", "Ctoncf", "Ca", "f"]))
    else:
        print(f'Sheet "{CA_SHEET}" not found: calcium series and 4 % TONCF stock skipped.')

    specimens = pd.concat(parts, ignore_index=True)
    specimens[["Calg", "Ctoncf", "Ca", "f"]] = specimens[["Calg", "Ctoncf", "Ca", "f"]].astype(float)
    return specimens


def summarise(specimens):
    g = specimens.groupby(["Calg", "Ctoncf", "Ca", "f"])["E_kPa"]
    s = g.agg(E_kPa="mean", SD_kPa="std", n="count").reset_index()
    s["Calg_final"] = (1 - s.f) * s.Calg
    s["Ctoncf_final"] = s.f * s.Ctoncf
    return s


def series(summary, calg, ctoncf, ca=CA):
    s = summary[np.isclose(summary.Calg, calg) & np.isclose(summary.Ctoncf, ctoncf) & np.isclose(summary.Ca, ca)]
    return s.sort_values("f")


def model_series(summary):
    out = []
    at_ca = summary[np.isclose(summary.Ca, CA)]
    for calg in sorted(at_ca[np.isclose(at_ca.Ctoncf, REF_CTONCF)].Calg.unique()):
        out.append((calg, REF_CTONCF, series(summary, calg, REF_CTONCF)))
    for ctoncf in sorted(at_ca.Ctoncf.unique()):
        if ctoncf > REF_CTONCF:
            for calg in sorted(at_ca[np.isclose(at_ca.Ctoncf, ctoncf)].Calg.unique()):
                out.append((calg, ctoncf, series(summary, calg, ctoncf)))
    return out

def x_coord(f, ctoncf):                              # Eq. (2): x = f min(C_TONCF, C_ref)
    return np.asarray(f, dtype=float) * min(ctoncf, C_REF)


def f_coord(x, ctoncf):                              # inverse of Eq. (2): mixing fraction f at position x
    return np.asarray(x, dtype=float) / min(ctoncf, C_REF)


def pos_pow(u, p):
    u = np.asarray(u, dtype=float)
    out = np.zeros_like(u)
    m = u > 0
    out[m] = u[m] ** p
    return out


def sigma(x):                                        # Eq. (19)
    return 1.0 / (1.0 + np.exp(-(x - PC1) / SIG_WIDTH))


def sigma_d1(x):
    s = sigma(x)
    return s * (1.0 - s) / SIG_WIDTH


def gauss(x):                                        # Eq. (17)
    return np.exp(-(x - MU) ** 2 / (2 * OMEGA ** 2))


def gauss_d1(x):                                     # Eq. (30)
    return -(x - MU) / OMEGA ** 2 * gauss(x)


def agg(x):                                          # H(x) = G(x) sigma(x)
    return gauss(x) * sigma(x)


def agg_d1(x):                                       # H'(x) = G'(x) sigma(x) + G(x) sigma'(x)
    return gauss_d1(x) * sigma(x) + gauss(x) * sigma_d1(x)


def backbone(x, A1, A2):
    return A1 * pos_pow(x - PC1, T1) - A2 * pos_pow(x - PC2, T2)


def backbone_d1(x, A1, A2):                          # Eq. (25)
    return A1 * T1 * pos_pow(x - PC1, T1 - 1) - A2 * T2 * pos_pow(x - PC2, T2 - 1)


def backbone_d2(x, A1, A2):                          # Eq. (27)
    return A1 * T1 * (T1 - 1) * pos_pow(x - PC1, T1 - 2) - A2 * T2 * (T2 - 1) * pos_pow(x - PC2, T2 - 2)


def disruption(ctoncf):                              # Eq. (14)
    return float(np.exp(-BETA * max(ctoncf - C_REF, 0.0)))


def csat(ctoncf):                                    # Eq. (22), expressed through C_sat at C_ref
    return C_SAT * min(ctoncf, C_REF) / C_REF


def baseline(calg):
    return 3.0 * PSI_A * B1 * calg

def amplitudes(calg, ctoncf, q, psi=None):
    psi = PSI if psi is None else psi
    A1 = 3.0 * psi * B1 * (calg + q * ALPHA * ctoncf) * disruption(ctoncf)    # Eq. (9)
    A2 = A2_RATIO * A1                                                     # Eq. (12)
    B = max(B2 * (csat(ctoncf) - calg), 0.0)                               # Eq. (20)
    return A1, A2, B


def model(x, A1, A2, B, y0=0.0):                     # Eq. (3)
    x = np.asarray(x, dtype=float)
    return y0 + backbone(x, A1, A2) - B * agg(x)


def predict(f, calg, ctoncf, q, psi=None):
    return model(x_coord(f, ctoncf), *amplitudes(calg, ctoncf, q, psi), y0=baseline(calg))

X_GRID = np.linspace(0.0, 2.8, 8000)

def x_ana(A1, A2):
    x = PC2 + 0.5
    for _ in range(60):
        x = max(x, PC2 + 1e-6)
        step = backbone_d1(np.array([x]), A1, A2)[0] / backbone_d2(np.array([x]), A1, A2)[0]
        x -= step
        if abs(step) < 1e-12:
            break
    return x

def x_num(A1, A2, B):
    return X_GRID[np.argmax(model(X_GRID, A1, A2, B))]

def k_coefficient(A1, A2):
    xa = np.array([x_ana(A1, A2)])
    return agg_d1(xa)[0] / backbone_d2(xa, A1, A2)[0]


def fit_exponents(summary, calg=REF_CALG):
    global T1, T2
    s = series(summary, calg, REF_CTONCF)
    x, y = x_coord(s.f.values, REF_CTONCF), s.E_kPa.values
    y0 = float(s[np.isclose(s.f, 0.0)].E_kPa.iloc[0])

    def residuals(p):
        A1, B, t1, t2 = p
        return y0 + A1 * pos_pow(x - PC1, t1) - A2_RATIO * A1 * pos_pow(x - PC2, t2) - B * agg(x) - y

    starts = [(300.0, 150.0, t1, t2) for t1 in (0.2, 0.5, 1.0) for t2 in (1.0, 1.5, 2.5)]
    best = min((least_squares(residuals, p0, bounds=([0, 0, 0.05, 0.5], [3000, 1000, 2, 4])) for p0 in starts),
               key=lambda r: r.cost)
    T1, T2 = float(best.x[2]), float(best.x[3])
    return T1, T2


def dip_amplitudes(summary, q, psi=None, which=None):
    h = float(agg(np.array([MU]))[0])
    out = []
    for calg, ctoncf, s in model_series(summary):
        if not np.isclose(ctoncf, REF_CTONCF) or (which is not None and calg not in which):
            continue
        A1, A2, _ = amplitudes(calg, ctoncf, q, psi)
        backbone_mu = baseline(calg) + float(backbone(np.array([MU]), A1, A2)[0])
        E_mu = np.interp(MU, x_coord(s.f.values, ctoncf), s.E_kPa.values)
        out.append((calg, (backbone_mu - E_mu) / h))
    return out


def _fit_psi_q(summary, train):
    data = [(calg, series(summary, calg, REF_CTONCF)) for calg in train]

    def residuals(p):
        return np.concatenate([predict(s.f.values, calg, REF_CTONCF, p[1], psi=p[0]) - s.E_kPa.values
                               for calg, s in data])

    fits = [least_squares(residuals, [p0, q0], bounds=([0.01, 0.0], [3.0, 30.0]))
            for p0 in (0.2, 0.35, 0.6) for q0 in (1.0, 3.0, 8.0)]
    return min(fits, key=lambda r: r.cost)

def calibrate(summary, train=TRAIN, n_iter=30, refit_exponents=True):
    global PSI, PSI_A, C_SAT
    if refit_exponents:
        fit_exponents(summary)
    PSI_A = float(np.mean([series(summary, c, REF_CTONCF).query("f == 0").E_kPa.iloc[0] / (3 * B1 * c)
                           for c in train]))
    for _ in range(n_iter):
        best = _fit_psi_q(summary, train)
        PSI, q = float(best.x[0]), float(best.x[1])
        new = float(np.mean([calg + B / B2 for calg, B in dip_amplitudes(summary, q, which=train)]))
        done = abs(new - C_SAT) < 1e-6
        C_SAT = new
        if done:
            break
    cov = np.linalg.pinv(best.jac.T @ best.jac) * (2 * best.cost / (len(best.fun) - 2))
    se = np.sqrt(np.diag(cov))
    FIT.update(psi=PSI, q=q, psi_se=float(se[0]), q_se=float(se[1]),
               corr=float(cov[0, 1] / (se[0] * se[1])), psi_A=PSI_A, C_sat=C_SAT, train=list(train),
               t1=T1, t2=T2, Pc1=float(PC1))
    return q


def series_amplitudes(summary):
    out = []
    for calg, ctoncf, s in model_series(summary):
        if not np.isclose(ctoncf, REF_CTONCF):
            continue
        x, y = x_coord(s.f.values, ctoncf), s.E_kPa.values
        y0 = float(s[np.isclose(s.f, 0.0)].E_kPa.iloc[0])
        fit = least_squares(lambda p: y0 + backbone(x, p[0], A2_RATIO * p[0]) - p[1] * agg(x) - y, [300.0, 150.0])
        cov = np.linalg.pinv(fit.jac.T @ fit.jac) * (2 * fit.cost / (len(fit.fun) - 2))
        out.append((calg, float(fit.x[0]), float(np.sqrt(cov[0, 0]))))
    return out


def goodness_of_fit(s, calg, ctoncf, q):
    pred = predict(s.f.values, calg, ctoncf, q)
    meas = s.E_kPa.values
    r2 = 1.0 - np.sum((meas - pred) ** 2) / np.sum((meas - meas.mean()) ** 2)
    rmse = float(np.sqrt(np.mean((meas - pred) ** 2)))
    return float(r2), rmse

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
FIGSIZE = (7, 3.9)
EB = dict(capsize=3, elinewidth=1, capthick=1)
F_PLOT = np.linspace(0.0, 0.9, 3601)
STYLE = {
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "font.size": 15,
    "legend.fontsize": 12,
    "axes.edgecolor": "0.2",
    "axes.linewidth": 1.0,
    "grid.color": "#b0b0b0",
    "grid.alpha": 0.6,
    "grid.linewidth": 0.8,
}


def _style_axes(ax):
    ax.grid(True)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))


def _new(figsize=FIGSIZE):
    fig, ax = plt.subplots(figsize=figsize)
    _style_axes(ax)
    return fig, ax


def _legend_below(fig, ax, ncol=2, fontsize=11, height=0.8):
    w, h = fig.get_size_inches()
    fig.set_size_inches(w, h + height)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=ncol,
               fontsize=fontsize, frameon=False)
    return (0, height / (h + height), 1, 1)


def _finish(fig, name, rect=None):
    fig.tight_layout(rect=rect)
    if not SHOW.get(name, False):
        plt.close(fig)


def _quadratic_guide(ax, s, color):
    c = np.polyfit(s.f.values, s.E_kPa.values, 2)
    ff = np.linspace(s.f.min(), s.f.max(), 200)
    ax.plot(ff, np.polyval(c, ff), ":", color=color, lw=2)


def fig1_reference(summary):
    s = series(summary, REF_CALG, REF_CTONCF)
    fig, ax = _new()
    ax.errorbar(s.f, s.E_kPa, yerr=s.SD_kPa, color="#2d9e5a", marker="s", lw=2.5, ms=7, **EB)
    ax.set_xlabel("TONCF mixing fraction $f$")
    ax.set_ylabel("Young's modulus $E$ [kPa]")
    ax.set_ylim(bottom=0)
    _finish(fig, "Fig1_reference_series")


def fig2_calcium(summary):
    fig, ax = _new()
    for ctoncf, ca, col in [(2, 1, "#2ca02c"), (3, 1, "#ff7f0e"), (4, 1, "#1f77b4"), (4, 2, "#9467bd")]:
        s = series(summary, 5.0, ctoncf, ca)
        if s.empty:
            continue
        _quadratic_guide(ax, s, col)
        ax.errorbar(s.f, s.E_kPa, yerr=s.SD_kPa, color=col, marker="o", ms=8, ls="none", **EB,
                    label=f"5 % alg / {ctoncf:g} % TONCF / {ca:g} % CaCl$_2$")
    ax.set_xticks([0.25, 0.5, 0.75])
    ax.set_xlabel("TONCF mixing fraction $f$")
    ax.set_ylabel("Young's modulus [kPa]")
    ax.set_ylim(0, None)
    ax.legend()
    _finish(fig, "Fig2a_calcium_series")

    s = summary[np.isclose(summary.Calg, 5.0) & np.isclose(summary.Ctoncf, 3.0) & np.isclose(summary.f, 0.5)]
    s = s.sort_values("Ca")
    fig, ax = _new()
    ax.errorbar(s.Ca, s.E_kPa, yerr=s.SD_kPa, color="#137b8a", marker="o", ms=8, lw=2.5, **EB)
    i = int(np.argmax(np.diff(s.E_kPa.values)))
    xa, ya = s.Ca.values[i + 1], s.E_kPa.values[i + 1]
    ax.annotate("step\nincrease", xy=(xa, ya), xytext=(xa - 0.3, ya + 15), ha="center",
                arrowprops=dict(arrowstyle="->", color="0.3"))
    ax.set_xticks(s.Ca.values)
    ax.set_xlabel("CaCl$_2$ concentration [% w/v]")
    ax.set_ylabel("Young's modulus [kPa]")
    ax.set_ylim(0, s.E_kPa.max() * 1.3)
    _finish(fig, "Fig2b_calcium_step")


def fig3_stock(summary):
    fig, ax = _new()
    for ctoncf, col in [(3.0, "#1f77b4"), (4.0, "#d62728")]:
        s = series(summary, 5.0, ctoncf)
        s = s[s.f.isin([0.25, 0.5, 0.75])]
        _quadratic_guide(ax, s, col)
        ax.errorbar(s.f, s.E_kPa, yerr=s.SD_kPa, color=col, marker="o", ms=8, ls="none", **EB,
                    label=f"{ctoncf:g} % TONCF stock")
    ax.set_xticks([0.25, 0.5, 0.75])
    ax.set_xlabel("TONCF mixing fraction $f$")
    ax.set_ylabel("Young's modulus [kPa]")
    ax.set_ylim(0, None)
    ax.legend(loc="upper center")
    _finish(fig, "Fig3_TONCF_stock")


def fig4_disruption():
    ct = np.linspace(2.5, 6, 200)
    fig, ax = _new()
    ax.plot(ct, [disruption(c) for c in ct], "k-", lw=2,
            label=rf"$d_{{fac}} = \exp[-{BETA:.3f}\,(C_{{TONCF}} - {C_REF:g})]$")
    ax.axvline(C_REF, color="gray", ls=":", lw=2, ymax=0.95, label=rf"$C_{{ref}}$ = {C_REF:g} %")
    for c, col, (dx, dy) in [(3.0, "#1f77b4", (3.55, 0.30)), (4.0, "#ff7f0e", (5.10, 0.30))]:
        d = disruption(c)
        ax.scatter([c], [d], s=110, color=col, zorder=5)
        ax.annotate(rf"$C_{{TONCF}}$ = {c:g} %" + "\n" + rf"$d_{{fac}}$ = {d:.3f} ({(1 - d) * 100:.1f} % red.)",
                    xy=(c, d), xytext=(dx, dy), ha="center", va="center", fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"),
                    arrowprops=dict(arrowstyle="-", color="gray"))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("TONCF stock concentration $C_{TONCF}$ [% w/v]")
    ax.set_ylabel("Disruption factor $d_{fac}$")
    ax.legend()
    _finish(fig, "Fig4_disruption_factor")


def fig5_curves(q):
    fig, ax = _new()
    levels = [(5.0, "-", 2, COLORS[0]), (10.0, "-", 2, COLORS[1]), (15.0, "-", 2, COLORS[2]),
              (C_SAT, "-", 3.5, "red"), (1.2 * C_SAT, "--", 2, COLORS[4]), (1.5 * C_SAT, "--", 2, COLORS[5])]
    for calg, ls, lw, col in levels:
        A1, A2, B = amplitudes(calg, REF_CTONCF, q)
        if np.isclose(calg, C_SAT):
            lab = rf"$C_{{Alg}} = C_{{sat}}$ = {calg:.2f} % (transition)"
        elif calg > C_SAT:
            lab = rf"$C_{{Alg}}$ = {calg:.1f} % (above $C_{{sat}}$, $B$ = 0)"
        else:
            lab = rf"$C_{{Alg}}$ = {calg:.1f} % ($B$ = {B:.1f} kPa)"
        ax.plot(F_PLOT, model(x_coord(F_PLOT, REF_CTONCF), A1, A2, B, baseline(calg)), ls, lw=lw, color=col,
                label=lab)
    ax.set_xlabel("TONCF mixing fraction $f$")
    ax.set_ylabel("Young's modulus $E$ [kPa]")
    _finish(fig, "Fig5_predicted_curves", rect=_legend_below(fig, ax))


def fig6_amplitudes(q):
    c = np.linspace(1, 1.3 * C_SAT, 300)
    fig, ax = _new()
    ax.plot(c, [amplitudes(ci, REF_CTONCF, q)[0] for ci in c], "b-", lw=2.5, label="$A_1$ (reinforcement)")
    ax.plot(c, [amplitudes(ci, REF_CTONCF, q)[2] for ci in c], "r--", lw=2.5, label="$B$ (agglomeration)")
    for i, calg in enumerate([5.0, 6.25, 7.5, 8.75, 10.0]):
        A1, _, B = amplitudes(calg, REF_CTONCF, q)
        ax.scatter([calg], [A1], s=110, color=COLORS[i], zorder=5)
        ax.scatter([calg], [B], s=110, color=COLORS[i], marker="D", zorder=5)
    ax.axvline(C_SAT, color="gray", ls="--", lw=2, label=rf"$C_{{sat}}$ = {C_SAT:.2f} %")
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Alginate stock concentration [% w/v]")
    ax.set_ylabel("Parameter value [kPa]")
    ax.legend()
    _finish(fig, "Fig6_amplitudes")


def fig7_dip_amplitudes(summary, q):
    pts = dip_amplitudes(summary, q)
    tr = [p for p in pts if p[0] in FIT["train"]]
    cbar, bbar = np.mean([p[0] for p in tr]), np.mean([p[1] for p in tr])
    c = np.linspace(0, 1.15 * C_SAT, 300)
    fig, ax = _new()
    ax.plot(c, np.maximum(bbar - B2 * (c - cbar), 0.0), "k-", lw=2, label="slope $-b_2$ (theory)")
    for i, (calg, B) in enumerate(pts):
        ax.scatter([calg], [B], s=110, color=COLORS[i], zorder=5, label=f"{calg:g} % alg")
    ax.axvline(C_SAT, color="red", ls="--", lw=1.5, label=rf"$C_{{sat}}$ = {C_SAT:.2f} % (computed)")
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Alginate stock concentration [% w/v]")
    ax.set_ylabel("$B$ [kPa]")
    ax.legend(fontsize=11)
    _finish(fig, "Fig7_dip_amplitude")


def fig8_peak_positions(q):
    c = np.linspace(5, 1.2 * C_SAT, 200)
    xn, xa, xp = [], [], []
    for ci in c:
        A1, A2, B = amplitudes(ci, REF_CTONCF, q)
        xa_i = x_ana(A1, A2)
        xn.append(f_coord(x_num(A1, A2, B), REF_CTONCF))
        xa.append(f_coord(xa_i, REF_CTONCF))
        xp.append(f_coord(xa_i + k_coefficient(A1, A2) * B, REF_CTONCF))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    for ax in (ax1, ax2):
        _style_axes(ax)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax1.plot(c, xn, lw=2.5, label="$X_{NUM}$")
    ax1.plot(c, xp, "--", lw=2.5, color="#2ca02c", label="$X_{ANA} + kB$")
    ax1.plot(c, xa, ":", lw=2.5, color="#d62728", label="$X_{ANA}$")
    for ax in (ax1, ax2):
        ax.axvline(C_SAT, color="gray", ls="--", lw=1, alpha=0.7)
    ax1.plot([], [], color="gray", ls="--", lw=1, label=rf"$C_{{sat}}$ = {C_SAT:.2f} %")
    ax1.set_ylabel("Mixing fraction $f$")
    ax1.legend()
    ax2.plot(c, np.array(xn) - np.array(xp), color="#ff7f0e", lw=2)
    ax2.axhline(0, color="gray", ls="--", lw=0.8)
    ax2.set_xlabel("Alginate stock concentration [% w/v]")
    ax2.set_ylabel("error")
    _finish(fig, "Fig8_peak_positions")


def fig9_calibration(summary, q):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7.4))
    for ax in (ax1, ax2):
        _style_axes(ax)
    c = np.linspace(0, 11, 100)
    ax1.plot(c, 3 * PSI * B1 * (c + q * ALPHA * REF_CTONCF), "k-", lw=2, label="$A_1$, Eq. (9)")
    amps = series_amplitudes(summary)
    ax1.errorbar([a[0] for a in amps], [a[1] for a in amps], yerr=[a[2] for a in amps], marker="o", ms=9,
                 color="#1f77b4", ls="none", label="$A_1$ fitted to each series", **EB)
    ax1.text(0.97, 0.06, f"slope $3\\psi b_1$  $\\rightarrow$  $\\psi$ = {PSI:.3f}\n"
                         f"intercept $3\\psi b_1 q\\,\\alpha\\,C_{{TONCF}}$  $\\rightarrow$  $q$ = {q:.2f}",
             transform=ax1.transAxes, ha="right", va="bottom", fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))
    ax1.set_xlim(0, 11)
    ax1.set_ylim(0, None)
    ax1.set_ylabel("$A_1$ [kPa]")
    ax1.legend(loc="upper left", fontsize=11)
    ax1.set_title("(a)", loc="left", fontsize=15)

    ax2.plot(c, 3 * PSI_A * B1 * c, "k-", lw=2, label="$y_0 = 3\\psi_A b_1 C_{Alg}$")
    f0 = [(calg, s[np.isclose(s.f, 0.0)]) for calg, ctoncf, s in model_series(summary) if np.isclose(ctoncf, REF_CTONCF)]
    f0 = [(calg, s0) for calg, s0 in f0 if not s0.empty]
    ax2.errorbar([v[0] for v in f0], [v[1].E_kPa.iloc[0] for v in f0], yerr=[v[1].SD_kPa.iloc[0] for v in f0],
                 marker="s", ms=9, color="#2d9e5a", ls="none", label="measured, $f$ = 0", **EB)
    ax2.text(0.97, 0.06, f"$\\psi_A$ = {PSI_A:.3f}", transform=ax2.transAxes, ha="right", va="bottom", fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))
    ax2.set_xlim(0, 11)
    ax2.set_ylim(0, None)
    ax2.set_xlabel("Alginate stock concentration [% w/v]")
    ax2.set_ylabel("$E$ at $f$ = 0 [kPa]")
    ax2.legend(loc="upper left", fontsize=11)
    ax2.set_title("(b)", loc="left", fontsize=15)
    _finish(fig, "Fig9_calibration")


def fig10_validation(summary, q):
    fig, ax = _new()
    alg_series = [m for m in model_series(summary) if np.isclose(m[1], REF_CTONCF)]
    for i, (calg, ctoncf, s) in enumerate(alg_series):
        ax.errorbar(s.f, s.E_kPa, yerr=s.SD_kPa, color=COLORS[i], marker="o", ms=5, ls="none", **EB)
        ax.plot(F_PLOT, predict(F_PLOT, calg, ctoncf, q), color=COLORS[i], lw=2.2, label=f"{calg:g} % alg. stock")
    ax.set_xlabel("TONCF mixing fraction $f$")
    ax.set_ylabel("Young's modulus $E$ [kPa]")
    ax.set_ylim(bottom=-5)
    _finish(fig, "Fig10_model_validation", rect=_legend_below(fig, ax, ncol=3))


def fig11_stock_prediction(summary, q):
    stock = [m for m in model_series(summary) if m[1] > REF_CTONCF]
    if not stock:
        return
    calg, ctoncf, s = stock[0]
    fig, ax = _new(figsize=(7, 4.8))
    ax.plot(F_PLOT, predict(F_PLOT, calg, ctoncf, q), color="gray", lw=2)
    ax.errorbar(s.f, s.E_kPa, yerr=s.SD_kPa, color="red", marker="o", ms=11, ls="none", **EB, label="measured")
    ax.scatter(s.f, predict(s.f.values, calg, ctoncf, q), color="black", marker="D", s=120, zorder=6,
               label="predicted")
    ax.set_xlabel("TONCF mixing fraction $f$")
    ax.set_ylabel("Young's modulus $E$ [kPa]")
    ax.legend(loc="upper left")
    _finish(fig, "Fig11_stock_prediction")

def main(path=None):
    specimens = load_measurements(path)
    summary = summarise(specimens)
    summary.round(3).to_csv(HERE / "data_summary.csv", index=False)

    q = calibrate(summary)
    gamma = C_SAT * N_ALG / (N_TONCF * C_REF)                  # Eq. (24)

    print("Fixed from the ionic-crosslinking framework")
    print(f"    aff = {AFF:.4f}   b1 = {B1:.3f} kPa/%   b2 = {B2:.3f} kPa/%   beta = {BETA:.4f} %^-1")
    print("Fixed from excluded-volume theory")
    print(f"    Pc1 = 100 rho V_fib/V_ex = {PC1:.4f}   (V_fib/V_ex = {PHI_C:.4e}; R = {R_FIB:g} nm, L = {L_FIB:g} nm, "
          f"rho = {RHO_TONCF:g} g/cm3)")
    print("Fixed from the geometry of the reference curve")
    print(f"    Pc2 = {PC2}   mu = {MU}   omega = {OMEGA:.4f}")
    print(f"    t1 = {T1:.3f}   t2 = {T2:.3f}   (shape of the {REF_CALG:g} % reference curve, Pc1 and Pc2 fixed)")
    print(f"Determined from the training series {FIT['train']}")
    print(f"    psi   = {FIT['psi']:.3f}   (fitted)")
    print(f"    q     = {FIT['q']:.2f}    (fitted)")
    print(f"    psi_A = {PSI_A:.4f}   (fibre-free gels)        ratio psi/psi_A = {PSI / PSI_A:.1f}")
    print(f"    C_sat = {C_SAT:.2f} %   (dip amplitudes)        gamma = {gamma:.2f}")

    print("\nFibre-free gels: measured vs y0 = 3 psi_A b1 C_Alg")
    for calg, ctoncf, s in model_series(summary):
        s0 = s[np.isclose(s.f, 0.0)]
        if not s0.empty:
            print(f"    {calg:>5g} % alginate:  measured {s0.E_kPa.iloc[0]:5.1f} +/- {s0.SD_kPa.iloc[0]:4.1f} kPa"
                  f"   y0 = {baseline(calg):5.1f} kPa")

    print("\nGoodness of fit (formulation means)")
    for calg, ctoncf, s in model_series(summary):
        r2, rmse = goodness_of_fit(s, calg, ctoncf, q)
        role = "training" if (calg in FIT["train"] and np.isclose(ctoncf, REF_CTONCF)) else "withheld"
        print(f"    {calg:>5g} % alginate, {ctoncf:g} % TONCF stock ({role:8s}):  R2 = {r2:.3f}   RMSE = {rmse:5.1f} kPa")

    A1, A2, B = amplitudes(REF_CALG, REF_CTONCF, q)
    xa, xn, k = x_ana(A1, A2), x_num(A1, A2, B), k_coefficient(A1, A2)
    fa, fp, fn = (f_coord(v, REF_CTONCF) for v in (xa, xa + k * B, xn))
    print(f"\nPeak positions (7.5 % series): X_ANA = {fa:.4f}   X_ANA + kB = {fp:.4f}   X_NUM = {fn:.4f}"
          f"  (TONCF mixing fraction)")

    article_figures(summary, q)


def article_figures(summary, q):
    with plt.style.context(["default", STYLE]):
        fig1_reference(summary)
        if (~np.isclose(summary.Ca, CA)).any():
            fig2_calcium(summary)
        if (summary.Ctoncf > REF_CTONCF).any():
            fig3_stock(summary)
        fig4_disruption()
        fig5_curves(q)
        fig6_amplitudes(q)
        fig7_dip_amplitudes(summary, q)
        fig8_peak_positions(q)
        fig9_calibration(summary, q)
        fig10_validation(summary, q)
        fig11_stock_prediction(summary, q)
        if plt.get_fignums():
            plt.show()


SHOW = {
    "Fig1_reference_series": True,
    "Fig2a_calcium_series": True,
    "Fig2b_calcium_step": True,
    "Fig3_TONCF_stock": True,
    "Fig4_disruption_factor": True,
    "Fig5_predicted_curves": True,
    "Fig6_amplitudes": True,
    "Fig7_dip_amplitude": True,
    "Fig8_peak_positions": True,
    "Fig9_calibration": True,
    "Fig10_model_validation": True,
    "Fig11_stock_prediction": True,
}

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].lower().endswith((".xlsx", ".xlsm", ".xls")) else None
    main(arg)
