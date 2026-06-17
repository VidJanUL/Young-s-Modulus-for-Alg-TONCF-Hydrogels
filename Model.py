"""
Local / working version: inline data, all diagnostic plots, shown on screen.

Same model and methodology as the GitHub version (model.py), just packaged as a
single self-contained script you can run and copy figures from:
  - parameters fixed from theory (A) and from the reference-curve geometry (B),
  - q (the ONLY fitted parameter) fit on the 7.5 % alginate reference curve,
  - C_sat COMPUTED from the falling agglomeration amplitude B and the theory
    slope b3 (self-consistent with q; never fitted),
  - everything else predicted with q held fixed.
"""

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

np.random.seed(0)

# ====================================================================
# (A) Fixed from theory / ionic-crosslinking framework
# ====================================================================
R_gas      = 8.314
T          = 293
Nion_TONCF = 1.5     # mmol/g
Nion_Alg   = 2.4     # mmol/g
a_ion      = 0.55
Ca         = 2.0     # crosslinker concentration, %
Ct_ref     = 3.0     # reference TONCF stock concentration, %

affinity = np.exp(-a_ion / Ca)
b1 = Nion_Alg   * 1e-3 * R_gas * T * 10 / 2 * affinity      # alginate amplitude
b3 = Nion_TONCF * 1e-3 * R_gas * T * 10 / 2 * affinity      # TONCF amplitude = slope of B vs alginate
beta  = Nion_TONCF / (Nion_Alg * Ct_ref * affinity**2)     # disruption rate
alpha = Nion_TONCF / Nion_Alg

# ====================================================================
# (B) Fixed from the geometry of the 7.5 % reference curve
# ====================================================================
A2_r = 2.0                       # A2 = 2*A1
Pc1  = 0.02                      # percolation onset
Pc2  = 1.918                     # second, geometric threshold
mu   = 1.63                      # agglomeration (Gaussian) centre
t1   = 0.526                     # reinforcement exponent
t2   = 1.437                     # reorganisation exponent
Xv1, Xv2 = 0.975, 2.09           # the two peak positions of the reference curve
sigma = ((Xv2 - Xv1) / 10) ** 0.5

# ====================================================================
# (D) C_sat is COMPUTED below (not fixed). This is only the iteration seed.
# ====================================================================
csat = 20.0

REF_CALG   = 7.5     # reference alginate concentration (calibration curve)
REF_CTONCF = 3.0

# ====================================================================
# Data (E in Pa). Each dataset: TONCF stock, alginate, list of (f, E)
# ====================================================================
datasets = [
    dict(Ctoncf=3.0, Calg=5.0,  data=[
        (0,13114),(0.1,110334),(0.2,197028),(0.25,227495),(0.3,231813),
        (0.4,171339),(0.5,125330),(0.6,147580),(0.7,255812),(0.75,239351),(0.8,187603),(0.9,70580)]),
    dict(Ctoncf=3.0, Calg=6.25, data=[
        (0,18527),(0.1,135746),(0.2,229967),(0.25,260434),(0.3,264752),
        (0.4,213220),(0.5,177211),(0.6,199461),(0.7,298848),(0.75,282887),(0.8,219640),(0.9,84580)]),
    dict(Ctoncf=3.0, Calg=7.5,  data=[
        (0,23688),(0.1,160907),(0.2,257411),(0.25,287877),(0.3,292196),
        (0.4,260606),(0.5,224597),(0.6,246847),(0.7,343216),(0.75,326755),(0.8,248007),(0.9,100580)]),
    dict(Ctoncf=3.0, Calg=8.75, data=[
        (0,29853),(0.1,177073),(0.2,289597),(0.25,320063),(0.3,319384),
        (0.4,297977),(0.5,279968),(0.6,312218),(0.7,386773),(0.75,370312),(0.8,285564),(0.9,114137)]),
    dict(Ctoncf=3.0, Calg=10.0, data=[
        (0,35767),(0.1,192986),(0.2,308478),(0.25,338945),(0.3,337264),
        (0.4,326855),(0.5,331846),(0.6,373096),(0.7,426667),(0.75,410205),(0.8,325458),(0.9,144030)]),
    dict(Ctoncf=4.0, Calg=5.0, data=[(0.25,150000),(0.50,86000),(0.75,178000)]),  # higher TONCF stock
]

# baseline modulus y0(Calg) from the f=0 measurements (linear)
_base = [(d['Calg'], d['data'][0][1] / 1000) for d in datasets if d['data'][0][0] == 0]
_y0_poly = np.polyfit([c for c, _ in _base], [y for _, y in _base], 1)
def get_y0(ds):
    return float(np.polyval(_y0_poly, ds['Calg']))


# ====================================================================
# Model
# ====================================================================
def activation(x): return 1 / (1 + np.exp(-(x - Pc1) / 0.08))
def gaussian(x):   return np.exp(-(x - mu)**2 / (2 * sigma**2))

def model(x, A1, A2, B, y0):
    return (y0 + A1 * np.maximum(x - Pc1, 0)**t1
               - A2 * np.maximum(x - Pc2, 0)**t2
               - B  * gaussian(x) * activation(x))

def analytical_peak(A1, A2):
    x = Pc2 + 0.5
    for _ in range(40):
        x = max(x, Pc2 + 1e-9); dx1 = max(x - Pc1, 1e-9); dx2 = max(x - Pc2, 1e-9)
        F  = A1*t1*dx1**(t1-1) - A2*t2*dx2**(t2-1)
        dF = A1*t1*(t1-1)*dx1**(t1-2) - A2*t2*(t2-1)*dx2**(t2-2)
        x -= F / (dF + 1e-9)
    return x

def Gprime(x):  return -(x - mu) / sigma**2 * gaussian(x)
def Gsecond(x): return ((x - mu)**2 - sigma**2) / sigma**4 * gaussian(x)
def Ysecond(x, A1, A2, B):
    dx1 = np.maximum(x - Pc1, 1e-9); dx2 = np.maximum(x - Pc2, 1e-9)
    return A1*t1*(t1-1)*dx1**(t1-2) - A2*t2*(t2-1)*dx2**(t2-2) - B*Gsecond(x)

def csat_eff(Ctoncf):
    # C_sat scales with TONCF stock up to the reference, then saturates
    return csat * min(Ctoncf, Ct_ref) / Ct_ref

def get_params(ds, q):
    Ctoncf, Calg = ds['Ctoncf'], ds['Calg']
    d_fac = np.exp(-beta * max(Ctoncf - Ct_ref, 0.0))
    A1 = b1 * (Calg + q * Nion_TONCF * Ctoncf / Nion_Alg) * d_fac
    A2 = A2_r * A1
    B  = max(b3 * (csat_eff(Ctoncf) - Calg), 0.0)
    return A1, A2, B, csat_eff(Ctoncf), d_fac


# ====================================================================
# Calibration:  q is the ONLY fitted parameter (on the reference curve);
#               C_sat is COMPUTED from the falling B + theory slope b3.
# ====================================================================
REF = next(d for d in datasets if d['Ctoncf'] == REF_CTONCF and d['Calg'] == REF_CALG)

def fit_q():
    y0 = get_y0(REF); v = np.array(REF['data']); x = v[:, 0] * 3; y = v[:, 1] / 1000
    def residuals(p):
        q, = p
        A1, A2, B, _, _ = get_params(REF, q)
        res = list(model(x, A1, A2, B, y0) - y)
        xs = np.linspace(0, 2.8, 4000); xn = xs[np.argmax(model(xs, A1, A2, B, y0))]
        xa = analytical_peak(A1, A2); k = Gprime(xn) / (Ysecond(xn, A1, A2, B) + 1e-9)
        res.append((xn - (xa + k * B)) * 500)
        return np.array(res)
    best = None
    for q0 in np.linspace(1, 10, 6):
        r = least_squares(residuals, [q0], bounds=([0], [20]), max_nfev=40000)
        if best is None or r.cost < best.cost: best = r
    return float(best.x[0])

def compute_csat(q):
    # B at each 3% dataset = dip depth below the reinforcement-only curve at x=mu
    w = gaussian(mu) * activation(mu); est = []
    for ds in datasets:
        if ds['Ctoncf'] != Ct_ref: continue
        A1 = b1 * (ds['Calg'] + q * Nion_TONCF * Ct_ref / Nion_Alg)
        A2 = A2_r * A1
        reinf = get_y0(ds) + A1*max(mu-Pc1, 0)**t1 - A2*max(mu-Pc2, 0)**t2
        v = np.array(sorted(ds['data'])); E_mu = np.interp(mu, v[:, 0]*3, v[:, 1]/1000)
        B = (reinf - E_mu) / w
        est.append(ds['Calg'] + B / b3)     # C_sat = C_Alg + B/b3 (theory slope)
    return float(np.mean(est))

# self-consistency loop (only q is ever fitted)
for _ in range(12):
    q = fit_q()
    csat = compute_csat(q)

print("(A) theory:      b1=%.3f  b3=%.3f kPa/%%   beta=%.4f   affinity=%.4f" % (b1, b3, beta, affinity))
print("(B) ref. curve:  Pc1=%g Pc2=%g mu=%g t1=%g t2=%g sigma=%.4f" % (Pc1, Pc2, mu, t1, t2, sigma))
print("(C) FITTED:      q = %.4f      <-- the only least-squares parameter" % q)
print("(D) COMPUTED:    C_sat = %.2f %%  (from falling B + theory slope b3)" % csat)

print("\nPer-dataset (q fixed):")
for ds in datasets:
    A1, A2, B, cs, d = get_params(ds, q)
    print("  Ct=%g%% Calg=%g%%  A1=%.1f  B=%.1f  d_fac=%.3f" % (ds['Ctoncf'], ds['Calg'], A1, B, d))


# ====================================================================
# PLOTS
# ====================================================================
colors5 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# PLOT 1: reference calibration + predicted alginate series
fig1, ax1 = plt.subplots(figsize=(9, 6))
for i, ds in enumerate([d for d in datasets if d['Ctoncf'] == 3.0]):
    A1, A2, B, cs, d = get_params(ds, q); y0 = get_y0(ds)
    v = np.array(ds['data']); xs = np.linspace(0, 2.8, 400)
    is_ref = ds['Calg'] == REF_CALG
    ax1.scatter(v[:, 0], v[:, 1] / 1000, color=colors5[i], s=25, zorder=5)
    ax1.plot(xs / 3, model(xs, A1, A2, B, y0), color=colors5[i], lw=2.6 if is_ref else 1.6,
             label="%g%% alg%s" % (ds['Calg'], " (calibration)" if is_ref else " (prediction)"))
ax1.set_xlabel("TONCF fraction"); ax1.set_ylabel("E [kPa]")
ax1.set_title("Calibration on 7.5%% alginate, prediction of the rest  (q=%.3f)" % q)
ax1.legend(fontsize=10); ax1.grid(alpha=0.3)

# PLOT 2: Ct=4% prediction
fig2, ax2 = plt.subplots(figsize=(7, 5))
ds4 = next(d for d in datasets if d['Ctoncf'] == 4.0)
A1, A2, B, cs, d = get_params(ds4, q); y0 = get_y0(ds4); xs = np.linspace(0, 2.8, 400)
xm = [p[0] for p in ds4['data']]; ym = [p[1] / 1000 for p in ds4['data']]
yp = [model(np.array([x * 3]), A1, A2, B, y0)[0] for x in xm]
ax2.scatter(xm, ym, color='red', s=80, zorder=5, label='measured')
ax2.scatter(xm, yp, color='black', s=80, zorder=5, marker='D', label='predicted')
ax2.plot(xs / 3, model(xs, A1, A2, B, y0), 'k-', lw=1.5, alpha=0.5)
ax2.set_xlabel("TONCF fraction"); ax2.set_ylabel("E [kPa]")
ax2.set_title("Ct=4%% prediction (Calg=5%%) - no re-fitting\nd_fac=%.3f  A1=%.1f  B=%.1f" % (d, A1, B))
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

# PLOT 3: disruption factor
fig3, ax3 = plt.subplots(figsize=(7, 4))
Ctr = np.linspace(2.5, 6, 200)
ax3.plot(Ctr, [np.exp(-beta * max(c - Ct_ref, 0)) for c in Ctr], 'k-', lw=2,
         label='d = exp(-%.3f(Ct-%g))' % (beta, Ct_ref))
ax3.axvline(Ct_ref, color='gray', ls=':', lw=1.5, label='Ct_ref=%g%%' % Ct_ref)
for c, pos in zip([3.0, 4.0], [(3.55, 0.30), (5.10, 0.30)]):
    dv = np.exp(-beta * max(c - Ct_ref, 0)); ax3.scatter([c], [dv], s=80, zorder=5)
    ax3.annotate("Ct=%g%%\nd=%.3f (%.1f%% red.)" % (c, dv, (1 - dv) * 100), xy=(c, dv), xytext=pos,
                 fontsize=9, ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9),
                 arrowprops=dict(arrowstyle='->', color='gray', lw=1))
ax3.set_xlabel("Ctoncf [%]"); ax3.set_ylabel("disruption factor"); ax3.set_ylim(0, 1.05)
ax3.set_title("Network disruption factor (beta=%.4f, theory)" % beta)
ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

# PLOT 4: C_sat from the falling (computed) B
fig4, ax4 = plt.subplots(figsize=(7, 5))
w = gaussian(mu) * activation(mu); cc, BB = [], []
for ds in [d for d in datasets if d['Ctoncf'] == 3.0]:
    A1 = b1 * (ds['Calg'] + q * Nion_TONCF * Ct_ref / Nion_Alg); A2 = A2_r * A1
    reinf = get_y0(ds) + A1*max(mu-Pc1, 0)**t1 - A2*max(mu-Pc2, 0)**t2
    v = np.array(sorted(ds['data'])); E_mu = np.interp(mu, v[:, 0]*3, v[:, 1]/1000)
    cc.append(ds['Calg']); BB.append((reinf - E_mu) / w)
crange = np.linspace(0, csat * 1.15, 300)
ax4.plot(crange, np.maximum(np.mean(BB) - b3 * (crange - np.mean(cc)), 0), 'k-', lw=2,
         label='slope -b3 (theory)')
for i, (c, B) in enumerate(zip(cc, BB)):
    ax4.scatter([c], [B], color=colors5[i], s=70, zorder=5, label='%g%% alg (computed B)' % c)
ax4.axvline(csat, color='red', ls='--', lw=1.5, label='C_sat=%.1f%% (computed)' % csat)
ax4.axhline(0, color='gray', ls=':', lw=1)
ax4.set_xlabel("Alginate concentration [%]"); ax4.set_ylabel("B [kPa]")
ax4.set_title("C_sat from the falling B: dip vanishes at ~%.1f%% alginate" % csat)
ax4.legend(fontsize=9); ax4.grid(alpha=0.3)

# PLOT 5: A1 and B vs alginate concentration
fig5, ax5 = plt.subplots(figsize=(7, 5))
cf = np.linspace(1, csat * 1.3, 300)
ax5.plot(cf, [b1 * (c + q * Nion_TONCF * Ct_ref / Nion_Alg) for c in cf], 'b-', lw=2,
         label='$A_1$ (reinforcement)')
ax5.plot(cf, [max(b3 * (csat - c), 0) for c in cf], 'r--', lw=2, label='$B$ (agglomeration)')
ax5.axvline(csat, color='gray', ls='--', lw=1.5, label='C_sat=%.1f%%' % csat)
ax5.axhline(0, color='gray', ls=':', lw=1)
for i, ds in enumerate([d for d in datasets if d['Ctoncf'] == 3.0]):
    c = ds['Calg']
    ax5.scatter([c], [b1 * (c + q * Nion_TONCF * Ct_ref / Nion_Alg)], color=colors5[i], s=60, zorder=5)
    ax5.scatter([c], [max(b3 * (csat - c), 0)], color=colors5[i], s=60, zorder=5, marker='D')
ax5.set_xlabel("Alginate concentration [%]"); ax5.set_ylabel("Parameter value [kPa]")
ax5.set_title("A1 (circles) and B (diamonds) vs alginate concentration")
ax5.legend(fontsize=9); ax5.grid(alpha=0.3)

# PLOT 6: X_NUM / X_ANA / X_ANA+kB with error panel
fig6, (ax6a, ax6b) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                  gridspec_kw={'height_ratios': [3, 1]})
cp = np.linspace(5, csat * 1.2, 200); xa_l, xn_l, xp_l = [], [], []
for c in cp:
    A1 = b1 * (c + q * Nion_TONCF * Ct_ref / Nion_Alg); A2 = A2_r * A1
    B = max(b3 * (csat - c), 0); xs = np.linspace(0, 2.8, 8000)
    xn = xs[np.argmax(model(xs, A1, A2, B, 0))]; xa = analytical_peak(A1, A2)
    k = Gprime(xn) / (Ysecond(xn, A1, A2, B) + 1e-9)
    xn_l.append(xn / 3); xa_l.append(xa / 3); xp_l.append((xa + k * B) / 3)
ax6a.plot(cp, xn_l, '#1f77b4', lw=2, label='$X_{NUM}$')
ax6a.plot(cp, xp_l, '#2ca02c', lw=2, ls='--', label='$X_{ANA}+kB$')
ax6a.plot(cp, xa_l, '#d62728', lw=2, ls=':', label='$X_{ANA}$')
ax6a.axvline(csat, color='gray', ls='--', lw=1, alpha=0.6, label='C_sat=%.1f%%' % csat)
ax6a.set_ylabel("TONCF fraction"); ax6a.set_title("Peak-position self-consistency (Xana / Xnum)")
ax6a.legend(fontsize=9); ax6a.grid(alpha=0.3)
ax6b.plot(cp, np.array(xn_l) - np.array(xp_l), '#ff7f0e', lw=1.5)
ax6b.axhline(0, color='gray', lw=0.8, ls='--'); ax6b.axvline(csat, color='gray', ls='--', lw=1, alpha=0.6)
ax6b.set_ylabel("error"); ax6b.set_xlabel("Alginate concentration [%]"); ax6b.grid(alpha=0.3)

# PLOT 7: model curves approaching and crossing csat
fig7, ax7 = plt.subplots(figsize=(9, 6))
showcase = [5.0, 10.0, 15.0, csat, csat * 1.2, csat * 1.5]
cols = ['#1f77b4', '#ff7f0e', '#2ca02c', 'red', '#9467bd', '#8c564b']
xd = np.linspace(0, 2.8, 500)
for c, col in zip(showcase, cols):
    A1 = b1 * (c + q * Nion_TONCF * Ct_ref / Nion_Alg); A2 = A2_r * A1
    B = max(b3 * (csat - c), 0); y0 = float(np.polyval(_y0_poly, c))
    if abs(c - csat) < 0.1: lbl, lw, ls = 'Calg=csat=%.1f%% (transition)' % c, 2.5, '-'
    elif c > csat:          lbl, lw, ls = 'Calg=%.1f%% (above csat, B=0)' % c, 1.5, '--'
    else:                   lbl, lw, ls = 'Calg=%.1f%% (B=%.1f)' % (c, B), 1.5, '-'
    ax7.plot(xd / 3, model(xd, A1, A2, B, y0), color=col, lw=lw, ls=ls, label=lbl)
ax7.set_xlabel("TONCF fraction"); ax7.set_ylabel("E [kPa]")
ax7.set_title("Model curves approaching and crossing C_sat=%.1f%%" % csat)
ax7.legend(fontsize=9); ax7.grid(alpha=0.3)

plt.show()