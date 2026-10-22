import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
DATA_FILE = None
MAIN_SHEET = "5-10% Alg"
CA_SHEET = "TONCF_Ca2+ Variations"
OUT = HERE / "statistics_results"

REF_CTONCF = 3.0      # TONCF stock concentration of the alginate series (% w/v)
CA = 2.0              # crosslinker concentration of the alginate series (%)
P1, DIP, P2 = 0.25, 0.50, 0.70          # positions fixed from the 7.5 % reference series

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
STYLE = {
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "font.size": 11,
    "legend.fontsize": 9,
    "axes.edgecolor": "0.2",
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.color": "#b0b0b0",
    "grid.alpha": 0.6,
    "grid.linewidth": 0.8,
}


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
        if p.name.startswith("~$"):
            continue
        try:
            with pd.ExcelFile(p) as xl:
                if _find_sheet(xl.sheet_names, MAIN_SHEET):
                    found.append(p)
        except Exception:
            pass
    if not found:
        raise FileNotFoundError(f'No Excel workbook with the sheet "{MAIN_SHEET}" in {HERE}. '
                                f"Set DATA_FILE at the top of Statistical_Analysis.py or give the path on the command line.")
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
        main = xl.parse(main_name)
        ca = xl.parse(ca_name) if ca_name else None
    print(f"Data: {path.name if path.resolve().parent == HERE else path}")

    main = main.rename(columns={main.columns[0]: "Calg", main.columns[1]: "f"})
    main["Ctoncf"], main["Ca"] = REF_CTONCF, CA
    parts = [_long_format(main, ["Calg", "Ctoncf", "Ca", "f"])]
    if ca is not None:
        ca = ca.rename(columns={ca.columns[0]: "Calg", ca.columns[1]: "Ctoncf",
                                ca.columns[2]: "Ca", ca.columns[3]: "f"})
        parts.append(_long_format(ca, ["Calg", "Ctoncf", "Ca", "f"]))
    else:
        print(f'Sheet "{CA_SHEET}" not found: tests for Figs. 2 and 3 skipped.')

    specimens = pd.concat(parts, ignore_index=True)
    specimens[["Calg", "Ctoncf", "Ca", "f"]] = specimens[["Calg", "Ctoncf", "Ca", "f"]].astype(float)
    return specimens


def tukey_table(values, groups):
    res = pairwise_tukeyhsd(values, groups.astype(str))
    t = pd.DataFrame(res.summary().data[1:], columns=res.summary().data[0])
    return t.rename(columns={"p-adj": "p_adj"})


def analyse(spec):
    g = spec.groupby(["Calg", "Ctoncf", "Ca", "f"]).E_kPa
    desc = g.agg(mean_kPa="mean", SD_kPa="std", n="count").reset_index()
    desc["CV_percent"] = 100 * desc.SD_kPa / desc.mean_kPa
    desc["Calg_final"] = (1 - desc.f) * desc.Calg
    desc["Ctoncf_final"] = desc.f * desc.Ctoncf

    main_set = spec[np.isclose(spec.Ca, CA) & np.isclose(spec.Ctoncf, REF_CTONCF)].copy()
    anova_rows, table1_rows = [], []
    for alg, s in main_set.groupby("Calg"):
        groups = [x.E_kPa.values for _, x in s.groupby("f")]
        F, p = stats.f_oneway(*groups)
        lev = stats.levene(*groups, center="median").pvalue
        fit = smf.ols("E_kPa ~ C(f)", data=s).fit()
        mse, dfe = fit.mse_resid, fit.df_resid
        anova_rows.append(dict(alginate_stock=alg, F=F, df_between=len(groups) - 1, df_within=int(dfe),
                               p=p, levene_p=lev))
        mu, n = s.groupby("f").E_kPa.mean(), s.groupby("f").size()
        sd = s.groupby("f").E_kPa.std()

        def contrast(a, b):
            se = np.sqrt(mse * (1 / n[a] + 1 / n[b]))
            t = (mu[a] - mu[b]) / se
            return mu[a] - mu[b], stats.t.sf(t, dfe)

        d1, p1 = contrast(P1, DIP)
        d2, p2 = contrast(P2, DIP)
        table1_rows.append(dict(
            alginate_stock=alg, n=int(n.iloc[0]),
            E_f025=mu[P1], SD_f025=sd[P1], E_f050=mu[DIP], SD_f050=sd[DIP], E_f070=mu[P2], SD_f070=sd[P2],
            diff_max1_minus_dip=d1, p_dip_lt_max1=p1, diff_max2_minus_dip=d2, p_dip_lt_max2=p2,
            dip_depth_percent=100 * (1 - mu[DIP] / ((mu[P1] + mu[P2]) / 2))))
    table1 = pd.DataFrame(table1_rows)
    m = 2 * len(table1)                               # Bonferroni: two planned contrasts per series
    table1.insert(table1.columns.get_loc("diff_max2_minus_dip"), "p_bonferroni_dip_lt_max1",
                  np.minimum(1.0, m * table1.p_dip_lt_max1))
    table1.insert(table1.columns.get_loc("dip_depth_percent"), "p_bonferroni_dip_lt_max2",
                  np.minimum(1.0, m * table1.p_dip_lt_max2))

    two_way = smf.ols("E_kPa ~ C(Calg) * C(f)", data=main_set).fit()

    rows = []
    other = spec[~(np.isclose(spec.Ca, CA) & np.isclose(spec.Ctoncf, REF_CTONCF))]
    for (ct, ca), s in other[other.f.isin([0.25, 0.5, 0.75])].groupby(["Ctoncf", "Ca"]):
        if s.f.nunique() < 3:
            continue
        tk = tukey_table(s.E_kPa, s.f)
        rows += [dict(figure="2a/3", alginate_stock=5.0, toncf_stock=ct, crosslinker=ca,
                      anova_p=stats.f_oneway(*[x.E_kPa for _, x in s.groupby("f")]).pvalue,
                      comparison=f"f {r.group1} vs {r.group2}", mean_difference=r.meandiff, tukey_p=r.p_adj)
                 for r in tk.itertuples()]
    ca_set = other[np.isclose(other.Ctoncf, REF_CTONCF) & np.isclose(other.f, 0.5)][["Ca", "E_kPa"]]
    if not ca_set.empty:
        ref_2 = main_set[np.isclose(main_set.Calg, 5.0) & np.isclose(main_set.f, 0.5)][["E_kPa"]].assign(Ca=CA)
        ca_set = pd.concat([ca_set, ref_2])
        tk = tukey_table(ca_set.E_kPa, ca_set.Ca)
        p_ca = stats.f_oneway(*[x.E_kPa for _, x in ca_set.groupby("Ca")]).pvalue
        rows += [dict(figure="2b", alginate_stock=5.0, toncf_stock=3.0, crosslinker=None, anova_p=p_ca,
                      comparison=f"Ca {r.group1} vs {r.group2}", mean_difference=r.meandiff, tukey_p=r.p_adj)
                 for r in tk.itertuples()]

    return dict(desc=desc, main_set=main_set, other=other, ca_set=ca_set,
                anova=pd.DataFrame(anova_rows), table1=table1,
                fig23=pd.DataFrame(rows), resid=two_way.resid, fitted=two_way.fittedvalues,
                shapiro_p=stats.shapiro(two_way.resid).pvalue)


def write_tables(res):
    OUT.mkdir(exist_ok=True)
    res["desc"].round(3).to_csv(OUT / "descriptives.csv", index=False)
    res["anova"].to_csv(OUT / "anova_per_series.csv", index=False)
    res["table1"].to_csv(OUT / "table1_peak_dip_tests.csv", index=False)
    if not res["fig23"].empty:
        res["fig23"].to_csv(OUT / "fig2_fig3_tests.csv", index=False)
    print(f"\nResults written to {OUT.name}/")


def print_summary(spec, res):
    desc = res["desc"]
    fib = desc[np.isclose(desc.Ca, CA) & np.isclose(desc.Ctoncf, REF_CTONCF) & (desc.f > 0)]
    print(f"{len(spec)} specimens, {len(desc)} formulations")
    print(f"CV of fibre-containing formulations (alginate series): {fib.CV_percent.min():.1f}-"
          f"{fib.CV_percent.max():.1f} % (median {fib.CV_percent.median():.1f} %)")
    print(f"Shapiro-Wilk on residuals of the two-way model: p = {res['shapiro_p']:.2f}")
    print("\nOne-way ANOVA per alginate series (factor f)")
    for r in res["anova"].itertuples():
        print(f"  {r.alginate_stock:>5g} %:  F({r.df_between}, {r.df_within}) = {r.F:6.1f}   p = {r.p:.1e}"
              f"   Levene p = {r.levene_p:.2f}")
    print("\nTable 1: dip (f = 0.50) vs maxima (f = 0.25, 0.70); one-sided contrasts, pooled error "
          "(Bonferroni-corrected p in brackets)")
    for r in res["table1"].itertuples():
        print(f"  {r.alginate_stock:>5g} %:  {r.E_f025:6.1f} | {r.E_f050:6.1f} | {r.E_f070:6.1f} kPa   "
              f"p1 = {r.p_dip_lt_max1:.1e} ({r.p_bonferroni_dip_lt_max1:.1e})  "
              f"p2 = {r.p_dip_lt_max2:.1e} ({r.p_bonferroni_dip_lt_max2:.1e})  dip depth {r.dip_depth_percent:.0f} %")


# ----------------------------------------------------------------------------- figures
def _finish(fig, name, tight=True):
    if tight:
        fig.tight_layout()
    if not SHOW.get(name, False):
        plt.close(fig)


def _sci(p):
    if p >= 0.001:
        return f"{p:.2f}"
    m, e = f"{p:.1e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


def fig_r1(res):
    ms, t1 = res["main_set"], res["table1"].set_index("alginate_stock")
    cols = ["#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    ink, ink2, muted, grid, base = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    rc = {"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "font.size": 8.5, "axes.edgecolor": base,
          "axes.labelcolor": ink2, "xtick.color": muted, "ytick.color": muted,
          "axes.spines.top": False, "axes.spines.right": False}
    with plt.style.context(["default", rc]):
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.3), sharey=True)
        axes = axes.ravel()
        for i, alg in enumerate(sorted(ms.Calg.unique())):
            ax, col = axes[i], cols[i % len(cols)]
            s = ms[np.isclose(ms.Calg, alg)]
            g = s.groupby("f").E_kPa
            mu, sd = g.mean(), g.std()
            for fx in (P1, DIP, P2):
                ax.axvline(fx, color=grid, lw=1.0, ls=(0, (3, 2)), zorder=0)
            for f, grp in s.groupby("f"):
                k = len(grp)
                jit = np.linspace(-0.014, 0.014, k) if k > 1 else np.zeros(1)
                ax.scatter(f + jit, grp.E_kPa, s=16, color=col, alpha=0.55, lw=0, zorder=2)
            c = col if i > 0 else "#3987e5"
            ax.errorbar(mu.index, mu.values, yerr=sd.values, color=c, lw=1.6, elinewidth=0.9, capsize=2.2, marker="o",
                        ms=4.2, mfc="white", mec=c, mew=1.2, zorder=3)
            p1, p2 = t1.loc[alg, "p_dip_lt_max1"], t1.loc[alg, "p_dip_lt_max2"]
            txt = rf"dip < 1st max: $p = {_sci(p1)}$" + "\n" + rf"dip < 2nd max: $p = {_sci(p2)}$"
            ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=7.2, color=ink2, linespacing=1.35)
            ax.set_title(f"{alg:g} % alginate stock (n = {int(g.size().iloc[0])})", loc="left", fontsize=8.8, color=ink, pad=4)
            ax.set_xlim(-0.05, 0.95)
            ax.set_xticks([0, 0.25, 0.5, 0.7, 0.9], ["0", "0.25", "0.5", "0.7", "0.9"])
            ax.set_ylim(0, 595)
            ax.set_yticks([0, 100, 200, 300, 400, 500])
            ax.grid(axis="y", color=grid, lw=0.6)
            ax.set_axisbelow(True)
            if i in (0, 3):
                ax.set_ylabel("Young's modulus $E$ [kPa]")
            if i >= 2:
                ax.set_xlabel("TONCF mixing fraction $f$")
        key = axes[5]
        key.axis("off")
        handles = [Line2D([], [], ls="none", marker="o", ms=5, mfc="#256abf", mec="none", alpha=0.55, label="individual specimen"),
                   Line2D([], [], color="#256abf", lw=1.6, marker="o", ms=4.2, mfc="white", mec="#256abf", mew=1.2,
                          label="mean ± SD"),
                   Line2D([], [], color=grid, lw=1.2, ls=(0, (3, 2)), label="pre-specified positions\n($f$ = 0.25, 0.50, 0.70)")]
        key.legend(handles=handles, loc="upper left", frameon=False, fontsize=7.8, handlelength=2.2, borderaxespad=0.2)
        key.text(0.02, 0.38, "p-values: pre-specified one-sided\ncontrasts with the pooled\nwithin-series error (one-way\n"
                 "ANOVA).", transform=key.transAxes,
                 va="top", ha="left", fontsize=7.2, color=ink2, linespacing=1.35)
        fig.tight_layout(h_pad=1.4, w_pad=0.8)
        _finish(fig, "FigR1_individual_specimens", tight=False)


def fig_peak_dip(res):
    t1 = res["table1"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    w = 0.25
    x = np.arange(len(t1))
    for j, (col, sdcol, lab, c) in enumerate([("E_f025", "SD_f025", f"first maximum (f = {P1})", "#1f77b4"),
                                             ("E_f050", "SD_f050", f"dip (f = {DIP})", "#d62728"),
                                             ("E_f070", "SD_f070", f"second maximum (f = {P2})", "#2ca02c")]):
        ax.bar(x + (j - 1) * w, t1[col], w, yerr=t1[sdcol], capsize=3, color=c, alpha=0.85, label=lab)
    top = (t1[["E_f025", "E_f070"]].values + t1[["SD_f025", "SD_f070"]].values).max(1)
    for xi, r, y in zip(x, t1.itertuples(), top):
        ax.text(xi, y + 12, f"p₁ = {r.p_dip_lt_max1:.0e}\np₂ = {r.p_dip_lt_max2:.0e}", ha="center", va="bottom",
                fontsize=9)
    ax.set_xticks(x, [f"{a:g} %" for a in t1.alginate_stock])
    ax.set_xlabel("alginate stock")
    ax.set_ylabel("E [kPa]")
    ax.set_ylim(0, top.max() * 1.3)
    ax.set_title("Dip vs maxima: one-sided contrasts with pooled error (p₁: dip < first max, p₂: dip < second max)",
                 fontsize=10)
    ax.legend(loc="upper left", ncol=3)
    ax.grid(axis="x", visible=False)
    _finish(fig, "S1_peak_dip_tests")


def fig_residuals(res):
    r, fv = res["resid"], res["fitted"]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.3))
    (osm, osr), (slope, icpt, _) = stats.probplot(r, dist="norm")
    ax1.scatter(osm, osr, s=12, color="#1f77b4")
    ax1.plot(osm, slope * np.asarray(osm) + icpt, "k-", lw=1)
    ax1.set_title(f"Normal Q-Q of residuals (Shapiro-Wilk p = {res['shapiro_p']:.2f})", fontsize=10)
    ax1.set_xlabel("theoretical quantiles")
    ax1.set_ylabel("residual [kPa]")
    ax2.scatter(fv, r, s=12, color="#1f77b4")
    ax2.axhline(0, color="k", lw=1)
    ax2.set_title("Residuals vs fitted (two-way model)", fontsize=10)
    ax2.set_xlabel("fitted E [kPa]")
    ax2.set_ylabel("residual [kPa]")
    a = res["anova"]
    ax3.bar([f"{x:g} %" for x in a.alginate_stock], a.levene_p, color=COLORS[:len(a)])
    ax3.axhline(0.05, color="red", ls="--", lw=1)
    ax3.text(len(a) - 0.5, 0.06, "p = 0.05", color="red", ha="right", va="bottom", fontsize=9,
             bbox=dict(fc="white", ec="none", pad=1))
    ax3.set_title("Levene test (equal variances) per series", fontsize=10)
    ax3.set_xlabel("alginate stock")
    ax3.set_ylabel("p")
    ax3.set_ylim(0, 1)
    _finish(fig, "S2_residuals")


def fig_fig2_fig3(res):
    other, ca_set, f23 = res["other"], res["ca_set"], res["fig23"]
    if f23.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    sub = other[other.f.isin([0.25, 0.5, 0.75])]
    fig2a_colors = {(2.0, 1.0): "#2ca02c", (3.0, 1.0): "#ff7f0e", (4.0, 1.0): "#1f77b4", (4.0, 2.0): "#9467bd"}
    for (ct, ca), s in sub.groupby(["Ctoncf", "Ca"]):
        if s.f.nunique() < 3:
            continue
        m = s.groupby("f").E_kPa.agg(["mean", "std"])
        p = f23[(f23.figure == "2a/3") & np.isclose(f23.toncf_stock, ct) & np.isclose(f23.crosslinker, ca)].anova_p.iloc[0]
        ax1.errorbar(m.index, m["mean"], yerr=m["std"], marker="o", capsize=3, color=fig2a_colors.get((ct, ca), "0.4"),
                     label=f"{ct:g} % TONCF / {ca:g} % Ca$^{{2+}}$: ANOVA p = {p:.1e}")
    ax1.set_xticks([0.25, 0.5, 0.75])
    ax1.set_xlabel("TONCF mixing fraction $f$")
    ax1.set_ylabel("E [kPa]")
    ax1.set_ylim(bottom=0)
    ax1.set_title("Figs. 2a and 3 (5 % alginate)", fontsize=11)
    ax1.legend()
    if not ca_set.empty:
        m = ca_set.groupby("Ca").E_kPa.agg(["mean", "std"])
        ax2.errorbar(m.index, m["mean"], yerr=m["std"], marker="o", capsize=3, color="#137b8a", lw=2)
        t = f23[f23.figure == "2b"]
        lv = list(m.index)
        for a, b in zip(lv[:-1], lv[1:]):
            row = t[t.comparison.isin([f"Ca {a} vs {b}", f"Ca {b} vs {a}"])]
            if row.empty:
                continue
            y = max(m.loc[a, "mean"] + m.loc[a, "std"], m.loc[b, "mean"] + m.loc[b, "std"]) + 6
            pv = row.tukey_p.iloc[0]
            ax2.text((a + b) / 2, y, "p < 0.001" if pv < 0.001 else f"p = {pv:.3f}", ha="center", fontsize=9)
        ax2.set_xticks(lv)
        ax2.set_ylim(0, (m["mean"] + m["std"]).max() * 1.3)
        p = t.anova_p.iloc[0]
        ax2.set_title(f"Fig. 2b (f = 0.5): ANOVA p = {p:.1e}; Tukey p for neighbours", fontsize=11)
        ax2.set_xlabel("Ca$^{2+}$ concentration [% w/v]")
        ax2.set_ylabel("E [kPa]")
    _finish(fig, "S3_fig2_fig3_tests")


def figures(res):
    with plt.style.context(["default", STYLE]):
        fig_r1(res)
        fig_peak_dip(res)
        fig_residuals(res)
        fig_fig2_fig3(res)
        if plt.get_fignums():
            plt.show()


def main(path=None):
    spec = load_measurements(path)
    res = analyse(spec)
    print_summary(spec, res)
    if SAVE_TABLES:
        write_tables(res)
    figures(res)


SAVE_TABLES = True        # write the results to statistics_results/*.csv
SHOW = {
    "FigR1_individual_specimens": True,   # Fig. R1 of the response: every specimen, mean ± SD, dip vs maxima
    "S1_peak_dip_tests": True,            # Table 1: dip vs both maxima
    "S2_residuals": True,                 # normality (Q-Q, Shapiro-Wilk) and equal variances (Levene)
    "S3_fig2_fig3_tests": True,           # ANOVA and Tukey for the formulations of Figs. 2 and 3
}

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].lower().endswith((".xlsx", ".xlsm", ".xls")) else None
    main(arg)
