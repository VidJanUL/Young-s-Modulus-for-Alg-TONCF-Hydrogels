# Young's modulus of alginate–TONCF composite hydrogels: data and model

Data and code for the paper *"Origin and prediction of the M-shaped Young's modulus response of
alginate hydrogels reinforced with TEMPO-oxidised cellulose nanofibres"* (citation and DOI will be
added on publication).

The repository contains the Young's modulus of every tested specimen, raw force–displacement data
with the script that converts them into Young's moduli, the statistical analysis of the M-shaped
response, and the model that describes how the modulus depends on the TONCF mixing fraction *f*
and on the alginate, TONCF and calcium concentrations.

## Contents

| File | Description |
|------|-------------|
| `Data.xlsx` | Young's modulus of every individual specimen (Pa), 252 specimens in 75 formulations |
| `Raw_UTM_data.xlsx` | Raw force–displacement records of three specimens, one sheet per specimen |
| `Model.py` | Model, calibration, prediction and Figs. 1b–11 of the paper |
| `Statistical_Analysis.py` | Statistical analysis of the M-shaped response (Table 1 of the paper) |
| `Raw_UTM_data_analysis.py` | Stress–strain curves, Young's moduli and tensile strengths from the raw records (Fig. 1a) |
| `LICENSE` | MIT licence |

## Requirements

Python 3.10 or newer with

```
pip install numpy pandas scipy statsmodels matplotlib openpyxl
```

The results in the paper were produced with Python 3.13, numpy 2.5, pandas 3.0, scipy 1.18,
statsmodels 0.15 and matplotlib 3.11. Other recent versions (tested: pandas 2 and 3, matplotlib
3.9 and 3.11) give identical numbers and visually identical figures.

## Usage

Keep the scripts and the two workbooks in one folder and run

```
python Model.py                   # model parameters and goodness of fit; displays Figs. 1b-11
python Statistical_Analysis.py    # statistics of the M-shaped response; displays Fig. R1 and S1-S3
python Raw_UTM_data_analysis.py   # stress-strain curves and Young's moduli of the raw records (Fig. 1a)
```

- `Model.py` and `Statistical_Analysis.py` read `Data.xlsx` from their own folder. To use a
  workbook elsewhere, set `DATA_FILE` at the top of the script or give the path on the command
  line (`python Model.py path/to/workbook.xlsx`).
- Means and standard deviations are always calculated from the individual specimens; the mean
  and SD columns of the workbook are not used.
- The scripts display their figures and do not write image files. At the end of `Model.py` and
  `Statistical_Analysis.py`, `SHOW` selects the figures that are displayed.
- `Model.py` also writes `data_summary.csv` (mean, SD and *n* of every formulation, with final
  concentrations). `Statistical_Analysis.py` writes its result tables to `statistics_results/`
  unless `SAVE_TABLES = False`.
- In `Raw_UTM_data_analysis.py`, `sheets` lists the specimens that are analysed and plotted.

## Data

`Data.xlsx` contains one row per formulation and one column per specimen (`n = 1` … `n = 4`),
with the Young's modulus in Pa.

- **`5-10% Alg`**: alginate stock concentrations of 5, 6.25, 7.5, 8.75 and 10 % (w/v), 3 % (w/v)
  TONCF stock, 2 % calcium, twelve mixing fractions from 0 to 0.9; *n* = 3 per formulation
  (*n* = 4 for 7.5 %). Columns: alginate stock [%], TONCF mixing fraction, `n = 1` … `n = 4`.
- **`TONCF_Ca2+ Variations`**: 5 % (w/v) alginate stock with 2, 3 and 4 % (w/v) TONCF stocks and
  1–2 % calcium; *n* = 4. Columns: alginate stock [%], TONCF stock [%], Ca²⁺ [%], TONCF mixing
  fraction, `n = 1` … `n = 4`.

The composites were prepared by blending the alginate and TONCF stocks by volume, so the final
concentrations (% w/v) are `C_TONCF = f · C_TONCF,stock` and `C_Alg = (1 − f) · C_Alg,stock`.

## Raw tensile data

`Raw_UTM_data.xlsx` contains the records of three specimens (5 % alginate, *f* = 0.25; 7.5 %
alginate, *f* = 0.50; 10 % alginate, *f* = 0.75; each `n = 1`), with the columns `Reading`,
`Load [N]`, `Travel [mm]` and `Time`.

`Raw_UTM_data_analysis.py` converts each record into a stress–strain curve, determines the Young's
modulus and plots the three curves with their fitted lines (Fig. 1a of the paper):

- engineering stress = load / (thickness × width), with the thickness and width measured on each
  specimen (listed in the script);
- engineering strain = travel / gauge length, with a gauge length of 30 mm;
- Young's modulus = slope of a least-squares line through the linear region of the curve, taken
  as the data points between 40 % and 75 % of the record;
- tensile strength = maximum stress.

For the three specimens the script reproduces the values in `Data.xlsx` ("5%Alg 0,25f n=1", "7,5%Alg 0,5f n=1" and "10%Alg 0,75f n=1").

## Statistical analysis

`Statistical_Analysis.py` tests whether the peak–dip–peak structure exceeds the scatter between
specimens:

- one-way ANOVA per alginate series, with the mixing fraction as the factor;
- two pre-specified one-sided contrasts per series, using the pooled within-series error: modulus
  at the dip (*f* = 0.50) lower than at the first maximum (*f* = 0.25) and lower than at the second
  maximum (*f* = 0.70); the positions are those of the 7.5 % reference series and are applied
  unchanged to every series;
- Bonferroni correction for the ten contrasts;
- Shapiro–Wilk test of the residuals (normality) and Levene's test within each series (equal
  variances);
- one-way ANOVA with Tukey's HSD test for the formulations of Figs. 2 and 3 of the paper.

## Model

```
E(x) = y0 + A1 <x − Pc1>+^t1 − A2 <x − Pc2>+^t2 − B · G(x) · σ(x),     <u>+ = max(u, 0),  x = 3f

y0 = 3 ψ_A b1 C_Alg
A1 = 3 ψ b1 (C_Alg + q (N_TONCF / N_Alg) C_TONCF) d_fac,    A2 = 2 A1
B  = b2 max(C_sat(C_TONCF) − C_Alg, 0),    C_sat(C_TONCF) = C_sat · min(C_TONCF, C_ref) / C_ref
```

`x = 3f` is dimensionless; for the 3 % TONCF stock its value equals the final TONCF concentration
in % (w/v). `Pc1`, `Pc2`, `μ` and `ω` are on the same scale. `b1 = N_Alg (RT/2) aff` and
`b2 = N_TONCF (RT/2) aff` are ionic cross-link (shear-modulus) scales in kPa per % (w/v); the
factor 3 converts the shear modulus to the Young's modulus (E = 3G), and `ψ` and `ψ_A` are the
effective network factors of the fibre-containing and the fibre-free gel.

| Parameter | Origin |
|-----------|--------|
| `b1`, `b2`, `aff`, `β` | ionic cross-linking framework |
| `Pc1` | excluded-volume theory: `Pc1 = 100 · ρ · V_fib / V_ex` = 0.28 (R = 15 nm, L = 8 µm, ρ = 1.5 g/cm³) |
| `Pc2`, `μ`, `ω` | geometry of the 7.5 % reference curve |
| `t1`, `t2` | fit of the model to the shape of the 7.5 % reference curve, with `Pc1`, `Pc2`, `μ`, `ω` and `A2 = 2 A1` fixed |
| `ψ_A` | fibre-free gels of the 5, 7.5 and 10 % series |
| `ψ`, `q` | joint least-squares fit to the 5, 7.5 and 10 % series |
| `C_sat` | dip amplitudes of the 5, 7.5 and 10 % series (`B = b2 (C_sat − C_Alg)`) |

`ψ`, `q` and `C_sat` are iterated to self-consistency. The 6.25 and 8.75 % alginate series and the
5 % alginate series prepared with the 4 % TONCF stock are not used in the calibration and are
predicted. The peak-position check (Eq. 33 of the paper) uses the first-order coefficient
`k = H'(X_ANA) / E0''(X_ANA)`, with `H = G·σ`, evaluated at the peak `X_ANA` of the power-law backbone.

`Model.py` prints all parameters and the goodness of fit of every series and displays the figures
of the paper:

| Figure | Content |
|--------|---------|
| 1b | Reference series (7.5 % alginate stock); panel 1a is drawn by `Raw_UTM_data_analysis.py` |
| 2a, 2b | Effect of the calcium concentration |
| 3 | Effect of the TONCF stock concentration |
| 4 | Network-disruption factor |
| 5 | Predicted curves across alginate concentrations |
| 6 | Amplitudes A1 and B versus alginate concentration |
| 7 | Dip amplitude B and the saturation concentration C_sat |
| 8 | Peak positions X_NUM and X_ANA + kB |
| 9 | Determination of ψ, q and ψ_A |
| 10 | Model and measurements for the five alginate series |
| 11 | Prediction for the 4 % TONCF stock |

## Licence

MIT licence; see `LICENSE`.
