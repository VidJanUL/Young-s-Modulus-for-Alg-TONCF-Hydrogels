from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# load data from Excel (one sheet per specimen)
path = Path(__file__).resolve().parent / "Raw_UTM_data.xlsx"
sheets = ["5%Alg 0,25f n=1", "7,5%Alg 0,5f n=1", "10%Alg 0,75f n=1"]   # specimens to analyse 

length = 0.030       # gauge length, fixed [m]
dimensions = {       # width, thickness [m], measured on each specimen
    "5%Alg 0,25f n=1": (0.00578, 0.00284),
    "7,5%Alg 0,5f n=1": (0.00611, 0.00305),
    "10%Alg 0,75f n=1": (0.00561, 0.00274),
}
labels = {           # legend of the stress-strain plot
    "5%Alg 0,25f n=1": "5 % alg., $f$ = 0.25",
    "7,5%Alg 0,5f n=1": "7.5 % alg., $f$ = 0.50",
    "10%Alg 0,75f n=1": "10 % alg., $f$ = 0.75",
}
colors = ["#1f77b4", "#2ca02c", "#d62728"]

# same figure style as Model.py
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

with plt.style.context(["default", STYLE]):
    fig, ax = plt.subplots(figsize=(7, 5.0))
    for sheet, color in zip(sheets, colors):
        data = pd.read_excel(path, sheet_name=sheet)
        width, thickness = dimensions[sheet]
        cross_sectional_area = thickness * width

        # convert the data to stress and strain
        data['Strain'] = data['Travel [mm]'] / 1000 / length
        data['Stress [Pa]'] = data['Load [N]'] / cross_sectional_area

        # The linear region is somewhere around 40%–75% of our measurements
        linear_region = data.loc[int(len(data) * 0.40):int(len(data) * 0.75)]
        slope, intercept, r_value, p_value, std_err = linregress(linear_region['Strain'], linear_region['Stress [Pa]'])
        youngs_modulus = slope  # given in pascals

        tensile_strength = data['Stress [Pa]'].max()

        print(sheet)
        print("  Young's modulus is:", youngs_modulus, "Pa")
        print("  Tensile strength is:", tensile_strength, "Pa")

        # stress-strain curve (kPa) and the fitted line over the fitting interval
        ax.plot(data['Strain'], data['Stress [Pa]'] / 1000, color=color, lw=1.5,
                label=f"{labels[sheet]}, $E$ = {youngs_modulus / 1000:.0f} kPa")
        ax.plot(linear_region['Strain'], (slope * linear_region['Strain'] + intercept) / 1000, color="black", ls="--", lw=2)

    ax.plot([], [], color="black", ls="--", lw=2, label="linear fit, 40–75 % of the record")
    ax.set_xlabel("Engineering strain [–]")
    ax.set_ylabel("Engineering stress [kPa]")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=11, frameon=False)
    fig.tight_layout()
    plt.show()
