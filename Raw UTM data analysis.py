import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# load data from Excel
path = r"C:\Users\\Data.xlsx"
sheet = "Name of the Sheet"   # name of the sheet, or its index (0 = first sheet, 1 = second, ...)

data = pd.read_excel(path, sheet_name=sheet)

length = 0.030       # this dimension is fixed [m]
thickness = 0.00111  # 0,00578 for "5%Alg 0,25f n=1", 0,00611 for "7,5%Alg 0,5f n=1", 0,00561" for "10%Alg 0,75f n=1"
width = 0.00111    # 0,00284 for "5%Alg 0,25f n=1", 0,00305 for "7,5%Alg 0,5f n=1", 0,00274" for "10%Alg 0,75f n=1"

cross_sectional_area = thickness * width

# convert the data to stress and strain
data['Strain'] = data['Travel [mm]'] / 1000 / length  # assumes length is already in the correct units
data['Stress [Pa]'] = data['Load [N]'] / cross_sectional_area

# Plot stress-strain
plt.figure(figsize=(10, 6))
plt.plot(data['Strain'], data['Stress [Pa]'], marker='o', linestyle='-')
plt.title('Stress-Strain Curve')
plt.xlabel('Strain')
plt.ylabel('Stress (Pa)')
plt.grid(True)
plt.show()

# The linear region is somewhere around 40%–75% of our measurements
linear_region = data.loc[int(len(data) * 0.40):int(len(data) * 0.75)]
slope, intercept, r_value, p_value, std_err = linregress(linear_region['Strain'], linear_region['Stress [Pa]'])
youngs_modulus = slope  # given in pascals

tensile_strength = data['Stress [Pa]'].max()

print("Young's modulus is:", youngs_modulus, "Pa")
print("Tensile strength is:", tensile_strength, "Pa")
