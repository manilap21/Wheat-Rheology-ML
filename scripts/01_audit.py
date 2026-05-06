import pandas as pd

# Load the data
df = pd.read_csv('data_raw/your_file_name.csv') # Change this to your actual file name!

# 1. Check Rheology Targets
rheology_targets = ['Rmax', 'DDT', 'Protein'] # Add whatever traits you have
print("--- Rheology Check ---")
print(df[rheology_targets].describe())

# 2. Check Spectral Features
# Assuming columns 5 onwards are your FTIR wavenumbers
spectral_data = df.iloc[:, 5:]
print("\n--- Spectral Check ---")
print(f"Wavelengths found: {spectral_data.shape[1]}")
print(f"Range: {spectral_data.columns[0]} to {spectral_data.columns[-1]}")

# 3. Check for Missing Data (The 'Pipeline Killer')
print("\n--- Missing Values ---")
print(df.isnull().sum().sum())