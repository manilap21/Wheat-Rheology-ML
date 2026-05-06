import pandas as pd
import os

# 1. Load the data
# Let's start with Set A (SSA)
spectra = pd.read_csv('SSA_FTIR_data.csv')
reference = pd.read_csv('SSA_reference_data.csv')

# 2. Join the data
# Assuming both files have a common column like 'Sample_ID' or 'Index'
# If they don't have a shared ID, we assume they are in the same order
if 'Sample_ID' in spectra.columns:
    df = pd.merge(spectra, reference, on='Sample_ID')
else:
    print("No common ID found, merging by row index...")
    df = pd.concat([spectra, reference], axis=1)

# 3. Quick Data Health Audit
print("--- Dataset Dimensions ---")
print(f"Total Samples: {df.shape[0]}")
print(f"Total Columns: {df.shape[1]}")

# 4. Identify your target (Rmax)
# Double check the exact spelling in your SSA_reference_data.csv
target_col = 'Rmax' 

if target_col in df.columns:
    print(f"\n--- Target Statistics ({target_col}) ---")
    print(df[target_col].describe())
else:
    print(f"\nError: {target_col} not found. Available columns: {list(reference.columns)}")

# Save the merged file so we don't have to do this again
df.to_csv('SSA_merged_baseline.csv', index=False)
print("\nSuccess: SSA_merged_baseline.csv created.")