import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os

# --- SETTINGS ---
INPUT_FILE = 'SSA_merged_baseline.csv'
OUTPUT_FILE = 'data/processed/SSA_processed.csv'
os.makedirs('data/processed', exist_ok=True)
os.makedirs('plots/preprocessing', exist_ok=True)

def process_dataset_1():
    print(f"🔄 Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # Identify spectral columns (numbers) and metadata
    spectral_cols = [col for col in df.columns if col.replace('.', '', 1).isdigit()]
    meta_cols = [col for col in df.columns if not col.replace('.', '', 1).isdigit()]

    X = df[spectral_cols].values
    wavenumbers = np.array([float(c) for c in spectral_cols])
    
    # Sort for plotting
    sort_idx = np.argsort(wavenumbers)
    w_sorted = wavenumbers[sort_idx]
    X_sorted = X[:, sort_idx]

    # --- STEP 1: SNV (Standard Normal Variate) ---
    # Removes the vertical "Spaghetti" shifts
    X_snv = (X_sorted - np.mean(X_sorted, axis=1).reshape(-1, 1)) / np.std(X_sorted, axis=1).reshape(-1, 1)

    # --- STEP 2: Savitzky-Golay 1st Derivative ---
    # Sharpen peaks. Window=11 is a safe bet for FTIR, poly=2
    X_sg = savgol_filter(X_snv, window_length=11, polyorder=2, deriv=1)

    # Save the output
    df_clean = pd.concat([
        df[meta_cols].reset_index(drop=True), 
        pd.DataFrame(X_sg, columns=w_sorted.astype(str))
    ], axis=1)
    
    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ SSA Processed and saved to {OUTPUT_FILE}")

    # --- STEP 3: The "James Validation" Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(w_sorted, X_sg[0], label='Sample 1 (Processed)', color='teal')
    plt.plot(w_sorted, X_sg[10], label='Sample 11 (Processed)', color='orange')
    plt.gca().invert_xaxis()
    plt.title("Dataset 1: After SNV + SG Derivative")
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("d(Absorbance)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/preprocessing/SSA_clean_check.png')
    print("📈 Validation plot saved: plots/preprocessing/SSA_clean_check.png")

if __name__ == "__main__":
    process_dataset_1()