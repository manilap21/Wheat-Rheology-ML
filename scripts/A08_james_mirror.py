import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def james_mirror_model():
    df = pd.read_csv('data/processed/SSA_processed.csv')
    
    # James's Secret Weapon: The A5 Region (1480 - 1180)
    # We filter columns to only include those in the A5 range
    spectral_cols = [c for c in df.columns if c.replace('.', '', 1).isdigit()]
    a5_cols = [c for c in spectral_cols if 1180 <= float(c) <= 1480]
    
    X = df[a5_cols]
    y = df['Protein'] # Let's start with Protein to see if we can hit his 0.97

    # Split and use James's component count for Protein
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = PLSRegression(n_components=3) 
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"🎯 James Mirror (Protein A5) R²: {r2_score(y_test, y_pred):.3f}")

if __name__ == "__main__":
    james_mirror_model()