import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load data
df = pd.read_csv('SSA_merged_baseline.csv')

# 2. Define Target
target_to_predict = 'Rmax'

# 3. Clean X (Features) 
# We need to drop ANY column that isn't a number.
# This keeps the wavenumbers and drops 'Unnamed: 0', 'Variety', etc.
X = df.select_dtypes(include=[np.number]).drop(columns=['Rmax', 'Protein', 'Absorption', 'Extensibility', 'DDT', 'Stability'], errors='ignore')

# If 'Unnamed: 0' is still there, drop it too
if 'Unnamed: 0' in X.columns:
    X = X.drop(columns=['Unnamed: 0'])

y = df[target_to_predict]

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model
print(f"Training Baseline Random Forest for {target_to_predict}...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Results
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print(f"\n--- SUCCESS: {target_to_predict} Baseline ---")
print(f"Number of Spectral Features used: {X.shape[1]}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")