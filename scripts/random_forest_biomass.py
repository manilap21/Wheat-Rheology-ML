import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Load your PCA features (The DNA markers)
print("📂 Loading PCA features...")
X = np.load('sorghum_pca_results.npy') 

# 2. Load your Traits (The Phenotype)
print("📂 Loading traits from Full_traits_dataset_499.csv...")
traits_df = pd.read_csv('Full_traits_dataset_499.csv')

# Based on your screenshot, the target column is 'Yield'
y = traits_df['Yield'].values 

# 3. Train/Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train the Random Forest
print("🌲 Training the Random Forest... this will be fast!")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 5. Make Predictions
predictions = rf.predict(X_test)

# 6. Evaluate Accuracy
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print("-" * 30)
print(f"✅ R^2 Score: {r2:.3f}")
print(f"✅ Mean Absolute Error: {mae:.3f}")
print("-" * 30)

# 7. Save the results and the model
joblib.dump(rf, 'sorghum_yield_rf_model.pkl')
results_df = pd.DataFrame({'Actual_Yield': y_test, 'Predicted_Yield': predictions})
results_df.to_csv('rf_prediction_results.csv', index=False)

# 8. Visualize: Predicted vs Actual
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.6, color='forestgreen')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Yield (metric ton/ha)')
plt.ylabel('Predicted Yield (metric ton/ha)')
plt.title(f'Sorghum Yield Prediction (R^2: {r2:.2f})')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('biomass_prediction_plot.png')

print("📊 Prediction plot saved as biomass_prediction_plot.png")
print("🚀 Done! You are ready for your 3 PM meeting.")