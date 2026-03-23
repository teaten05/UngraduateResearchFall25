"""
Symbolic Regression for Drone Synchronization
Week 2 - Initial Run
"""

import numpy as np
import pandas as pd
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

print("🔬 Starting Symbolic Regression...")
print("="*60)

# Load data
data = np.load('research_output/run_TIMESTAMP/ml_data.npz')  # UPDATE THIS
X = data['X']
y = data['y']
feature_names = list(data['feature_names'])

print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"   Target (r_final) range: [{y.min():.3f}, {y.max():.3f}]")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

# Symbolic Regression
print(f"\n🚀 Running symbolic regression (this will take 15-30 min)...")
print("   Iterations: 40")
print("   Searching for interpretable formulas...\n")

model = PySRRegressor(
    niterations=40,
    binary_operators=["+", "*", "/", "-"],
    unary_operators=["sqrt", "log", "square"],
    maxsize=12,
    populations=30,
    population_size=50,
    ncycles_per_iteration=550,
    procs=4,  # Use 4 cores
    verbosity=1,
    random_state=42
)

# Fit model
model.fit(X_train, y_train, variable_names=feature_names)

# Evaluate
print("\n" + "="*60)
print("📈 RESULTS")
print("="*60)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"\n📊 Performance:")
print(f"   Train R²:  {train_r2:.4f}")
print(f"   Test R²:   {test_r2:.4f}")
print(f"   Train RMSE: {train_rmse:.4f}")
print(f"   Test RMSE:  {test_rmse:.4f}")

if test_r2 > 0.70:
    print("\n🎉 SUCCESS! R² > 0.70 achieved!")
elif test_r2 > 0.60:
    print("\n👍 Good start! Need some tuning to reach R² > 0.70")
else:
    print("\n⚠️  Need more tuning. Try longer run or different parameters.")

# Best equation
print(f"\n🎯 Best Equation:")
print(model.sympy())

# Show top 5 equations
print(f"\n📋 Top 5 Equations (by score):")
print(model.equations_[['complexity', 'loss', 'score']].head(10))

# Save results
model.equations_.to_csv('symbolic_regression_equations.csv', index=False)
print("\n💾 Saved equations to 'symbolic_regression_equations.csv'")

# Plot predictions vs actual
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Train
axes[0].scatter(y_train, y_pred_train, alpha=0.5, s=20)
axes[0].plot([y_train.min(), y_train.max()], 
             [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual r_final')
axes[0].set_ylabel('Predicted r_final')
axes[0].set_title(f'Train Set (R² = {train_r2:.3f})')
axes[0].grid(True, alpha=0.3)

# Test
axes[1].scatter(y_test, y_pred_test, alpha=0.5, s=20, color='orange')
axes[1].plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Actual r_final')
axes[1].set_ylabel('Predicted r_final')
axes[1].set_title(f'Test Set (R² = {test_r2:.3f})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('predictions_vs_actual.png', dpi=150)
print("💾 Saved plot to 'predictions_vs_actual.png'")

print("\n" + "="*60)
print("✅ Symbolic regression complete!")
print("="*60)