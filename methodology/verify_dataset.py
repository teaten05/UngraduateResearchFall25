import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('research_output/dataset/data.csv')  # Replace TIMESTAMP

print(f" Dataset Summary:")
print(f"   Samples: {len(df)}")
print(f"   Nodes (N): {df['num_nodes'].iloc[0]}")
print(f"   Features: {df.shape[1] - 3}")  # Minus r_final, efficiency, radius
print(f"   r_final range: [{df['r_final'].min():.3f}, {df['r_final'].max():.3f}]")
print(f"   Mean r_final: {df['r_final'].mean():.3f}")

# Quick correlation check
print(f"\n🔍 Top correlations with r_final:")
correlations = df.corr()['r_final'].sort_values(ascending=False)
print(correlations.head(10))

# Save for later
print("\n✅ Data looks good! Ready for symbolic regression.")