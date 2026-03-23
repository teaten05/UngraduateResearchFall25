import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('research_output/dataset/data.csv')

# Check sample count
print(f"Total samples: {len(df)}")  # Should be ~1300-1400

# Check N
print(f"Nodes per network: {df['num_nodes'].iloc[0]}")  # Should be 150

# Check r_final range
print(f"r_final range: [{df['r_final'].min():.3f}, {df['r_final'].max():.3f}]")

# Plot λ₂ correlation
plt.scatter(df['lambda_2'], df['r_final'], alpha=0.5)
plt.xlabel('λ₂')
plt.ylabel('r_final')
plt.title(f'N={len(df)} samples')
plt.savefig('validation.png')
print("✅ Saved validation.png")