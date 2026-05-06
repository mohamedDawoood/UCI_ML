import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("heart.csv")

# Fix target: 0 = Low Risk, 1+ = High Risk
df['target'] = (df['num'] > 0).astype(int)

# Drop useless columns
df.drop(columns=['id', 'dataset', 'num'], inplace=True)

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(f"Samples:  {df.shape[0]}")
print(f"Features: {df.shape[1]}")
print("\nColumn Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nClass Distribution:\n", df['target'].value_counts())
print("\nBasic Stats:\n", df.describe())

# --- Histograms (numeric only) ---
df.select_dtypes(include='number').hist(figsize=(14, 10), color="#2196F3", edgecolor="black")
plt.suptitle("Feature Distributions", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("histograms.png", dpi=150)
plt.show()

# --- Correlation Heatmap (numeric only) ---
plt.figure(figsize=(12, 8))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("heatmap.png", dpi=150)
plt.show()

print("\nEDA Done. Plots saved.")