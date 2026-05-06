import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

X_train = np.load("X_train.npy")
X_test  = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test  = np.load("y_test.npy")

with open("lr_model.pkl",  "rb") as f: lr  = pickle.load(f)
with open("knn_model.pkl", "rb") as f: knn = pickle.load(f)

# --- Decision Tree ---
dt = DecisionTreeClassifier(random_state=42, max_depth=5)
dt.fit(X_train, y_train)

with open("dt_model.pkl", "wb") as f:
    pickle.dump(dt, f)

# --- Comparison Table ---
models = {
    "Logistic Regression": lr,
    "KNN":                 knn,
    "Decision Tree":       dt,
}

results = []
for name, model in models.items():
    preds = model.predict(X_test)
    results.append({
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_test,  preds), 4),
        "Precision": round(precision_score(y_test, preds), 4),
        "Recall":    round(recall_score(y_test,    preds), 4),
        "F1-Score":  round(f1_score(y_test,        preds), 4),
    })

df_results = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(df_results.to_string(index=False))

best = df_results.loc[df_results["F1-Score"].idxmax(), "Model"]
print(f"\n✅ Best Model (by F1-Score): {best}")

with open("best_model_name.txt", "w") as f:
    f.write(best)

# --- Confusion Matrices ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor("#0D1117")

for ax, (name, model) in zip(axes, models.items()):
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Low Risk", "High Risk"],
                yticklabels=["Low Risk", "High Risk"])
    ax.set_title(name, fontsize=12, fontweight="bold", color="white")
    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("Actual", color="white")
    ax.tick_params(colors="white")

plt.suptitle("Confusion Matrices", fontsize=15, fontweight="bold", color="white")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, facecolor="#0D1117")
plt.show()

# --- Bar Chart ---
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
x = np.arange(len(metrics))
width = 0.25
colors = ["#58A6FF", "#FF9800", "#4CAF50"]

fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor("#0D1117")
ax.set_facecolor("#161B22")

for i, (_, row) in enumerate(df_results.iterrows()):
    vals = [row[m] for m in metrics]
    bars = ax.bar(x + i * width, vals, width, label=row["Model"], color=colors[i])
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom",
                fontsize=8, color="white")

ax.set_xticks(x + width)
ax.set_xticklabels(metrics, fontsize=12, color="white")
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score", fontsize=12, color="white")
ax.set_title("Models Comparison", fontsize=14, fontweight="bold", color="white")
ax.tick_params(colors="white")
ax.legend(facecolor="#0D1117", labelcolor="white")
ax.grid(axis="y", linestyle="--", alpha=0.3, color="white")

plt.tight_layout()
plt.savefig("comparison_chart.png", dpi=150, facecolor="#0D1117")
plt.show()

print("\nconfusion_matrices.png and comparison_chart.png saved.")