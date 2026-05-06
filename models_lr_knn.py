import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X_train = np.load("X_train.npy")
X_test  = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test  = np.load("y_test.npy")

# --- Logistic Regression ---
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

print("=" * 50)
print("LOGISTIC REGRESSION")
print("=" * 50)
print(f"Accuracy: {accuracy_score(y_test, lr_preds):.4f}")
print(classification_report(y_test, lr_preds, target_names=["Low Risk", "High Risk"]))

# --- KNN ---
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_preds = knn.predict(X_test)

print("=" * 50)
print("KNN (K=5)")
print("=" * 50)
print(f"Accuracy: {accuracy_score(y_test, knn_preds):.4f}")
print(classification_report(y_test, knn_preds, target_names=["Low Risk", "High Risk"]))

# --- Save Models ---
with open("lr_model.pkl", "wb") as f:
    pickle.dump(lr, f)

with open("knn_model.pkl", "wb") as f:
    pickle.dump(knn, f)

print("lr_model.pkl and knn_model.pkl saved.")