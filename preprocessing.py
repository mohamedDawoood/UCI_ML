import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

df = pd.read_csv("heart.csv")

# --- Fix Target ---
df['target'] = (df['num'] > 0).astype(int)
df.drop(columns=['id', 'dataset', 'num'], inplace=True)

# --- Encode Categorical Columns FIRST ---
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

# --- Split Features / Target ---
X = df.drop("target", axis=1)
y = df["target"]

with open("feature_names.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

# --- Impute Missing Values (after encoding) ---
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

with open("imputer.pkl", "wb") as f:
    pickle.dump(imputer, f)

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

# --- Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# --- Save Arrays ---
np.save("X_train.npy", X_train_scaled)
np.save("X_test.npy",  X_test_scaled)
np.save("y_train.npy", y_train.values)
np.save("y_test.npy",  y_test.values)

print("=" * 50)
print("PREPROCESSING DONE")
print("=" * 50)
print(f"Train samples : {X_train_scaled.shape[0]}")
print(f"Test samples  : {X_test_scaled.shape[0]}")
print(f"Features      : {X_train_scaled.shape[1]}")
print(f"NaN remaining : {np.isnan(X_train_scaled).sum()}")
print("\nAll files saved.")