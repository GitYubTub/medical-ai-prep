import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OneHotEncoder

# 1) Load the dataset
df = pd.read_csv("crc_dataset1.csv")

print("\n=== First 5 rows ===")
print(df.head())

print("\n=== Column names ===")
print(df.columns)


# 2) Choose target column
# --------------------------
# Build features (numeric + categorical)
# --------------------------

TARGET_COL = "CRC_Risk"
y = df[TARGET_COL]

# Numeric features
X_numeric = df.select_dtypes(include=["number"]).drop(
    columns=[TARGET_COL, "Participant_ID"]
)

# Categorical features (these are NOT real text)
cat_cols = ["Gender", "Lifestyle", "Family_History_CRC", "Pre-existing Conditions"]

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

X_cat = encoder.fit_transform(df[cat_cols])

# Combine numeric + categorical
X = np.hstack([X_numeric.values, X_cat])

# Debug prints
print("X_numeric shape:", X_numeric.shape)
print("X_cat shape:", X_cat.shape)
print("X combined shape:", X.shape)

# 4) Train/Test split (do this BEFORE oversampling)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nClass counts BEFORE oversampling (train):")
print(y_train.value_counts())

X_train_bal = X_train
y_train_bal = y_train

print("\nClass counts AFTER oversampling (train):")
print(y_train_bal.value_counts())


# 6) Train a simple model (Logistic Regression)
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=5,
    min_samples_leaf=15,
    class_weight={0: 1, 1: 2},  # CRC is rarer, so weight it higher
    random_state=42
)

model.fit(X_train_bal, y_train_bal)


# 7) Predict on test set
y_prob = model.predict_proba(X_test)[:, 1]

# Apply decision threshold
threshold = 0.6
y_pred = (y_prob >= threshold).astype(int)

# 8) Evaluate like the paper
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Sensitivity (Recall for class 1)
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

# Specificity (True Negative Rate)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print("\n=== RESULTS ===")
print("Confusion Matrix:")
print(cm)

y_prob = model.predict_proba(X_test)[:, 1]
print("Average predicted CRC probability:", y_prob.mean())

print("Predicted counts:")
print(pd.Series(y_pred).value_counts())

print(f"\nAccuracy:     {acc:.4f}")
print(f"F1-score:     {f1:.4f}")
print(f"Sensitivity:  {sensitivity:.4f}")
print(f"Specificity:  {specificity:.4f}")
