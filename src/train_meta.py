import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# =========================
# CONFIG
# =========================

TEST_FOLD = 5

# =========================
# LOAD OOF
# =========================

df = pd.read_csv(
    "data/oof_predictions.csv"
)

print("Fold distribution:")
print(
    df["fold_id"]
    .value_counts()
    .sort_index()
)

# =========================
# SPLIT BY FOLD
# =========================

train_df = df[
    df["fold_id"] != TEST_FOLD
]

test_df = df[
    df["fold_id"] == TEST_FOLD
]

print(
    f"\nTrain samples: {len(train_df)}"
)

print(
    f"Test samples: {len(test_df)}"
)

# =========================
# FEATURES
# =========================

FEATURES = [
    "prob_nb",
    "prob_svm",
    "prob_rf"
]

X_train = train_df[
    FEATURES
]

y_train = train_df[
    "label"
]

X_test = test_df[
    FEATURES
]

y_test = test_df[
    "label"
]

# =========================
# TRAIN META MODEL
# =========================

meta = LogisticRegression(
    random_state=42,
    max_iter=1000
)

meta.fit(
    X_train,
    y_train
)

# =========================
# SAVE MODEL
# =========================

joblib.dump(
    meta,
    "models/meta_model.pkl"
)

print(
    "\nSaved: models/meta_model.pkl"
)

# =========================
# EVALUATE
# =========================

y_pred = meta.predict(
    X_test
)

y_prob = meta.predict_proba(
    X_test
)[:, 1]

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

auc = roc_auc_score(
    y_test,
    y_prob
)

print("\n===== META MODEL RESULTS =====")

print(
    f"Accuracy : {accuracy:.6f}"
)

print(
    f"Precision: {precision:.6f}"
)

print(
    f"Recall   : {recall:.6f}"
)

print(
    f"F1 Score : {f1:.6f}"
)

print(
    f"AUC      : {auc:.6f}"
)