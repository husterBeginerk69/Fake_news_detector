import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

TEST_FOLD = 5

FEATURES = [
    "prob_nb",
    "prob_svm",
    "prob_rf"
]

# =========================
# LOAD TEST FOLD
# =========================

df = pd.read_csv(
    "data/oof_predictions.csv"
)

test_df = df[
    df["fold_id"] == TEST_FOLD
]

X_test = test_df[
    FEATURES
]

y_test = test_df[
    "label"
]

# =========================
# LOAD META MODEL
# =========================

meta = joblib.load(
    "models/meta_model.pkl"
)

# =========================
# PREDICT
# =========================

y_pred = meta.predict(
    X_test
)

y_prob = meta.predict_proba(
    X_test
)[:, 1]

# =========================
# METRICS
# =========================

print("\n===== META MODEL RESULTS =====")

print(
    "Accuracy:",
    accuracy_score(
        y_test,
        y_pred
    )
)

print(
    "Precision:",
    precision_score(
        y_test,
        y_pred
    )
)

print(
    "Recall:",
    recall_score(
        y_test,
        y_pred
    )
)

print(
    "F1:",
    f1_score(
        y_test,
        y_pred
    )
)

print(
    "AUC:",
    roc_auc_score(
        y_test,
        y_prob
    )
)