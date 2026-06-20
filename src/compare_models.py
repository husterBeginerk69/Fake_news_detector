import os
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# =========================
# CONFIG
# =========================

TEST_FOLD = 5

FEATURES = [
    "prob_nb",
    "prob_svm",
    "prob_rf"
]

# =========================
# OUTPUT FOLDER
# =========================

os.makedirs(
    "results",
    exist_ok=True
)

# =========================
# LOAD TEST FOLD
# =========================

df = pd.read_csv(
    "data/oof_predictions.csv"
)

df = df[
    df["fold_id"] == TEST_FOLD
]

y = df["label"]

# =========================
# LEVEL-1 MODELS
# =========================

models = {
    "Naive Bayes":
        df["prob_nb"],

    "SVM":
        df["prob_svm"],

    "Random Forest":
        df["prob_rf"]
}

# =========================
# META MODEL
# =========================

meta = joblib.load(
    "models/meta_model.pkl"
)

meta_prob = meta.predict_proba(
    df[FEATURES]
)[:, 1]

models["Stacking"] = meta_prob

# =========================
# EVALUATE
# =========================

rows = []

for name, probs in models.items():

    pred = (
        probs >= 0.5
    ).astype(int)

    rows.append({

        "Model":
            name,

        "Accuracy":
            accuracy_score(
                y,
                pred
            ),

        "Precision":
            precision_score(
                y,
                pred
            ),

        "Recall":
            recall_score(
                y,
                pred
            ),

        "F1":
            f1_score(
                y,
                pred
            ),

        "AUC":
            roc_auc_score(
                y,
                probs
            )
    })

result = pd.DataFrame(
    rows
)

# =========================
# SORT BY F1
# =========================

result = result.sort_values(
    by="F1",
    ascending=False
)

print(
    "\n===== MODEL COMPARISON =====\n"
)

print(
    result.round(4)
)

# =========================
# SAVE
# =========================

result.to_csv(
    "results/model_comparison.csv",
    index=False
)

print(
    "\nSaved: results/model_comparison.csv"
)