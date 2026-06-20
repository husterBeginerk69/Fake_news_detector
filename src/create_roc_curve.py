import os
import pandas as pd
import joblib

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

os.makedirs("results", exist_ok=True)

df = pd.read_csv("data/oof_predictions.csv")

X = df[
    [
        "prob_nb",
        "prob_svm",
        "prob_rf"
    ]
]

y = df["label"]

meta = joblib.load(
    "models/meta_model.pkl"
)

prob = meta.predict_proba(X)[:,1]

RocCurveDisplay.from_predictions(
    y,
    prob
)

plt.savefig(
    "results/roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

print("ROC curve saved!")