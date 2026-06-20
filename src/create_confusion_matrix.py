import os
import pandas as pd
import joblib

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

os.makedirs("results", exist_ok=True)

df = pd.read_csv("data/oof_predictions.csv")

X = df[
    ["prob_nb", "prob_svm", "prob_rf"]
]

y = df["label"]

meta = joblib.load(
    "models/meta_model.pkl"
)

pred = meta.predict(X)

ConfusionMatrixDisplay.from_predictions(
    y,
    pred
)

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

print("Saved!")