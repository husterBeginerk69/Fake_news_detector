import pandas as pd
import joblib

from sklearn.metrics import *

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

pred = meta.predict(X)

prob = meta.predict_proba(X)[:,1]

print(
    "ACC",
    accuracy_score(y,pred)
)

print(
    "F1",
    f1_score(y,pred)
)

print(
    "AUC",
    roc_auc_score(y,prob)
)