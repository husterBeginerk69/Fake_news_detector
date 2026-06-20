import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/oof_predictions.csv")

X = df[
    [
        "prob_nb",
        "prob_svm",
        "prob_rf"
    ]
]

y = df["label"]

meta = LogisticRegression(
    random_state=42,
    max_iter=1000
)

meta.fit(X,y)

joblib.dump(
    meta,
    "models/meta_model.pkl"
)

print("Done")