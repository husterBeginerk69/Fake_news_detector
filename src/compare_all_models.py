import pandas as pd
import joblib

from sklearn.metrics import *

df = pd.read_csv("data/oof_predictions.csv")

y = df["label"]

rows = []

# NB
for name, col in {
    "Naive Bayes":"prob_nb",
    "SVM":"prob_svm",
    "Random Forest":"prob_rf"
}.items():

    prob = df[col]

    pred = (prob >= 0.5).astype(int)

    rows.append({
        "Model":name,
        "Accuracy":accuracy_score(y,pred),
        "Precision":precision_score(y,pred),
        "Recall":recall_score(y,pred),
        "F1":f1_score(y,pred),
        "AUC":roc_auc_score(y,prob)
    })

# STACKING

meta = joblib.load(
    "models/meta_model.pkl"
)

X = df[
    [
        "prob_nb",
        "prob_svm",
        "prob_rf"
    ]
]

pred = meta.predict(X)

prob = meta.predict_proba(X)[:,1]

rows.append({
    "Model":"Stacking (LR)",
    "Accuracy":accuracy_score(y,pred),
    "Precision":precision_score(y,pred),
    "Recall":recall_score(y,pred),
    "F1":f1_score(y,pred),
    "AUC":roc_auc_score(y,prob)
})

result = pd.DataFrame(rows)

print(result)

result.to_csv(
    "results/final_metrics.csv",
    index=False
)