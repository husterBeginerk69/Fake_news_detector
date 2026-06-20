import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

df = pd.read_csv("data/oof_predictions.csv")

y = df["label"]

models = {
    "Naive Bayes": df["prob_nb"],
    "SVM": df["prob_svm"],
    "Random Forest": df["prob_rf"]
}

rows = []

for name, probs in models.items():

    pred = (probs >= 0.5).astype(int)

    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "AUC": roc_auc_score(y, probs)
    })

result = pd.DataFrame(rows)

print(result)

result.to_csv(
    "results/level1_metrics.csv",
    index=False
)