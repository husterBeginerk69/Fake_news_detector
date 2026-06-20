import pandas as pd


TEST_FOLD = 5

df = pd.read_csv(
    "data/oof_predictions.csv"
)

df = df[
    df["fold_id"] == TEST_FOLD
]

X = df[
    [
        "prob_nb",
        "prob_svm",
        "prob_rf"
    ]
]

y = df["label"]