import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression

TEST_FOLD = 5

df = pd.read_csv(
    "data/oof_predictions.csv"
)

train_df = df[
    df["fold_id"] != TEST_FOLD
]

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

meta = LogisticRegression(
    random_state=42,
    max_iter=1000
)

meta.fit(
    X_train,
    y_train
)

joblib.dump(
    meta,
    "models/meta_model.pkl"
)

print(
    "Meta model saved to models/meta_model.pkl"
)