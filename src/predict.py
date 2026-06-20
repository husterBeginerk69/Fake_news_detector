import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack

# Load artifacts

vec_title = joblib.load(
    "models/full_vectorizer_title.pkl"
)

vec_body = joblib.load(
    "models/full_vectorizer_body.pkl"
)

svd = joblib.load(
    "models/full_svd_rf.pkl"
)

nb = joblib.load(
    "models/final_model_nb.pkl"
)

svm = joblib.load(
    "models/final_model_svm.pkl"
)

rf = joblib.load(
    "models/final_model_rf.pkl"
)

meta = joblib.load(
    "models/meta_model.pkl"
)


def predict_news(title, body):

    title_vec = vec_title.transform(
        [title]
    )

    body_vec = vec_body.transform(
        [body]
    )

    X = hstack([
        title_vec,
        body_vec
    ])

    # Level 1

    prob_nb = nb.predict_proba(X)[0][1]

    prob_svm = svm.predict_proba(X)[0][1]

    X_rf = svd.transform(X)

    prob_rf = rf.predict_proba(X_rf)[0][1]

    # Level 2

    meta_input = pd.DataFrame({
        "prob_nb": [prob_nb],
        "prob_svm": [prob_svm],
        "prob_rf": [prob_rf]
    })

    final_prob = meta.predict_proba(
        meta_input
    )[0][1]

    prediction = (
        "Fake"
        if final_prob >= 0.5
        else "Real"
    )

    return {
        "prediction": prediction,
        "confidence": float(final_prob),
        "prob_nb": float(prob_nb),
        "prob_svm": float(prob_svm),
        "prob_rf": float(prob_rf)
    }


if __name__ == "__main__":

    result = predict_news(
        "Breaking News",
        "This is a test article."
    )

    print(result)