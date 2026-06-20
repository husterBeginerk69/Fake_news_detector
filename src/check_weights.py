import joblib
import pandas as pd

meta = joblib.load("models/meta_model.pkl")

weights = pd.DataFrame({
    "Feature": ["prob_nb", "prob_svm", "prob_rf"],
    "Weight": meta.coef_[0]
})

weights.to_csv(
    "results/meta_weights.csv",
    index=False
)

print(weights)