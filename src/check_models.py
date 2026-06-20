import joblib

print("===== SVM =====")

svm = joblib.load(
    "models/final_model_svm.pkl"
)

print(type(svm))
print(svm)

print("\n===== RF =====")

rf = joblib.load(
    "models/final_model_rf.pkl"
)

print(type(rf))
print(rf)