import joblib

print(type(joblib.load("models/full_vectorizer_title.pkl")))
print(type(joblib.load("models/full_vectorizer_body.pkl")))
print(type(joblib.load("models/full_svd_rf.pkl")))