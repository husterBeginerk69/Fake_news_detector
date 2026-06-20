# Tên file: predict.py
import joblib
import pandas as pd
from scipy.sparse import hstack

# Gọi hàm làm sạch từ file preprocessing.py vừa tạo
from src.preprocessing import preprocess_text

# Tải các mô hình
vec_title = joblib.load("models/full_vectorizer_title.pkl")
vec_body = joblib.load("models/full_vectorizer_body.pkl")
svd = joblib.load("models/full_svd_rf.pkl")

nb = joblib.load("models/final_model_nb.pkl")
svm = joblib.load("models/final_model_svm.pkl")
rf = joblib.load("models/final_model_rf.pkl")
meta = joblib.load("models/meta_model.pkl")


def predict_news(title, body):
    # 1. TIỀN XỬ LÝ (Sử dụng hàm đã import)
    clean_title, clean_body = preprocess_text(title, body)

    # 2. VECTOR HÓA
    title_vec = vec_title.transform([clean_title])
    body_vec = vec_body.transform([clean_body])
    X = hstack([title_vec, body_vec])

    # 3. DỰ ĐOÁN CẤP ĐỘ 1 (Level 1)
    prob_nb = nb.predict_proba(X)[0][1]
    prob_svm = svm.predict_proba(X)[0][1]
    
    X_rf = svd.transform(X)
    prob_rf = rf.predict_proba(X_rf)[0][1]

    # 4. DỰ ĐOÁN META (Level 2)
    meta_input = pd.DataFrame({
        "prob_nb": [prob_nb],
        "prob_svm": [prob_svm],
        "prob_rf": [prob_rf]
    })

    final_prob = meta.predict_proba(meta_input)[0][1]
    prediction = "Fake" if final_prob >= 0.5 else "Real"

    # ===== META EXPLANATION =====

    scores = {
        "Naive Bayes": prob_nb,
        "SVM": prob_svm,
        "Random Forest": prob_rf
    }

    dominant_model = max(
       scores,
       key=scores.get
    )

    explanation = (
        f"Meta model relied most on "
        f"{dominant_model} "
        f"({scores[dominant_model]:.1%}) "
        f"when making this prediction."
    )

    return {
        "prediction": prediction,
        "fake_probability": float(final_prob),
        "prob_nb": float(prob_nb),
        "prob_svm": float(prob_svm),
        "prob_rf": float(prob_rf),
        "explanation": explanation
    }

if __name__ == "__main__":
    result = predict_news(
        "Breaking News",
        "This is a test article."
    )
    print(result)