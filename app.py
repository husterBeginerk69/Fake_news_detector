from flask import Flask, render_template, request, jsonify

from src.predict import predict_news

app = Flask(__name__)


# =========================
# HOME
# =========================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================
# PREDICT
# =========================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    data = request.get_json()

    title = data.get(
        "title",
        ""
    )

    body = data.get(
        "body",
        ""
    )

    if not title.strip() and not body.strip():

        return jsonify({

            "error":
                "Please enter article content"

        }), 400

    result = predict_news(
        title,
        body
    )

    return jsonify({

        "prediction":
            result["prediction"],

        "confidence":
            round(
                float(result["confidence"]),
                4
            ),

        "level1_models": {

            "naive_bayes":
                round(
                    float(result["prob_nb"]),
                    4
                ),

            "svm":
                round(
                    float(result["prob_svm"]),
                    4
                ),

            "random_forest":
                round(
                    float(result["prob_rf"]),
                    4
                )
        },

        "ensemble":
            "Stacking (NB + SVM + RF → Logistic Regression)"

    })


# =========================
# HEALTH CHECK
# =========================

@app.route("/health")
def health():

    return jsonify({

        "status": "ok"

    })


# =========================
# RUN
# =========================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )