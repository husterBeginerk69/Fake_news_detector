# Fake News Detection using Machine Learning and Ensemble Learning

## Overview

This project implements a Fake News Detection system using traditional Machine Learning methods and Ensemble Learning. The system classifies news articles as **Real** or **Fake** based on textual content.

The project evaluates four classification models:

* Naive Bayes (NB)
* Support Vector Machine (SVM)
* Random Forest (RF)
* Stacking Ensemble (Logistic Regression Meta-Learner)

In addition, the project includes:

* Ablation Study
* McNemar Statistical Test
* Explainable AI (XAI)
* Flask-based Web Application

---

## Project Structure

```text
project/
│
├── app.py
├── predict.py
├── requirements.txt
│
├── models/
│   ├── *.pkl
│
├── results/
│   ├── model_comparison.csv
│   ├── results_ablation_delta.csv
│   ├── results_mcnemar_stacking_vs_svm.csv
│   ├── results_xai_nb_top_features.png
│   ├── results_xai_svm_top_features.png
│   └── results_xai_meta_weights.png
│
├── static/
├── templates/
│
└── README.md
```

---

## Dataset

The project uses the ISOT Fake News Dataset consisting of:

* Real News
* Fake News

The dataset is preprocessed before feature extraction and model training.

---

## Methodology

### Text Preprocessing

* Lowercasing
* Remove HTML tags
* Remove URLs
* Remove punctuation
* Remove special characters
* Normalize whitespace

### Feature Extraction

* TF-IDF Vectorization
* Truncated SVD (for Random Forest)

### Models

1. Naive Bayes
2. Support Vector Machine (Linear SVM)
3. Random Forest
4. Stacking Ensemble

The Stacking model combines predictions from the three base learners using Logistic Regression as the meta-learner.

---

## Experimental Results

### Text Clean Variant

| Model         | Accuracy | F1     |
| ------------- | -------- | ------ |
| SVM           | 0.9717   | 0.9683 |
| Stacking      | 0.9714   | 0.9679 |
| Random Forest | 0.9208   | 0.9105 |
| Naive Bayes   | 0.9056   | 0.8951 |

### Text No Dateline Variant

| Model         | Accuracy | F1     |
| ------------- | -------- | ------ |
| Stacking      | 0.9688   | 0.9650 |
| SVM           | 0.9685   | 0.9647 |
| Random Forest | 0.9162   | 0.9048 |
| Naive Bayes   | 0.9033   | 0.8925 |

---

## Explainable AI (XAI)

The project provides model interpretation through:

* Top influential words for Naive Bayes
* Top influential words for SVM
* Meta-model coefficient analysis for Stacking

The analysis helps explain why SVM and Stacking often produce highly similar predictions.

---

## Installation

```bash
git clone <repository-url>
cd project
pip install -r requirements.txt
```

---

## Running the Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

in your browser.

---

## Key Findings

* SVM achieved the best overall performance.
* Stacking performed nearly identically to SVM.
* McNemar testing showed no statistically significant difference between SVM and Stacking.
* Removing noisy textual components slightly improved performance.
* XAI analysis revealed that the meta-learner heavily relied on SVM predictions, explaining the strong agreement between the two models.

---

## Author

Course Project – Fake News Detection using Machine Learning and Ensemble Learning.
