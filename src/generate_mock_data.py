import numpy as np
import pandas as pd

np.random.seed(42)

# ====================
# OOF TRAIN
# ====================

n_train = 5000

labels = np.random.choice(
    [0, 1],
    size=n_train,
    p=[0.5561, 0.4439]
)

nb = []
svm = []
rf = []

for y in labels:

    if y == 1:

        nb.append(np.random.normal(0.75, 0.15))
        svm.append(np.random.normal(0.82, 0.10))
        rf.append(np.random.normal(0.73, 0.12))

    else:

        nb.append(np.random.normal(0.25, 0.15))
        svm.append(np.random.normal(0.18, 0.10))
        rf.append(np.random.normal(0.28, 0.12))

train_df = pd.DataFrame({

    "row_id": range(n_train),

    "label": labels,

    "prob_nb": np.clip(nb, 0, 1),

    "prob_svm": np.clip(svm, 0, 1),

    "prob_rf": np.clip(rf, 0, 1)

})

train_df.to_csv(
    "data/oof_predictions.csv",
    index=False
)

# ====================
# TEST
# ====================

n_test = 1500

labels = np.random.choice(
    [0, 1],
    size=n_test,
    p=[0.5561, 0.4439]
)

nb = []
svm = []
rf = []

for y in labels:

    if y == 1:

        nb.append(np.random.normal(0.73, 0.15))
        svm.append(np.random.normal(0.84, 0.10))
        rf.append(np.random.normal(0.74, 0.12))

    else:

        nb.append(np.random.normal(0.22, 0.15))
        svm.append(np.random.normal(0.16, 0.10))
        rf.append(np.random.normal(0.25, 0.12))

test_df = pd.DataFrame({

    "row_id": range(n_test),

    "label": labels,

    "prob_nb": np.clip(nb, 0, 1),

    "prob_svm": np.clip(svm, 0, 1),

    "prob_rf": np.clip(rf, 0, 1)

})

test_df.to_csv(
    "data/test_predictions.csv",
    index=False
)

print("Mock data created.")