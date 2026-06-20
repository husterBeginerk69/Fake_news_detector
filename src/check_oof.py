# src/check_oof.py

import pandas as pd

df = pd.read_csv("data/oof_predictions.csv")

print(df.head())
print(df.shape)
print(df.isnull().sum())
print(df.describe())