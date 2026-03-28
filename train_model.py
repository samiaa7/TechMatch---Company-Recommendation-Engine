# ================================
# TRAINING + PREPROCESSING SCRIPT
# ================================

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# =====================================
# 1. LOAD RAW DATA
# =====================================
print("Loading dataset...")

df = pd.read_csv("employee_reviews.csv", encoding="latin1")
print("Original Shape:", df.shape)

# =====================================
# 2. DATA CLEANING
# =====================================

# --- STEP 1: clean column names (remove extra spaces + lowercase)
df.columns = [c.strip() for c in df.columns]

# These are the actual columns in your dataset
numeric_cols = [
    "overall-ratings",
    "work-balance-stars",
    "culture-values-stars",
    "career-opportunities-stars",
    "comp-benefit-stars",
    "senior-management-stars"
]

text_cols = ["company", "pros", "cons"]

# --- STEP 2: Remove duplicate rows
df.drop_duplicates(inplace=True)

# --- STEP 3: Replace dirty missing values
df.replace(["none", "None", "N/A", "na", ""], np.nan, inplace=True)

# --- STEP 4: Convert numeric columns
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# --- STEP 5: Fill numeric NaN with median
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

# --- STEP 6: Clean text columns
for col in text_cols:
    df[col] = df[col].astype(str).str.strip()

# --- STEP 7: Remove rows with missing company
df = df[df["company"].notna()]

print("After cleaning shape:", df.shape)

# =====================================
# 3. AGGREGATE TO COMPANY LEVEL
# =====================================

print("Aggregating company-level scores...")

company_df = df.groupby("company").agg({
    "work-balance-stars": "mean",
    "culture-values-stars": "mean",
    "comp-benefit-stars": "mean",
    "career-opportunities-stars": "mean",
    "senior-management-stars": "mean",
    "overall-ratings": "mean"
}).reset_index()

company_df.to_csv("cleaned_data.csv", index=False)
print("Saved cleaned_data.csv")

# =====================================
# 4. DEFINE ML INPUT + TARGET
# =====================================

X = company_df.drop(columns=["overall-ratings"])
y = company_df["overall-ratings"]

categorical_cols = ["company"]

numeric_cols_for_model = [
    "work-balance-stars",
    "culture-values-stars",
    "comp-benefit-stars",
    "career-opportunities-stars",
    "senior-management-stars"
]

# =====================================
# 5. PREPROCESSING PIPELINE
# =====================================

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", MinMaxScaler(), numeric_cols_for_model)
    ]
)

# =====================================
# 6. XGBOOST MODEL
# =====================================

model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.08,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", model)
])

# =====================================
# 7. TRAIN–TEST SPLIT + TRAIN
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")
pipeline.fit(X_train, y_train)

# =====================================
# 8. MODEL EVALUATION
# =====================================

score = pipeline.score(X_test, y_test)
print("Model R² Score:", round(score, 4))

# =====================================
# 9. SAVE MODEL + FEATURE ORDER
# =====================================

joblib.dump(pipeline, "model.pkl")
print("Saved model.pkl")

feature_order = {
    "categorical": categorical_cols,
    "numeric": numeric_cols_for_model
}

with open("feature_order.json", "w") as f:
    json.dump(feature_order, f, indent=4)

print("Saved feature_order.json")

print("\n============================")
print("TRAINING + CLEANING COMPLETE")
print("============================")
