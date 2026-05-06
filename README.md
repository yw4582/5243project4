# Home Credit Default Risk – Project 4

End-to-end machine learning pipeline for predicting loan default risk using the [Home Credit dataset](https://www.kaggle.com/datasets/megancrenshaw/home-credit-default-risk).

**Live Dashboard:** https://5243project4-hgcccpnmfvz6jr5cpmefqd.streamlit.app/

---

## Project Structure

```
project4/
├── home_credit_pipeline.py        # Task 1 + Task 2 + Task 3 (data → modeling-ready dataset)
├── home_credit_modeling.py        # Task 4 (supervised modeling)
├── bonus_dashboard.py             # Streamlit interactive dashboard
├── data/
│   └── raw/                       # Place raw CSVs here
│       ├── application_train.csv  (required)
│       ├── bureau.csv             (optional)
│       └── previous_application.csv (optional)
└── outputs/
    ├── processed/                 # Task 1 cleaned dataset
    ├── task2/                     # EDA outputs + unsupervised features
    ├── task3/                     # Feature-engineered modeling dataset
    └── task4/                     # Model results + figures
```

---

## Data

| Source | Link |
|--------|------|
| Raw dataset (Kaggle) | https://www.kaggle.com/datasets/megancrenshaw/home-credit-default-risk |
| Cleaned dataset (Google Drive) | https://drive.google.com/file/d/1z6bRNZbhuSRJY1gz7cJLm_HsHIo6VUN9/view?usp=sharing |

If you already have the cleaned dataset, you can skip Task 1 and start from Task 3 directly.

---

## Install Dependencies

```bash
pip install pandas numpy scipy scikit-learn matplotlib lightgbm xgboost streamlit kagglehub statsmodels
```

---

## Pipeline Overview

### `home_credit_pipeline.py` — Tasks 1, 2, 3

This script runs the full preprocessing pipeline in one file.

**Task 1 – Data Cleaning & Merging**
- Loads `application_train.csv`, `bureau.csv`, `previous_application.csv`
- Standardizes column names, removes duplicates, imputes missing values
- Aggregates support tables to customer level and merges into one dataset
- Output: `outputs/processed/home_credit_task1_cleaned.csv` (307,511 × 439)

**Task 2 – EDA & Unsupervised Learning**
- Computes descriptive statistics and feature-target correlations (Pearson + Spearman)
- Runs Mann-Whitney U tests and chi-square tests for feature significance
- Applies PCA (2 components) + KMeans clustering (k=3) on a 20,000-row sample
- Key finding: default rate is 8.07%, imbalance ratio 11:1 — AUC not accuracy should be used
- Output: `outputs/task2/` including figures and `task2_unsupervised_features_sample.csv`

**Task 3 – Feature Engineering & Preprocessing**
- Fixes known data anomaly: `DAYS_EMPLOYED = 365243` (sentinel for "not employed")
- Engineers 22 domain-driven features (credit ratios, age/employment, EXT_SOURCE aggregations, behavioral signals)
- Encodes categoricals, winsorizes outliers (1st–99th pct), scales with StandardScaler
- Two-stage feature selection: correlation dedup (|r| ≥ 0.95) → LightGBM importance Top-80
- Output: `outputs/task3/processed/home_credit_task3_modeling_ready.csv` (307,511 × 82)

**Run commands:**

```bash
# Full pipeline from raw data
python home_credit_pipeline.py --data-dir data/raw

# Start from existing cleaned CSV (skip Task 1 & 2)
python home_credit_pipeline.py --task task3 --task1-input "outputs/processed/home_credit_task1_cleaned.csv"

# Skip LightGBM if not installed
python home_credit_pipeline.py --task task3 --task1-input "outputs/processed/home_credit_task1_cleaned.csv" --no-lgbm

# Auto-download raw data from Kaggle
python home_credit_pipeline.py --auto-download --task all
```

---

### `home_credit_modeling.py` — Task 4

Trains and compares three classification models on the Task 3 output.

**Models:** Logistic Regression, Random Forest, XGBoost (+ LightGBM if installed)

**Evaluation:** 5-fold stratified cross-validation + held-out test set (80/20 split)

**Metrics:** AUC, F1, Precision, Recall, PR-AUC

**Results:**

| Model | AUC | F1 | Recall | PR-AUC |
|-------|-----|----|--------|--------|
| Logistic Regression | 0.7623 | 0.2696 | 0.6900 | 0.2416 |
| Random Forest | 0.7575 | 0.2780 | 0.6304 | 0.2441 |
| **XGBoost** | **0.7809** | **0.2942** | **0.6779** | **0.2730** |

**Selected model: XGBoost** — highest AUC, F1, and PR-AUC with lowest CV variance (±0.0020).

Output: `outputs/task4/` including ROC curves, PR curves, confusion matrices, feature importance plots.

**Run command:**

```bash
python home_credit_modeling.py --input "outputs/task3/processed/home_credit_task3_modeling_ready.csv"
```

---

### `bonus_dashboard.py` — Interactive Dashboard

Streamlit app for interactive exploration of the pipeline results.

**Live:** https://5243project4-hgcccpnmfvz6jr5cpmefqd.streamlit.app/

**Run locally:**

```bash
streamlit run bonus_dashboard.py
```

---

## Key Design Decisions

- **Median imputation** for numeric missing values — robust to skewed credit data distributions
- **UNKNOWN encoding** for categorical missing values — preserves missingness as a signal
- **Ratio features** instead of raw amounts — relative burden generalizes across income levels
- **EXT_SOURCE aggregations** (mean, min, max, std, product) — captures both level and bureau disagreement
- **AUC as primary metric** — accuracy is misleading at 11:1 class imbalance (92% by always predicting "no default")
- **PCA/cluster features excluded from Task 4** — generated on a 20k sample only; merging would introduce missing values for 287k rows

---

## Output Reference

| File | Description |
|------|-------------|
| `outputs/processed/home_credit_task1_cleaned.csv` | Cleaned merged dataset |
| `outputs/task3/processed/home_credit_task3_modeling_ready.csv` | **Use for modeling (Task 4)** |
| `outputs/task3/processed/home_credit_task3_scaler.pkl` | Fitted StandardScaler — reuse on test set |
| `outputs/task4/task4_results.json` | All model metrics |
| `outputs/task4/figures/` | All plots |
