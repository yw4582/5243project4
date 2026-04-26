# Task 3 README: Feature Engineering and Preprocessing

## Purpose

Task 3 transforms the cleaned Home Credit dataset from Task 1 into a modeling-ready feature set. It fixes known data anomalies, engineers domain-driven features, encodes categorical variables, handles outliers, scales numeric features, and applies a two-stage feature selection process to reduce dimensionality while preserving predictive power.

The output is a compact and efficient dataset ready for supervised learning in Task 4.

---

## How to Run

From the project directory:

```bash
python home_credit_project4_task3.py --task1-input home_credit_task1_cleaned.csv
```

Skip LightGBM feature selection if needed:

```bash
python home_credit_project4_task3.py --task1-input home_credit_task1_cleaned.csv --no-lgbm
```

Include Task 2 unsupervised features (optional):

```bash
python home_credit_project4_task3.py `
  --task1-input home_credit_task1_cleaned.csv `
  --task2-unsup outputs/task2/task2_unsupervised_features_sample.csv
```

---

## Input Dataset

* **Primary input:** `home_credit_task1_cleaned.csv`
* Shape: **307,511 rows × 439 columns**

Optional:

* `outputs/task2/task2_unsupervised_features_sample.csv` (PCA + clustering features)

---

## Anomaly Fix

The value `DAYS_EMPLOYED = 365243` is a known placeholder representing missing or special cases.

* Rows affected: **55,374**
* Fix:

  * Replaced with median
  * Added binary flag: `DAYS_EMPLOYED_ANOM`

This preserves the anomaly as an informative signal.

---

## Feature Engineering

A total of **22 domain-driven features** were created:

### Credit Burden Ratios

* CREDIT_INCOME_RATIO
* ANNUITY_INCOME_RATIO
* CREDIT_ANNUITY_RATIO
* GOODS_CREDIT_RATIO
* DOWN_PAYMENT_PROXY

### Age & Employment

* AGE_YEARS
* EMPLOYED_YEARS
* DAYS_EMPLOYED_PERC
* AGE_CREDIT_INTERACTION

### External Credit Scores (EXT_SOURCE)

* EXT_SOURCE_MEAN / MIN / MAX / STD
* EXT_SOURCE_PROD
* EXT_SOURCE_WEIGHTED

### Behavioral & Risk Signals

* DOCUMENT_COUNT
* DAYS_LAST_PHONE_CHANGE_ABS
* SOCIAL_CIRCLE_DEFAULT_RATE
* CREDIT_ENQUIRIES_FLAG

### Bureau & Previous Applications

* BUREAU_CREDIT_RATIO
* BUREAU_OVERDUE_FLAG
* PREV_CREDIT_RATIO

---

## Preprocessing Steps

### Categorical Encoding

* Binary encoded: 3 columns
* One-hot encoded: 11 columns
* Dropped high-cardinality columns: 2

---

### Outlier Handling

* Method: Winsorization (1st–99th percentile)
* Applied to: 311 numeric columns

---

### Scaling

* Method: StandardScaler
* Applied to: 385 numeric features
* Saved as:
  `outputs/task3/processed/home_credit_task3_scaler.pkl`

---

## Feature Selection

### Stage A: Correlation Deduplication

* Threshold: |r| ≥ 0.95
* Dropped: 77 features
* Remaining: 422 features

---

### Stage B: LightGBM Feature Importance 

* Method: Gain-based importance ranking
* Selected: **Top 80 features**
* Final feature set is optimized for:

  * Predictive performance
  * Reduced redundancy
  * Better interpretability

---

## Output Files

### Core Outputs

* `home_credit_task3_modeling_ready.csv` → final dataset (**use for Task 4**)
* `home_credit_task3_scaler.pkl` → fitted scaler
* `home_credit_task3_feature_metadata.csv` → feature tracking & importance
* `task3_feature_engineering_log.json` → full pipeline log
* `task3_summary.txt` → human-readable summary

### Figures

* Feature importance plot
* Distribution plots (key features)
* Correlation heatmap
* Cluster risk visualization

---

## Latest Run Summary

```
Input rows:              307,511
Input columns:           439
DAYS_EMPLOYED anomalies: 55,374
New features engineered: 22
After correlation:       422 features
Final selected features: 80
Final dataset shape:     307,511 × 82
```

---

## Key Design Decisions

* Ratio features capture **relative financial burden**, which generalizes better than raw values.
* EXT_SOURCE features are the **strongest predictors**, enhanced via aggregation and interaction.
* Anomalies are preserved via flags instead of removed.
* Winsorization prevents extreme values from distorting scaling.
* Two-stage feature selection:

  * Removes redundancy (correlation)
  * Keeps only the most predictive features (LightGBM)

---

## Task 4 Handoff

Use the following dataset:

```
outputs/task3/processed/home_credit_task3_modeling_ready.csv
```

Example:

```python
import pandas as pd

df = pd.read_csv("outputs/task3/processed/home_credit_task3_modeling_ready.csv")
X = df.drop(columns=["TARGET", "SK_ID_CURR"])
y = df["TARGET"]
```

### Notes

* Target is highly imbalanced (~8% default rate)
* Use metrics such as:

  * ROC-AUC
  * Precision / Recall
  * F1-score
* Consider class weighting or resampling techniques

---

## Summary

Task 3 reduces the dataset from over **400 features to 80 high-quality predictors**, producing a compact, efficient, and interpretable dataset ready for machine learning modeling.
