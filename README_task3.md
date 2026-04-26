# Task 3 README: Feature Engineering and Preprocessing

## Purpose

Task 3 transforms the cleaned Home Credit dataset from Task 1 into a modeling-ready feature set. It fixes known data anomalies, engineers 22 new credit-risk features based on domain knowledge, encodes categorical variables, clips outliers, scales numeric features, and removes redundant features through correlation-based deduplication. The output is a single CSV file ready for supervised learning in Task 4.

## How to Run

From the project directory:

```bash
python home_credit_project4_task3.py --task1-input home_credit_task1_cleaned.csv
```

If LightGBM is not installed, add `--no-lgbm` to skip importance-based feature selection:

```bash
python home_credit_project4_task3.py --task1-input home_credit_task1_cleaned.csv --no-lgbm
```

To also use Task 2 unsupervised features (PCA, KMeans):

```bash
python home_credit_project4_task3.py \
  --task1-input home_credit_task1_cleaned.csv \
  --task2-unsup outputs/task2/task2_unsupervised_features_sample.csv
```

Optional arguments:

- `--top-k-features 80`: keep top K features by LightGBM importance (default: 80, requires LightGBM)
- `--corr-threshold 0.95`: drop one of any feature pair with absolute Pearson correlation above this value (default: 0.95)
- `--output-dir outputs`: root directory for all outputs (default: `outputs`)

## Input Dataset

Task 3 uses:

```text
home_credit_task1_cleaned.csv
```

The latest run processed:

```text
307,511 rows x 439 columns
```

Optionally merges:

```text
outputs/task2/task2_unsupervised_features_sample.csv
```

This file adds PCA_1, PCA_2, CLUSTER_ID, and DIST_TO_CLUSTER_* features from Task 2. If it is not found, the merge step is skipped automatically.

## Anomaly Fix

`DAYS_EMPLOYED = 365243` is a documented encoding error in the raw dataset meaning "not employed / retired" rather than actual work tenure. Without fixing this, models would treat these applicants as having approximately 1,000 years of employment.

Latest run:

- Rows affected: `55,374`
- Fix applied: replaced with column median
- New column added: `DAYS_EMPLOYED_ANOM = 1` for affected rows, preserving the signal

## Feature Engineering

Task 3 engineers the following new features when their source columns are available. All features are guarded against zero denominators, infinite values, and missing values.

**Credit Burden Ratios**

- `CREDIT_INCOME_RATIO = AMT_CREDIT / AMT_INCOME_TOTAL` — loan amount relative to annual income; higher values indicate heavier debt load
- `ANNUITY_INCOME_RATIO = AMT_ANNUITY / AMT_INCOME_TOTAL` — monthly repayment as a fraction of income (debt-service ratio)
- `CREDIT_ANNUITY_RATIO = AMT_CREDIT / AMT_ANNUITY` — estimated repayment term in months
- `GOODS_CREDIT_RATIO = AMT_GOODS_PRICE / AMT_CREDIT` — goods price to loan amount, a proxy for loan-to-value ratio
- `DOWN_PAYMENT_PROXY = AMT_CREDIT - AMT_GOODS_PRICE` — difference between credit granted and goods price

**Age and Employment Features**

- `AGE_YEARS = -DAYS_BIRTH / 365.25` — applicant age in years
- `EMPLOYED_YEARS = -DAYS_EMPLOYED / 365.25` — employment tenure in years
- `DAYS_EMPLOYED_PERC = DAYS_EMPLOYED / DAYS_BIRTH` — employment tenure as a fraction of total age
- `AGE_CREDIT_INTERACTION = AGE_YEARS × CREDIT_INCOME_RATIO` — interaction between age and debt burden

**External Credit Score Aggregations**

EXT_SOURCE_1, EXT_SOURCE_2, and EXT_SOURCE_3 are scores from external credit bureaus and are the strongest predictors in this dataset. Multiple aggregations capture different aspects of the signal:

- `EXT_SOURCE_MEAN` — average of available scores
- `EXT_SOURCE_MIN` — lowest score (worst-case bureau assessment)
- `EXT_SOURCE_MAX` — highest score
- `EXT_SOURCE_STD` — standard deviation; high disagreement between bureaus may indicate risk
- `EXT_SOURCE_PROD` — product of all three scores; amplifies the risk signal when all scores are jointly low
- `EXT_SOURCE_WEIGHTED` — weighted combination with EXT_SOURCE_2 at 50% weight, EXT_SOURCE_1 and EXT_SOURCE_3 at 25% each

**Other Risk Signals**

- `DOCUMENT_COUNT` — total number of documents provided by the applicant
- `DAYS_LAST_PHONE_CHANGE_ABS` — absolute days since last phone number change
- `SOCIAL_CIRCLE_DEFAULT_RATE = DEF_30_CNT_SOCIAL_CIRCLE / OBS_30_CNT_SOCIAL_CIRCLE` — default rate within the applicant's social circle
- `CREDIT_ENQUIRIES_FLAG` — binary flag for more than 3 credit bureau enquiries in the past year

**Bureau and Previous Application Features** (available when Task 1 merged supporting tables)

- `BUREAU_CREDIT_RATIO` — historical bureau credit relative to current application credit
- `BUREAU_OVERDUE_FLAG` — binary flag for any overdue bureau credit amount
- `PREV_CREDIT_RATIO` — previous application credit amounts relative to current credit

Latest run: **22 new features engineered**

## Preprocessing Steps

**Categorical Encoding**

- Binary columns (2 unique values): label-encoded to 0/1
- Low-cardinality columns (3–10 unique values): one-hot encoded with `drop_first=True`
- High-cardinality columns (>10 unique values): dropped to avoid excessive sparse dimensions

Latest run: binary encoded `3`, one-hot encoded `11`, dropped `2`

**Outlier Clipping**

All numeric columns winsorized to the 1st–99th percentile range to prevent extreme values from distorting the scaler and model training.

Latest run: `311` columns clipped

**Scaling**

StandardScaler applied to all non-binary numeric features, producing zero mean and unit variance. Binary columns (0/1 only) are excluded from scaling. The fitted scaler is saved as `scaler.pkl` and must be reused on the test set in Task 4 without refitting to avoid data leakage.

Latest run: `385` columns scaled

**Feature Selection**

Stage A — Correlation deduplication: for any pair of features with absolute Pearson correlation ≥ 0.95, the one with lower variance is dropped.

Stage B — LightGBM importance ranking (requires LightGBM): features ranked by gain importance, top K kept. Skip with `--no-lgbm` if LightGBM is not installed.

Latest run: dropped `77` correlated features, final dataset has `422` features (Stage B skipped)

## Output Files

Main files:

- `outputs/task3/processed/home_credit_task3_modeling_ready.csv`: final modeling-ready dataset — **use this for Task 4**
- `outputs/task3/processed/home_credit_task3_scaler.pkl`: fitted StandardScaler — reuse on test set in Task 4
- `outputs/task3/processed/home_credit_task3_feature_metadata.csv`: feature provenance, LightGBM importance, variance, and missing rate
- `outputs/task3/task3_summary.txt`: human-readable pipeline report
- `outputs/task3/task3_feature_engineering_log.json`: machine-readable processing log

Figures:

- `outputs/task3/figures/task3_top30_feature_importance.png`
- `outputs/task3/figures/task3_dist_CREDIT_INCOME_RATIO.png`
- `outputs/task3/figures/task3_dist_ANNUITY_INCOME_RATIO.png`
- `outputs/task3/figures/task3_dist_EXT_SOURCE_MEAN.png`
- `outputs/task3/figures/task3_dist_AGE_YEARS.png`
- `outputs/task3/figures/task3_dist_EMPLOYED_YEARS.png`
- `outputs/task3/figures/task3_cluster_default_rate.png`
- `outputs/task3/figures/task3_top15_correlation_heatmap.png`

## Latest Run Summary

```
Input rows:              307,511
Input columns:           439
DAYS_EMPLOYED anomalies: 55,374
New features engineered: 22
Dropped (correlation):   77
Final feature count:     422
Output shape:            307,511 rows x 424 columns
```

## Task 4 Handoff

For Task 4 supervised modeling:

- Load `outputs/task3/processed/home_credit_task3_modeling_ready.csv` as the input dataset.
- Split TARGET and SK_ID_CURR from the feature matrix before training.
- Apply the saved `scaler.pkl` to the test split — do not refit the scaler on test data.
- The target is imbalanced (default rate ~8%, imbalance ratio ~11:1). Evaluate models with AUC, PR-AUC, F1, recall, and precision rather than accuracy alone.
- Consider class weighting or resampling (e.g. SMOTE) to handle the imbalance.

```python
import pandas as pd

df = pd.read_csv("outputs/task3/processed/home_credit_task3_modeling_ready.csv")
X = df.drop(columns=["TARGET", "SK_ID_CURR"])
y = df["TARGET"]
```
