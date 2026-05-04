# Task 4 README: Supervised Modeling

## Purpose

Task 4 builds and compares supervised learning models to predict Home Credit loan default risk. Using the modeling-ready dataset from Task 3, three classification models are trained, evaluated with 5-fold cross-validation, and compared across multiple metrics. The best model is selected based on AUC, PR-AUC, and cross-validation stability.

## How to Run

From the project directory:

```bash
python "Home credit project4 task4.py" --input "outputs\task3\processed\home_credit_task3_modeling_ready.csv"
```

Optional arguments:

- `--input`: path to the Task 3 modeling-ready CSV (required if not in default location)
- `--output-dir`: root directory for outputs (default: `outputs/task4`)
- `--test-size 0.2`: fraction of data held out for testing (default: 0.2)
- `--cv-folds 5`: number of cross-validation folds (default: 5)
- `--random-state 42`: random seed for reproducibility (default: 42)

## Input Dataset

Task 4 uses:

```text
outputs/task3/processed/home_credit_task3_modeling_ready.csv
```

Latest run:

```text
307,511 rows x 82 columns (80 features + TARGET + SK_ID_CURR)
Default rate: 8.07%
Train set: 246,008 rows
Test set:  61,503 rows
```

47 columns with residual NaN values were filled with column medians at load time.

## Models

Three models were trained and compared:

**1. Logistic Regression** (baseline)
- Interpretable, fast, linear decision boundary
- Uses `class_weight="balanced"` to handle class imbalance
- Good for understanding which features push predictions in each direction

**2. Random Forest**
- Ensemble of 200 decision trees, max depth 10
- Handles non-linear relationships
- Uses `class_weight="balanced"`

**3. XGBoost**
- Gradient boosting with 300 estimators
- `scale_pos_weight=11` to account for the 11:1 class imbalance ratio
- Generally the strongest performer on tabular data

Note: LightGBM was not installed in this environment and was skipped. Install with `pip install lightgbm` to include it.

## Results

### Cross-Validation (5-fold Stratified)

| Model | CV AUC | CV AUC Std | CV F1 | CV F1 Std |
|-------|--------|------------|-------|-----------|
| Logistic Regression | 0.7608 | ±0.0032 | 0.2693 | ±0.0019 |
| Random Forest | 0.7538 | ±0.0026 | 0.2769 | ±0.0030 |
| XGBoost | 0.7776 | ±0.0020 | 0.2916 | ±0.0022 |

### Test Set Results

| Model | AUC | F1 | Precision | Recall | PR-AUC |
|-------|-----|----|-----------|--------|--------|
| Logistic Regression | 0.7623 | 0.2696 | 0.1675 | 0.6900 | 0.2416 |
| Random Forest | 0.7575 | 0.2780 | 0.1784 | 0.6304 | 0.2441 |
| **XGBoost** | **0.7809** | **0.2942** | **0.1879** | **0.6779** | **0.2730** |

### Selected Model: XGBoost

Selection criterion: 60% test AUC + 20% CV AUC + 20% PR-AUC

| Model | Weighted Score |
|-------|---------------|
| XGBoost | 0.6787 |
| Logistic Regression | 0.6579 |
| Random Forest | 0.6541 |

## Why These Metrics

The dataset is heavily imbalanced — only 8.07% of applicants defaulted, giving an 11:1 class ratio. This means:

- **Accuracy is misleading**: a model that always predicts "no default" would achieve 92% accuracy while being completely useless for identifying defaulters.
- **AUC (ROC-AUC)** is the primary metric. It measures how well the model separates defaulters from non-defaulters regardless of threshold, and is robust to class imbalance.
- **PR-AUC** (Precision-Recall AUC) is used as a secondary metric. It is more sensitive to performance on the minority class (defaulters) than ROC-AUC.
- **F1** balances precision and recall. The low F1 values (~0.27–0.29) are expected given the imbalance and do not indicate a poorly performing model.
- **Recall** measures what fraction of actual defaulters the model catches. Logistic Regression has the highest recall (0.69), meaning it identifies 69% of all defaulters.
- **CV AUC std** measures stability. XGBoost has the lowest standard deviation (±0.0020), indicating consistent performance across folds.

## Output Files

Figures:

- `outputs/task4/figures/task4_roc_curves.png` — ROC curves for all models
- `outputs/task4/figures/task4_pr_curves.png` — Precision-Recall curves
- `outputs/task4/figures/task4_metrics_comparison.png` — all metrics side by side
- `outputs/task4/figures/task4_cv_auc_comparison.png` — cross-validation AUC with error bars
- `outputs/task4/figures/task4_confusion_matrices.png` — confusion matrices for all models
- `outputs/task4/figures/task4_feature_importance_*.png` — top 30 feature importances (tree models)

Results:

- `outputs/task4/task4_results.json` — all metrics in machine-readable format
- `outputs/task4/task4_summary.txt` — human-readable summary report

## Task 5/6 Handoff

For Task 5 (Model Evaluation & Selection) and reporting:

- The selected model is **XGBoost** with test AUC = 0.7809
- All metric comparisons are in `task4_results.json`
- All figures are ready to use in the report directly
- Key trade-off to discuss: Logistic Regression has slightly higher recall (0.69 vs 0.68 for XGBoost) and is more interpretable, but XGBoost wins on every other metric and is more robust
- For a credit default prediction context, recall matters — missing a defaulter is costly — so this trade-off is worth discussing in the report
