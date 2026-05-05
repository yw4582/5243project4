# Part 5 Model Comparison and Selection README

## What Part 5 does

Part 5 compares the supervised models from Part 4, recalculates the weighted model-selection score, ranks the models, and explains why XGBoost is selected as the final model for loan default prediction.

## Input files used

- `task4_summary.txt`
- `task4_roc_curves.png`
- `task4_pr_curves.png`
- `task4_metrics_comparison.png`
- `task4_cv_auc_comparison.png`
- `task4_confusion_matrices.png`
- `task4_feature_importance_xgboost.png`
- `task4_feature_importance_random_forest.png`

The script first looks for `task4_results.json`. If that file is missing, it prints a clear warning and falls back to `task4_summary.txt`.

## Output files created

- `outputs/part5/part5_model_comparison.csv`
- `outputs/part5/part5_model_comparison.md`
- `outputs/part5/part5_summary.md`
- `outputs/part5/part5_README.md`
- `outputs/part5/part5_final_report_paragraph.txt`
- `outputs/part5/figures/part5_roc_curves.png`
- `outputs/part5/figures/part5_precision_recall_curves.png`
- `outputs/part5/figures/part5_metric_comparison_barplot.png`
- `outputs/part5/figures/part5_cv_auc_comparison.png`
- `outputs/part5/figures/part5_confusion_matrices.png`
- `outputs/part5/figures/part5_feature_importance_xgboost.png`
- `outputs/part5/figures/part5_feature_importance_random_forest.png`
- `outputs/part5/figures/part5_weighted_selection_score.png`
- `outputs/part5/figures/part5_model_rank_summary.png`

## How to run the code

From the project root, run:

```bash
python part5_model_selection_comparison.py
```

The script creates missing folders automatically and saves all Part 5 outputs under `outputs/part5/`.

## Required Python packages

- pandas
- numpy
- matplotlib
- seaborn

The script also uses standard Python libraries: `json`, `pathlib`, `re`, and `shutil`.

## Model-selection logic

The weighted selection score follows the Part 4 criterion:

```text
Weighted score = 0.60 * Test AUC + 0.20 * CV AUC + 0.20 * Test PR-AUC
```

This criterion emphasizes test-set ranking performance while still accounting for cross-validation stability and minority-class performance.

## Final selected model

XGBoost is selected as the final model because it has the highest weighted selection score, highest test AUC, highest PR-AUC, highest F1-score, and strongest cross-validation AUC. Precision remains relatively low, which is common in imbalanced credit-risk settings, so the model should be used as a risk-ranking and screening tool rather than a perfect individual-level predictor.
