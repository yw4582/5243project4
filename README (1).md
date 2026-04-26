# Task 2 README: Exploratory Data Analysis and Unsupervised Learning

## Purpose

Task 2 analyzes the cleaned Home Credit customer-level dataset produced by Task 1. It performs exploratory data analysis, adds interpretable credit-risk ratio features, checks target imbalance, identifies important feature-target relationships, and creates unsupervised learning features that can support Task 3 modeling.

## How to Run

From the project directory:

```bash
python home_credit_project4.py --task task2
```

If the cleaned Task 1 file is in the current project folder instead of `outputs/processed/`, run:

```bash
python home_credit_project4.py --task task2 --task2-input home_credit_task1_cleaned.csv
```

On systems where `python` is not available, use `python3` with the same arguments.

## Input Dataset

Task 2 uses:

```text
home_credit_task1_cleaned.csv
```

The latest run analyzed:

```text
307,511 rows x 444 columns
```

## Feature Engineering

Task 2 creates the following interpretable credit-risk features when their source columns are available:

- `CREDIT_INCOME_RATIO = AMT_CREDIT / AMT_INCOME_TOTAL`
- `ANNUITY_INCOME_RATIO = AMT_ANNUITY / AMT_INCOME_TOTAL`
- `CREDIT_ANNUITY_RATIO = AMT_CREDIT / AMT_ANNUITY`
- `GOODS_CREDIT_RATIO = AMT_GOODS_PRICE / AMT_CREDIT`
- `DAYS_EMPLOYED_PERC = DAYS_EMPLOYED / DAYS_BIRTH`

The script handles missing source columns, zero denominators, infinite values, and missing values safely.

## Key Findings

The target is imbalanced:

- Default rate: `0.0807`
- Majority-to-minority imbalance ratio: `11.39`
- `TARGET=0`: `282,686`
- `TARGET=1`: `24,825`

Because of this imbalance, Task 3 should evaluate models with AUC, recall, precision, F1, or PR-AUC rather than accuracy alone.

Top Pearson correlations with `TARGET` from the latest run:

- `EXT_SOURCE_2`: `-0.1603`
- `EXT_SOURCE_3`: `-0.1559`
- `EXT_SOURCE_1`: `-0.0989`
- `BUREAU_DAYS_CREDIT_MEAN`: `0.0897`
- `BUREAU_CREDIT_ACTIVE_CLOSED`: `-0.0794`

## Unsupervised Learning

Task 2 uses standardized numeric features, PCA, and KMeans clustering.

Latest run:

- PCA components used: `2`
- PC1 + PC2 explained variance: `13.80%`
- Final KMeans clusters: `3`
- Sample size used for unsupervised learning: `20,000`
- Cluster counts: `0=8,698`, `1=1,678`, `2=9,624`

The file `task2_kmeans_k_selection.csv` reports inertia and silhouette scores for `k=2` through `k=8`. The final model keeps `k=3` for interpretability.

## Output Files

Main tables:

- `task2_numeric_descriptive_stats.csv`: numeric summary statistics, missing counts, and missing rates
- `task2_outlier_summary.csv`: IQR-based outlier diagnostics for numeric features
- `task2_feature_target_correlations.csv`: Pearson correlations with `TARGET`
- `task2_feature_target_spearman_correlations.csv`: Spearman correlations with `TARGET`
- `task2_statistical_tests.csv`: Mann-Whitney U tests with Benjamini-Hochberg adjusted p-values
- `task2_categorical_associations.csv`: chi-square tests and Cramer's V for categorical features
- `task2_kmeans_k_selection.csv`: KMeans inertia and silhouette diagnostics for `k=2` to `k=8`
- `task2_cluster_profiles.csv`: cluster-level averages, cluster sizes, cluster percentages, and default rates when available
- `task2_unsupervised_features_sample.csv`: Task 3-ready PCA, cluster, and distance features
- `task2_eda_unsupervised_summary.json`: machine-readable summary of Task 2 outputs
- `task2_insights.txt`: report-ready insights
- `task2_brief_summary.txt`: concise Task 2 output summary

Figures:

- `figures/task2_target_distribution.png`
- `figures/task2_top_feature_correlation_heatmap.png`
- `figures/task2_top15_target_correlations.png`
- `figures/task2_pca_kmeans_clusters.png`
- `figures/task2_dist_by_target_EXT_SOURCE_1.png`
- `figures/task2_dist_by_target_EXT_SOURCE_2.png`
- `figures/task2_dist_by_target_EXT_SOURCE_3.png`
- `figures/task2_dist_by_target_BUREAU_DAYS_CREDIT_MEAN.png`
- `figures/task2_dist_by_target_BUREAU_CREDIT_ACTIVE_CLOSED.png`

## Task 3 Handoff

For Task 3 modeling:

- Use the cleaned Task 1 dataset as the main modeling table.
- Merge `task2_unsupervised_features_sample.csv` back by `SK_ID_CURR` when using the sampled unsupervised features.
- Consider using `PCA_1`, `PCA_2`, `CLUSTER_ID`, and `DIST_TO_CLUSTER_0` through `DIST_TO_CLUSTER_2` as additional model features.
- Prioritize features with strong Pearson/Spearman correlations, statistically significant Mann-Whitney results, and interpretable credit-risk meaning.
- Evaluate classification models with imbalance-aware metrics: AUC, recall, precision, F1, and PR-AUC.
