# Home Credit Project 4 (Python) - Task 1 + Task 2

This project now implements:

- **Task 1: Data Acquisition & Preparation**
  - Data loading from local files or `kagglehub`
  - Data cleaning and table-level cleaning logs
  - Merging support tables to build one customer-level dataset
- **Task 2: EDA + Unsupervised Learning**
  - Descriptive statistics, correlations, and statistical tests
  - Visualizations for class balance and feature relationships
  - PCA + KMeans for structure discovery and Task 3 feature ideas

## 1) Data Source

You can run in either mode:

- **Local CSV mode** (`--data-dir data/raw`)
- **Auto-download mode** (`--auto-download`) using Kaggle slug (default: `megancrenshaw/home-credit-default-risk`)

If using local files, put at least:

- `application_train.csv` (required)

Optional but recommended:

- `bureau.csv`
- `previous_application.csv`

## 2) Install Dependencies

```bash
pip install pandas numpy scipy scikit-learn matplotlib kagglehub
```

## 3) Run Commands

From `project4/`:

- Run **Task 1 only**:

```bash
python home_credit_project4.py --task task1 --data-dir data/raw --output-dir outputs
```

- Run **Task 2 only** (uses Task 1 cleaned dataset by default):

```bash
python home_credit_project4.py --task task2 --task2-input outputs/processed/home_credit_task1_cleaned.csv --output-dir outputs
```

- Run **Task 1 + Task 2 together**:

```bash
python home_credit_project4.py --task all --data-dir data/raw --output-dir outputs
```

- Auto-download + run all:

```bash
python home_credit_project4.py --task all --auto-download --output-dir outputs
```

Optional:

- `--dataset-slug "megancrenshaw/home-credit-default-risk"`
- `--sample-size 20000` (Task 2 PCA/KMeans sample size)

## 4) What Task 1 Does

Task 1 produces the main cleaned customer-level dataset for downstream tasks:

- Standardizes column names
- Removes duplicate rows
- Normalizes placeholder missing values
- Converts numeric-like text columns
- Imputes missing values (`median` for numeric, `UNKNOWN` for categorical)
- Aggregates optional one-to-many tables by `SK_ID_CURR`
- Merges aggregated features into `application_train.csv`

### Task 1 outputs

Under `outputs/`:

- `processed/home_credit_task1_cleaned.csv`  
  Main dataset for Task 2+ (in current run: `307511 x 439`)
- `task1_raw_data_summary.json`  
  Source, table list, target distribution, final shape
- `task1_preparation_summary.json`  
  Detailed cleaning log (before/after missing rate, imputation counts, duplicate ratios, rationale)
- `task1_brief_summary.txt`  
  Human-readable summary for report/presentation

## 5) What Task 2 Does

Task 2 performs EDA and unsupervised analysis on the Task 1 cleaned dataset:

- Builds numeric descriptive statistics
- Computes feature-target correlations
- Runs statistical tests:
  - Mann-Whitney U (+ Cohen's d) for numeric variables
  - Chi-square (+ Cramer's V) for categorical variables
- Generates visualizations:
  - Target distribution
  - Top-feature correlation heatmap
  - Target-group distribution plots for key features
  - PCA 2D projection with KMeans clusters
- Exports unsupervised features for Task 3 ideas (`PCA_1`, `PCA_2`, `CLUSTER_ID`, `DIST_TO_CLUSTER_*`)

### Task 2 outputs

Under `outputs/task2/`:

- `task2_numeric_descriptive_stats.csv`
- `task2_feature_target_correlations.csv`
- `task2_statistical_tests.csv`
- `task2_categorical_associations.csv`
- `task2_cluster_profiles.csv`
- `task2_unsupervised_features_sample.csv`
- `task2_eda_unsupervised_summary.json`
- `task2_insights.txt`
- `task2_brief_summary.txt`
- `figures/*.png` (all Task 2 charts)

## 6) Which file to use next

For Task 3+ modeling pipeline, use:

- `outputs/processed/home_credit_task1_cleaned.csv`

Task 2 outputs are supporting analysis artifacts used to justify feature engineering and modeling decisions.
