#!/usr/bin/env python3
"""
Project 4 - Task 1: Data Acquisition & Preparation (Home Credit).

This script does ONLY the following:
1) Identify/obtain dataset tables from a credible source folder
2) Clean data (missing values, formatting, inconsistencies)
3) Produce a brief summary of raw data and preparation steps
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Home Credit Project 4 pipeline (Task 1 and Task 2).")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing Home Credit CSV files (optional if using kagglehub).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where outputs (summaries + cleaned data) are saved.",
    )
    parser.add_argument(
        "--dataset-slug",
        type=str,
        default="megancrenshaw/home-credit-default-risk",
        help="Kaggle dataset slug used by kagglehub when auto-downloading.",
    )
    parser.add_argument(
        "--auto-download",
        action="store_true",
        help="Auto download dataset via kagglehub when local files are missing.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["task1", "task2", "all"],
        default="all",
        help="Which task(s) to run: task1, task2, or all.",
    )
    parser.add_argument(
        "--task2-input",
        type=Path,
        default=Path("outputs/processed/home_credit_task1_cleaned.csv"),
        help="Input cleaned dataset path for Task 2.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20000,
        help="Max rows sampled for PCA/KMeans to keep Task 2 runtime manageable.",
    )
    return parser.parse_args()


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean.columns = [
        c.strip().upper().replace(" ", "_").replace("/", "_").replace("-", "_")
        for c in clean.columns
    ]
    return clean


def resolve_csv_path(data_dir: Path, name: str) -> Optional[Path]:
    direct = data_dir / name
    if direct.exists():
        return direct
    matches = list(data_dir.rglob(name))
    if matches:
        return matches[0]
    return None


def load_csv(data_dir: Path, name: str) -> pd.DataFrame:
    path = resolve_csv_path(data_dir, name)
    if path is None:
        raise FileNotFoundError(f"Missing required file: {name} under {data_dir}")
    return sanitize_columns(pd.read_csv(path))


def maybe_download_with_kagglehub(dataset_slug: str) -> Path:
    try:
        import kagglehub  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "kagglehub is not installed. Run: pip install kagglehub"
        ) from exc

    print(f"[INFO] Downloading dataset via kagglehub: {dataset_slug}")
    downloaded_path = Path(kagglehub.dataset_download(dataset_slug))
    print(f"[INFO] kagglehub dataset path: {downloaded_path}")
    return downloaded_path


def coerce_numeric_object_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    """Convert object columns to numeric when most non-null values are numeric-like."""
    out = df.copy()
    converted_cols: List[str] = []
    for col in out.select_dtypes(include=["object"]).columns:
        series = out[col].astype(str).str.strip()
        # Treat blank and common placeholders as missing.
        series = series.replace({"": np.nan, "NA": np.nan, "N/A": np.nan, "None": np.nan, "null": np.nan})
        numeric_try = pd.to_numeric(series, errors="coerce")
        non_null = series.notna().sum()
        if non_null == 0:
            out[col] = series
            continue
        numeric_ratio = numeric_try.notna().sum() / non_null
        if numeric_ratio >= 0.90:
            out[col] = numeric_try
            converted_cols.append(col)
        else:
            out[col] = series
    return out, converted_cols


def clean_table(df: pd.DataFrame, table_name: str) -> tuple[pd.DataFrame, Dict[str, object]]:
    """Apply Task-1 cleaning rules and return cleaned table + cleaning log."""
    cleaned = sanitize_columns(df)
    raw_rows, raw_cols = cleaned.shape
    raw_missing_total = int(cleaned.isna().sum().sum())
    raw_cells = int(raw_rows * raw_cols) if raw_rows and raw_cols else 0
    raw_missing_ratio = (raw_missing_total / raw_cells) if raw_cells else 0.0
    duplicate_rows = int(cleaned.duplicated().sum())
    if duplicate_rows > 0:
        cleaned = cleaned.drop_duplicates()

    # Standardize string placeholders and trim text.
    obj_cols = cleaned.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        cleaned[col] = cleaned[col].astype(str).str.strip()
        cleaned[col] = cleaned[col].replace(
            {"": np.nan, "NA": np.nan, "N/A": np.nan, "None": np.nan, "null": np.nan}
        )

    cleaned, converted_cols = coerce_numeric_object_columns(cleaned)

    # Impute missing values with simple, reproducible rules.
    num_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = cleaned.select_dtypes(exclude=[np.number]).columns.tolist()
    num_imputed_cells = 0
    cat_imputed_cells = 0
    for col in num_cols:
        if cleaned[col].isna().any():
            num_imputed_cells += int(cleaned[col].isna().sum())
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())
    for col in cat_cols:
        if cleaned[col].isna().any():
            cat_imputed_cells += int(cleaned[col].isna().sum())
            cleaned[col] = cleaned[col].fillna("UNKNOWN")

    missing_after = int(cleaned.isna().sum().sum())
    clean_rows, clean_cols = cleaned.shape
    clean_cells = int(clean_rows * clean_cols) if clean_rows and clean_cols else 0
    missing_ratio_after = (missing_after / clean_cells) if clean_cells else 0.0
    duplicate_ratio = (duplicate_rows / raw_rows) if raw_rows else 0.0
    log = {
        "table": table_name,
        "raw_shape": [int(raw_rows), int(raw_cols)],
        "clean_shape": [int(clean_rows), int(clean_cols)],
        "duplicate_rows_removed": duplicate_rows,
        "duplicate_rows_removed_ratio": round(float(duplicate_ratio), 6),
        "missing_values_before_cleaning": raw_missing_total,
        "missing_rate_before_cleaning": round(float(raw_missing_ratio), 6),
        "missing_values_imputed_numeric": int(num_imputed_cells),
        "missing_values_imputed_categorical": int(cat_imputed_cells),
        "object_to_numeric_columns": converted_cols,
        "missing_values_after_cleaning": missing_after,
        "missing_rate_after_cleaning": round(float(missing_ratio_after), 6),
        "outlier_rows_removed": 0,
        "outlier_handling_note": "No rows were removed as outliers in Task 1; outlier treatment is deferred to Task 2+ modeling stage.",
    }
    return cleaned, log


def build_master_dataset(data_dir: Path) -> tuple[pd.DataFrame, Dict[str, object]]:
    app_raw = load_csv(data_dir, "application_train.csv")
    if TARGET_COL not in app_raw.columns:
        raise ValueError(f"{TARGET_COL} not found in application_train.csv")
    app, app_log = clean_table(app_raw, "application_train.csv")

    # Optional supporting tables for richer complexity in Task 1.
    optional_tables = [
        ("bureau.csv", "BUREAU"),
        ("previous_application.csv", "PREV"),
    ]

    merged = app.copy()
    source_log: Dict[str, object] = {"main_table": app_log, "optional_tables": []}

    for file_name, prefix in optional_tables:
        table_path = data_dir / file_name
        if table_path.exists():
            raw = load_csv(data_dir, file_name)
            cleaned, cleaning_log = clean_table(raw, file_name)
            agg = aggregate_table(cleaned, ID_COL, prefix)
            merged = merged.merge(agg, on=ID_COL, how="left")
            print(f"[INFO] Merged {file_name}: +{agg.shape[1] - 1} aggregated features")
            source_log["optional_tables"].append(
                {
                    "table": file_name,
                    "prefix": prefix,
                    "cleaning": cleaning_log,
                    "aggregated_shape": [int(agg.shape[0]), int(agg.shape[1])],
                }
            )
        else:
            print(f"[WARN] Optional file not found, skipping: {file_name}")

    # Remove duplicate columns if any.
    merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged, source_log


def aggregate_table(df: pd.DataFrame, id_col: str, prefix: str) -> pd.DataFrame:
    """Aggregate one-to-many support tables to one row per customer."""
    if id_col not in df.columns:
        raise ValueError(f"{id_col} not found in {prefix} table.")

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != id_col]
    cat_cols = [c for c in df.select_dtypes(exclude=[np.number]).columns if c != id_col]

    agg_parts: List[pd.DataFrame] = []
    if numeric_cols:
        num_agg = df.groupby(id_col)[numeric_cols].agg(["mean", "max", "min", "std"]).reset_index()
        num_agg.columns = [
            id_col if c[0] == id_col else f"{prefix}_{c[0]}_{c[1]}".upper()
            for c in num_agg.columns.to_flat_index()
        ]
        agg_parts.append(num_agg)
    if cat_cols:
        cat_dummies = pd.get_dummies(df[[id_col] + cat_cols], columns=cat_cols, dummy_na=True)
        cat_agg = cat_dummies.groupby(id_col).mean().reset_index()
        cat_agg.columns = [id_col] + [f"{prefix}_{c}".upper() for c in cat_agg.columns if c != id_col]
        agg_parts.append(cat_agg)

    if not agg_parts:
        return df[[id_col]].drop_duplicates().copy()

    result = agg_parts[0]
    for part in agg_parts[1:]:
        result = result.merge(part, on=id_col, how="left")
    return result


def write_task1_outputs(
    df: pd.DataFrame, source_log: Dict[str, object], data_dir: Path, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Save cleaned, merged dataset for downstream tasks.
    cleaned_path = processed_dir / "home_credit_task1_cleaned.csv"
    df.to_csv(cleaned_path, index=False)

    raw_summary = {
        "source_directory": str(data_dir.resolve()),
        "credible_source_note": (
            "This dataset should be cited as Kaggle Home Credit Default Risk "
            "(public competition dataset)."
        ),
        "tables_used": {
            "required": ["application_train.csv"],
            "optional_if_available": ["bureau.csv", "previous_application.csv"],
        },
        "shape_after_preparation": [int(df.shape[0]), int(df.shape[1])],
        "target_distribution": df[TARGET_COL].value_counts(dropna=False).to_dict()
        if TARGET_COL in df.columns
        else {},
    }

    prep_summary = {
        "raw_data_summary": raw_summary,
        "cleaning_strategy_rationale": [
            "Median imputation is robust to skewed numeric distributions and extreme values common in credit-risk variables.",
            "Categorical missing values are encoded as 'UNKNOWN' so missingness is preserved as potentially informative signal.",
            "Type coercion for numeric-like text prevents silent parsing inconsistencies in downstream preprocessing.",
            "Aggregation of one-to-many tables to customer-level avoids label leakage from row multiplication and keeps one row per SK_ID_CURR.",
            "Outlier filtering is intentionally deferred to Task 2+ so EDA and model comparison can evaluate transformations transparently.",
        ],
        "cleaning_steps_taken": [
            "Standardized all column names to uppercase snake_case style.",
            "Removed duplicate rows within each source table.",
            "Trimmed string values and normalized placeholder missing values (e.g., '', NA, N/A, None, null).",
            "Converted numeric-like object columns to numeric data types.",
            "Imputed missing numeric values with median and missing categorical values with 'UNKNOWN'.",
            "Aggregated one-to-many support tables to customer-level features using mean/max/min/std and category proportions.",
            "Merged prepared support tables into application_train using SK_ID_CURR.",
            "Removed duplicated columns generated during merge.",
        ],
        "table_level_cleaning_log": source_log,
        "dataset_for_task2_and_beyond": {
            "path": "outputs/processed/home_credit_task1_cleaned.csv",
            "definition": "Primary cleaned customer-level modeling dataset for Task 2+.",
            "shape": [int(df.shape[0]), int(df.shape[1])],
        },
    }

    (output_dir / "task1_raw_data_summary.json").write_text(json.dumps(raw_summary, indent=2))
    (output_dir / "task1_preparation_summary.json").write_text(json.dumps(prep_summary, indent=2))

    # Human-readable short brief for report.
    lines = [
        "Project 4 Task 1 Brief Summary",
        "=============================",
        f"Source directory: {data_dir.resolve()}",
        "Credible source: Kaggle Home Credit Default Risk",
        f"Prepared dataset shape: {df.shape[0]} rows x {df.shape[1]} columns",
        "Task 2+ dataset: outputs/processed/home_credit_task1_cleaned.csv "
        f"({df.shape[0]} x {df.shape[1]})",
        "",
        "Cleaning completed:",
        "- Standardized column names",
        "- Removed duplicates",
        "- Normalized formatting and missing-value placeholders",
        "- Converted numeric-like text columns",
        "- Imputed missing values",
        "- Aggregated and merged supporting tables (if available)",
        "",
        "Rationale for key cleaning choices:",
        "- Numeric missing values were imputed by median for robustness to skew/outliers.",
        "- Categorical missing values were imputed as UNKNOWN to preserve missingness signal.",
        "- Outlier removal was deferred to Task 2+ for transparent EDA/modeling decisions.",
        "",
        f"Cleaned dataset saved to: {cleaned_path.resolve()}",
    ]
    (output_dir / "task1_brief_summary.txt").write_text("\n".join(lines))


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def run_task2(input_csv: Path, output_dir: Path, sample_size: int) -> None:
    print("[INFO] Task 2 - Exploratory Data Analysis (EDA) + Unsupervised Learning")
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Task 2 input dataset not found: {input_csv}. "
            "Run Task 1 first, or pass --task2-input to an existing cleaned CSV."
        )

    task2_dir = output_dir / "task2"
    figs_dir = task2_dir / "figures"
    task2_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Keep matplotlib cache writable in restricted environments.
    os.environ.setdefault("MPLCONFIGDIR", str((output_dir / ".mplconfig").resolve()))
    import matplotlib.pyplot as plt

    df = pd.read_csv(input_csv)
    n_rows, n_cols = int(df.shape[0]), int(df.shape[1])

    engineered_features: List[str] = []
    engineered_columns: Dict[str, pd.Series] = {}

    def safe_add_ratio(new_col: str, numerator: str, denominator: str) -> None:
        if numerator not in df.columns or denominator not in df.columns:
            return
        num = pd.to_numeric(df[numerator], errors="coerce")
        den = pd.to_numeric(df[denominator], errors="coerce").replace(0, np.nan)
        ratio = (num / den).replace([np.inf, -np.inf], np.nan)
        median = ratio.median(skipna=True)
        fill_value = float(median) if pd.notna(median) else 0.0
        engineered_columns[new_col] = ratio.fillna(fill_value)
        engineered_features.append(new_col)

    safe_add_ratio("CREDIT_INCOME_RATIO", "AMT_CREDIT", "AMT_INCOME_TOTAL")
    safe_add_ratio("ANNUITY_INCOME_RATIO", "AMT_ANNUITY", "AMT_INCOME_TOTAL")
    safe_add_ratio("CREDIT_ANNUITY_RATIO", "AMT_CREDIT", "AMT_ANNUITY")
    safe_add_ratio("GOODS_CREDIT_RATIO", "AMT_GOODS_PRICE", "AMT_CREDIT")
    safe_add_ratio("DAYS_EMPLOYED_PERC", "DAYS_EMPLOYED", "DAYS_BIRTH")
    if engineered_columns:
        df = df.drop(columns=list(engineered_columns), errors="ignore")
        df = pd.concat([df, pd.DataFrame(engineered_columns, index=df.index)], axis=1)
    else:
        df = df.copy()
    n_rows, n_cols = int(df.shape[0]), int(df.shape[1])

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [c for c in numeric_cols if c not in [TARGET_COL, ID_COL]]
    if not numeric_features:
        raise ValueError("Task 2 requires numeric feature columns beyond TARGET/SK_ID_CURR.")

    target_distribution: Dict[str, int] = {}
    class_imbalance: Dict[str, object] = {}
    target_numeric: Optional[pd.Series] = None
    if TARGET_COL in df.columns:
        target_distribution = {str(k): int(v) for k, v in df[TARGET_COL].value_counts(dropna=False).items()}
        target_numeric = pd.to_numeric(df[TARGET_COL], errors="coerce")
        target_counts = target_numeric.value_counts(dropna=True)
        if 0 in target_counts.index and 1 in target_counts.index:
            majority_count = int(target_counts.max())
            minority_count = int(target_counts.min())
            imbalance_ratio = float(majority_count / minority_count) if minority_count > 0 else float("inf")
            class_imbalance = {
                "default_rate": float(target_numeric.mean()),
                "imbalance_ratio_majority_to_minority": imbalance_ratio,
                "majority_class_count": majority_count,
                "minority_class_count": minority_count,
                "interpretation": (
                    "Task 3 should evaluate models using AUC, recall, precision, F1, "
                    "or PR-AUC rather than accuracy alone because the target is imbalanced."
                ),
            }

    # Descriptive stats and correlations.
    desc = df[numeric_features].describe().T
    desc["missing_count"] = df[numeric_features].isna().sum()
    desc["missing_rate"] = desc["missing_count"] / n_rows
    desc_path = task2_dir / "task2_numeric_descriptive_stats.csv"
    desc.to_csv(desc_path)

    outlier_rows: List[Dict[str, object]] = []
    for col in numeric_features:
        series = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            outlier_rows.append(
                {
                    "feature": col,
                    "q1": np.nan,
                    "q3": np.nan,
                    "iqr": np.nan,
                    "lower_bound": np.nan,
                    "upper_bound": np.nan,
                    "outlier_count": 0,
                    "non_missing_count": 0,
                    "outlier_rate": 0.0,
                }
            )
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = int(((series < lower_bound) | (series > upper_bound)).sum())
        outlier_rows.append(
            {
                "feature": col,
                "q1": q1,
                "q3": q3,
                "iqr": float(iqr),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "outlier_count": outlier_count,
                "non_missing_count": int(series.shape[0]),
                "outlier_rate": float(outlier_count / series.shape[0]) if series.shape[0] else 0.0,
            }
        )
    outlier_path = task2_dir / "task2_outlier_summary.csv"
    pd.DataFrame(outlier_rows).sort_values(by="outlier_rate", ascending=False).to_csv(outlier_path, index=False)

    if TARGET_COL in df.columns:
        corr_input = df[numeric_features].apply(pd.to_numeric, errors="coerce")
        target_for_corr = pd.to_numeric(df[TARGET_COL], errors="coerce")
        corr_to_target = (
            corr_input.corrwith(target_for_corr, method="pearson")
            .sort_values(key=lambda s: s.abs(), ascending=False)
        )
        corr_path = task2_dir / "task2_feature_target_correlations.csv"
        corr_to_target.to_csv(corr_path, header=["pearson_corr_with_target"])
        spearman_corr_to_target = (
            corr_input.corrwith(target_for_corr, method="spearman")
            .sort_values(key=lambda s: s.abs(), ascending=False)
        )
        spearman_corr_path = task2_dir / "task2_feature_target_spearman_correlations.csv"
        spearman_corr_to_target.to_csv(spearman_corr_path, header=["spearman_corr_with_target"])
    else:
        corr_to_target = pd.Series(dtype=float)
        spearman_corr_to_target = pd.Series(dtype=float)
        corr_path = None
        spearman_corr_path = None

    # Advanced statistical comparisons by target group.
    stats_tests_path = task2_dir / "task2_statistical_tests.csv"
    selected_numeric_for_tests = corr_to_target.head(12).index.tolist() if not corr_to_target.empty else numeric_features[:12]
    stats_rows: List[Dict[str, object]] = []
    if target_numeric is not None and set(target_numeric.dropna().unique().tolist()) >= {0, 1}:
        g0 = df[target_numeric == 0]
        g1 = df[target_numeric == 1]
        for col in selected_numeric_for_tests:
            x0 = pd.to_numeric(g0[col], errors="coerce").dropna()
            x1 = pd.to_numeric(g1[col], errors="coerce").dropna()
            if len(x0) < 20 or len(x1) < 20:
                continue
            stat, p_val = mannwhitneyu(x0, x1, alternative="two-sided")
            pooled_std = float(np.sqrt((x0.var(ddof=1) + x1.var(ddof=1)) / 2)) if (x0.var(ddof=1) + x1.var(ddof=1)) > 0 else 0.0
            effect_d = float((x1.mean() - x0.mean()) / pooled_std) if pooled_std > 0 else 0.0
            stats_rows.append(
                {
                    "feature": col,
                    "test": "mann_whitney_u",
                    "p_value": float(p_val),
                    "group0_mean": float(x0.mean()),
                    "group1_mean": float(x1.mean()),
                    "effect_size_cohens_d": effect_d,
                }
            )
    stats_columns = [
        "feature",
        "test",
        "p_value",
        "p_value_adj_bh",
        "group0_mean",
        "group1_mean",
        "effect_size_cohens_d",
        "abs_effect_size",
    ]
    if stats_rows:
        stats_df = pd.DataFrame(stats_rows)
        stats_df["p_value_adj_bh"] = multipletests(stats_df["p_value"], method="fdr_bh")[1]
        stats_df["abs_effect_size"] = stats_df["effect_size_cohens_d"].abs()
        stats_df = stats_df.sort_values(by=["abs_effect_size", "p_value_adj_bh"], ascending=[False, True])
        stats_df = stats_df[stats_columns]
    else:
        stats_df = pd.DataFrame(columns=stats_columns)
    stats_df.to_csv(stats_tests_path, index=False)

    cat_assoc_path = task2_dir / "task2_categorical_associations.csv"
    cat_rows: List[Dict[str, object]] = []
    if TARGET_COL in df.columns:
        cat_cols = [c for c in df.select_dtypes(exclude=[np.number]).columns if c not in [ID_COL]]
        for col in cat_cols[:20]:
            contingency = pd.crosstab(df[col].fillna("UNKNOWN"), df[TARGET_COL])
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                continue
            chi2, p_val, _, _ = chi2_contingency(contingency)
            n = contingency.to_numpy().sum()
            r, k = contingency.shape
            denom = max(1, min(r - 1, k - 1))
            cramers_v = float(np.sqrt((chi2 / n) / denom)) if n > 0 else 0.0
            cat_rows.append(
                {
                    "feature": col,
                    "test": "chi_square",
                    "p_value": float(p_val),
                    "cramers_v": cramers_v,
                    "n_levels": int(contingency.shape[0]),
                }
            )
    cat_assoc_df = pd.DataFrame(
        cat_rows,
        columns=["feature", "test", "p_value", "cramers_v", "n_levels"],
    )
    if not cat_assoc_df.empty:
        cat_assoc_df = cat_assoc_df.sort_values(by="p_value", ascending=True)
    cat_assoc_df.to_csv(cat_assoc_path, index=False)

    # Visuals.
    if TARGET_COL in df.columns:
        plt.figure(figsize=(6, 4))
        df[TARGET_COL].value_counts().sort_index().plot(kind="bar")
        plt.title("Target Distribution")
        plt.xlabel(TARGET_COL)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(figs_dir / "task2_target_distribution.png", dpi=140)
        plt.close()

    top_corr_cols = corr_to_target.head(6).index.tolist() if not corr_to_target.empty else numeric_features[:6]
    if top_corr_cols:
        plt.figure(figsize=(10, 6))
        corr_matrix = df[top_corr_cols].corr(numeric_only=True)
        im = plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(top_corr_cols)), top_corr_cols, rotation=45, ha="right", fontsize=8)
        plt.yticks(range(len(top_corr_cols)), top_corr_cols, fontsize=8)
        plt.title("Correlation Heatmap (Top Numeric Features)")
        plt.tight_layout()
        plt.savefig(figs_dir / "task2_top_feature_correlation_heatmap.png", dpi=140)
        plt.close()

    top15_corr_fig_path = figs_dir / "task2_top15_target_correlations.png"
    if not corr_to_target.empty:
        top15_corr = corr_to_target.dropna().head(15).sort_values()
        if not top15_corr.empty:
            plt.figure(figsize=(9, 6))
            colors = ["#5b8fd9" if value >= 0 else "#d9825b" for value in top15_corr.values]
            plt.barh(top15_corr.index, top15_corr.values, color=colors)
            plt.axvline(0, color="black", linewidth=0.8)
            plt.title("Top 15 Absolute Pearson Correlations with TARGET")
            plt.xlabel("Pearson correlation")
            plt.tight_layout()
            plt.savefig(top15_corr_fig_path, dpi=140)
            plt.close()

    # Advanced visuals: target-wise distributions for top features.
    top_dist_cols = selected_numeric_for_tests[:5]
    for col in top_dist_cols:
        if col not in df.columns or TARGET_COL not in df.columns:
            continue
        tmp = df[[col, TARGET_COL]].copy()
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
        q_low, q_high = tmp[col].quantile([0.01, 0.99]).tolist()
        tmp[col] = tmp[col].clip(lower=q_low, upper=q_high)
        target_for_plot = pd.to_numeric(tmp[TARGET_COL], errors="coerce")
        g0_all = tmp[target_for_plot == 0][col].dropna()
        g1_all = tmp[target_for_plot == 1][col].dropna()
        if g0_all.empty or g1_all.empty:
            continue
        g0 = g0_all.sample(n=min(15000, len(g0_all)), random_state=42)
        g1 = g1_all.sample(n=min(15000, len(g1_all)), random_state=42)
        plt.figure(figsize=(7, 4))
        plt.hist(g0, bins=40, alpha=0.55, density=True, label="TARGET=0")
        plt.hist(g1, bins=40, alpha=0.55, density=True, label="TARGET=1")
        plt.title(f"Distribution by Target: {col}")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figs_dir / f"task2_dist_by_target_{col}.png", dpi=140)
        plt.close()

    # Unsupervised: PCA + KMeans.
    use_cols = numeric_features.copy()
    sample_n = min(sample_size, n_rows)
    if sample_n < n_rows:
        sample_indices = df.sample(n=sample_n, random_state=42).index
        sample_df = df.loc[sample_indices, use_cols].copy()
    else:
        sample_indices = df.index
        sample_df = df[use_cols].copy()
    sample_df = sample_df.replace([np.inf, -np.inf], np.nan)
    sample_df = sample_df.fillna(sample_df.median(numeric_only=True)).fillna(0)
    # Clamp extreme values and remove constant columns for numerical stability.
    sample_df = sample_df.clip(lower=sample_df.quantile(0.01), upper=sample_df.quantile(0.99), axis=1)
    nunique = sample_df.nunique(dropna=False)
    stable_cols = nunique[nunique > 1].index.tolist()
    if len(stable_cols) < 2:
        raise ValueError("Task 2 unsupervised stage requires at least two non-constant numeric features.")
    sample_df = sample_df[stable_cols]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(sample_df.values)
    x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    pca = PCA(n_components=2, random_state=42)
    x_pca = pca.fit_transform(x_scaled)
    explained_2d = pca.explained_variance_ratio_

    k_selection_rows: List[Dict[str, object]] = []
    for k in range(2, 9):
        if k > len(x_scaled):
            k_selection_rows.append({"k": k, "inertia": np.nan, "silhouette_score": np.nan})
            continue
        k_model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = k_model.fit_predict(x_scaled)
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1 and len(unique_labels) < len(labels):
            sil_score = silhouette_score(
                x_scaled,
                labels,
                sample_size=min(5000, len(labels)),
                random_state=42,
            )
        else:
            sil_score = np.nan
        k_selection_rows.append(
            {
                "k": k,
                "inertia": float(k_model.inertia_),
                "silhouette_score": float(sil_score) if pd.notna(sil_score) else np.nan,
            }
        )
    k_selection_path = task2_dir / "task2_kmeans_k_selection.csv"
    pd.DataFrame(k_selection_rows).to_csv(k_selection_path, index=False)

    selected_k = 3
    kmeans = KMeans(n_clusters=selected_k, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(x_scaled)
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    dist_to_centers = kmeans.transform(x_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c=clusters, s=8, alpha=0.6, cmap="tab10")
    plt.title(f"PCA Projection with KMeans Clusters (k={selected_k})")
    plt.xlabel(f"PC1 ({explained_2d[0] * 100:.2f}% var)")
    plt.ylabel(f"PC2 ({explained_2d[1] * 100:.2f}% var)")
    plt.tight_layout()
    plt.savefig(figs_dir / "task2_pca_kmeans_clusters.png", dpi=140)
    plt.close()

    # Cluster profiling and target relationship.
    cluster_profile = pd.DataFrame({"CLUSTER_ID": clusters})
    for col in selected_numeric_for_tests[:8]:
        if col in sample_df.columns:
            cluster_profile[col] = sample_df[col].values
    if TARGET_COL in df.columns:
        cluster_profile[TARGET_COL] = pd.to_numeric(df.loc[sample_indices, TARGET_COL], errors="coerce").values
    cluster_profile_summary = cluster_profile.groupby("CLUSTER_ID").mean(numeric_only=True)
    cluster_sizes = pd.Series(clusters).value_counts().sort_index()
    cluster_profile_summary["cluster_size"] = cluster_sizes
    cluster_profile_summary["cluster_percentage"] = cluster_sizes / len(clusters)
    cluster_profile_path = task2_dir / "task2_cluster_profiles.csv"
    cluster_profile_summary.to_csv(cluster_profile_path)

    # Save Task 3-ready unsupervised features for sampled rows.
    task3_unsup_path = task2_dir / "task2_unsupervised_features_sample.csv"
    unsup_features_dict: Dict[str, object] = {}
    if ID_COL in df.columns:
        unsup_features_dict[ID_COL] = df.loc[sample_indices, ID_COL].values
    else:
        unsup_features_dict["SAMPLE_INDEX"] = sample_indices.to_numpy()
    unsup_features_dict.update(
        {
            "PCA_1": x_pca[:, 0],
            "PCA_2": x_pca[:, 1],
            "CLUSTER_ID": clusters,
        }
    )
    unsup_features = pd.DataFrame(unsup_features_dict)
    for i in range(dist_to_centers.shape[1]):
        unsup_features[f"DIST_TO_CLUSTER_{i}"] = dist_to_centers[:, i]
    unsup_features.to_csv(task3_unsup_path, index=False)

    unsup_summary = {
        "method": "PCA + KMeans",
        "sample_size_used": int(sample_n),
        "n_numeric_features_used": int(len(stable_cols)),
        "pca_explained_variance_ratio": {
            "pc1": float(round(explained_2d[0], 6)),
            "pc2": float(round(explained_2d[1], 6)),
            "pc1_pc2_total": float(round(explained_2d[0] + explained_2d[1], 6)),
        },
        "kmeans": {
            "n_clusters": selected_k,
            "cluster_counts": {str(int(k)): int(v) for k, v in cluster_counts.items()},
            "inertia": float(round(kmeans.inertia_, 3)),
            "k_selection_file": str(k_selection_path),
        },
    }

    eda_summary = {
        "input_dataset": str(input_csv),
        "shape": [n_rows, n_cols],
        "target_distribution": target_distribution,
        "class_imbalance": class_imbalance,
        "engineered_features": engineered_features,
        "eda_outputs": {
            "numeric_descriptive_stats_csv": str(desc_path),
            "outlier_summary_csv": str(outlier_path),
            "feature_target_correlations_csv": str(corr_path) if corr_path else None,
            "feature_target_spearman_correlations_csv": str(spearman_corr_path) if spearman_corr_path else None,
            "statistical_tests_csv": str(stats_tests_path),
            "categorical_associations_csv": str(cat_assoc_path),
            "cluster_profiles_csv": str(cluster_profile_path),
            "kmeans_k_selection_csv": str(k_selection_path),
            "task3_ready_unsupervised_features_sample_csv": str(task3_unsup_path),
            "figures_directory": str(figs_dir),
        },
        "unsupervised_learning": unsup_summary,
        "task3_prep_note": (
            "Task 3 should consume the Task 1 cleaned dataset, merge Task 2 unsupervised "
            "features back by SK_ID_CURR when available, and use Task 2 insights for feature "
            "transformations and selection."
        ),
    }
    write_json(task2_dir / "task2_eda_unsupervised_summary.json", eda_summary)

    # Narrative insights for report/presentation.
    top_corr_lines = []
    if not corr_to_target.empty:
        for feat, val in corr_to_target.head(5).items():
            top_corr_lines.append(f"- {feat}: correlation with TARGET = {val:.4f}")
    top_stats_lines = []
    if not stats_df.empty:
        for _, row in stats_df.head(5).iterrows():
            top_stats_lines.append(
                f"- {row['feature']}: p={row['p_value']:.3e}, Cohen's d={row['effect_size_cohens_d']:.3f}"
            )
    cluster_rate_note = ""
    if TARGET_COL in cluster_profile_summary.columns:
        cluster_rate_note = (
            "Cluster-level default rates (sample): "
            + ", ".join(
                [f"cluster {idx}={val:.4f}" for idx, val in cluster_profile_summary[TARGET_COL].items()]
            )
        )

    corr_section = top_corr_lines if top_corr_lines else ["- Correlation table unavailable."]
    stats_section = top_stats_lines if top_stats_lines else ["- Statistical test table unavailable."]
    default_rate_note = "Default rate unavailable because TARGET is missing or not binary."
    imbalance_ratio_note = "Imbalance ratio unavailable because TARGET is missing or not binary."
    if class_imbalance:
        default_rate_note = f"Default rate: {class_imbalance['default_rate']:.4f}."
        imbalance_ratio_note = (
            "Imbalance ratio (majority/minority): "
            f"{class_imbalance['imbalance_ratio_majority_to_minority']:.2f}."
        )

    insights_lines = [
        "Task 2 Key Insights (EDA + Unsupervised)",
        "=======================================",
        f"Dataset analyzed: {input_csv} ({n_rows} x {n_cols})",
        "",
        "1) Global structure",
        f"- Class imbalance remains visible: TARGET=1 count is {target_distribution.get('1', target_distribution.get('1.0', 'N/A'))} vs TARGET=0 count {target_distribution.get('0', target_distribution.get('0.0', 'N/A'))}.",
        f"- {default_rate_note}",
        f"- {imbalance_ratio_note}",
        "- Task 3 should evaluate models with AUC, recall, precision, F1, or PR-AUC rather than accuracy alone because the target is imbalanced.",
        f"- Engineered credit-risk features added when source columns were available: {', '.join(engineered_features) if engineered_features else 'none'}.",
        f"- Numeric feature space used for unsupervised analysis: {len(stable_cols)} columns.",
        "",
        "2) Strongest feature-target relationships (Pearson)",
    ]
    insights_lines.extend(corr_section)
    insights_lines.extend([
        "",
        "3) Statistical evidence of group differences (Mann-Whitney U)",
    ])
    insights_lines.extend(stats_section)
    insights_lines.extend([
        "",
        "4) Unsupervised structure",
        f"- PCA PC1+PC2 explains {(explained_2d[0] + explained_2d[1]) * 100:.2f}% variance in sampled standardized data.",
        f"- KMeans(k={selected_k}) cluster counts: {unsup_summary['kmeans']['cluster_counts']}.",
        f"- {cluster_rate_note}" if cluster_rate_note else "- Cluster-level target relationship unavailable.",
        "",
        "5) Task 3 feature engineering implications",
        "- Merge task2_unsupervised_features_sample.csv back to the modeling dataset by SK_ID_CURR when that identifier exists.",
        f"- Reuse unsupervised features: PCA_1, PCA_2, CLUSTER_ID, and DIST_TO_CLUSTER_0 through DIST_TO_CLUSTER_{selected_k - 1}.",
        "- Prioritize features with strong correlation and significant distribution shifts between TARGET groups.",
    ])
    (task2_dir / "task2_insights.txt").write_text("\n".join(insights_lines))

    brief_lines = [
        "Project 4 Task 2 Brief Summary",
        "=============================",
        f"Input dataset: {input_csv}",
        f"Shape analyzed: {n_rows} rows x {n_cols} columns",
        f"Numeric features analyzed: {len(numeric_features)}",
        f"Unsupervised methods: PCA + KMeans(k={selected_k}), sample size={sample_n}",
        (
            "PCA explained variance (PC1+PC2): "
            f"{(explained_2d[0] + explained_2d[1]) * 100:.2f}%"
        ),
        f"Cluster counts: {unsup_summary['kmeans']['cluster_counts']}",
        "",
        "Key output files:",
        f"- {desc_path}",
        f"- {task2_dir / 'task2_outlier_summary.csv'}",
        f"- {task2_dir / 'task2_feature_target_correlations.csv'}",
        f"- {task2_dir / 'task2_feature_target_spearman_correlations.csv'}",
        f"- {task2_dir / 'task2_statistical_tests.csv'}",
        f"- {task2_dir / 'task2_categorical_associations.csv'}",
        f"- {task2_dir / 'task2_kmeans_k_selection.csv'}",
        f"- {task2_dir / 'task2_cluster_profiles.csv'}",
        f"- {task2_dir / 'task2_unsupervised_features_sample.csv'}",
        f"- {task2_dir / 'task2_eda_unsupervised_summary.json'}",
        f"- {task2_dir / 'task2_insights.txt'}",
        f"- {figs_dir / 'task2_target_distribution.png'}",
        f"- {figs_dir / 'task2_top_feature_correlation_heatmap.png'}",
        f"- {figs_dir / 'task2_top15_target_correlations.png'}",
        f"- {figs_dir / 'task2_pca_kmeans_clusters.png'}",
    ]
    (task2_dir / "task2_brief_summary.txt").write_text("\n".join(brief_lines))
    print(f"[INFO] Task 2 outputs saved to: {task2_dir.resolve()}")


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    data_dir: Path = args.data_dir
    task2_input: Path = args.task2_input

    if args.task in ("task1", "all"):
        print("[INFO] Task 1 - Data Acquisition & Preparation")
        required_file = "application_train.csv"
        required_path = resolve_csv_path(data_dir, required_file)
        if required_path is None and args.auto_download:
            data_dir = maybe_download_with_kagglehub(args.dataset_slug)

        required_path = resolve_csv_path(data_dir, required_file)
        if required_path is None:
            raise FileNotFoundError(
                f"Could not find {required_file}. "
                f"Checked data directory: {data_dir}. "
                "Either place files locally or re-run with --auto-download."
            )

        print("[INFO] Loading, cleaning, and merging source tables...")
        df, source_log = build_master_dataset(data_dir)
        write_task1_outputs(df, source_log, data_dir, output_dir)
        print(f"[INFO] Task 1 outputs saved to: {output_dir.resolve()}")
        task2_input = output_dir / "processed" / "home_credit_task1_cleaned.csv"

    if args.task in ("task2", "all"):
        run_task2(task2_input, output_dir, sample_size=args.sample_size)


if __name__ == "__main__":
    main()
