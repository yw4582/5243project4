#!/usr/bin/env python3
"""
Home Credit Default Risk – Unified Pipeline
Project 4: Task 1 + Task 2 + Task 3

Runs the full preprocessing pipeline in one script:
  Task 1 – Data cleaning & merging
  Task 2 – EDA, PCA, KMeans clustering
  Task 3 – Feature engineering, encoding, scaling, feature selection

Usage:
  # Run all three tasks end-to-end
  python home_credit_pipeline.py --data-dir "D:\\data\\home_credit"

  # Run only Task 3 on an existing cleaned CSV
  python home_credit_pipeline.py --task task3 --task1-input "D:\\...\\home_credit_task1_cleaned.csv"

  # Skip LightGBM in Task 3 (if not installed)
  python home_credit_pipeline.py --data-dir "D:\\data" --no-lgbm

Output of Task 3 (use this for Task 4 modeling):
  outputs/task3/processed/home_credit_task3_modeling_ready.csv
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_COL            = "TARGET"
ID_COL                = "SK_ID_CURR"
DAYS_EMPLOYED_SENTINEL = 365243
PASSTHROUGH_COLS      = {TARGET_COL, ID_COL}


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Home Credit – Unified Pipeline (Tasks 1–3)")
    p.add_argument("--task", choices=["task1", "task2", "task3", "all"], default="all",
                   help="Which task(s) to run (default: all).")
    p.add_argument("--data-dir", type=Path, default=Path("data/raw"),
                   help="Folder containing raw CSV files (application_train.csv etc).")
    p.add_argument("--task1-input", type=Path,
                   default=Path("outputs/processed/home_credit_task1_cleaned.csv"),
                   help="Pre-existing Task 1 cleaned CSV (skip Task 1 if provided with --task task2/task3).")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"),
                   help="Root output directory.")
    p.add_argument("--sample-size", type=int, default=20000,
                   help="Rows sampled for PCA/KMeans in Task 2 (default: 20000).")
    p.add_argument("--top-k-features", type=int, default=80,
                   help="Top-K features to keep after LightGBM selection in Task 3.")
    p.add_argument("--corr-threshold", type=float, default=0.95,
                   help="Correlation threshold for deduplication in Task 3.")
    p.add_argument("--no-lgbm", action="store_true",
                   help="Skip LightGBM feature selection in Task 3.")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ══════════════════════════════════════════════════════════════════════════════
def setup_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        c.strip().upper().replace(" ", "_").replace("/", "_").replace("-", "_")
        for c in df.columns
    ]
    return df


def resolve_csv(data_dir: Path, name: str) -> Optional[Path]:
    direct = data_dir / name
    if direct.exists():
        return direct
    matches = list(data_dir.rglob(name))
    return matches[0] if matches else None


def load_raw_csv(data_dir: Path, name: str) -> pd.DataFrame:
    path = resolve_csv(data_dir, name)
    if path is None:
        raise FileNotFoundError(f"Cannot find {name} under {data_dir}")
    return sanitize_columns(pd.read_csv(path))


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    d = den.replace(0, np.nan)
    r = num / d
    return r.replace([np.inf, -np.inf], np.nan)


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 – Data Cleaning & Merging
# ══════════════════════════════════════════════════════════════════════════════
def coerce_numeric_objects(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    converted = []
    for col in out.select_dtypes(include=["object"]).columns:
        s = out[col].astype(str).str.strip().replace(
            {"": np.nan, "NA": np.nan, "N/A": np.nan, "None": np.nan, "null": np.nan}
        )
        numeric = pd.to_numeric(s, errors="coerce")
        non_null = s.notna().sum()
        if non_null > 0 and numeric.notna().sum() / non_null >= 0.90:
            out[col] = numeric
            converted.append(col)
        else:
            out[col] = s
    return out, converted


def clean_table(df: pd.DataFrame, name: str) -> Tuple[pd.DataFrame, Dict]:
    df = sanitize_columns(df)
    raw_shape = df.shape
    raw_missing = int(df.isna().sum().sum())

    # Remove duplicates
    n_dup = int(df.duplicated().sum())
    if n_dup:
        df = df.drop_duplicates()

    # Normalize string placeholders
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip().replace(
            {"": np.nan, "NA": np.nan, "N/A": np.nan, "None": np.nan, "null": np.nan}
        )

    df, converted = coerce_numeric_objects(df)

    # Impute
    num_imputed = cat_imputed = 0
    for col in df.select_dtypes(include=[np.number]).columns:
        n = int(df[col].isna().sum())
        if n:
            df[col] = df[col].fillna(df[col].median())
            num_imputed += n
    for col in df.select_dtypes(exclude=[np.number]).columns:
        n = int(df[col].isna().sum())
        if n:
            df[col] = df[col].fillna("UNKNOWN")
            cat_imputed += n

    log = {
        "table": name,
        "raw_shape": list(raw_shape),
        "clean_shape": list(df.shape),
        "duplicates_removed": n_dup,
        "missing_before": raw_missing,
        "numeric_imputed": num_imputed,
        "categorical_imputed": cat_imputed,
        "missing_after": int(df.isna().sum().sum()),
        "numeric_columns_converted": converted,
    }
    return df, log


def aggregate_support_table(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if ID_COL not in df.columns:
        raise ValueError(f"{ID_COL} not in {prefix} table")
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != ID_COL]
    cat_cols = [c for c in df.select_dtypes(exclude=[np.number]).columns if c != ID_COL]
    parts = []
    if num_cols:
        agg = df.groupby(ID_COL)[num_cols].agg(["mean", "max", "min", "std"]).reset_index()
        agg.columns = [
            ID_COL if c[0] == ID_COL else f"{prefix}_{c[0]}_{c[1]}".upper()
            for c in agg.columns.to_flat_index()
        ]
        parts.append(agg)
    if cat_cols:
        dummies = pd.get_dummies(df[[ID_COL] + cat_cols], columns=cat_cols, dummy_na=True)
        cat_agg = dummies.groupby(ID_COL).mean().reset_index()
        cat_agg.columns = [ID_COL] + [f"{prefix}_{c}".upper() for c in cat_agg.columns if c != ID_COL]
        parts.append(cat_agg)
    if not parts:
        return df[[ID_COL]].drop_duplicates()
    result = parts[0]
    for part in parts[1:]:
        result = result.merge(part, on=ID_COL, how="left")
    return result


def run_task1(data_dir: Path, output_dir: Path) -> Path:
    print("\n" + "="*60)
    print("TASK 1 – Data Cleaning & Merging")
    print("="*60)

    app_raw = load_raw_csv(data_dir, "application_train.csv")
    if TARGET_COL not in app_raw.columns:
        raise ValueError(f"{TARGET_COL} not found in application_train.csv")

    app, app_log = clean_table(app_raw, "application_train.csv")
    print(f"[INFO] application_train: {app_raw.shape} → {app.shape}")

    merged = app.copy()
    support_logs = []
    for fname, prefix in [("bureau.csv", "BUREAU"), ("previous_application.csv", "PREV")]:
        path = resolve_csv(data_dir, fname)
        if path:
            raw = load_raw_csv(data_dir, fname)
            cleaned, slog = clean_table(raw, fname)
            agg = aggregate_support_table(cleaned, prefix)
            merged = merged.merge(agg, on=ID_COL, how="left")
            print(f"[INFO] Merged {fname}: +{agg.shape[1]-1} features")
            support_logs.append(slog)
        else:
            print(f"[WARN] {fname} not found, skipping.")

    merged = merged.loc[:, ~merged.columns.duplicated()]
    print(f"[INFO] Final merged shape: {merged.shape}")

    proc_dir = output_dir / "processed"
    setup_dirs(proc_dir)
    out_path = proc_dir / "home_credit_task1_cleaned.csv"
    merged.to_csv(out_path, index=False)

    summary = {
        "main_table": app_log,
        "support_tables": support_logs,
        "final_shape": list(merged.shape),
        "output": str(out_path),
    }
    write_json(output_dir / "task1_summary.json", summary)

    lines = [
        "Task 1 Summary",
        "==============",
        f"Input:  application_train.csv + bureau.csv + previous_application.csv",
        f"Output: {out_path}",
        f"Shape:  {merged.shape[0]:,} rows × {merged.shape[1]} columns",
        "",
        "Cleaning steps:",
        "  - Standardized column names to uppercase snake_case",
        "  - Removed duplicate rows",
        "  - Replaced placeholders (NA, None, null) with NaN",
        "  - Converted numeric-like text columns",
        "  - Imputed numeric missing with median, categorical with UNKNOWN",
        "  - Aggregated bureau/previous_application to customer level",
        "  - Merged all tables on SK_ID_CURR",
    ]
    (output_dir / "task1_brief_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Task 1 output saved: {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 – EDA + PCA + KMeans
# ══════════════════════════════════════════════════════════════════════════════
def run_task2(input_csv: Path, output_dir: Path, sample_size: int) -> Path:
    print("\n" + "="*60)
    print("TASK 2 – EDA + PCA + KMeans")
    print("="*60)

    task2_dir = output_dir / "task2"
    figs_dir  = task2_dir / "figures"
    setup_dirs(task2_dir, figs_dir)
    os.environ.setdefault("MPLCONFIGDIR", str((output_dir / ".mplconfig").resolve()))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(input_csv)
    n_rows, n_cols = df.shape
    print(f"[INFO] Loaded: {n_rows:,} rows × {n_cols} cols")

    # ── Engineered ratios for EDA ─────────────────────────────────────────────
    def safe_add_ratio(new_col, num_col, den_col):
        if num_col not in df.columns or den_col not in df.columns:
            return
        num = pd.to_numeric(df[num_col], errors="coerce")
        den = pd.to_numeric(df[den_col], errors="coerce").replace(0, np.nan)
        r = (num / den).replace([np.inf, -np.inf], np.nan)
        df[new_col] = r.fillna(r.median())

    safe_add_ratio("CREDIT_INCOME_RATIO",  "AMT_CREDIT",    "AMT_INCOME_TOTAL")
    safe_add_ratio("ANNUITY_INCOME_RATIO", "AMT_ANNUITY",   "AMT_INCOME_TOTAL")
    safe_add_ratio("CREDIT_ANNUITY_RATIO", "AMT_CREDIT",    "AMT_ANNUITY")
    safe_add_ratio("GOODS_CREDIT_RATIO",   "AMT_GOODS_PRICE","AMT_CREDIT")
    safe_add_ratio("DAYS_EMPLOYED_PERC",   "DAYS_EMPLOYED", "DAYS_BIRTH")

    numeric_features = [c for c in df.select_dtypes(include=[np.number]).columns
                        if c not in [TARGET_COL, ID_COL]]

    # ── Target distribution ───────────────────────────────────────────────────
    target_dist = {}
    class_imbalance = {}
    if TARGET_COL in df.columns:
        tgt = pd.to_numeric(df[TARGET_COL], errors="coerce")
        target_dist = {str(k): int(v) for k, v in tgt.value_counts(dropna=True).items()}
        counts = tgt.value_counts(dropna=True)
        if 0 in counts.index and 1 in counts.index:
            class_imbalance = {
                "default_rate": float(tgt.mean()),
                "imbalance_ratio": float(counts.max() / counts.min()),
                "TARGET_0": int(counts[0]),
                "TARGET_1": int(counts[1]),
            }
        plt.figure(figsize=(6, 4))
        tgt.value_counts().sort_index().plot(kind="bar", color=["#4CAF50", "#F44336"])
        plt.title("Target Distribution")
        plt.xlabel("TARGET"); plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(figs_dir / "task2_target_distribution.png", dpi=140)
        plt.close()

    # ── Correlations with TARGET ──────────────────────────────────────────────
    corr_to_target = pd.Series(dtype=float)
    if TARGET_COL in df.columns:
        tgt_num = pd.to_numeric(df[TARGET_COL], errors="coerce")
        corr_to_target = (
            df[numeric_features].apply(pd.to_numeric, errors="coerce")
            .corrwith(tgt_num, method="pearson")
            .sort_values(key=lambda s: s.abs(), ascending=False)
        )
        corr_to_target.to_csv(task2_dir / "task2_feature_target_correlations.csv",
                               header=["pearson_corr_with_target"])

        # Top 15 bar chart
        top15 = corr_to_target.dropna().head(15).sort_values()
        colors = ["#5b8fd9" if v >= 0 else "#d9825b" for v in top15.values]
        plt.figure(figsize=(9, 6))
        plt.barh(top15.index, top15.values, color=colors)
        plt.axvline(0, color="black", linewidth=0.8)
        plt.title("Top 15 Pearson Correlations with TARGET")
        plt.xlabel("Pearson correlation")
        plt.tight_layout()
        plt.savefig(figs_dir / "task2_top15_target_correlations.png", dpi=140)
        plt.close()

    # ── Descriptive stats ─────────────────────────────────────────────────────
    desc = df[numeric_features].describe().T
    desc["missing_count"] = df[numeric_features].isna().sum()
    desc["missing_rate"]  = desc["missing_count"] / n_rows
    desc.to_csv(task2_dir / "task2_numeric_descriptive_stats.csv")

    # ── Mann-Whitney U tests ──────────────────────────────────────────────────
    test_cols = corr_to_target.head(12).index.tolist() if not corr_to_target.empty else numeric_features[:12]
    stats_rows = []
    if TARGET_COL in df.columns:
        tgt_num = pd.to_numeric(df[TARGET_COL], errors="coerce")
        g0 = df[tgt_num == 0]
        g1 = df[tgt_num == 1]
        for col in test_cols:
            x0 = pd.to_numeric(g0[col], errors="coerce").dropna()
            x1 = pd.to_numeric(g1[col], errors="coerce").dropna()
            if len(x0) < 20 or len(x1) < 20:
                continue
            _, p = mannwhitneyu(x0, x1, alternative="two-sided")
            pooled = float(np.sqrt((x0.var(ddof=1) + x1.var(ddof=1)) / 2))
            d = float((x1.mean() - x0.mean()) / pooled) if pooled > 0 else 0.0
            stats_rows.append({"feature": col, "p_value": float(p),
                                "group0_mean": float(x0.mean()),
                                "group1_mean": float(x1.mean()),
                                "cohens_d": d})
    pd.DataFrame(stats_rows).sort_values("p_value").to_csv(
        task2_dir / "task2_statistical_tests.csv", index=False)

    # ── Correlation heatmap ───────────────────────────────────────────────────
    top6 = corr_to_target.head(6).index.tolist() if not corr_to_target.empty else numeric_features[:6]
    if top6:
        corr_mat = df[top6].corr(numeric_only=True)
        plt.figure(figsize=(9, 7))
        im = plt.imshow(corr_mat, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(top6)), top6, rotation=45, ha="right", fontsize=8)
        plt.yticks(range(len(top6)), top6, fontsize=8)
        plt.title("Correlation Heatmap (Top Features)")
        plt.tight_layout()
        plt.savefig(figs_dir / "task2_top_feature_correlation_heatmap.png", dpi=140)
        plt.close()

    # ── Distribution plots by target ──────────────────────────────────────────
    if TARGET_COL in df.columns:
        tgt_num = pd.to_numeric(df[TARGET_COL], errors="coerce")
        for col in test_cols[:5]:
            if col not in df.columns:
                continue
            tmp = pd.to_numeric(df[col], errors="coerce")
            lo, hi = tmp.quantile(0.01), tmp.quantile(0.99)
            tmp = tmp.clip(lo, hi)
            g0s = tmp[tgt_num == 0].dropna().sample(n=min(15000, int((tgt_num==0).sum())), random_state=42)
            g1s = tmp[tgt_num == 1].dropna().sample(n=min(15000, int((tgt_num==1).sum())), random_state=42)
            plt.figure(figsize=(7, 4))
            plt.hist(g0s, bins=40, alpha=0.55, density=True, label="TARGET=0 (repaid)", color="#2196F3")
            plt.hist(g1s, bins=40, alpha=0.55, density=True, label="TARGET=1 (default)", color="#F44336")
            plt.title(f"Distribution by Repayment Status: {col}")
            plt.xlabel(col); plt.ylabel("Density"); plt.legend()
            plt.tight_layout()
            plt.savefig(figs_dir / f"task2_dist_by_target_{col}.png", dpi=140)
            plt.close()

    # ── PCA + KMeans ──────────────────────────────────────────────────────────
    sample_n = min(sample_size, n_rows)
    if sample_n < n_rows:
        sample_idx = df.sample(n=sample_n, random_state=42).index
    else:
        sample_idx = df.index
    sample_df = df.loc[sample_idx, numeric_features].copy()
    sample_df = sample_df.replace([np.inf, -np.inf], np.nan)
    sample_df = sample_df.fillna(sample_df.median()).fillna(0)
    sample_df = sample_df.clip(lower=sample_df.quantile(0.01),
                               upper=sample_df.quantile(0.99), axis=1)
    stable = sample_df.columns[sample_df.nunique() > 1].tolist()
    sample_df = sample_df[stable]

    scaler_pca = StandardScaler()
    x_scaled = scaler_pca.fit_transform(sample_df.values)
    x_scaled = np.nan_to_num(x_scaled)

    pca = PCA(n_components=2, random_state=42)
    x_pca = pca.fit_transform(x_scaled)
    explained = pca.explained_variance_ratio_

    # KMeans k selection
    k_rows = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(x_scaled)
        sil = silhouette_score(x_scaled, labels,
                               sample_size=min(5000, len(labels)), random_state=42) \
              if len(np.unique(labels)) > 1 else np.nan
        k_rows.append({"k": k, "inertia": float(km.inertia_),
                       "silhouette_score": float(sil) if pd.notna(sil) else np.nan})
    pd.DataFrame(k_rows).to_csv(task2_dir / "task2_kmeans_k_selection.csv", index=False)

    # Final KMeans k=3
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(x_scaled)
    dist_centers = kmeans.transform(x_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c=clusters, s=8, alpha=0.6, cmap="tab10")
    plt.title("PCA Projection with KMeans Clusters (k=3)")
    plt.xlabel(f"PC1 ({explained[0]*100:.2f}% var)")
    plt.ylabel(f"PC2 ({explained[1]*100:.2f}% var)")
    plt.tight_layout()
    plt.savefig(figs_dir / "task2_pca_kmeans_clusters.png", dpi=140)
    plt.close()

    # Cluster profiles
    cluster_profile = pd.DataFrame({"CLUSTER_ID": clusters})
    for col in test_cols[:8]:
        if col in sample_df.columns:
            cluster_profile[col] = sample_df[col].values
    if TARGET_COL in df.columns:
        cluster_profile[TARGET_COL] = pd.to_numeric(
            df.loc[sample_idx, TARGET_COL], errors="coerce").values
    cluster_profile.groupby("CLUSTER_ID").mean(numeric_only=True).to_csv(
        task2_dir / "task2_cluster_profiles.csv")

    # Save unsupervised features for Task 3
    unsup_path = task2_dir / "task2_unsupervised_features_sample.csv"
    unsup = pd.DataFrame({
        ID_COL: df.loc[sample_idx, ID_COL].values if ID_COL in df.columns else sample_idx,
        "PCA_1": x_pca[:, 0],
        "PCA_2": x_pca[:, 1],
        "CLUSTER_ID": clusters,
    })
    for i in range(dist_centers.shape[1]):
        unsup[f"DIST_TO_CLUSTER_{i}"] = dist_centers[:, i]
    unsup.to_csv(unsup_path, index=False)

    unsup_summary = {
        "sample_size": int(sample_n),
        "pca_explained_pc1": float(explained[0]),
        "pca_explained_pc2": float(explained[1]),
        "pca_explained_total": float(explained[0] + explained[1]),
        "kmeans_k": 3,
        "cluster_counts": {str(int(k)): int(v)
                           for k, v in pd.Series(clusters).value_counts().sort_index().items()},
        "class_imbalance": class_imbalance,
        "top5_correlations_with_target": corr_to_target.head(5).to_dict(),
    }
    write_json(task2_dir / "task2_eda_unsupervised_summary.json", unsup_summary)

    lines = [
        "Task 2 Summary",
        "==============",
        f"Input:  {input_csv}",
        f"Shape:  {n_rows:,} rows × {n_cols} cols",
        f"Default rate: {class_imbalance.get('default_rate', 'N/A')}",
        f"Imbalance ratio: {class_imbalance.get('imbalance_ratio', 'N/A')}",
        "",
        f"PCA: PC1+PC2 explains {(explained[0]+explained[1])*100:.2f}% of variance",
        f"KMeans k=3 cluster counts: {unsup_summary['cluster_counts']}",
        "",
        "Top 5 correlations with TARGET:",
    ] + [f"  {k}: {v:.4f}" for k, v in corr_to_target.head(5).items()] + [
        "",
        f"Unsupervised features saved: {unsup_path}",
    ]
    (task2_dir / "task2_brief_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Task 2 complete. Outputs: {task2_dir}")
    return unsup_path


# ══════════════════════════════════════════════════════════════════════════════
# TASK 3 – Feature Engineering, Encoding, Scaling, Selection
# ══════════════════════════════════════════════════════════════════════════════
def run_task3(task1_csv: Path, unsup_csv: Optional[Path],
              output_dir: Path, top_k: int,
              corr_threshold: float, use_lgbm: bool) -> None:
    print("\n" + "="*60)
    print("TASK 3 – Feature Engineering & Preprocessing")
    print("="*60)

    task3_dir = output_dir / "task3"
    proc_dir  = task3_dir / "processed"
    figs_dir  = task3_dir / "figures"
    setup_dirs(task3_dir, proc_dir, figs_dir)

    log: Dict = {}

    # ── Step 1: Load & merge ──────────────────────────────────────────────────
    df = pd.read_csv(task1_csv)
    print(f"[INFO] Loaded Task 1 data: {df.shape[0]:,} × {df.shape[1]}")
    log["task1_shape"] = list(df.shape)
    log["unsup_merged"] = False

    if unsup_csv and unsup_csv.exists():
        unsup = pd.read_csv(unsup_csv)
        if ID_COL in unsup.columns:
            unsup_cols = [c for c in unsup.columns if c != ID_COL]
            df = df.merge(unsup[[ID_COL] + unsup_cols], on=ID_COL, how="left")
            print(f"[INFO] Merged {len(unsup_cols)} unsupervised features from Task 2.")
            log["unsup_merged"] = True
            log["unsup_features_added"] = unsup_cols
    else:
        print("[WARN] Task 2 unsupervised features not found; skipping merge.")

    # ── Step 2: Fix anomalies ─────────────────────────────────────────────────
    if "DAYS_EMPLOYED" in df.columns:
        n_anom = int((df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).sum())
        df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).astype(np.int8)
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_SENTINEL, np.nan)
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].fillna(df["DAYS_EMPLOYED"].median())
        print(f"[INFO] Fixed DAYS_EMPLOYED sentinel: {n_anom:,} rows affected.")
        log["days_employed_anomaly_fix"] = n_anom

    # ── Step 3: Engineer features ─────────────────────────────────────────────
    print("[INFO] Engineering domain features...")
    new_features = []

    def add(name, series, desc):
        s = series.replace([np.inf, -np.inf], np.nan)
        med = s.median()
        df[name] = s.fillna(med if pd.notna(med) else 0.0)
        new_features.append(name)
        print(f"[INFO]   + {name}")

    def has(*cols):
        return all(c in df.columns for c in cols)

    if has("AMT_CREDIT", "AMT_INCOME_TOTAL"):
        add("CREDIT_INCOME_RATIO",  safe_ratio(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"]),
            "Loan / income")
    if has("AMT_ANNUITY", "AMT_INCOME_TOTAL"):
        add("ANNUITY_INCOME_RATIO", safe_ratio(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"]),
            "Monthly repayment / income")
    if has("AMT_CREDIT", "AMT_ANNUITY"):
        add("CREDIT_ANNUITY_RATIO", safe_ratio(df["AMT_CREDIT"], df["AMT_ANNUITY"]),
            "Loan term proxy")
    if has("AMT_GOODS_PRICE", "AMT_CREDIT"):
        add("GOODS_CREDIT_RATIO",   safe_ratio(df["AMT_GOODS_PRICE"], df["AMT_CREDIT"]),
            "LTV proxy")
    if has("AMT_CREDIT", "AMT_GOODS_PRICE"):
        add("DOWN_PAYMENT_PROXY",   df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"],
            "Implicit down-payment")
    if has("DAYS_BIRTH"):
        add("AGE_YEARS", (-df["DAYS_BIRTH"] / 365.25).clip(lower=0), "Age in years")
    if has("DAYS_EMPLOYED"):
        add("EMPLOYED_YEARS", (-df["DAYS_EMPLOYED"].clip(upper=0) / 365.25), "Work tenure")
    if has("DAYS_EMPLOYED", "DAYS_BIRTH"):
        add("DAYS_EMPLOYED_PERC", safe_ratio(df["DAYS_EMPLOYED"], df["DAYS_BIRTH"]),
            "Work fraction of life")
    if has("DAYS_BIRTH") and "AGE_YEARS" in df.columns and "CREDIT_INCOME_RATIO" in df.columns:
        add("AGE_CREDIT_INTERACTION", df["AGE_YEARS"] * df["CREDIT_INCOME_RATIO"],
            "Age × credit burden")

    ext_cols = [c for c in ["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"] if c in df.columns]
    if len(ext_cols) >= 2:
        ext = df[ext_cols].apply(pd.to_numeric, errors="coerce")
        add("EXT_SOURCE_MEAN",     ext.mean(axis=1),          "Mean external credit score")
        add("EXT_SOURCE_MIN",      ext.min(axis=1),           "Min external credit score")
        add("EXT_SOURCE_MAX",      ext.max(axis=1),           "Max external credit score")
        add("EXT_SOURCE_STD",      ext.std(axis=1).fillna(0), "Std of external scores")
        add("EXT_SOURCE_PROD",     ext.prod(axis=1),          "Product of external scores")
        if len(ext_cols) == 3:
            add("EXT_SOURCE_WEIGHTED",
                0.5*ext["EXT_SOURCE_2"] + 0.25*ext.get("EXT_SOURCE_1",0) + 0.25*ext.get("EXT_SOURCE_3",0),
                "Weighted external score")

    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    if doc_cols:
        add("DOCUMENT_COUNT", df[doc_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1),
            "Docs provided")

    bureau_mean = next((c for c in df.columns if "BUREAU" in c and "AMT_CREDIT_SUM_MEAN" in c), None)
    bureau_ovd  = next((c for c in df.columns if "BUREAU" in c and "AMT_CREDIT_SUM_OVERDUE" in c), None)
    if bureau_mean and has("AMT_CREDIT"):
        add("BUREAU_CREDIT_RATIO", safe_ratio(df[bureau_mean], df["AMT_CREDIT"]),
            "Bureau credit vs current")
    if bureau_ovd:
        add("BUREAU_OVERDUE_FLAG",
            (pd.to_numeric(df[bureau_ovd], errors="coerce") > 0).astype(np.int8),
            "Any overdue bureau credit")

    prev_mean = next((c for c in df.columns
                      if "PREV" in c and c.endswith("_MEAN") and "AMT_CREDIT" in c), None)
    if prev_mean and has("AMT_CREDIT"):
        add("PREV_CREDIT_RATIO", safe_ratio(pd.to_numeric(df[prev_mean], errors="coerce"),
                                            df["AMT_CREDIT"]), "Prev credit vs current")

    if has("DAYS_LAST_PHONE_CHANGE"):
        add("DAYS_LAST_PHONE_CHANGE_ABS", df["DAYS_LAST_PHONE_CHANGE"].abs(),
            "Days since phone change")
    if has("OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE"):
        obs = pd.to_numeric(df["OBS_30_CNT_SOCIAL_CIRCLE"], errors="coerce").replace(0, np.nan)
        dft = pd.to_numeric(df["DEF_30_CNT_SOCIAL_CIRCLE"], errors="coerce")
        add("SOCIAL_CIRCLE_DEFAULT_RATE", safe_ratio(dft, obs), "Social circle default rate")
    if has("AMT_REQ_CREDIT_BUREAU_YEAR"):
        add("CREDIT_ENQUIRIES_FLAG",
            (pd.to_numeric(df["AMT_REQ_CREDIT_BUREAU_YEAR"], errors="coerce") > 3).astype(np.int8),
            ">3 credit enquiries in past year")

    print(f"[INFO] Total new features engineered: {len(new_features)}")
    log["engineered_features"] = new_features

    # ── Step 4: Encode categoricals ───────────────────────────────────────────
    obj_cols = [c for c in df.select_dtypes(include="object").columns
                if c not in PASSTHROUGH_COLS]
    binary_enc, ohe_enc, dropped_hc = [], [], []
    for col in obj_cols:
        n = df[col].nunique(dropna=False)
        if n <= 2:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            binary_enc.append(col)
        elif n <= 10:
            dummies = pd.get_dummies(df[col].astype(str), prefix=col, drop_first=True, dtype=np.int8)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            ohe_enc.append(col)
        else:
            df = df.drop(columns=[col])
            dropped_hc.append(col)
    print(f"[INFO] Encoding – binary: {len(binary_enc)}, OHE: {len(ohe_enc)}, dropped: {len(dropped_hc)}")
    log["encoding"] = {"binary": binary_enc, "ohe": ohe_enc, "dropped": dropped_hc}

    # ── Step 5: Clip outliers ─────────────────────────────────────────────────
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in PASSTHROUGH_COLS]
    clipped = 0
    for col in num_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.dropna().nunique() <= 2:
            continue
        lo, hi = float(s.quantile(0.01)), float(s.quantile(0.99))
        if lo < hi:
            df[col] = s.clip(lo, hi)
            clipped += 1
    print(f"[INFO] Clipped outliers in {clipped} columns.")
    log["clipped_columns"] = clipped

    # ── Step 6: Scale ─────────────────────────────────────────────────────────
    scale_cols = []
    for col in num_cols:
        if col in df.columns:
            uv = df[col].dropna().unique()
            if not set(uv).issubset({0, 1, 0.0, 1.0}):
                scale_cols.append(col)
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols].astype(float))
    scaler_path = proc_dir / "home_credit_task3_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[INFO] Scaled {len(scale_cols)} features. Scaler saved: {scaler_path}")
    log["scaled_columns"] = len(scale_cols)

    # ── Step 7: Feature selection ─────────────────────────────────────────────
    feat_cols = [c for c in df.columns if c not in PASSTHROUGH_COLS]
    X = df[feat_cols].select_dtypes(include=[np.number])
    kept = X.columns.tolist()

    # Stage A: correlation dedup
    print(f"[INFO] Stage A: correlation dedup (threshold={corr_threshold})...")
    corr_mat = X.corr().abs()
    upper = corr_mat.where(np.triu(np.ones(corr_mat.shape, dtype=bool), k=1))
    drop_corr = [col for col in upper.columns if any(upper[col] >= corr_threshold)]
    kept = [c for c in kept if c not in drop_corr]
    print(f"[INFO]   Dropped {len(drop_corr)} correlated features; {len(kept)} remain.")

    # Stage B: LightGBM importance
    lgbm_imp = {}
    if use_lgbm and TARGET_COL in df.columns:
        try:
            from lightgbm import LGBMClassifier
            print(f"[INFO] Stage B: LightGBM importance (top_k={top_k})...")
            lgbm = LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  class_weight="balanced", random_state=42,
                                  n_jobs=-1, verbose=-1)
            lgbm.fit(df[kept].fillna(0).values, df[TARGET_COL].values)
            lgbm_imp = dict(zip(kept, lgbm.feature_importances_.tolist()))
            if top_k > 0:
                kept = sorted(lgbm_imp, key=lgbm_imp.get, reverse=True)[:top_k]
            print(f"[INFO]   Kept top {len(kept)} features by LightGBM importance.")
        except ImportError:
            print("[WARN] LightGBM not installed; skipping. Run: pip install lightgbm")

    log["feature_selection"] = {
        "dropped_by_correlation": len(drop_corr),
        "final_feature_count": len(kept),
        "lgbm_used": use_lgbm and bool(lgbm_imp),
    }

    # Feature metadata
    meta_rows = []
    for col in X.columns.tolist():
        meta_rows.append({
            "feature": col,
            "selected": col in kept,
            "dropped_corr": col in drop_corr,
            "lgbm_importance": lgbm_imp.get(col, np.nan),
            "variance": float(df[col].var()) if col in df.columns else np.nan,
        })
    meta_df = pd.DataFrame(meta_rows).sort_values("lgbm_importance", ascending=False)

    # ── Step 8: Save outputs ──────────────────────────────────────────────────
    passthrough = [c for c in PASSTHROUGH_COLS if c in df.columns]
    df_out = df[passthrough + kept].copy()
    out_csv = proc_dir / "home_credit_task3_modeling_ready.csv"
    df_out.to_csv(out_csv, index=False)
    meta_df.to_csv(proc_dir / "home_credit_task3_feature_metadata.csv", index=False)
    write_json(task3_dir / "task3_log.json", log)
    print(f"[INFO] Modeling dataset saved: {out_csv}  ({df_out.shape[0]:,} × {df_out.shape[1]})")

    # ── Step 9: Plots ─────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Feature importance
        imp_df = meta_df[meta_df["lgbm_importance"].notna()].head(30)
        if not imp_df.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(imp_df["feature"][::-1], imp_df["lgbm_importance"][::-1], color="#4C72B0")
            ax.set_xlabel("LightGBM Importance (Gain)")
            ax.set_title("Top 30 Features – LightGBM Importance")
            plt.tight_layout()
            fig.savefig(figs_dir / "task3_top30_feature_importance.png", dpi=140)
            plt.close(fig)

        # Distribution plots
        tgt = pd.to_numeric(df_out.get(TARGET_COL, pd.Series()), errors="coerce")
        for col in ["CREDIT_INCOME_RATIO","ANNUITY_INCOME_RATIO","EXT_SOURCE_MEAN",
                    "AGE_YEARS","EMPLOYED_YEARS"]:
            if col not in df_out.columns or tgt.empty:
                continue
            s = pd.to_numeric(df_out[col], errors="coerce").clip(
                df_out[col].quantile(0.01), df_out[col].quantile(0.99))
            g0 = s[tgt==0].dropna().sample(min(15000,int((tgt==0).sum())), random_state=42)
            g1 = s[tgt==1].dropna().sample(min(15000,int((tgt==1).sum())), random_state=42)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(g0, bins=50, alpha=0.55, density=True, label="TARGET=0 (repaid)", color="#2196F3")
            ax.hist(g1, bins=50, alpha=0.55, density=True, label="TARGET=1 (default)", color="#F44336")
            ax.set_xlabel(col); ax.set_ylabel("Density")
            ax.set_title(f"Distribution by Repayment Status: {col}"); ax.legend()
            plt.tight_layout()
            fig.savefig(figs_dir / f"task3_dist_{col}.png", dpi=140)
            plt.close(fig)

        print(f"[INFO] Figures saved: {figs_dir}")
    except Exception as e:
        print(f"[WARN] Plotting failed (non-fatal): {e}")

    # ── Summary report ────────────────────────────────────────────────────────
    lines = [
        "Task 3 Summary – Feature Engineering & Preprocessing",
        "=" * 55,
        f"Input:   {task1_csv}",
        f"Output:  {out_csv}",
        f"Shape:   {df_out.shape[0]:,} rows × {df_out.shape[1]} columns",
        f"Features for Task 4: {len(kept)}",
        "",
        f"Anomaly fix: DAYS_EMPLOYED sentinel replaced in {log.get('days_employed_anomaly_fix',0):,} rows",
        f"New features engineered: {len(new_features)}",
        f"Encoding – binary: {len(binary_enc)}, OHE: {len(ohe_enc)}, dropped: {len(dropped_hc)}",
        f"Outlier clipping: {clipped} columns winsorized (1st–99th pct)",
        f"Scaling: {len(scale_cols)} columns (StandardScaler)",
        f"Feature selection: dropped {len(drop_corr)} correlated, kept top {len(kept)}",
        "",
        "Outputs:",
        f"  {out_csv}        ← use for Task 4",
        f"  {scaler_path}    ← apply to test set in Task 4",
        f"  {proc_dir}/home_credit_task3_feature_metadata.csv",
        f"  {figs_dir}/",
    ]
    (task3_dir / "task3_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[DONE] Task 3 complete.")
    print(f"  Modeling dataset → {out_csv}")
    print(f"  Scaler           → {scaler_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    args = parse_args()
    setup_dirs(args.output_dir)

    task1_csv = args.task1_input
    unsup_csv = None

    if args.task in ("task1", "all"):
        task1_csv = run_task1(args.data_dir, args.output_dir)

    if args.task in ("task2", "all"):
        if not task1_csv.exists():
            raise FileNotFoundError(
                f"Task 1 output not found: {task1_csv}\n"
                "Run with --task all, or provide --task1-input."
            )
        unsup_csv = run_task2(task1_csv, args.output_dir, args.sample_size)

    if args.task == "task3":
        # When running task3 alone, look for task2 output automatically
        default_unsup = args.output_dir / "task2" / "task2_unsupervised_features_sample.csv"
        unsup_csv = default_unsup if default_unsup.exists() else None

    if args.task in ("task3", "all"):
        if not task1_csv.exists():
            raise FileNotFoundError(
                f"Task 1 output not found: {task1_csv}\n"
                "Run with --task all, or provide --task1-input."
            )
        run_task3(
            task1_csv   = task1_csv,
            unsup_csv   = unsup_csv,
            output_dir  = args.output_dir,
            top_k       = args.top_k_features,
            corr_threshold = args.corr_threshold,
            use_lgbm    = not args.no_lgbm,
        )

    print("\n" + "="*60)
    print("Pipeline complete.")
    print("="*60)


if __name__ == "__main__":
    main()
