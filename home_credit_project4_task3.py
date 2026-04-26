#!/usr/bin/env python3
"""
Project 4 - Task 3: Feature Engineering & Preprocessing (Home Credit).

This script consumes the Task 1 cleaned dataset and Task 2 unsupervised features,
then produces a fully engineered, preprocessed modeling-ready dataset for Task 4+.

Pipeline:
  1. Load Task 1 cleaned data + Task 2 unsupervised features
  2. Fix known data anomalies (e.g. DAYS_EMPLOYED sentinel)
  3. Engineer domain-specific features (ratios, interactions, aggregates)
  4. Encode categorical variables (binary label-encode, multi-class one-hot)
  5. Clip outliers (1st–99th percentile) on skewed numerics
  6. Scale numeric features with StandardScaler
  7. Feature selection via LightGBM importance + correlation-based deduplication
  8. Save final dataset + feature metadata + visualisations + summary report

Outputs (all under --output-dir/task3/):
  processed/home_credit_task3_modeling_ready.csv   ← use this in Task 4
  processed/home_credit_task3_scaler.pkl           ← fitted StandardScaler
  processed/home_credit_task3_feature_metadata.csv ← feature provenance & importance
  figures/                                          ← plots
  task3_summary.txt                                 ← human-readable report
  task3_feature_engineering_log.json               ← machine-readable log
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── constants ─────────────────────────────────────────────────────────────────
TARGET_COL = "TARGET"
ID_COL     = "SK_ID_CURR"

# DAYS_EMPLOYED = 365243 is a known sentinel for "not employed / retired".
DAYS_EMPLOYED_SENTINEL = 365243

# Columns skipped during encoding / scaling
PASSTHROUGH_COLS = {TARGET_COL, ID_COL}


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Home Credit Project 4 – Task 3: Feature Engineering")
    p.add_argument("--task1-input", type=Path,
                   default=Path("outputs/processed/home_credit_task1_cleaned.csv"),
                   help="Cleaned dataset from Task 1.")
    p.add_argument("--task2-unsup", type=Path,
                   default=Path("outputs/task2/task2_unsupervised_features_sample.csv"),
                   help="Unsupervised features (PCA / KMeans) from Task 2.")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"),
                   help="Root output directory.")
    p.add_argument("--top-k-features", type=int, default=80,
                   help="Keep at most this many features after selection (0 = keep all).")
    p.add_argument("--corr-threshold", type=float, default=0.95,
                   help="Drop one of any pair of features with |corr| >= this value.")
    p.add_argument("--no-lgbm", action="store_true",
                   help="Skip LightGBM importance step (use if LightGBM not installed).")
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────
def load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"[Task3] {label} not found: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {label}: {df.shape[0]:,} rows × {df.shape[1]} cols  ({path})")
    return df


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """Compute num/den, replacing zeros in denominator and infinities with NaN."""
    d = den.replace(0, np.nan)
    r = num / d
    return r.replace([np.inf, -np.inf], np.nan)


# ── Step 1: load & merge ──────────────────────────────────────────────────────
def load_and_merge(task1_path: Path, unsup_path: Path) -> Tuple[pd.DataFrame, Dict]:
    df = load_csv(task1_path, "Task-1 cleaned dataset")

    log: Dict = {
        "task1_shape": list(df.shape),
        "unsup_merged": False,
        "unsup_features_added": [],
    }

    if unsup_path.exists():
        unsup = load_csv(unsup_path, "Task-2 unsupervised features")
        merge_key = ID_COL if ID_COL in unsup.columns else None
        if merge_key:
            unsup_cols = [c for c in unsup.columns if c != merge_key]
            before = df.shape[1]
            df = df.merge(unsup[[merge_key] + unsup_cols], on=merge_key, how="left")
            added = [c for c in df.columns if c in unsup_cols]
            print(f"[INFO] Merged {len(added)} unsupervised features (PCA_1, PCA_2, CLUSTER_ID, DIST_TO_CLUSTER_*)")
            log["unsup_merged"] = True
            log["unsup_features_added"] = added
            log["shape_after_merge"] = list(df.shape)
    else:
        print(f"[WARN] Task-2 unsup features not found at {unsup_path}; skipping merge.")

    return df, log


# ── Step 2: fix known anomalies ───────────────────────────────────────────────
def fix_anomalies(df: pd.DataFrame, log: Dict) -> pd.DataFrame:
    """
    DAYS_EMPLOYED = 365243 is a documented sentinel meaning 'not employed'.
    Replace with NaN, then re-impute with median so downstream models don't
    treat it as a very large employment tenure.
    Add a binary flag to preserve the anomaly as a feature.
    """
    anomaly_fixes: List[str] = []

    if "DAYS_EMPLOYED" in df.columns:
        n_anomalous = int((df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).sum())
        df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).astype(np.int8)
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_SENTINEL, np.nan)
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].fillna(df["DAYS_EMPLOYED"].median())
        anomaly_fixes.append(
            f"DAYS_EMPLOYED: {n_anomalous} sentinel values (365243) replaced with median; "
            "DAYS_EMPLOYED_ANOM flag added."
        )
        print(f"[INFO] Fixed DAYS_EMPLOYED sentinel: {n_anomalous:,} rows affected.")

    log["anomaly_fixes"] = anomaly_fixes
    return df


# ── Step 3: domain feature engineering ───────────────────────────────────────
def engineer_features(df: pd.DataFrame, log: Dict) -> pd.DataFrame:
    """
    Create credit-risk domain features. Each feature is guarded so it is only
    added when its source columns are present, making the script robust to
    datasets that include only application_train without the optional tables.
    """
    new_features: List[Dict] = []

    def add(name: str, series: pd.Series, description: str) -> None:
        """Safely add a feature, filling NaN with median."""
        s = series.replace([np.inf, -np.inf], np.nan)
        median = s.median()
        df[name] = s.fillna(median if pd.notna(median) else 0.0)
        new_features.append({"feature": name, "description": description, "source": "engineered"})
        print(f"[INFO]   + {name}")

    def cols_exist(*names: str) -> bool:
        return all(c in df.columns for c in names)

    print("[INFO] Engineering domain features...")

    # ── Credit load ratios ────────────────────────────────────────────────────
    if cols_exist("AMT_CREDIT", "AMT_INCOME_TOTAL"):
        add("CREDIT_INCOME_RATIO",
            safe_ratio(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"]),
            "Loan amount relative to annual income – higher = heavier debt load.")

    if cols_exist("AMT_ANNUITY", "AMT_INCOME_TOTAL"):
        add("ANNUITY_INCOME_RATIO",
            safe_ratio(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"]),
            "Monthly repayment burden as a fraction of income (debt-service ratio).")

    if cols_exist("AMT_CREDIT", "AMT_ANNUITY"):
        add("CREDIT_ANNUITY_RATIO",
            safe_ratio(df["AMT_CREDIT"], df["AMT_ANNUITY"]),
            "Loan term proxy: how many monthly payments to repay the loan.")

    if cols_exist("AMT_GOODS_PRICE", "AMT_CREDIT"):
        add("GOODS_CREDIT_RATIO",
            safe_ratio(df["AMT_GOODS_PRICE"], df["AMT_CREDIT"]),
            "LTV proxy: goods price / credit amount; <1 means goods cost less than credit.")

    if cols_exist("AMT_CREDIT", "AMT_GOODS_PRICE"):
        add("DOWN_PAYMENT_PROXY",
            df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"],
            "Difference between credit granted and goods price (implicit down-payment).")

    # ── Age & tenure ──────────────────────────────────────────────────────────
    if cols_exist("DAYS_BIRTH"):
        add("AGE_YEARS",
            (-df["DAYS_BIRTH"] / 365.25).clip(lower=0),
            "Applicant age in years (DAYS_BIRTH is stored as negative).")

    if cols_exist("DAYS_EMPLOYED"):
        add("EMPLOYED_YEARS",
            (-df["DAYS_EMPLOYED"].clip(upper=0) / 365.25),
            "Employment tenure in years (only for employed applicants).")

    if cols_exist("DAYS_EMPLOYED", "DAYS_BIRTH"):
        add("DAYS_EMPLOYED_PERC",
            safe_ratio(df["DAYS_EMPLOYED"], df["DAYS_BIRTH"]),
            "Employment fraction of life: how long applicant has been employed relative to age.")

    if cols_exist("DAYS_BIRTH") and "AGE_YEARS" in df.columns:
        add("AGE_CREDIT_INTERACTION",
            df["AGE_YEARS"] * df.get("CREDIT_INCOME_RATIO", pd.Series(0, index=df.index)),
            "Interaction: older applicants with high credit load may have different risk profile.")

    # ── External credit scores (EXT_SOURCE_1/2/3) ────────────────────────────
    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
    if len(ext_cols) >= 2:
        ext = df[ext_cols].apply(pd.to_numeric, errors="coerce")
        add("EXT_SOURCE_MEAN",
            ext.mean(axis=1),
            "Mean of available external credit scores – strong single predictor in literature.")
        add("EXT_SOURCE_MIN",
            ext.min(axis=1),
            "Minimum external credit score – captures worst-case bureau assessment.")
        add("EXT_SOURCE_MAX",
            ext.max(axis=1),
            "Maximum external credit score.")
        add("EXT_SOURCE_STD",
            ext.std(axis=1).fillna(0),
            "Std dev of external scores – disagreement between bureaus may indicate risk.")
        add("EXT_SOURCE_PROD",
            ext.prod(axis=1),
            "Product of external scores – jointly-low scores amplify the risk signal.")
        if len(ext_cols) == 3:
            add("EXT_SOURCE_WEIGHTED",
                (0.5 * ext["EXT_SOURCE_2"] + 0.25 * ext.get("EXT_SOURCE_1", 0)
                 + 0.25 * ext.get("EXT_SOURCE_3", 0)),
                "Weighted combo (EXT_SOURCE_2 carries most weight per competition insights).")

    # ── Document completeness ─────────────────────────────────────────────────
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    if doc_cols:
        add("DOCUMENT_COUNT",
            df[doc_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1),
            "Total number of documents provided – more documents may signal diligence.")

    # ── Bureau / previous application aggregates (if merged in Task 1) ───────
    bureau_amt_mean = next((c for c in df.columns if "BUREAU" in c and "AMT_CREDIT_SUM_MEAN" in c), None)
    bureau_overdue  = next((c for c in df.columns if "BUREAU" in c and "AMT_CREDIT_SUM_OVERDUE" in c), None)

    if bureau_amt_mean and cols_exist("AMT_CREDIT"):
        add("BUREAU_CREDIT_RATIO",
            safe_ratio(df[bureau_amt_mean], df["AMT_CREDIT"]),
            "Ratio of historical bureau credit to current application credit.")

    if bureau_overdue:
        add("BUREAU_OVERDUE_FLAG",
            (pd.to_numeric(df[bureau_overdue], errors="coerce") > 0).astype(np.int8),
            "Binary flag: applicant has any overdue bureau credit amount.")

    prev_count = next((c for c in df.columns if "PREV" in c and "SK_ID_PREV_COUNT" in c.upper()
                        or ("PREV" in c and c.endswith("_MEAN") and "AMT_CREDIT" in c)), None)
    if prev_count and cols_exist("AMT_CREDIT"):
        add("PREV_CREDIT_RATIO",
            safe_ratio(pd.to_numeric(df[prev_count], errors="coerce"), df["AMT_CREDIT"]),
            "Previous application credit amounts relative to current credit.")

    # ── Application meta-features ─────────────────────────────────────────────
    if cols_exist("DAYS_LAST_PHONE_CHANGE"):
        add("DAYS_LAST_PHONE_CHANGE_ABS",
            df["DAYS_LAST_PHONE_CHANGE"].abs(),
            "Absolute days since last phone change – recent changes may indicate instability.")

    if cols_exist("OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE"):
        obs = pd.to_numeric(df["OBS_30_CNT_SOCIAL_CIRCLE"], errors="coerce").replace(0, np.nan)
        dft = pd.to_numeric(df["DEF_30_CNT_SOCIAL_CIRCLE"], errors="coerce")
        add("SOCIAL_CIRCLE_DEFAULT_RATE",
            safe_ratio(dft, obs),
            "Fraction of social circle that defaulted within 30 days – peer-effect signal.")

    if cols_exist("AMT_REQ_CREDIT_BUREAU_YEAR"):
        add("CREDIT_ENQUIRIES_FLAG",
            (pd.to_numeric(df["AMT_REQ_CREDIT_BUREAU_YEAR"], errors="coerce") > 3).astype(np.int8),
            "Flag: more than 3 credit enquiries in past year – may indicate credit-seeking.")

    log["engineered_features"] = new_features
    log["n_engineered"] = len(new_features)
    print(f"[INFO] Total new features engineered: {len(new_features)}")
    return df


# ── Step 4: encode categoricals ───────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame, log: Dict) -> pd.DataFrame:
    """
    - Binary categoricals (2 unique values)  → LabelEncoder (0/1)
    - Low-cardinality multi-class (3–10)     → One-Hot Encoding (drop_first=True)
    - High-cardinality (>10)                 → Drop (too many dummies, risk of sparsity)
    UNKNOWN values introduced in Task 1 are treated as a valid category level.
    """
    from sklearn.preprocessing import LabelEncoder

    obj_cols = [c for c in df.select_dtypes(include="object").columns
                if c not in PASSTHROUGH_COLS]

    binary_encoded: List[str] = []
    ohe_encoded: List[str]    = []
    dropped_high_card: List[str] = []

    for col in obj_cols:
        n_unique = df[col].nunique(dropna=False)

        if n_unique <= 2:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            binary_encoded.append(col)

        elif n_unique <= 10:
            dummies = pd.get_dummies(df[col].astype(str), prefix=col, drop_first=True, dtype=np.int8)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            ohe_encoded.append(col)

        else:
            df = df.drop(columns=[col])
            dropped_high_card.append(col)

    print(f"[INFO] Encoding – binary: {len(binary_encoded)}, OHE: {len(ohe_encoded)}, "
          f"dropped (high-card): {len(dropped_high_card)}")
    log["encoding"] = {
        "binary_label_encoded": binary_encoded,
        "one_hot_encoded": ohe_encoded,
        "dropped_high_cardinality": dropped_high_card,
    }
    return df


# ── Step 5: clip outliers ─────────────────────────────────────────────────────
def clip_outliers(df: pd.DataFrame, log: Dict) -> pd.DataFrame:
    """
    Winsorize numeric features to [1st, 99th] percentile.
    Skips TARGET, ID, and binary (0/1 only) columns.
    """
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in PASSTHROUGH_COLS]

    clipped_count = 0
    for col in num_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        unique_vals = s.dropna().unique()
        if len(unique_vals) <= 2:   # binary flag – skip
            continue
        lo, hi = float(s.quantile(0.01)), float(s.quantile(0.99))
        if lo < hi:
            df[col] = s.clip(lower=lo, upper=hi)
            clipped_count += 1

    print(f"[INFO] Clipped outliers in {clipped_count} numeric columns (1st–99th pct).")
    log["outlier_clipping"] = {
        "method": "winsorize 1st-99th percentile",
        "columns_clipped": clipped_count,
    }
    return df


# ── Step 6: scale ─────────────────────────────────────────────────────────────
def scale_features(
    df: pd.DataFrame,
    log: Dict,
    scaler_path: Path,
) -> Tuple[pd.DataFrame, object]:
    """
    StandardScaler applied to all numeric non-target/non-id columns.
    Saves fitted scaler for use in Task 4 (train/test consistency).
    Binary columns (0/1 only) are excluded from scaling – they are already
    on the right scale and scaling would add noise.
    """
    import pickle
    from sklearn.preprocessing import StandardScaler

    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in PASSTHROUGH_COLS]

    # Exclude binary (0/1) columns from scaling
    scale_cols = []
    for col in num_cols:
        unique_vals = df[col].dropna().unique()
        if not set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            scale_cols.append(col)

    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols].astype(float))

    # Persist fitted scaler
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"[INFO] Scaled {len(scale_cols)} numeric features with StandardScaler.")
    print(f"[INFO] Scaler saved to: {scaler_path}")
    log["scaling"] = {
        "method": "StandardScaler (zero mean, unit variance)",
        "columns_scaled": len(scale_cols),
        "scaler_path": str(scaler_path),
    }
    return df, scaler


# ── Step 7: feature selection ─────────────────────────────────────────────────
def select_features(
    df: pd.DataFrame,
    log: Dict,
    top_k: int,
    corr_threshold: float,
    use_lgbm: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Two-stage selection:
      Stage A – Correlation deduplication: for any pair |corr| >= threshold, drop
                the one with lower variance.
      Stage B – LightGBM importance ranking: keep top-k by gain importance.
                If LightGBM is unavailable or --no-lgbm is set, skip Stage B.
    Returns (filtered_df, feature_metadata_df).
    """
    feature_cols = [c for c in df.columns if c not in PASSTHROUGH_COLS]
    X = df[feature_cols].select_dtypes(include=[np.number])
    kept_cols = X.columns.tolist()

    # Stage A: correlation deduplication
    print(f"[INFO] Feature selection – Stage A: correlation dedup (threshold={corr_threshold})...")
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
    to_drop_corr: List[str] = []
    for col in upper.columns:
        if any(upper[col] >= corr_threshold):
            to_drop_corr.append(col)
    kept_cols = [c for c in kept_cols if c not in to_drop_corr]
    print(f"[INFO]   Dropped {len(to_drop_corr)} highly correlated features; {len(kept_cols)} remain.")

    # Stage B: LightGBM importance
    lgbm_importances: Dict[str, float] = {}
    if use_lgbm and TARGET_COL in df.columns:
        try:
            from lightgbm import LGBMClassifier
            print(f"[INFO] Feature selection – Stage B: LightGBM importance (top_k={top_k})...")
            X_sel = df[kept_cols].fillna(0).values
            y     = df[TARGET_COL].values
            lgbm  = LGBMClassifier(
                n_estimators=200,
                num_leaves=31,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            lgbm.fit(X_sel, y)
            lgbm_importances = dict(zip(kept_cols, lgbm.feature_importances_.tolist()))
            if top_k > 0:
                sorted_features = sorted(lgbm_importances, key=lgbm_importances.get, reverse=True)
                kept_cols = sorted_features[:top_k]
                print(f"[INFO]   Kept top {len(kept_cols)} features by LightGBM gain importance.")
        except ImportError:
            print("[WARN] LightGBM not installed; skipping importance-based selection. "
                  "Install with: pip install lightgbm")

    # Build feature metadata table
    all_num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    meta_rows = []
    for col in all_num_cols:
        meta_rows.append({
            "feature": col,
            "selected": col in kept_cols,
            "dropped_corr": col in to_drop_corr,
            "lgbm_importance": lgbm_importances.get(col, np.nan),
            "variance": float(df[col].var()),
            "missing_rate": float(df[col].isna().mean()),
        })
    meta_df = pd.DataFrame(meta_rows).sort_values("lgbm_importance", ascending=False)

    log["feature_selection"] = {
        "stage_a_corr_threshold": corr_threshold,
        "dropped_by_correlation": len(to_drop_corr),
        "stage_b_lgbm_top_k": top_k,
        "final_feature_count": len(kept_cols),
    }

    # Keep passthrough cols + selected features
    passthrough_present = [c for c in PASSTHROUGH_COLS if c in df.columns]
    df_selected = df[passthrough_present + kept_cols].copy()
    return df_selected, meta_df


# ── Step 8: visualisations ────────────────────────────────────────────────────
def make_plots(df: pd.DataFrame, meta_df: pd.DataFrame, figs_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str((figs_dir.parent / ".mplconfig").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── 1. Top-30 feature importances ────────────────────────────────────────
    imp = meta_df[meta_df["lgbm_importance"].notna()].head(30).copy()
    if not imp.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(imp["feature"][::-1], imp["lgbm_importance"][::-1], color="#4C72B0")
        ax.set_xlabel("LightGBM Feature Importance (Gain)", fontsize=11)
        ax.set_title("Top 30 Selected Features – LightGBM Importance", fontsize=13)
        plt.tight_layout()
        fig.savefig(figs_dir / "task3_top30_feature_importance.png", dpi=140)
        plt.close(fig)

    # ── 2. Engineered feature distributions vs TARGET ─────────────────────────
    engineered_to_plot = [
        c for c in ["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "EXT_SOURCE_MEAN",
                    "AGE_YEARS", "EMPLOYED_YEARS", "PAYMENT_RATE", "EXT_SOURCE_PROD"]
        if c in df.columns and TARGET_COL in df.columns
    ]
    for col in engineered_to_plot[:5]:
        fig, ax = plt.subplots(figsize=(7, 4))
        s = pd.to_numeric(df[col], errors="coerce")
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        s = s.clip(lo, hi)
        tgt = pd.to_numeric(df[TARGET_COL], errors="coerce")
        g0 = s[tgt == 0].dropna().sample(min(15_000, int((tgt == 0).sum())), random_state=42)
        g1 = s[tgt == 1].dropna().sample(min(15_000, int((tgt == 1).sum())), random_state=42)
        ax.hist(g0, bins=50, alpha=0.55, density=True, label="TARGET=0 (repaid)", color="#2196F3")
        ax.hist(g1, bins=50, alpha=0.55, density=True, label="TARGET=1 (default)", color="#F44336")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution by Repayment Status: {col}")
        ax.legend()
        plt.tight_layout()
        fig.savefig(figs_dir / f"task3_dist_{col}.png", dpi=140)
        plt.close(fig)

    # ── 3. Cluster vs default rate (if CLUSTER_ID present) ────────────────────
    if "CLUSTER_ID" in df.columns and TARGET_COL in df.columns:
        cluster_rates = (
            df.groupby("CLUSTER_ID")[TARGET_COL]
            .agg(["mean", "count"])
            .rename(columns={"mean": "default_rate", "count": "n"})
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(cluster_rates["CLUSTER_ID"].astype(str),
                      cluster_rates["default_rate"], color="#FF9800")
        ax.set_xlabel("KMeans Cluster ID")
        ax.set_ylabel("Default Rate")
        ax.set_title("Default Rate by KMeans Cluster (from Task 2 Unsupervised Features)")
        for bar, (_, row) in zip(bars, cluster_rates.iterrows()):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"n={int(row['n']):,}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        fig.savefig(figs_dir / "task3_cluster_default_rate.png", dpi=140)
        plt.close(fig)

    # ── 4. Correlation heatmap of top-15 selected features ────────────────────
    top_feats = meta_df[meta_df["selected"]].head(15)["feature"].tolist()
    top_feats = [c for c in top_feats if c in df.columns]
    if len(top_feats) >= 4:
        fig, ax = plt.subplots(figsize=(11, 9))
        corr = df[top_feats].corr()
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(top_feats)))
        ax.set_yticks(range(len(top_feats)))
        ax.set_xticklabels(top_feats, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(top_feats, fontsize=7)
        ax.set_title("Correlation Matrix – Top 15 Selected Features")
        plt.tight_layout()
        fig.savefig(figs_dir / "task3_top15_correlation_heatmap.png", dpi=140)
        plt.close(fig)

    print(f"[INFO] Figures saved to: {figs_dir}")


# ── Step 9: summary report ────────────────────────────────────────────────────
def write_summary(df: pd.DataFrame, log: Dict, task3_dir: Path) -> None:
    n_rows, n_cols = df.shape
    n_features = n_cols - len([c for c in PASSTHROUGH_COLS if c in df.columns])

    lines = [
        "Project 4 – Task 3: Feature Engineering & Preprocessing Summary",
        "=" * 65,
        "",
        f"Final modeling dataset shape: {n_rows:,} rows × {n_cols} columns",
        f"Features available for Task 4 modeling: {n_features}",
        "",
        "── Step 1: Data Loading ─────────────────────────────────────────",
        f"  Task-1 input shape:         {log['task1_shape'][0]:,} × {log['task1_shape'][1]}",
        f"  Unsupervised features merged: {log['unsup_merged']}",
        f"  Unsupervised features added: {', '.join(log['unsup_features_added']) or 'none'}",
        "",
        "── Step 2: Anomaly Fixes ────────────────────────────────────────",
    ]
    for fix in log.get("anomaly_fixes", []):
        lines.append(f"  • {fix}")

    lines += [
        "",
        "── Step 3: Domain Feature Engineering ──────────────────────────",
        f"  New features created: {log.get('n_engineered', 0)}",
    ]
    for feat in log.get("engineered_features", []):
        lines.append(f"  • {feat['feature']}: {feat['description']}")

    enc = log.get("encoding", {})
    lines += [
        "",
        "── Step 4: Categorical Encoding ─────────────────────────────────",
        f"  Binary label-encoded:  {len(enc.get('binary_label_encoded', []))} columns",
        f"  One-hot encoded:       {len(enc.get('one_hot_encoded', []))} columns",
        f"  Dropped (high-card>10):{len(enc.get('dropped_high_cardinality', []))} columns",
        "",
        "── Step 5: Outlier Clipping ─────────────────────────────────────",
        f"  Method: Winsorize (1st–99th percentile)",
        f"  Columns clipped: {log.get('outlier_clipping', {}).get('columns_clipped', 0)}",
        "",
        "── Step 6: Scaling ──────────────────────────────────────────────",
        f"  Method: StandardScaler (mean=0, std=1)",
        f"  Columns scaled: {log.get('scaling', {}).get('columns_scaled', 0)}",
        f"  Scaler saved to: {log.get('scaling', {}).get('scaler_path', 'N/A')}",
        "",
        "── Step 7: Feature Selection ─────────────────────────────────────",
        f"  Stage A – Correlation dedup (|r| ≥ {log['feature_selection']['stage_a_corr_threshold']}): "
        f"dropped {log['feature_selection']['dropped_by_correlation']}",
        f"  Stage B – LightGBM top-k (k={log['feature_selection']['stage_b_lgbm_top_k']}): "
        f"kept {log['feature_selection']['final_feature_count']}",
        "",
        "── Outputs ──────────────────────────────────────────────────────",
        f"  Modeling dataset:   outputs/task3/processed/home_credit_task3_modeling_ready.csv",
        f"  Scaler:             outputs/task3/processed/home_credit_task3_scaler.pkl",
        f"  Feature metadata:   outputs/task3/processed/home_credit_task3_feature_metadata.csv",
        f"  Figures:            outputs/task3/figures/",
        f"  Log (JSON):         outputs/task3/task3_feature_engineering_log.json",
        "",
        "── Rationale for Key Decisions ──────────────────────────────────",
        "  • Ratio features (CREDIT_INCOME_RATIO etc.) capture relative burden,",
        "    which is more predictive than raw amounts across applicants of",
        "    different income levels.",
        "  • EXT_SOURCE features are the strongest predictors in this dataset;",
        "    their mean, product, and std capture cooperative and discordant",
        "    bureau assessments simultaneously.",
        "  • DAYS_EMPLOYED sentinel (365243) replaced with median + binary flag,",
        "    preserving the anomaly as an informative signal.",
        "  • Winsorization avoids distorting StandardScaler with extreme values.",
        "  • Correlation dedup removes near-redundant features before LightGBM",
        "    importance ranking, preventing importance dilution.",
        "  • Scaler fitted on training data only (here: full cleaned dataset);",
        "    in Task 4, apply the same scaler to the test split without refitting.",
    ]

    report_path = task3_dir / "task3_summary.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Summary report saved to: {report_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # ── output directory setup ─────────────────────────────────────────────
    task3_dir   = args.output_dir / "task3"
    proc_dir    = task3_dir / "processed"
    figs_dir    = task3_dir / "figures"
    for d in [task3_dir, proc_dir, figs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    log: Dict = {}

    # Step 1: load & merge
    df, load_log = load_and_merge(args.task1_input, args.task2_unsup)
    log.update(load_log)

    # Step 2: fix anomalies
    df = fix_anomalies(df, log)

    # Step 3: engineer features
    df = engineer_features(df, log)

    # Step 4: encode categoricals
    df = encode_categoricals(df, log)

    # Step 5: clip outliers
    df = clip_outliers(df, log)

    # Step 6: scale
    scaler_path = proc_dir / "home_credit_task3_scaler.pkl"
    df, _ = scale_features(df, log, scaler_path)

    # Step 7: feature selection
    df, meta_df = select_features(
        df,
        log,
        top_k=args.top_k_features,
        corr_threshold=args.corr_threshold,
        use_lgbm=(not args.no_lgbm),
    )

    # Step 8: save outputs
    out_csv = proc_dir / "home_credit_task3_modeling_ready.csv"
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Modeling-ready dataset saved: {out_csv}  "
          f"({df.shape[0]:,} rows × {df.shape[1]} cols)")

    meta_path = proc_dir / "home_credit_task3_feature_metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    print(f"[INFO] Feature metadata saved: {meta_path}")

    write_json(task3_dir / "task3_feature_engineering_log.json", log)

    # Step 9: plots
    try:
        make_plots(df, meta_df, figs_dir)
    except Exception as e:
        print(f"[WARN] Plotting failed (non-fatal): {e}")

    # Step 10: summary
    write_summary(df, log, task3_dir)

    print("\n[DONE] Task 3 complete.")
    print(f"  Modeling dataset → {out_csv}")
    print(f"  Scaler           → {scaler_path}")
    print(f"  Feature metadata → {meta_path}")
    print(f"  Figures          → {figs_dir}")
    print(f"  Summary          → {task3_dir / 'task3_summary.txt'}")


if __name__ == "__main__":
    main()