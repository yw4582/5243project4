import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TARGET_COL = "TARGET"
ID_COL     = "SK_ID_CURR"


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path,
                   default=Path("outputs/task3/processed/home_credit_task3_modeling_ready.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/task4"))
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────
def setup_dirs(base: Path):
    for d in [base, base / "figures", base / "models"]:
        d.mkdir(parents=True, exist_ok=True)


def load_data(path: Path):
    print(f"[INFO] Loading dataset: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    drop_cols = [c for c in [TARGET_COL, ID_COL] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[TARGET_COL]
    # Fill any remaining NaN with column median (safety net)
    nan_cols = X.columns[X.isna().any()].tolist()
    if nan_cols:
        print(f"[INFO] Filling NaN in {len(nan_cols)} columns with median.")
        X = X.fillna(X.median())
    print(f"[INFO] Features: {X.shape[1]}  |  Default rate: {y.mean():.2%}")
    return X, y


# ── model definitions ─────────────────────────────────────────────────────────
def get_models():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    try:
        from xgboost import XGBClassifier
        scale_pos_weight = 11  # ~ratio of negative to positive class
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric="auc",
            verbosity=0,
        )
        print("[INFO] XGBoost loaded.")
    except ImportError:
        print("[WARN] XGBoost not installed; skipping. pip install xgboost")

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300,
            num_leaves=63,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        print("[INFO] LightGBM loaded.")
    except ImportError:
        print("[WARN] LightGBM not installed; skipping. pip install lightgbm")

    return models


# ── cross-validation ──────────────────────────────────────────────────────────
def cross_validate_models(models, X_train, y_train, cv_folds, random_state):
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import make_scorer, f1_score

    print(f"\n[INFO] Running {cv_folds}-fold stratified cross-validation...")
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_results = {}

    for name, model in models.items():
        print(f"[INFO]   {name}...")
        auc_scores = cross_val_score(model, X_train, y_train, cv=skf,
                                     scoring="roc_auc", n_jobs=-1)
        f1_scores  = cross_val_score(model, X_train, y_train, cv=skf,
                                     scoring=make_scorer(f1_score), n_jobs=-1)
        cv_results[name] = {
            "cv_auc_mean":  float(auc_scores.mean()),
            "cv_auc_std":   float(auc_scores.std()),
            "cv_f1_mean":   float(f1_scores.mean()),
            "cv_f1_std":    float(f1_scores.std()),
        }
        print(f"         AUC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}  "
              f"F1: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")

    return cv_results


# ── train & evaluate on test set ──────────────────────────────────────────────
def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    from sklearn.metrics import (
        roc_auc_score, f1_score, precision_score, recall_score,
        average_precision_score, confusion_matrix, classification_report
    )

    print("\n[INFO] Training models on full training set and evaluating on test set...")
    results = {}
    trained_models = {}
    predictions = {}

    for name, model in models.items():
        print(f"[INFO]   Training {name}...")
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        metrics = {
            "test_auc":       float(roc_auc_score(y_test, y_prob)),
            "test_f1":        float(f1_score(y_test, y_pred)),
            "test_precision": float(precision_score(y_test, y_pred)),
            "test_recall":    float(recall_score(y_test, y_pred)),
            "test_pr_auc":    float(average_precision_score(y_test, y_prob)),
        }
        results[name]       = metrics
        trained_models[name] = model
        predictions[name]   = {"y_prob": y_prob, "y_pred": y_pred}

        print(f"         AUC={metrics['test_auc']:.4f}  "
              f"F1={metrics['test_f1']:.4f}  "
              f"Precision={metrics['test_precision']:.4f}  "
              f"Recall={metrics['test_recall']:.4f}  "
              f"PR-AUC={metrics['test_pr_auc']:.4f}")

    return results, trained_models, predictions


# ── plots ─────────────────────────────────────────────────────────────────────
def make_plots(models_dict, predictions, y_test, test_results, cv_results,
               feature_names, figs_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
    )

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0"]

    # ── 1. ROC Curves ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, preds), color in zip(predictions.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, preds["y_prob"])
        auc = test_results[name]["test_auc"]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", color=color, lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – All Models")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(figs_dir / "task4_roc_curves.png", dpi=140)
    plt.close(fig)

    # ── 2. Precision-Recall Curves ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, preds), color in zip(predictions.items(), colors):
        prec, rec, _ = precision_recall_curve(y_test, preds["y_prob"])
        pr_auc = test_results[name]["test_pr_auc"]
        ax.plot(rec, prec, label=f"{name} (PR-AUC={pr_auc:.4f})", color=color, lw=2)
    ax.axhline(y=y_test.mean(), color="k", linestyle="--", lw=1, label="Baseline (random)")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves – All Models")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(figs_dir / "task4_pr_curves.png", dpi=140)
    plt.close(fig)

    # ── 3. Metrics Comparison Bar Chart ───────────────────────────────────────
    model_names = list(test_results.keys())
    metrics_to_plot = ["test_auc", "test_f1", "test_precision", "test_recall", "test_pr_auc"]
    metric_labels   = ["AUC", "F1", "Precision", "Recall", "PR-AUC"]

    x = np.arange(len(model_names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        vals = [test_results[m][metric] for m in model_names]
        bars = ax.bar(x + i * width, vals, width, label=label, color=colors[i])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score"); ax.set_title("Model Performance Comparison")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(figs_dir / "task4_metrics_comparison.png", dpi=140)
    plt.close(fig)

    # ── 4. CV AUC Comparison ──────────────────────────────────────────────────
    cv_names = list(cv_results.keys())
    cv_means = [cv_results[n]["cv_auc_mean"] for n in cv_names]
    cv_stds  = [cv_results[n]["cv_auc_std"]  for n in cv_names]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(cv_names, cv_means, yerr=cv_stds, capsize=5,
                  color=colors[:len(cv_names)], alpha=0.85)
    for bar, v in zip(bars, cv_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("CV AUC (mean ± std)")
    ax.set_title("Cross-Validation AUC Comparison")
    plt.tight_layout()
    fig.savefig(figs_dir / "task4_cv_auc_comparison.png", dpi=140)
    plt.close(fig)

    # ── 5. Confusion Matrices ─────────────────────────────────────────────────
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]
    for ax, (name, preds) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, preds["y_pred"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=["Repaid", "Default"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=10)
    plt.suptitle("Confusion Matrices", fontsize=12)
    plt.tight_layout()
    fig.savefig(figs_dir / "task4_confusion_matrices.png", dpi=140)
    plt.close(fig)

    # ── 6. Feature Importance (tree models only) ──────────────────────────────
    for name, model in models_dict.items():
        if not hasattr(model, "feature_importances_"):
            continue
        imp = pd.Series(model.feature_importances_, index=feature_names)
        top30 = imp.nlargest(30)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(top30.index[::-1], top30.values[::-1], color="#4C72B0")
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Top 30 Feature Importances – {name}")
        plt.tight_layout()
        fname = name.lower().replace(" ", "_")
        fig.savefig(figs_dir / f"task4_feature_importance_{fname}.png", dpi=140)
        plt.close(fig)

    print(f"[INFO] Figures saved to: {figs_dir}")


# ── select best model ─────────────────────────────────────────────────────────
def select_best_model(test_results, cv_results):
    """
    Select best model based on a weighted score:
      60% test AUC + 20% CV AUC + 20% PR-AUC
    AUC is the primary metric for imbalanced classification.
    """
    scores = {}
    for name in test_results:
        test_auc = test_results[name]["test_auc"]
        pr_auc   = test_results[name]["test_pr_auc"]
        cv_auc   = cv_results.get(name, {}).get("cv_auc_mean", test_auc)
        scores[name] = 0.6 * test_auc + 0.2 * cv_auc + 0.2 * pr_auc

    best = max(scores, key=scores.get)
    print(f"\n[INFO] Best model: {best} (weighted score: {scores[best]:.4f})")
    return best, scores


# ── summary report ────────────────────────────────────────────────────────────
def write_summary(test_results, cv_results, best_model, selection_scores,
                  X_train, X_test, y_train, y_test, out_dir):
    lines = [
        "Project 4 – Task 4: Supervised Modeling Summary",
        "=" * 55,
        "",
        f"Training set: {X_train.shape[0]:,} rows",
        f"Test set:     {X_test.shape[0]:,} rows",
        f"Features:     {X_train.shape[1]}",
        f"Default rate (train): {y_train.mean():.2%}",
        f"Default rate (test):  {y_test.mean():.2%}",
        "",
        "── Cross-Validation Results (5-fold Stratified) ──────────",
    ]
    for name, res in cv_results.items():
        lines.append(f"  {name}:")
        lines.append(f"    CV AUC: {res['cv_auc_mean']:.4f} ± {res['cv_auc_std']:.4f}")
        lines.append(f"    CV F1:  {res['cv_f1_mean']:.4f} ± {res['cv_f1_std']:.4f}")

    lines += ["", "── Test Set Results ──────────────────────────────────────"]
    for name, res in test_results.items():
        lines.append(f"  {name}:")
        lines.append(f"    AUC:       {res['test_auc']:.4f}")
        lines.append(f"    F1:        {res['test_f1']:.4f}")
        lines.append(f"    Precision: {res['test_precision']:.4f}")
        lines.append(f"    Recall:    {res['test_recall']:.4f}")
        lines.append(f"    PR-AUC:    {res['test_pr_auc']:.4f}")

    lines += [
        "",
        "── Model Selection ───────────────────────────────────────",
        f"  Selected model: {best_model}",
        f"  Selection criterion: 60% test AUC + 20% CV AUC + 20% PR-AUC",
        "",
        "  Weighted scores:",
    ]
    for name, score in sorted(selection_scores.items(), key=lambda x: -x[1]):
        lines.append(f"    {name}: {score:.4f}")

    lines += [
        "",
        "── Rationale ─────────────────────────────────────────────",
        "  Primary metric is AUC because the dataset is imbalanced",
        "  (default rate ~8%). Accuracy would be misleading since a",
        "  model predicting 'no default' always gets 92% accuracy.",
        "  PR-AUC is included as a secondary metric because it is",
        "  more sensitive to performance on the minority class.",
        "  CV AUC ensures the model generalises and is not overfit.",
        "",
        "── Outputs ───────────────────────────────────────────────",
        "  figures/task4_roc_curves.png",
        "  figures/task4_pr_curves.png",
        "  figures/task4_metrics_comparison.png",
        "  figures/task4_cv_auc_comparison.png",
        "  figures/task4_confusion_matrices.png",
        "  figures/task4_feature_importance_*.png",
        "  task4_results.json",
        "  task4_summary.txt",
    ]

    path = out_dir / "task4_summary.txt"
    path.write_text("\n".join(lines))
    print(f"[INFO] Summary saved to: {path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    setup_dirs(args.output_dir)

    # 1. Load data
    X, y = load_data(args.input)
    feature_names = X.columns.tolist()

    # 2. Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )
    print(f"[INFO] Train: {X_train.shape[0]:,}  Test: {X_test.shape[0]:,}")

    # 3. Get models
    models = get_models()

    # 4. Cross-validation
    cv_results = cross_validate_models(
        models, X_train, y_train, args.cv_folds, args.random_state
    )

    # 5. Train & evaluate on test set
    test_results, trained_models, predictions = train_and_evaluate(
        models, X_train, X_test, y_train, y_test
    )

    # 6. Select best model
    best_model, selection_scores = select_best_model(test_results, cv_results)

    # 7. Plots
    try:
        make_plots(trained_models, predictions, y_test,
                   test_results, cv_results, feature_names,
                   args.output_dir / "figures")
    except Exception as e:
        print(f"[WARN] Plotting failed (non-fatal): {e}")

    # 8. Save results JSON
    results_json = {
        "best_model": best_model,
        "selection_scores": selection_scores,
        "cv_results": cv_results,
        "test_results": test_results,
    }
    json_path = args.output_dir / "task4_results.json"
    json_path.write_text(json.dumps(results_json, indent=2))
    print(f"[INFO] Results saved to: {json_path}")

    # 9. Summary report
    write_summary(test_results, cv_results, best_model, selection_scores,
                  X_train, X_test, y_train, y_test, args.output_dir)

    # 10. Done
    print(f"\n[DONE] Task 4 complete.")
    print(f"  Best model:  {best_model}")
    print(f"  Test AUC:    {test_results[best_model]['test_auc']:.4f}")
    print(f"  Test F1:     {test_results[best_model]['test_f1']:.4f}")
    print(f"  Figures   →  {args.output_dir / 'figures'}")
    print(f"  Summary   →  {args.output_dir / 'task4_summary.txt'}")


if __name__ == "__main__":
    main()