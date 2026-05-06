# Task 7: Communication & Interpretation

## Project Goal and Problem Framing

This project builds an end-to-end machine learning pipeline for **credit default risk prediction** using the Home Credit dataset. The predictive objective is binary classification: given an applicant's financial profile and aggregated credit history, predict the likelihood of `TARGET=1` (default).

The problem is both practically important and analytically challenging. Home Credit serves populations that lack conventional credit histories, making standard scoring approaches insufficient. The dataset is severely class-imbalanced — only 8.07% of applicants default — creating a structural tension between sensitivity and specificity that shapes every decision in this pipeline.

The analysis was designed with two goals: (1) build a robust, reproducible model pipeline that explicitly handles class imbalance; and (2) communicate every design decision and trade-off in terms that are meaningful to both technical and non-specialist stakeholders.

---

## Pipeline Design: Not Just What, But Why

Each stage of the pipeline was designed with explicit reasoning — not just implementation.

**Task 1 — Data Acquisition & Preparation**
Three relational tables were merged into a single customer-level dataset (307,511 applicants, 439 features). Numeric missing values were imputed with the column median — robust to the right-skewed distributions common in credit data. Categorical missing values were encoded as `"UNKNOWN"` rather than discarded, because missingness itself can be an informative risk signal. Outlier removal was deferred intentionally, so EDA could observe true data distributions before any transformation was applied.

**Task 2 — EDA & Unsupervised Learning**
Statistical testing confirmed that the EXT_SOURCE credit scores are the strongest predictors of default (Pearson |r| up to 0.160, Cohen's d up to 0.562). The class imbalance finding — 91.9% non-default vs. 8.1% default — directly motivated the use of AUC and PR-AUC as primary metrics throughout modeling, since accuracy is misleading under this level of imbalance. PCA and K-means clustering (k=3) were applied on a random sample of 20,000 rows to explore latent structure efficiently; the resulting segmentation showed three customer groups with meaningfully different default rates (5.2%, 9.7%, 7.4%), providing independent unsupervised validation that risk-predictive structure exists in the data.

**Why PCA / cluster features were not used in the final supervised model.**  
PCA coordinates and cluster assignments are defined only for the sampled applicants used to fit the unsupervised models. Merging those columns back into the full dataset (307,511 rows) would leave roughly 287,511 rows without values unless we filled them in artificially or refit the unsupervised models on the entire population. Imputing such a large share of rows would invent structure and could bias downstream supervised learning; refitting PCA/KMeans on all rows—with correct train-only fitting to avoid leakage—was not integrated into our Task 4 pipeline for this submission. **We therefore did not include PCA or cluster-ID features as inputs to the final XGBoost model.** Task 2 still informed downstream choices: it reinforced leveraging external bureau-style scores (**EXT_SOURCE** family) and **credit-burden ratios** as dominant signals—consistent with what later supervised modeling and importance analysis confirmed.

**Task 3 — Feature Engineering & Preprocessing**
Twenty-two domain-driven features were engineered. Ratio features (e.g., `ANNUITY_INCOME_RATIO`) are more informative than raw amounts because they capture *relative* financial burden — a monthly payment of 20,000 means something very different for someone earning 30,000 versus 500,000 per month. EXT_SOURCE aggregations (mean, min, max, std, weighted) capture both the typical level and the disagreement between credit bureaus, the latter being an independent risk signal. A two-stage feature selection process (correlation deduplication → LightGBM importance ranking) reduced 461 features to 80, balancing predictive power with model efficiency.

**Tasks 4 & 5 — Supervised Modeling & Selection**
Three models were compared under 5-fold stratified cross-validation. XGBoost was selected as the final model (test AUC = 0.7809, PR-AUC = 0.2730, F1 = 0.294) based on a composite weighted score (0.60 × Test AUC + 0.20 × CV AUC + 0.20 × PR-AUC). It outperforms Logistic Regression and Random Forest across all composite metrics, with the lowest CV variance (±0.0020), indicating the most stable generalization. Logistic Regression is retained as an interpretability benchmark — its coefficients can be directly examined by regulators or stakeholders.

---

## Communication to a Non-Technical Audience

The model assigns every applicant a risk score between 0 and 1. It does not make final lending decisions — it helps loan officers **prioritize their review effort** by identifying which applications warrant the most scrutiny.

**Plain-language takeaway:** The model is a risk-ranking tool, not a decision-maker. A high score means "this application deserves a closer look," not "this applicant will definitely default."

**How to use it in practice:**
- Applications above a chosen risk threshold are flagged for manual review before approval.
- The threshold is set by the business, not the model. Institutions prioritizing loss minimization choose a lower threshold (catching more defaults, accepting more false alarms); those prioritizing approval volume choose a higher one.
- The model should be monitored over time and recalibrated as applicant behavior and macroeconomic conditions change.

---

## Reproducibility and Transparency

Every pipeline stage produces explicit, documented outputs — cleaned datasets, cleaning logs, EDA summaries, statistical test results, feature engineering logs, fitted scalers, and performance metrics. Model selection is documented both quantitatively (composite score with explicit weights) and qualitatively (model-by-model analysis of strengths and limitations). The accompanying Streamlit dashboard provides an interactive interface for exploring results without requiring programming knowledge.

---

## Limitations and Future Improvements

- **Threshold calibration:** A production deployment should optimize the decision threshold using an explicit cost function (cost of missed defaults vs. false alarms), not a generic default.
- **Probability calibration:** XGBoost's raw scores may not be well-calibrated probabilities. Platt scaling or isotonic regression could improve reliability when scores are used to set risk-tiered interest rates.
- **Temporal validation:** The current random train-test split may overestimate future generalization. A time-ordered split would provide a more realistic performance estimate.
- **Fairness auditing:** Model performance should be evaluated across demographic groups before deployment to identify potential disparate impact.
- **Explainability:** SHAP values would provide individualized explanations for each prediction, supporting regulatory adverse-action communication requirements.

---

## Final Conclusion

This project delivers a complete, transparent, and reproducible machine learning pipeline from raw data to a business-ready risk model. The final XGBoost model achieves competitive performance (test AUC = 0.7809) with documented trade-offs and clear paths for improvement. Its value lies not in replacing human judgment but in augmenting it — helping lending teams direct their expertise toward the applications where it is most needed.
