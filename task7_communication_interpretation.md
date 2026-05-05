# Task 7: Communication & Interpretation (Advanced-Level Narrative)

## Project Goal and Problem Framing

This project builds an end-to-end machine learning pipeline for **credit default risk prediction** using the Home Credit dataset.  
The predictive objective is binary classification: given applicant information and aggregated credit history, estimate the likelihood of `TARGET=1` (default).

The analysis is designed as a full workflow from raw data handling to final model selection, with two goals:

- Build a robust and reproducible model pipeline for imbalanced credit-risk data.
- Communicate decisions, trade-offs, and practical implications for non-specialist stakeholders.

## End-to-End Workflow Summary

The workflow follows Tasks 1-6:

1. Data acquisition and cleaning from a public, credible source.
2. EDA and unsupervised learning to identify structure and risk patterns.
3. Feature engineering and preprocessing to improve predictive signal quality.
4. Supervised model development with cross-validation.
5. Model comparison and final model selection based on quantitative and contextual criteria.
6. Integrated communication of findings, limitations, and recommendations.

## Task 1: Data Acquisition & Preparation - Interpretation

The project starts with `application_train.csv` and enriches the base table with optional one-to-many support tables (`bureau.csv`, `previous_application.csv`) by aggregating to customer level and merging on `SK_ID_CURR`.

Key preparation decisions:

- Standardized schema (column normalization, duplicate handling, placeholder normalization).
- Converted numeric-like text to numeric types to avoid downstream parsing inconsistency.
- Imputed missing values using reproducible rules (`median` for numeric, `UNKNOWN` for categorical).
- Preserved one row per applicant to avoid leakage from one-to-many expansion.

Interpretation:

- This stage emphasizes **data reliability over speed**.
- Aggregation choices prioritize stable customer-level modeling features.
- Missingness was treated as potentially informative rather than discarded.

## Task 2: EDA + Unsupervised Learning - Interpretation

EDA and statistical analysis were used to build understanding before modeling:

- Descriptive statistics and feature-target correlation profiling.
- Mann-Whitney U tests and effect sizes for numeric group differences.
- Chi-square and Cramer's V for categorical association.
- PCA + KMeans to examine latent structure and cluster-level behavior.

Interpretation:

- Strong class imbalance confirms the need for imbalance-aware evaluation.
- Risk-related patterns are not purely linear, supporting nonlinear model candidates.
- Unsupervised structure provides additional intuition for segmentation and feature design.

Challenge and resolution:

- High dimensional numeric space may be noisy and unstable for unsupervised methods.
- The pipeline addressed this with clipping, scaling, and stable numeric subset construction before PCA/KMeans.

## Task 3: Feature Engineering & Preprocessing - Interpretation

Task 3 transforms raw cleaned features into a compact modeling-ready dataset.

Important engineering choices:

- Sentinel anomaly handling (`DAYS_EMPLOYED=365243`) with both correction and anomaly flag retention.
- Domain ratio features (income burden, annuity burden, credit structure).
- External score aggregations (mean/min/max/std/product/weighted).
- Outlier winsorization (1st-99th percentile) for robust scaling.
- Mixed encoding strategy (binary label encoding + one-hot for low-cardinality fields).
- Two-stage feature reduction: correlation deduplication, then importance ranking.

Interpretation:

- The pipeline balances **predictive strength** and **operational interpretability**.
- Engineered ratios support cross-applicant comparability.
- Feature reduction mitigates over-redundancy and improves model stability.

## Task 4-5: Supervised Modeling, Evaluation, and Selection - Interpretation

Three supervised models were compared under stratified cross-validation and test evaluation:

- Logistic Regression
- Random Forest
- XGBoost

Primary metrics:

- ROC-AUC (primary ranking metric under imbalance)
- PR-AUC (minority-class sensitivity)
- F1, Precision, Recall
- CV mean/std for robustness

Final selection:

- **XGBoost** selected as final model.
- Best test AUC (`0.7809`), best PR-AUC (`0.2730`), best F1 (`0.2942`), and strongest CV AUC (`0.7776 +/- 0.0020`).

Interpretation:

- The selected model best balances discrimination and minority-class utility.
- Logistic Regression remains a useful benchmark for interpretability.
- Precision remains modest across models, so deployment should be framed as **risk screening/ranking**, not deterministic default labeling.

## Communication to a Non-Technical Audience

Plain-language takeaway:

- The model helps rank applicants by relative default risk.
- It should support, not replace, lending policy and human oversight.
- The system is most valuable for early warning and triage of high-risk applications.

Decision-use recommendation:

- Use model scores to prioritize review depth.
- Apply threshold tuning based on business tolerance for missed defaults vs false alarms.
- Monitor drift and recalibrate as applicant behavior and macro conditions change.

## Reproducibility and Transparency

The project provides reproducible artifacts across stages (cleaned data, summaries, metrics, and figures), and separates each task into explicit files and outputs.  
Model selection logic is documented quantitatively and contextually, not by a single metric only.

## Limitations and Future Improvements

- Current thresholding still reflects generic defaults; cost-sensitive threshold optimization can improve policy fit.
- Additional calibration (e.g., probability calibration curves) can improve risk-score reliability.
- Temporal validation and fairness checks should be added before production deployment.
- SHAP-based local explanations can strengthen interpretability for adverse-action style communication.

## Final Conclusion

This project delivers a full data science workflow with a coherent analytical narrative from raw data to final model selection.  
The final XGBoost model provides the strongest overall predictive performance for default risk ranking, while documented trade-offs and limitations ensure transparent and responsible interpretation.
