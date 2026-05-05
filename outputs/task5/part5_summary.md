# Part 5: Model Comparison and Selection

## Purpose

This section compares the supervised learning models developed in Part 4 and selects the final model for the loan default prediction task. The comparison considers predictive performance, minority-class performance, interpretability, robustness, and the practical context of credit-risk screening.

## Why accuracy is not enough

The dataset is highly imbalanced: only about 8.07% of applicants in both the training and test sets are labeled as default cases. A model that predicted every applicant as "Repaid" would achieve about 91.93% accuracy, but it would fail to identify any true default cases. For this reason, accuracy alone is misleading and is not sufficient for evaluating model quality in this project.

## Metrics used

ROC-AUC measures the model's overall ability to rank default cases above non-default cases across classification thresholds. PR-AUC is especially important because the positive class, "Default," is rare, so it focuses more directly on the tradeoff between precision and recall for the minority class. Recall matters because missing true default cases can be costly in a lending context. Precision matters because too many false alarms may reduce the practical usefulness of the model. F1-score balances precision and recall into a single metric. Cross-validation AUC checks whether model performance is stable across folds and helps assess generalizability beyond one train-test split.

## Model comparison

### Logistic Regression

Logistic Regression is the most interpretable model and provides a useful simple baseline. It produced strong recall (0.6900), meaning it identified a relatively large share of actual default cases, and its cross-validation AUC was stable (0.7608 +/- 0.0032). However, it had lower F1-score (0.2696) and PR-AUC (0.2416) than XGBoost, suggesting weaker minority-class performance. Because it is a linear model, it may also underfit nonlinear relationships and interactions in the credit-risk data.

### Random Forest

Random Forest captures nonlinear relationships and performed better than Logistic Regression on F1-score. It is also partly interpretable through feature-importance plots. However, it had the lowest test AUC among the three models and lower recall than both Logistic Regression and XGBoost. It is also less transparent than Logistic Regression, which makes it somewhat harder to explain in a credit-risk context.

### XGBoost

XGBoost had the strongest overall performance. It achieved the highest CV AUC, highest test AUC, highest PR-AUC, and highest F1-score. It also maintained high recall while improving precision relative to Logistic Regression: XGBoost's precision was 0.1879, compared with Logistic Regression's precision of 0.1675. This improvement matters because precision is still relatively low across all models, meaning many predicted defaults are false positives. That limitation is common in imbalanced credit-risk settings, but XGBoost provides the best balance among the models tested.

## Final model selection

XGBoost is selected as the final model. It has the best test AUC (0.7809), best PR-AUC (0.2730), best F1-score (0.2942), and strongest cross-validation AUC (0.7776 +/- 0.0020). Its CV AUC and test AUC are close, suggesting good generalization and no major overfitting signal. XGBoost keeps recall high at 0.6779 while improving precision to 0.1879, compared with Logistic Regression's precision of 0.1675. Because this project is focused on default prediction, identifying risky applicants is more important than maximizing overall accuracy.

## Interpretability discussion

XGBoost is less directly interpretable than Logistic Regression because it uses an ensemble of boosted decision trees rather than a single linear coefficient for each feature. However, feature-importance plots improve interpretability by showing which variables contribute most to model decisions. The Part 4 importance plots indicate that EXT_SOURCE_MEAN, EXT_SOURCE_MIN, EXT_SOURCE_MAX, education indicators, gender, document flags, credit ratios, and previous application features appear important. These variables are consistent with credit-risk intuition because they capture external risk scores, borrower characteristics, credit structure, and prior application behavior.

## Robustness and overfitting discussion

XGBoost's CV AUC is 0.7776 and its test AUC is 0.7809. The similarity between cross-validation and test performance suggests stable generalization. Random Forest performs slightly worse on AUC and PR-AUC, while Logistic Regression remains useful as an interpretable benchmark but does not perform as well on minority-class metrics. None of the models should be interpreted as perfect individual-level predictors, especially because precision remains modest. Instead, the selected XGBoost model is best viewed as a risk-ranking and screening model. Threshold tuning could be considered if the deployment goal prioritizes either higher recall or higher precision.

## Final report paragraph

Based on the model comparison, XGBoost was selected as the final supervised learning model for default prediction. It achieved the strongest overall performance, with the highest test ROC-AUC (0.7809), highest PR-AUC (0.2730), highest F1-score (0.2942), and strongest cross-validation AUC (0.7776 +/- 0.0020). These results suggest that XGBoost provides the best ranking ability and minority-class performance among the three models evaluated. Its test AUC is also close to its cross-validation AUC, which indicates stable generalization and no major overfitting signal. Although precision remains relatively low, this is common in imbalanced credit-risk settings where the positive class is rare. Therefore, the model should be interpreted as a useful risk-screening and ranking tool rather than a perfect individual-level predictor. Future implementation could include threshold tuning if the business objective places greater emphasis on recall or precision.
