# List of Figures

---

## Chapter 2: Data Foundations

| Figure | Title | File | Page |
|--------|-------|------|------|
| **2.1** | Data Quality Issues: Completeness, Duplicates, and Date Formats | `data_quality_issues.png` | ___ |

---

## Chapter 3: Building the Credit Model

| Figure | Title | File | Page |
|--------|-------|------|------|
| **3.1** | Confusion Matrix - SMOTE Model | `confusion_matrix.png` | ___ |
| **3.2** | Top 10 Most Important Features (Logistic Regression Coefficients) | `feature_importance.png` | ___ |
| **3.3** | ROC Curve - Credit Default Model (Baseline) | `roc_curve.png` | ___ |
| **3.4** | Precision-Recall Curve - Credit Default Model | `precision_recall_curve.png` | ___ |
| **3.5** | Top 10 Features - Random Forest (Gini Importance) | `rf_feature_importance.png` | ___ |
| **3.6** | Top 10 Features - XGBoost (Gain) | `xgb_feature_importance.png` | ___ |
| **3.7** | Model Comparison: Precision, Recall, and ROC-AUC | `model_comparison.png` | ___ |
| **3.8** | Precision-Recall Trade-off by Decision Threshold | `threshold_optimization.png` | ___ |
| **3.9** | Probability Distributions by Dataset (Train/Val/Test) | `probability_distributions.png` | ___ |
| **3.10** | Calibration Curves by Dataset (Train/Val/Test) | `calibration_curves.png` | ___ |
| **3.11** | SHAP Summary Plot - Feature Importance (Test Set) | `shap_summary_plot.png` | ___ |
| **3.12** | SHAP Dependence Plots - Top Features | `shap_dependence_plots.png` | ___ |
| **3.13** | SHAP Waterfall: Highest Probability Prediction (False Positive) | `shap_waterfall_highest_probability_prediction.png` | ___ |
| **3.14** | SHAP Waterfall: Correct Non-Default (True Negative) | `shap_waterfall_correct_non-default.png` | ___ |
| **3.15** | SHAP Waterfall: Missed Default (False Negative) | `shap_waterfall_missed_default_false_negative.png` | ___ |
| **3.16** | SHAP Waterfall Examples - Combined View | `shap_waterfall_examples.png` | ___ |

---

## Chapter 4: Fairness & Compliance

| Figure | Title | File | Page |
|--------|-------|------|------|
| **4.1** | Fairness Dashboard (6-Panel): Approval Rates, TPR, FPR, ECE, Metrics Summary | `fairness_dashboard.png` | ___ |
| **4.2** | Calibration Comparison: Original vs. Platt Scaling vs. Isotonic Regression | `calibration_comparison.png` | ___ |
| **4.3** | Accuracy vs. Fairness Trade-off (Mitigation Approaches) | `accuracy_fairness_tradeoff.png` | ___ |

---

## Figure Details and Captions

### Chapter 2 Figures

**Figure 2.1: Data Quality Issues**
Three-panel visualization summarizing data quality findings: (1) Overall Data Completeness showing missing data percentages by table - Accounts has ~2.5% missing, Transactions has ~10% missing, Balances has minimal missing data; (2) Duplicate Records by Table showing Transactions has the highest duplicate count (~250), Accounts has ~65, Balances has none; (3) Date Format Distribution in Accounts table showing inconsistent formats with 51.1% MM/DD/YYYY and 48.9% YYYY-MM-DD, requiring standardization.

---

### Chapter 3 Figures

**Figure 3.1: Confusion Matrix - SMOTE Model**
Shows the confusion matrix for the baseline logistic regression model trained on SMOTE-balanced data. The matrix reveals 282 true negatives, 179 false positives, 11 false negatives, and 5 true positives, highlighting the class imbalance challenge where the model correctly identifies most non-defaults but struggles with the minority default class.

**Figure 3.2: Top 10 Most Important Features (Logistic Regression)**
Displays logistic regression coefficients for the top 10 features. Green bars (positive coefficients) indicate features that increase default risk; red bars (negative coefficients) indicate features that decrease default risk. Credit history months and credit utilization percentage emerge as the strongest predictors.

**Figure 3.3: ROC Curve - Credit Default Model**
Receiver Operating Characteristic curve for the baseline model showing AUC = 0.524, only marginally better than random chance (dashed diagonal line). This poor discrimination ability motivates the model improvement efforts in Section 3.4.

**Figure 3.4: Precision-Recall Curve - Credit Default Model**
Precision-Recall curve showing Average Precision (AP) = 0.036, barely above the random baseline of 0.034 (equal to the default rate). Demonstrates that the baseline model provides minimal value for identifying defaults.

**Figure 3.5: Top 10 Features - Random Forest**
Feature importance from Random Forest model using Gini importance. FICO score dominates, followed by credit history months. Behavioral features like balance trends and transaction patterns also contribute significantly.

**Figure 3.6: Top 10 Features - XGBoost**
Feature importance from XGBoost model using gain. FICO score remains most important, but behavioral features (average monthly overdrafts, spending patterns) show stronger relative importance compared to Random Forest.

**Figure 3.7: Model Comparison**
Three-panel comparison of Baseline (LR + SMOTE), Random Forest, and XGBoost across Precision, Recall, and ROC-AUC. XGBoost achieves the best balance with 0.375 recall and 0.669 ROC-AUC, though precision remains challenging at 0.079.

**Figure 3.8: Precision-Recall Trade-off by Decision Threshold**
Shows how precision, recall, and F1 score vary with decision threshold for the tuned XGBoost model. The optimal threshold of 0.25 (red dashed line) maximizes F1 score, balancing the precision-recall trade-off.

**Figure 3.9: Probability Distributions by Dataset**
Histograms comparing predicted probability distributions for non-defaults (blue) and defaults (red) across training, validation, and test sets. Reveals the distribution shift problem: defaults receive much lower probabilities on test data than on training data.

**Figure 3.10: Calibration Curves by Dataset**
Reliability diagrams showing calibration across datasets. Training data (Brier Score: 0.010) is well-calibrated; validation (0.033) shows some deviation; test (0.065) reveals significant miscalibration, especially for higher probability bins.

**Figure 3.11: SHAP Summary Plot**
Beeswarm plot showing SHAP values for all features on the test set. Each dot represents one prediction; color indicates feature value (red = high, blue = low); horizontal position shows impact on prediction. FICO score has the strongest impact, with high scores (red) pushing predictions left (lower default risk).

**Figure 3.12: SHAP Dependence Plots**
Five dependence plots showing how individual feature values affect SHAP contributions. The FICO score plot (top-left) shows a clear negative relationship: higher FICO scores consistently reduce default predictions. Channel usage features show more complex, non-linear patterns.

**Figure 3.13: SHAP Waterfall - Highest Probability Prediction**
Individual explanation for the prediction with highest default probability (49.67%). Despite actual outcome being NO DEFAULT, high travel spending (+0.49) and restaurant spending (+0.37) pushed the prediction toward default. This is a false positive - a good borrower who would be wrongly denied.

**Figure 3.14: SHAP Waterfall - Correct Non-Default**
Individual explanation for a correctly identified non-default (2.95% probability). High transaction frequency (-0.69) and ATM usage (-0.56) strongly indicate low risk, leading to correct approval decision.

**Figure 3.15: SHAP Waterfall - Missed Default (False Negative)**
Individual explanation for a missed default (2.73% probability despite actual default). Despite moderate FICO score (+0.2 toward default), low transaction frequency and channel usage pushed the prediction below threshold, causing a false negative - a bad borrower who would be wrongly approved.

**Figure 3.16: SHAP Waterfall Examples - Combined**
Three-panel comparison showing SHAP waterfall plots for: (1) a missed default (false negative), (2) a correct non-default (true negative), and (3) the highest probability prediction. Illustrates how the same features contribute differently across individual cases.

---

### Chapter 4 Figures

**Figure 4.1: Fairness Dashboard (6-Panel)**
Comprehensive fairness visualization showing: (1) Approval rates by race with 4/5ths threshold, (2) True Positive Rates by race, (3) False Positive Rates by race, (4) Expected Calibration Error by race, (5) Summary fairness metrics (DIR, SPD, EOD, AOD) for Black vs. White comparison, and (6) Approval rates by gender. All groups pass the 4/5ths threshold (DIR ≥ 0.80).

**Figure 4.2: Calibration Comparison**
Three-panel visualization comparing calibration methods: (1) Probability distributions after calibration showing Original, Platt Scaling, and Isotonic approaches, (2) Calibration curves showing Isotonic achieves closest fit to perfect calibration line, (3) Precision and Recall comparison showing Isotonic Regression achieves best recall (0.67) while maintaining precision.

**Figure 4.3: Accuracy vs. Fairness Trade-off**
Scatter plot comparing four mitigation approaches on Accuracy (y-axis) vs. Disparate Impact Ratio (x-axis). Red dashed line marks DIR = 0.80 (4/5ths rule threshold). Shows that Calibrated and Original models achieve both high accuracy (~95%) and good fairness (DIR > 1.0), while Group Thresholds destroy accuracy (31%) despite achieving DIR = 0.88.

---

## File Inventory

### Uploaded Files (20 total)

| # | Original Filename | Assigned Figure | Chapter |
|---|-------------------|-----------------|---------|
| 1 | `data_quality_issues.png` | Figure 2.1 | Ch 2 |
| 2 | `confusion_matrix.png` | Figure 3.1 | Ch 3 |
| 3 | `feature_importance.png` | Figure 3.2 | Ch 3 |
| 4 | `roc_curve.png` | Figure 3.3 | Ch 3 |
| 5 | `precision_recall_curve.png` | Figure 3.4 | Ch 3 |
| 6 | `rf_feature_importance.png` | Figure 3.5 | Ch 3 |
| 7 | `xgb_feature_importance.png` | Figure 3.6 | Ch 3 |
| 8 | `model_comparison.png` | Figure 3.7 | Ch 3 |
| 9 | `threshold_optimization.png` | Figure 3.8 | Ch 3 |
| 10 | `probability_distributions.png` | Figure 3.9 | Ch 3 |
| 11 | `calibration_curves.png` | Figure 3.10 | Ch 3 |
| 12 | `shap_summary_plot.png` | Figure 3.11 | Ch 3 |
| 13 | `shap_dependence_plots.png` | Figure 3.12 | Ch 3 |
| 14 | `shap_waterfall_highest_probability_prediction.png` | Figure 3.13 | Ch 3 |
| 15 | `shap_waterfall_correct_non-default.png` | Figure 3.14 | Ch 3 |
| 16 | `shap_waterfall_missed_default_false_negative.png` | Figure 3.15 | Ch 3 |
| 17 | `shap_waterfall_examples.png` | Figure 3.16 | Ch 3 |
| 18 | `fairness_dashboard.png` | Figure 4.1 | Ch 4 |
| 19 | `calibration_comparison.png` | Figure 4.2 | Ch 4 |
| 20 | `accuracy_fairness_tradeoff.png` | Figure 4.3 | Ch 4 |

---

## Figures Using ASCII Diagrams (in chapter text)

The following conceptual diagrams are represented as ASCII art within the chapter text rather than as image files:

| Location | Diagram | Description |
|----------|---------|-------------|
| Ch 4.1 | Impossibility Theorem | Venn diagram showing conflict between fairness definitions |
| Ch 4.3 | Mitigation Strategies | Three-column comparison of Pre/In/Post processing |
| Ch 5.2 | EU AI Act Risk Tiers | Four-tier risk classification pyramid |
| Ch 5.3 | Three Pillars | Technical + Process + Culture framework |

These ASCII diagrams are effective for the technical audience and don't require separate image files.

---

## Recommended Figure Naming Convention

For final production, rename files to match figure numbers:

```
figures/
├── chapter2/
│   └── fig_2_01_data_quality_issues.png
├── chapter3/
│   ├── fig_3_01_confusion_matrix.png
│   ├── fig_3_02_feature_importance_lr.png
│   ├── fig_3_03_roc_curve.png
│   ├── fig_3_04_precision_recall_curve.png
│   ├── fig_3_05_rf_feature_importance.png
│   ├── fig_3_06_xgb_feature_importance.png
│   ├── fig_3_07_model_comparison.png
│   ├── fig_3_08_threshold_optimization.png
│   ├── fig_3_09_probability_distributions.png
│   ├── fig_3_10_calibration_curves.png
│   ├── fig_3_11_shap_summary.png
│   ├── fig_3_12_shap_dependence.png
│   ├── fig_3_13_shap_waterfall_fp.png
│   ├── fig_3_14_shap_waterfall_tn.png
│   ├── fig_3_15_shap_waterfall_fn.png
│   └── fig_3_16_shap_waterfall_combined.png
├── chapter4/
│   ├── fig_4_01_fairness_dashboard.png
│   ├── fig_4_02_calibration_comparison.png
│   └── fig_4_03_accuracy_fairness_tradeoff.png
```

---

*End of List of Figures*
