# Notebook Code Notes

Running list of observations, explanations, and potential callouts for readers.

| # | Notebook | Cell/Section | Note | Status |
|---|----------|-------------|------|--------|
| 1 | ch2 | §2.3.2 Date standardization | `default_date` uses `errors='coerce'` because most accounts never default — nulls are expected. `transaction_date` and `balance_date` omit `coerce` deliberately: these are required fields, so a parse failure should raise an error rather than silently produce NaT. | ℹ️ For book text |
| 2 | ch2 | §2.3.1 Cleaning | Print statements show cleaning results interactively, which is good for the notebook experience. Add callout: *"In production, these steps would be captured by the `DataQualityLogger` (see Appendix C) to maintain the audit trail required by SR 11-7."* | ℹ️ For book text |
| 3 | ch3 | §3.3.6 Threshold Exploration | `model_normal` replaced with `model_smote` and `y_prob_normal` with `y_prob_smote`. | ✅ Applied |
| 4 | ch3 | §3.3.13 Preliminary Fairness Check | DIR 1.16 looks acceptable, but only because model is over-denying everyone equally. Added as markdown teaching note after cell. | ✅ Applied |
| 5 | ch3 | §3.4.5 Model Comparison | Baseline 75% recall at 4.3% precision = flagging everyone. XGBoost advantage in ranking quality. Added as markdown teaching note after cell. | ✅ Applied |
| 6 | ch3 | §3.4.9 Threshold Optimization | Cliff at 0.30 reveals SMOTE distortion. Added as markdown teaching note after cell. | ✅ Applied |
| 7 | ch4 | §4.2.4 Fairness Analysis: Race | Equal approval ≠ equal treatment. 0% minority TPR on test. Added as markdown teaching note after test set cell. | ✅ Applied |
| 8 | ch4 | §4.2.4 Fairness Analysis: Race | Small sample caveat — 1–8 defaults per group. Combined with note 7 in single markdown cell. | ✅ Applied |
| 9 | ch4 | §4.2.6 Calibration Fairness | ECE is absolute value — doesn't indicate direction. Added as markdown teaching note after ECE summary. | ✅ Applied |
| 10 | ch4 | §4.3.1 Reweighting | Weight formula simplification `w = P(Y) / P(Y|A)` with concrete example. Enhanced docstring in code cell. | ✅ Applied |
| 11 | ch4 | §4.3.1 Reweighting | Binary race training vs four-group evaluation mismatch. ROC-AUC dropped for marginal DIR gain. Added as markdown teaching note. | ✅ Applied |

## Additional Code Fixes Applied

| Fix | Notebook | Description | Status |
|-----|----------|-------------|--------|
| Hardcoded val probabilities | ch3 cell 60 | `"Median: 0.140"` → `np.median(y_prob_best)` | ✅ |
| Hardcoded ratio | ch3 cell 60 | `0.14/y_prob_test.mean()` → `y_prob_best.mean()/y_prob_test.mean()` | ✅ |
| Metadata hardcoded | ch3 cell 62 | Same ratio fix in `improved_metadata['warnings']` | ✅ |
| Stale comment | ch4 cell 49 | `(0.5)` → `model threshold` | ✅ |
| Typo | ch4 cell 57 | `ffig` → `fig` | ✅ |
| Calibration conditional | ch3 cell 81 | Added `gap > 0` / else for honest assessment | ✅ (prior) |
| test_f1 variable | ch3 cell 60 | Added `test_f1 = f1_score(...)` | ✅ (prior) |
| Model card updates | ch3 cell 83 | "1 in 10", correct statuses, §3.5, "Chapter 4" | ✅ (prior) |
| Distribution shift counts | ch4 cell 39 | Added `n=` sample counts and `⚠️ (small sample)` | ✅ (prior) |
| Tier 3 thresholds | ch4 cell 36 | Added pass/fail indicators | ✅ (prior) |
| Updated conclusion | ch4 cell 56 | Data-driven findings replacing generic boilerplate | ✅ (prior) |
