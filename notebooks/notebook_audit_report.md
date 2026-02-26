# Notebook Audit Report
**Date:** 2026-02-23  
**Notebooks:** ch2_data_pipeline.ipynb (23 cells), ch3_modeling.ipynb (90 cells), ch4_fairness.ipynb (64 cells)

## Summary

All 11 code notes have been addressed. The notebooks are ready for a final run-through to verify outputs match expectations.

---

## Code Fixes Applied

### ch3_modeling.ipynb

| Fix | Cell | Description | Status |
|-----|------|-------------|--------|
| A | 60 (was 57) | Replaced hardcoded `"Median: 0.140, Mean: 0.142"` with `np.median(y_prob_best)` and `y_prob_best.mean()` | ✅ |
| A | 60 (was 57) | Replaced hardcoded `0.14/y_prob_test.mean()` with `y_prob_best.mean()/y_prob_test.mean()` | ✅ |
| B | 62 (was 59) | Replaced hardcoded `0.14` in metadata `distribution_shift` warning with `y_prob_best.mean()` | ✅ |
| Prior | 12 | `model_normal` → `model_smote` in threshold exploration | ✅ (already applied) |
| Prior | 78→81 | Calibration conditional: `gap > 0` with correct else branch | ✅ (already applied) |
| Prior | 57→60 | `test_f1` variable added, `f1_score` imported | ✅ (already applied) |
| Prior | 80→83 | Model card: "1 in 10", correct statuses, §3.5, "Chapter 4" | ✅ (already applied) |

### ch4_fairness.ipynb

| Fix | Cell | Description | Status |
|-----|------|-------------|--------|
| C | 49 (was 46) | Comment `(0.5)` → `model threshold` | ✅ |
| D | 57 (was 54) | Typo `ffig` → `fig` | ✅ |
| Prior | 37→39 | Distribution shift with sample counts and `⚠️ (small sample)` warnings | ✅ (already applied) |
| Prior | 34→36 | Tier 3 thresholds with pass/fail indicators | ✅ (already applied) |
| Prior | 53→56 | Updated conclusion with data-driven findings | ✅ (already applied) |

---

## Teaching Callouts Added (Markdown Cells)

### ch3_modeling.ipynb (+3 cells → 90 total)

| Note | After Cell | Topic | Content |
|------|-----------|-------|---------|
| 4 | 29 (was after 28) | §3.3.13 Preliminary Fairness | Fairness metrics acceptable when model uniformly bad. DIR 1.16 passes only because model over-denies everyone. |
| 5 | 47 (was after 45) | §3.4.5 Model Comparison | Baseline 75% recall at 4.3% precision = flagging everyone. XGBoost advantage in ranking quality. |
| 6 | 57 (was after 54) | §3.4.9 Threshold Optimization | Cliff at 0.30 reveals SMOTE distortion. Signal exists (ROC-AUC 0.658) but scale wrong. Motivates calibration. |

### ch4_fairness.ipynb (+3 cells → 64 total)

| Note | After Cell | Topic | Content |
|------|-----------|-------|---------|
| 7+8 | 22 (was after 21) | §4.2.4 Race Fairness (Test) | Equal approval ≠ equal treatment. 0% minority TPR. Small sample caveat (1–8 defaults per group). |
| 9 | 31 (was after 29) | §4.2.6 Calibration Fairness | ECE is absolute — doesn't indicate direction. Check per-bin table for overestimation vs underestimation. |
| 11 | 47 (was after 44) | §4.3.1 Reweighting | Binary race training, four-group evaluation mismatch. ROC-AUC dropped for metric that wasn't failing. |

### ch4_fairness.ipynb (Docstring Enhancement)

| Note | Cell | Topic | Content |
|------|------|-------|---------|
| 10 | 42 (was 40) | Reweighting formula | Added `w = P(Y) / P(Y|A)` simplification to docstring with concrete example (5%/2% = 2.5). |

---

## Items NOT in Notebooks (For Book Chapters)

These teaching points emerged in our discussion and should be incorporated into the manuscript text, not the notebook code:

1. **SMOTE base value explanation** — SHAP base value ≈ 50% (log-odds ≈ 0) because of SMOTE 50/50 training. Features push down from there.
2. **Harm through apparent approval** — Expanded framing for §4.2.4 chapter text.
3. **ECOA adverse action requirements** — Feature names must be human-readable for consumer notices (§3.5.10 commentary).
4. **Fairness-through-inaction** — Reweighting/calibration achieve EOD = 0 by approving everyone. The empty top-left corner. Best as a "Lessons Learned" section.
5. **Production monitoring architecture** — Validation as frozen baseline, test simulates ongoing production batches.

---

## Recommended Next Steps

1. **Re-run all three notebooks end-to-end** to verify outputs with the new cells and fixes
2. **Update manuscript chapters** with new figures and commentary
3. **Add "Lessons Learned" section** to end of Chapter 4
4. **Final consistency pass** — figure numbers, cross-references, table of contents
