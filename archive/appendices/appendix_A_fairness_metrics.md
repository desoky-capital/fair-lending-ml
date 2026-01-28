# Appendix A: Fairness Metrics Quick Reference

---

## A.1 Metric Definitions at a Glance

**Table A.1: Fairness Metrics at a Glance**

| Metric | What It Measures | One-Line Summary |
|--------|------------------|------------------|
| **Disparate Impact Ratio (DIR)** | Approval rate parity | Are groups approved at similar rates? |
| **Statistical Parity Difference (SPD)** | Approval rate gap | How big is the approval rate gap? |
| **Equal Opportunity Difference (EOD)** | True positive rate gap | Are defaults caught equally across groups? |
| **Average Odds Difference (AOD)** | Combined error rate gap | Are all errors distributed equally? |
| **Expected Calibration Error (ECE)** | Probability honesty | Do predictions match reality? |

---

## A.2 Formulas

### Disparate Impact Ratio (DIR)

```
DIR = P(Ŷ = approve | A = unprivileged) / P(Ŷ = approve | A = privileged)
```

**Python:**
```python
def disparate_impact_ratio(y_pred, protected_attr, unprivileged, privileged):
    unpriv_approval = (y_pred[protected_attr == unprivileged] == 0).mean()
    priv_approval = (y_pred[protected_attr == privileged] == 0).mean()
    return unpriv_approval / priv_approval if priv_approval > 0 else np.nan
```

---

### Statistical Parity Difference (SPD)

```
SPD = P(Ŷ = approve | A = unprivileged) - P(Ŷ = approve | A = privileged)
```

**Python:**
```python
def statistical_parity_difference(y_pred, protected_attr, unprivileged, privileged):
    unpriv_approval = (y_pred[protected_attr == unprivileged] == 0).mean()
    priv_approval = (y_pred[protected_attr == privileged] == 0).mean()
    return unpriv_approval - priv_approval
```

---

### Equal Opportunity Difference (EOD)

```
EOD = TPR(unprivileged) - TPR(privileged)

Where: TPR = P(Ŷ = 1 | Y = 1) = True Positive Rate
```

**Python:**
```python
def equal_opportunity_difference(y_true, y_pred, protected_attr, unprivileged, privileged):
    from sklearn.metrics import recall_score
    
    unpriv_mask = (protected_attr == unprivileged)
    priv_mask = (protected_attr == privileged)
    
    tpr_unpriv = recall_score(y_true[unpriv_mask], y_pred[unpriv_mask], zero_division=0)
    tpr_priv = recall_score(y_true[priv_mask], y_pred[priv_mask], zero_division=0)
    
    return tpr_unpriv - tpr_priv
```

---

### Average Odds Difference (AOD)

```
AOD = 0.5 × [(FPR_unpriv - FPR_priv) + (TPR_unpriv - TPR_priv)]

Where: 
  FPR = P(Ŷ = 1 | Y = 0) = False Positive Rate
  TPR = P(Ŷ = 1 | Y = 1) = True Positive Rate
```

**Python:**
```python
def average_odds_difference(y_true, y_pred, protected_attr, unprivileged, privileged):
    unpriv_mask = (protected_attr == unprivileged)
    priv_mask = (protected_attr == privileged)
    
    # TPR
    tpr_unpriv = recall_score(y_true[unpriv_mask], y_pred[unpriv_mask], zero_division=0)
    tpr_priv = recall_score(y_true[priv_mask], y_pred[priv_mask], zero_division=0)
    
    # FPR
    fpr_unpriv = ((y_pred[unpriv_mask] == 1) & (y_true[unpriv_mask] == 0)).sum() / \
                 (y_true[unpriv_mask] == 0).sum() if (y_true[unpriv_mask] == 0).sum() > 0 else 0
    fpr_priv = ((y_pred[priv_mask] == 1) & (y_true[priv_mask] == 0)).sum() / \
               (y_true[priv_mask] == 0).sum() if (y_true[priv_mask] == 0).sum() > 0 else 0
    
    return 0.5 * ((fpr_unpriv - fpr_priv) + (tpr_unpriv - tpr_priv))
```

---

### Expected Calibration Error (ECE)

```
ECE = Σ (|bin_accuracy - bin_confidence| × bin_weight)

Where bins group predictions by probability range
```

**Python:**
```python
def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            bin_accuracy = y_true[in_bin].mean()
            bin_confidence = y_prob[in_bin].mean()
            bin_weight = in_bin.sum() / len(y_true)
            ece += abs(bin_accuracy - bin_confidence) * bin_weight
    
    return ece
```

---

## A.3 Interpretation Guide

### Disparate Impact Ratio (DIR)

**Table A.2: Disparate Impact Ratio Interpretation**

| Value | Interpretation | Action |
|-------|----------------|--------|
| **DIR = 1.00** | Perfect parity | No action needed |
| **DIR ≥ 0.80** | Acceptable (passes 4/5ths rule) | Monitor |
| **DIR 0.70-0.80** | Concerning | Investigate, consider mitigation |
| **DIR < 0.70** | Severe disparity | Immediate action required |
| **DIR > 1.25** | Reverse disparity | Investigate (may favor unprivileged) |

**Note:** DIR < 1.0 means unprivileged group has lower approval rate.

---

### Statistical Parity Difference (SPD)

**Table A.3: Statistical Parity Difference Interpretation**

| Value | Interpretation | Action |
|-------|----------------|--------|
| **SPD = 0.00** | Perfect parity | No action needed |
| **|SPD| < 0.05** | Acceptable | Monitor |
| **|SPD| 0.05-0.10** | Moderate gap | Investigate |
| **|SPD| > 0.10** | Large gap | Mitigation needed |

**Note:** SPD < 0 means unprivileged group has lower approval rate.

---

### Equal Opportunity Difference (EOD)

**Table A.4: Equal Opportunity Difference Interpretation**

| Value | Interpretation | Action |
|-------|----------------|--------|
| **EOD = 0.00** | Equal detection rates | No action needed |
| **|EOD| < 0.05** | Acceptable | Monitor |
| **|EOD| 0.05-0.10** | Moderate gap | Investigate |
| **|EOD| > 0.10** | Unequal error burden | Mitigation needed |

**Note:** EOD < 0 means unprivileged group's defaults are caught less often.

---

### Average Odds Difference (AOD)

**Table A.5: Average Odds Difference Interpretation**

| Value | Interpretation | Action |
|-------|----------------|--------|
| **AOD = 0.00** | Errors equally distributed | No action needed |
| **|AOD| < 0.05** | Acceptable | Monitor |
| **|AOD| 0.05-0.10** | Moderate imbalance | Investigate |
| **|AOD| > 0.10** | Significant error disparity | Mitigation needed |

---

### Expected Calibration Error (ECE)

**Table A.6: Expected Calibration Error Interpretation**

| Value | Interpretation | Action |
|-------|----------------|--------|
| **ECE < 0.02** | Excellent calibration | No action needed |
| **ECE 0.02-0.05** | Good calibration | Monitor |
| **ECE 0.05-0.10** | Fair calibration | Consider recalibration |
| **ECE > 0.10** | Poor calibration | Recalibration required |

**Compare across groups:** If ECE differs by > 0.05 between groups, calibration is unfair.

---

## A.4 Thresholds and Standards

### Legal/Regulatory Thresholds

**Table A.7: Legal/Regulatory Thresholds**

| Standard | Threshold | Source |
|----------|-----------|--------|
| **4/5ths Rule** | DIR ≥ 0.80 | EEOC Uniform Guidelines, applied to ECOA |
| **Adverse Impact** | DIR < 0.80 triggers investigation | CFPB, DOJ |

---

### Industry Best Practices

**Table A.8: Industry Best Practices Thresholds**

| Metric | Minimum | Target | Excellent |
|--------|---------|--------|-----------|
| **DIR** | ≥ 0.80 | ≥ 0.90 | ≥ 0.95 |
| **|SPD|** | < 0.15 | < 0.10 | < 0.05 |
| **|EOD|** | < 0.15 | < 0.10 | < 0.05 |
| **|AOD|** | < 0.15 | < 0.10 | < 0.05 |
| **ECE** | < 0.10 | < 0.05 | < 0.02 |

---

### Prioritization Framework

**Table A.9: Fairness Metrics Prioritization Framework**

| Tier | Metrics | Requirement |
|------|---------|-------------|
| **Tier 1: Legal** | DIR ≥ 0.80 | MUST satisfy |
| **Tier 2: Business** | ECE < 0.05, \|EOD\| < 0.10 | SHOULD optimize |
| **Tier 3: Monitor** | SPD, AOD | Track over time |

---

## A.5 Quick Decision Tree

```
START: Is DIR ≥ 0.80?
  │
  ├─ NO → STOP. Fix immediately. (Legal requirement)
  │
  └─ YES → Is ECE similar across groups (diff < 0.05)?
           │
           ├─ NO → Apply calibration
           │
           └─ YES → Is |EOD| < 0.10?
                    │
                    ├─ NO → Investigate error distribution
                    │
                    └─ YES → ✓ Model passes fairness checks
                             Continue monitoring
```

---

## A.6 Terminology Reference

### Error Types in Credit Scoring

**Table A.10: Error Types in Credit Scoring**

| Term | Meaning | Impact |
|------|---------|--------|
| **True Positive (TP)** | Correctly predicted default | Avoided loss |
| **True Negative (TN)** | Correctly predicted non-default | Good customer approved |
| **False Positive (FP)** | Predicted default, actually paid | Good customer wrongly denied |
| **False Negative (FN)** | Predicted non-default, actually defaulted | Loss incurred |

### Rates

**Table A.11: Rate Formulas Reference**

| Rate | Formula | Meaning |
|------|---------|---------|
| **TPR (Recall)** | TP / (TP + FN) | % of actual defaults caught |
| **FPR** | FP / (FP + TN) | % of good borrowers wrongly denied |
| **Precision** | TP / (TP + FP) | % of predicted defaults that are correct |
| **Approval Rate** | (TN + FN) / Total | % of applicants approved |

---

## A.7 Sample Size Guidelines

**Table A.12: Sample Size Guidelines**

| Group Size | Reliability | Recommendation |
|------------|-------------|----------------|
| **n < 30** | Unreliable | Do not report metrics; aggregate groups |
| **n = 30-50** | Marginal | Report with confidence intervals |
| **n = 50-100** | Acceptable | Standard reporting |
| **n > 100** | Good | Reliable point estimates |

**Confidence interval formula (95%):**
```
CI = metric ± 1.96 × sqrt(metric × (1-metric) / n)
```

---

*End of Appendix A*
