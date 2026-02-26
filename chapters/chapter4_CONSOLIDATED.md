# Chapter 4: Fairness & Compliance

---

## Executive Summary

**What this chapter covers:**
- The multiple definitions of algorithmic fairness and why they conflict
- Concrete metrics: Disparate Impact Ratio, Equal Opportunity, Calibration
- Measuring bias in your credit model across protected groups
- Bias mitigation techniques: reweighting, threshold adjustment, calibration
- Production monitoring and regulatory documentation
- The critical insight: validation fairness ≠ production fairness

**Key takeaways:**
- Fairness has multiple competing definitions—you must choose and justify
- Mathematical impossibility results mean you can't satisfy all fairness criteria
- The 4/5ths rule (DIR ≥ 0.80) is the legal baseline for disparate impact
- Calibration often improves both accuracy AND fairness
- Fairness can drift over time—continuous monitoring is essential
- Documentation matters as much as the technical solution

**Time estimate:**
- Path A (Hands-On): 4-6 hours (reading + coding)
- Path B (Conceptual): 2-3 hours (reading only)

**What you'll build:**
- Fairness metric functions (DIR, SPD, EOD, AOD, ECE)
- Bias measurement dashboards
- Mitigation techniques (reweighting, group thresholds, calibration)
- Production monitoring system
- Regulatory documentation templates

---

## 4.1 Defining Algorithmic Fairness

Chapter 3 built a credit model that achieved strong validation performance but failed on the test set. But even if we'd achieved perfect accuracy, a critical question would remain: **Is the model fair?**

This question is harder than it sounds. Unlike accuracy—where higher is clearly better—fairness has multiple competing definitions. A model can be fair by one definition while being deeply unfair by another.

> 💡 **Key Insight:** Mathematical proofs show that certain combinations of fairness criteria are impossible to satisfy simultaneously. You can't optimize for everything—you must choose.

---

### 4.1.1 What Does "Fair" Mean?

Imagine three loan applicants with identical credit scores (680), incomes ($60,000), and DTI ratios (0.30). The only difference: Alice is White, Bob is Black, and Carlos is Hispanic. Your model assigns them different default probabilities:
- Alice: 8%
- Bob: 12%
- Carlos: 10%

**Is this fair?** The answer depends on which definition of fairness you use.

---

### Definition 1: Fairness Through Blindness (Anti-Classification)

**Principle:** Don't use protected characteristics (race, gender, age) as features.

**The problem:** If ZIP code, name patterns, or shopping behavior serve as proxies for race, your model can discriminate without ever seeing a "race" column. Courts have ruled that disparate impact—harm to protected groups regardless of intent—can violate fair lending laws even when protected characteristics aren't explicit features.

**Verdict:** Necessary but not sufficient.

---

### Definition 2: Demographic Parity (Statistical Parity)

**Principle:** Approve loans at equal rates across protected groups.

**Mathematical definition:**
```
P(Ŷ = approve | A = White) = P(Ŷ = approve | A = Black)
```

**In plain English:** If 70% of White applicants are approved, 70% of Black applicants should be approved.

**Strengths:**
- Simple to measure and explain
- Aligns with "disparate impact" doctrine (4/5ths rule)

**Weaknesses:**
- Ignores whether groups have different base rates of default
- May require approving higher-risk applicants from one group

**Legal threshold:** The 4/5ths rule states that if the approval rate for any group is less than 80% of the highest group's rate, there may be disparate impact.

```python
disparate_impact_ratio = black_approval_rate / white_approval_rate
# If DIR < 0.80, potential violation
```

---

### Definition 3: Equalized Odds

**Principle:** Equal true positive rates AND equal false positive rates across groups.

**About errors in credit scoring:**

**Table 4.1: Error Types in Credit Scoring**

| Reality | Prediction | Decision | Result | Name |
|---------|------------|----------|--------|------|
| Pays back | No default | **APPROVE** | Correct ✓ | True Negative |
| Pays back | Default | **DENY** | Wrong ✗ | False Positive |
| Defaults | No default | **APPROVE** | Wrong ✗ | False Negative |
| Defaults | Default | **DENY** | Correct ✓ | True Positive |

**Key insight:**
- **False Positive** = Deny a good borrower (they would have paid back)
- **False Negative** = Approve a bad borrower (they will default)

**In plain English:**
- Among people who will default, catch them at equal rates regardless of race (Equal TPR)
- Among people who won't default, falsely deny them at equal rates regardless of race (Equal FPR)

**Intuition:** The model's errors should be equally distributed across groups.

---

### Definition 4: Calibration (Predictive Parity)

**Principle:** When the model predicts a certain probability, that prediction should be equally accurate across groups.

**In plain English:** If the model predicts 15% default risk, about 15% of those people should actually default—regardless of whether they're White or Black.

**Example of miscalibration:**
```
White applicants predicted at 15% risk:
  - 14% actually default (well-calibrated ✓)

Black applicants predicted at 15% risk:
  - 28% actually default (under-predicting risk! ✗)
```

**Why miscalibration harms even when it seems like a "benefit":** Under-predicting risk means Black borrowers get approved for loans they can't afford, leading to defaults, credit damage, and financial hardship.

---

### 4.1.2 The Impossibility Result

**Theorem (Chouldechova, 2016; Kleinberg et al., 2016):**

When base rates differ between groups (e.g., Group A has 5% default rate, Group B has 10%), it is mathematically impossible to simultaneously achieve:
1. Equal false positive rates
2. Equal false negative rates  
3. Calibration

**Implication:** You must choose which fairness criteria to prioritize. There is no "fair across all definitions" solution when groups have different base rates.

> 🎓 **Teaching Note:** This impossibility result is not a technicality—it's fundamental. Any claim that a model is "fair" must specify which definition of fairness and acknowledge what was sacrificed.

---

### 4.1.3 Recommended Prioritization for Credit Scoring

**Tier 1: Must satisfy (legal compliance)**
- ✅ **Disparate Impact Ratio ≥ 0.80** for all protected groups
- ✅ **Anti-classification:** No protected characteristics as features
- ✅ **Explainability:** Can generate adverse action notices

**Tier 2: Should optimize (business + ethics)**
- ✅ **Calibration:** Similar ECE across groups (< 0.05 difference)
- ✅ **Equal Opportunity:** Similar TPR across groups (< 0.10 difference)

**Tier 3: Monitor (detect issues early)**
- 📊 Statistical Parity Difference
- 📊 Average Odds Difference
- 📊 Precision/Recall by group

---

## 4.2 Measuring Bias in Our Model

### 4.2.1 Fairness Metrics Implementation

Before calculating fairness metrics, we need to convert the model's probability predictions into binary decisions (0 or 1) by applying a threshold:

```python
# Model outputs probabilities
y_prob = best_model.predict_proba(X_val)[:, 1]  # Probability of default

# Apply threshold to get binary predictions
threshold = 0.20  # Optimized on validation data
y_pred = (y_prob >= threshold).astype(int)

# y_pred is now an array of 0s and 1s:
# 0 = predicted no default (approve)
# 1 = predicted default (deny/flag)
```

Now let's implement the key fairness metrics. Each metric takes `y_pred` (the binary predictions) and compares outcomes across protected groups:

```python
def disparate_impact_ratio(y_pred, protected_attr, unprivileged_value, privileged_value):
    """
    Disparate Impact Ratio: Are approval rates equal across groups?
    
    DIR = P(approved | unprivileged) / P(approved | privileged)
    
    Interpretation:
        DIR >= 0.80: Passes 4/5ths rule
        DIR < 0.80: Potential disparate impact violation
    """
    unprivileged_mask = (protected_attr == unprivileged_value)
    privileged_mask = (protected_attr == privileged_value)
    
    # Approval = predict no default (y_pred == 0)
    unprivileged_approval = (y_pred[unprivileged_mask] == 0).mean()
    privileged_approval = (y_pred[privileged_mask] == 0).mean()
    
    return unprivileged_approval / privileged_approval


def equal_opportunity_difference(y_true, y_pred, protected_attr, unprivileged, privileged):
    """
    Equal Opportunity Difference: Are true positive rates equal?
    
    EOD = TPR(unprivileged) - TPR(privileged)
    
    Interpretation:
        EOD = 0: Perfect equality
        |EOD| < 0.10: Generally acceptable
    """
    unprivileged_mask = (protected_attr == unprivileged)
    privileged_mask = (protected_attr == privileged)
    
    # TPR = P(predict default | actually defaults)
    tpr_unprivileged = recall_score(y_true[unprivileged_mask], 
                                     y_pred[unprivileged_mask])
    tpr_privileged = recall_score(y_true[privileged_mask], 
                                   y_pred[privileged_mask])
    
    return tpr_unprivileged - tpr_privileged


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Expected Calibration Error: How well do probabilities match reality?
    
    ECE = Σ |bin_accuracy - bin_confidence| × (bin_size / total)
    
    Interpretation:
        ECE < 0.05: Well calibrated
        ECE > 0.10: Poorly calibrated
    """
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

### 4.2.2 Analyzing Our Model

We implemented five fairness metrics in Section 4.2.1 (DIR, SPD, EOD, AOD, ECE), but we'll focus on three for our analysis:

- **DIR (Disparate Impact Ratio):** Legal compliance — the 4/5ths rule
- **EOD (Equal Opportunity Difference):** Equal treatment of actual defaulters
- **ECE (Calibration):** Probability honesty across groups

SPD and AOD are logged for monitoring but often correlate with DIR and EOD, so we omit them here to avoid redundancy. In your own work, choose metrics that align with your prioritization framework (Section 4.1.3).

```python
# Calculate metrics for validation set
print("FAIRNESS METRICS (Black vs White)")
print("="*50)

# Disparate Impact Ratio
dir_black = disparate_impact_ratio(y_pred_val, race_val, 'Black', 'White')
print(f"Disparate Impact Ratio: {dir_black:.3f}")
print(f"  Status: {'✓ PASS' if dir_black >= 0.80 else '✗ FAIL'}")

# Equal Opportunity Difference
eod_black = equal_opportunity_difference(y_val, y_pred_val, race_val, 'Black', 'White')
print(f"\nEqual Opportunity Difference: {eod_black:+.3f}")
print(f"  Status: {'✓ PASS' if abs(eod_black) < 0.10 else '⚠️ CONCERN'}")

# Calibration by group
ece_white = expected_calibration_error(y_val[race_val=='White'], 
                                        y_prob_val[race_val=='White'])
ece_black = expected_calibration_error(y_val[race_val=='Black'], 
                                        y_prob_val[race_val=='Black'])
print(f"\nCalibration (ECE):")
print(f"  White: {ece_white:.3f}")
print(f"  Black: {ece_black:.3f}")
print(f"  Difference: {abs(ece_white - ece_black):.3f}")
```

**Actual Results (Validation Set):**
```
FAIRNESS METRICS (Black vs White)
==================================================
Disparate Impact Ratio: 0.992
  Status: ✓ PASS

Equal Opportunity Difference: -0.182
  Status: ⚠️ CONCERN (exceeds 0.10 threshold)

Calibration (ECE):
  White: 0.041
  Black: 0.066
  Difference: 0.025
```

**The surface looks fair; the depths don't.** DIR passes comfortably — approval rates are roughly equal across groups (90–92%). But EOD fails: the model catches 18.2% of White defaults but 0% of Black defaults on validation. The model detects defaults unequally by race.

> 📌 **Teaching Note: Equal Approval ≠ Equal Treatment.** The model denies almost nobody, so approval rates look fair. But protection from bad loans is completely unequal — if you're Black and about to default, the model won't flag you, meaning you get approved for a loan you can't repay. That's harm through apparent approval.

> ⚠️ **Small Sample Caveat:** These EOD metrics are based on very small default counts per group — as few as 1 actual default for Black applicants in the validation set. One more or fewer default would swing EOD dramatically. In production, you'd need larger samples before drawing conclusions about differential TPR.

*Figure 4.1: Fairness Dashboard (6-Panel)*

The dashboard reveals the split personality of our fairness results. Tier 1 metrics (DIR, SPD) pass for all racial groups and both genders. But Tier 2 metrics tell a different story: EOD fails for race (−0.182 for Black vs White) though gender EOD results are inconsistent between validation (−0.250) and test (+0.014). The calibration panel shows ECE is highest for Black applicants (0.064 on test) — the model's probabilities are least honest for this group.

**Test set confirms the pattern:** DIR passes for all groups (0.995–1.004). But EOD is −0.176 for *every* minority group — the model only identifies defaults among White applicants. Black, Hispanic, and Asian defaulters all receive 0% TPR on test data.

### 4.2.3 Root Causes of Bias

**Data-level causes:**
- Historical bias in training data (past lending discrimination)
- Measurement bias (credit bureau data quality varies by population)
- Representation bias (some groups underrepresented)

**Model-level causes:**
- Proxy discrimination (ZIP code correlates with race)
- Threshold effects (single threshold may disadvantage some groups)
- SMOTE amplification (synthetic data may encode existing patterns)

**Critical question:** Does the distribution shift from Chapter 3 affect groups differently?

---

## 4.3 Bias Mitigation Techniques

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     BIAS MITIGATION STRATEGIES                          │
├─────────────────────────────────────────────────────────────────────────┤
│  PRE-PROCESSING          IN-PROCESSING           POST-PROCESSING        │
│  ───────────────         ─────────────           ───────────────        │
│  Fix the DATA            Fix the MODEL           Fix the OUTPUT         │
│  before training         during training         after prediction       │
│                                                                         │
│  • Reweighting           • Fairness              • Threshold            │
│  • Resampling              constraints             adjustment           │
│  • Feature removal       • Adversarial           • Calibration          │
│                            debiasing               by group             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 4.3.1 Pre-Processing: Reweighting

**Idea:** Give different weights to samples so protected groups have equal influence during training.

First, let's clarify what `prot_train` is:

```python
# prot_train is the protected attribute column (e.g., race or region)
# split alongside X_train and y_train. It is NOT used as a model feature —
# the model never sees it during training or prediction. We keep it 
# separate solely for fairness calculations: computing reweighting,
# measuring metrics by group, and auditing outcomes.

# Example split (from Chapter 3):
X_train = data[feature_cols]       # Features the model trains on
y_train = data['defaulted']        # Target variable
prot_train = data['region']        # Protected attribute (kept separate)
```

Now we can calculate reweighting:

```python
def calculate_reweighting_weights(y_true, protected_attr, privileged_value):
    """
    Calculate sample weights to balance outcomes across groups.
    
    Weight formula: w = P(Y) * P(A) / P(Y, A)
    
    Interpretation:
        weight > 1: Underrepresented, increase influence
        weight < 1: Overrepresented, reduce influence
    """
    # ... implementation (see Appendix C, Section C.4 for complete code)
    return weights

# Calculate weights using the protected attribute
weights = calculate_reweighting_weights(y_train, prot_train, privileged_value='A')

# Train with weights — model still only sees X_train, not prot_train
model_reweighted.fit(X_train, y_train, sample_weight=weights)
```

**When reweighting helps most:**
- Severe imbalance between groups
- Different default rates across groups
- Historical bias in training labels

**What happened with our model:** We reweighted using binary race labels (White/Black, 70/30 split). The result was sobering — ROC-AUC dropped from 0.658 to 0.589 for a marginal DIR improvement (0.992 → 1.007) on a metric that was already passing. The model essentially gave up on distinguishing defaulters from non-defaulters, approving nearly everyone.

> 📌 **Teaching Note:** Reweighting traded real predictive performance for a marginal improvement on a metric that wasn't failing. The actual fairness gaps were in EOD and TPR — which binary reweighting on a simplified race variable didn't address. In production, reweight on the actual group labels targeting the metric you're trying to fix.

---

### 4.3.2 In-Processing: Fairness-Constrained Training

**Idea:** Modify the model's training to penalize both prediction errors AND fairness violations during learning—not before or after, but as the model learns.

```python
from sklearn.linear_model import LogisticRegression

def train_with_fairness_constraint(X_train, y_train, prot_train, dir_threshold=0.80):
    """
    Train multiple models with different hyperparameters and
    select the one with best accuracy that meets fairness constraints.
    """
    best_model = None
    best_score = 0
    
    # C is the regularization strength parameter:
    # - Small C (0.01): Heavy regularization → simpler model, may underfit
    # - Large C (10.0): Light regularization → complex model, may overfit
    for C_val in [0.01, 0.1, 1.0, 10.0]:
        for class_wt in [None, 'balanced']:
            model = LogisticRegression(C=C_val, class_weight=class_wt,
                                       max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_train)
            
            # Check fairness constraint
            dir_score = disparate_impact_ratio(preds, prot_train,
                                               unprivileged_value='C',
                                               privileged_value='A')
            
            if dir_score >= dir_threshold:
                accuracy = (preds == y_train).mean()
                if accuracy > best_score:
                    best_score = accuracy
                    best_model = model
    
    return best_model

model_constrained = train_with_fairness_constraint(X_train, y_train, prot_train)
```

> 💡 **Note:** We use Logistic Regression here to demonstrate the in-processing concept because its `C` parameter provides a simple way to explore the accuracy-fairness trade-off. In practice, you would apply similar constrained optimization to your best-performing model (XGBoost) using libraries like Fairlearn's `ExponentiatedGradient` or `GridSearch` with fairness constraints.

---

### 4.3.3 Post-Processing: Threshold Adjustment

**Idea:** Use different decision thresholds for different groups to equalize approval rates.

```python
def find_threshold_for_parity(y_prob, protected_attr, target_approval_rate, group):
    """
    Find threshold that achieves target approval rate for a SPECIFIC group.
    """
    # Filter to just this group
    group_mask = (protected_attr == group)
    group_probs = y_prob[group_mask]
    
    thresholds = np.linspace(0, 1, 100)
    for thresh in thresholds:
        approval_rate = (group_probs < thresh).mean()  # approve if below threshold
        if approval_rate >= target_approval_rate:
            return thresh
    return 0.5

# Example: Find thresholds to achieve 85% approval rate for each group
target_rate = 0.85
thresh_A = find_threshold_for_parity(y_prob_val, prot_val, target_rate, group='A')
thresh_C = find_threshold_for_parity(y_prob_val, prot_val, target_rate, group='C')

print(f"Group A threshold: {thresh_A:.2f}")
print(f"Group C threshold: {thresh_C:.2f}")
# Different thresholds achieve same approval rate → demographic parity
```

**What happened with our model:** Group-specific thresholds equalized approval rates across groups (all within 90.0–90.7%) at only 2% accuracy cost. DIR improved to ~1.0. But EOD remained unchanged at −0.176 — adjusting the cutoff changes *who gets approved* but doesn't change *how well the model detects defaults per group*.

**⚠️ Trade-offs:**
- Legally controversial (explicit differential treatment by group)
- Only fixes approval rate parity, not differential default detection
- Preserves the model's ranking ability (ROC-AUC unchanged)

---

### 4.3.4 Post-Processing: Calibration

**Idea:** Adjust probabilities so they're honest across groups.

**How calibration works:** The model's raw probabilities are often miscalibrated — when it predicts 15% default risk, the actual default rate might be 50%. Calibration learns a mapping from raw probabilities to honest ones using validation data.

**Isotonic regression** (`method='isotonic'`) groups the model's validation predictions into bins and asks: "When the model predicted ~15%, what fraction actually defaulted?" If the answer is 50%, it remaps 0.15 → 0.50 for any future prediction. The mapping is constrained to be non-decreasing — higher raw probability always maps to equal or higher calibrated probability.

**Platt scaling** (`method='sigmoid'`) fits a logistic S-curve through the same mapping, learning just two parameters. It's smoother but less flexible — it assumes the miscalibration follows an S-shaped pattern. Use Platt when you have limited validation data; use isotonic when you have enough data for the step function to be reliable.

```python
from sklearn.calibration import CalibratedClassifierCV

# base_model is our tuned XGBoost from Chapter 3
base_model = grid_search.best_estimator_  # Already trained on X_train

# Step 1: Wrap the trained model (no calibration yet)
calibrated_model = CalibratedClassifierCV(
    base_model, 
    method='isotonic',  # Non-decreasing step function (more flexible)
    cv='prefit'         # 'prefit' means base_model is already trained
)

# Step 2: Learn the calibration mapping on validation data
# This learns: "When base_model predicts 0.15, what's the actual default rate?"
# If 6 validation accounts had predictions around 0.10-0.20 and 3 actually
# defaulted, the calibrated probability for that range becomes ~0.50
calibrated_model.fit(X_val, y_val)

# Step 3: Apply the learned mapping to test data
y_prob_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
# Now probabilities are "honest" — if it says 30%, roughly 30% actually default
```

**Why calibration often works best:**
- Improves both accuracy AND fairness
- Doesn't require explicit group treatment
- Makes probabilities meaningful for business decisions

**What happened with our model:** Group-specific calibration (isotonic regression per race group) achieved perfect fairness metrics on validation — ECE dropped to 0.000 for all groups, DIR = 1.0, EOD = 0.0. But on test data, ECE *increased* for every group (White: 0.019 → 0.025, Black: 0.064 → 0.086). The calibrators memorized the validation data (50–97 samples per group) rather than learning a robust mapping.

> 📌 **Teaching Note:** With 50–97 validation accounts per group, isotonic regression memorized noise rather than learning a generalizable mapping. Group-specific calibration requires either larger calibration sets or simpler methods (Platt scaling, with fewer parameters) to avoid overfitting.

*Figure 4.2: Calibration Comparison (Original vs. Platt vs. Isotonic)*

---

### 4.3.5 Comparing Approaches

**Table 4.2: Mitigation Approach Comparison (Test Set)**

*Evaluated on held-out test data — the honest measure. DIR ≥ 0.80 passes the 4/5ths rule. |EOD| < 0.10 indicates equal opportunity.*

| Approach | ROC-AUC | DIR (B/W) | |EOD| (B/W) | Assessment |
|----------|---------|-----------|-------------|------------|
| Original | 0.683 | 0.995 | 0.176 | Best ranking, EOD fails |
| Reweighted | 0.604 | 1.004 | 0.000 | EOD passes, ranking destroyed |
| Group Thresholds | 0.683 | 1.018 | 0.176 | Ranking preserved, EOD unchanged |
| Calibrated (by group) | 0.609 | 1.000 | 0.000 | EOD passes, ranking destroyed |

Figure 4.3 plots each approach on ROC-AUC vs |EOD|, making the trade-off visually clear:

*Figure 4.3: ROC-AUC vs. Fairness Trade-off (Validation and Test)*

**The ideal corner — top-left (high ROC-AUC, low EOD) — is empty.** No approach achieves both strong ranking and equal opportunity on test data. Two distinct clusters emerge:

- **Right side (Original, Group Thresholds):** Preserve ROC-AUC at 0.683 but fail EOD at 0.176. The model ranks well but detects defaults unequally.
- **Bottom-left (Reweighted, Calibrated):** Pass EOD at 0.0 but drop ROC-AUC to ~0.60. They achieved fairness by approving nearly everyone — fairness through inaction, not through better predictions.

Note how Calibrated collapsed from the top-left on validation (ROC-AUC 0.822, "perfect" fairness) to the bottom-left on test (ROC-AUC 0.609) — the overfitting we predicted from 50–97 samples per group.

> 💡 **Key Finding:** Group-specific thresholds are the most honest trade-off — they fix approval rate parity (DIR ≈ 1.0) at minimal cost while preserving ranking ability. Reweighting and calibration appear to "solve" EOD, but only by losing the ability to distinguish defaulters from non-defaulters. With a 6.5% default rate, approving everyone gives 93.5% accuracy automatically — high accuracy ≠ good model.

---

## 4.4 Production Monitoring

### 4.4.1 Why Monitoring Matters

**Critical insight from our analysis:** While DIR remained stable between validation and test (0.992 → 0.995), deeper metrics shifted dramatically. Gender EOD swung from −0.250 (validation) to +0.014 (test) — a 26-point reversal explained entirely by small default counts per group. Race EOD was more consistent (−0.182 → −0.176), giving confidence that finding is real, not a small-sample artifact.

Without monitoring, you'd never distinguish real fairness issues from statistical noise!

### 4.4.2 Monitoring Dashboard

```python
def create_fairness_snapshot(y_true, y_pred, y_prob, protected_attr, 
                              threshold, dataset_name):
    """
    Create a snapshot of fairness metrics for monitoring.
    """
    snapshot = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'threshold': threshold,
        'n_samples': len(y_true),
        'metrics': {}
    }
    
    # Calculate metrics for each group
    for group in ['White', 'Black', 'Hispanic', 'Asian']:
        mask = (protected_attr == group)
        if mask.sum() > 0:
            snapshot['metrics'][group] = {
                'n_samples': int(mask.sum()),
                'approval_rate': float((y_pred[mask] == 0).mean()),
                'default_rate': float(y_true[mask].mean()),
                'avg_probability': float(y_prob[mask].mean())
            }
    
    # Calculate DIR
    if 'White' in snapshot['metrics'] and 'Black' in snapshot['metrics']:
        white_rate = snapshot['metrics']['White']['approval_rate']
        black_rate = snapshot['metrics']['Black']['approval_rate']
        snapshot['dir_black_vs_white'] = black_rate / white_rate if white_rate > 0 else None
    
    return snapshot
```

### 4.4.3 Alert Thresholds

In production, you'd compute DIR on a rolling window of recent model decisions — typically 30 days of data to ensure sufficient sample sizes per group. A weekly batch job runs the `create_fairness_snapshot()` function against the latest decisions and compares DIR to the thresholds below. Alerts route to the model risk team, who triage based on severity. The 7-day investigation window for YELLOW reflects regulatory expectations for timely remediation; RED triggers an immediate review that may include temporarily reverting to the previous model version or applying manual overrides while the issue is diagnosed.

**Table 4.3: Monitoring Alert Thresholds**

| Level | DIR Range | Action |
|-------|-----------|--------|
| 🟢 GREEN | ≥ 0.90 | Normal monitoring |
| 🟡 YELLOW | 0.80-0.90 | Investigate within 7 days |
| 🔴 RED | < 0.80 | Immediate escalation |

---

## 4.5 Regulatory Documentation

### Required Documentation

```
════════════════════════════════════════════════════════════════════════
                    MODEL FAIRNESS DOCUMENTATION
════════════════════════════════════════════════════════════════════════

1. MODEL OVERVIEW
   Model Name:    Credit Default Prediction Model
   Purpose:       Predict probability of loan default for credit decisions
   
   Protected Characteristics Analyzed:
   • Race/Ethnicity (White, Black, Hispanic, Asian)
   • Gender (Male, Female)

2. FAIRNESS DEFINITION & METRICS
   Primary Definition: Disparate Impact (4/5ths rule)
   
   Metrics Measured:
   • Disparate Impact Ratio (DIR) - MUST be >= 0.80
   • Statistical Parity Difference (SPD)
   • Equal Opportunity Difference (EOD)
   • Expected Calibration Error (ECE) by group

3. MITIGATION MEASURES APPLIED
   • Group-specific threshold adjustment for approval rate parity
   • Probability calibration evaluated but overfit on small groups
   
   Trade-offs Accepted:
   • Group thresholds fix DIR at 2% accuracy cost; EOD gap remains
   • Root cause (differential default detection) requires better training data

4. ADVERSE ACTION NOTICES
   Method: SHAP-based feature contribution explanations
   Top 4 contributing features provided for each denial

5. ONGOING MONITORING PLAN
   Frequency:
   • Daily: Approval rates by group
   • Weekly: DIR, SPD metrics
   • Monthly: Full fairness audit

6. APPROVAL SIGNATURES
   Model Developer:     _____________ Date: _______
   Model Validator:     _____________ Date: _______
   Compliance Officer:  _____________ Date: _______
════════════════════════════════════════════════════════════════════════
```

---

## Key Takeaways

### Conceptual Lessons

1. **Multiple fairness definitions exist and conflict** - Demographic parity, equalized odds, calibration all capture different notions of "fair"

2. **Impossibility results are real** - You cannot simultaneously satisfy all fairness criteria when base rates differ

3. **Choose and justify your priorities** - There's no universally correct answer; document your trade-offs

### Technical Lessons

4. **The 4/5ths rule is the legal baseline** — DIR ≥ 0.80 is required; below triggers investigation

5. **Passing DIR doesn't mean passing fairness** — Our model passed DIR easily (approval rates nearly equal) while failing EOD completely (0% minority default detection)

6. **Group thresholds are the most honest post-processing** — They preserve model ranking (ROC-AUC unchanged) and fix approval parity at low accuracy cost, but cannot fix what the model never learned

7. **Calibration can overfit on small groups** — Group-specific isotonic regression with 50–97 samples memorized noise. Validation perfection (ECE = 0.000) collapsed on test data

8. **"Fairness through inaction" is a trap** — Both reweighting and calibration achieved EOD = 0 by approving nearly everyone. High accuracy with low ROC-AUC means the model stopped distinguishing defaulters from non-defaulters

### Process Lessons

9. **Always evaluate on test data** — Validation fairness results can be wildly optimistic (our calibration went from "perfect" to "worse than baseline")

10. **Compare metrics across datasets** — Consistent findings across val and test (like race EOD) are more trustworthy than inconsistent ones (like gender EOD)

11. **Documentation matters** — Regulators want to see your reasoning, not just your results

12. **Fairness from day one** — Harder to retrofit than to design in

---

## Common Pitfalls to Avoid

**Table 4.4: Common Fairness Pitfalls**

| Pitfall | Why It Fails | Fix |
|---------|--------------|-----|
| "We don't collect race, so no bias" | Proxies (ZIP code) can discriminate | Audit with external data |
| "Data shows real risk differences" | Historical data may reflect past discrimination | Question your data |
| "We optimized for fairness, done" | Fairness drifts over time; validation results can overfit | Evaluate on test data; continuous monitoring |
| "Perfect fairness is impossible, why try?" | Can improve any specific metric | Choose and measure |
| "We'll add fairness later" | Harder to retrofit | Design in from start |

---

## 4.6 Lessons Learned

This chapter attempted three approaches to fix fairness — reweighting, group-specific thresholds, and group-specific calibration. Here's what we learned:

### The Empty Top-Left Corner

When we plotted ROC-AUC against |EOD| for all four approaches on test data, the ideal corner (high performance, high fairness) was empty. Every approach either preserved the model's signal or achieved fairness metrics, but not both. This is the central tension of fair ML, and it's honest to show it.

### Fairness Through Inaction

Reweighting and calibration both achieved EOD = 0 — but by approving nearly everyone. With a 6.5% default rate, approving everyone gives 93.5% accuracy automatically. Their "high accuracy" came from denying almost nobody, not from making better predictions. A broken smoke detector that never beeps has a great track record in a building that rarely catches fire.

### The Root Cause Is the Model, Not the Post-Processing

The model fundamentally detects White defaults better than minority defaults. No post-processing can add signal that doesn't exist. Group thresholds can adjust *who gets approved*, but not *how well the model identifies risk per group*. The true fix requires more representative training data, features that generalize across groups, and larger samples for group-specific techniques.

### What Worked Best

Group-specific thresholds were the most honest intervention — they fixed what could be fixed (approval rate parity) at minimal cost (2% accuracy, 0% ROC-AUC loss), without pretending to fix what couldn't be fixed (differential default detection). Combined with transparent documentation of the remaining EOD gap and a plan to address it through better data, this is a defensible position.

### Small Samples Make Everything Fragile

With 1–8 actual defaults per racial group in validation, fairness metrics were inherently unstable. Gender EOD swung 26 points between datasets. Race EOD was more consistent — giving more confidence that finding was real. The lesson: consistency across datasets matters more than any single pass/fail result.

---

## Teaching Notes

### Learning Objectives

By the end of this chapter, learners should be able to:

**LO1: Recognize Fairness Definitions**
- Explain demographic parity, equalized odds, and calibration
- Articulate why these definitions conflict
- Apply the impossibility result to real scenarios

**LO2: Measure Fairness**
- Calculate DIR, EOD, SPD, and ECE
- Interpret metric values and thresholds
- Identify which metrics are passing/failing

**LO3: Mitigate Bias**
- Implement reweighting, threshold adjustment, and calibration
- Compare accuracy-fairness trade-offs
- Choose appropriate mitigation strategies

**LO4: Monitor and Document**
- Set up fairness monitoring dashboards
- Create regulatory documentation
- Establish alert thresholds and escalation procedures

### Discussion Questions

1. **The Impossibility Question:** If you can't satisfy all fairness criteria simultaneously, how do you decide which to prioritize? What role should affected communities play in this decision?

2. **The Calibration Trap:** Group-specific calibration looked perfect on validation but degraded on test. What minimum sample size per group would you require before trusting group-specific calibration? Would Platt scaling (2 parameters) have been more robust than isotonic regression?

3. **The Threshold Dilemma:** Group-specific thresholds preserved ranking ability while fixing approval parity, but couldn't fix EOD. Is there ever a case where differential thresholds by group are the right answer despite the legal controversy?

4. **The Empty Corner:** No approach achieved both high ROC-AUC and low EOD. If the root cause is the model not learning minority default patterns, what data or features would you need to fill that top-left corner?

### Key Terms Introduced

**Table 4.5: Key Terms - Fairness & Compliance**

| Term | Definition |
|------|------------|
| **Disparate Impact** | Disproportionate harm to protected groups, regardless of intent |
| **4/5ths Rule** | Approval rate for any group must be ≥ 80% of highest group |
| **Demographic Parity** | Equal approval rates across groups |
| **Equalized Odds** | Equal TPR and FPR across groups |
| **Calibration** | Predicted probabilities match actual frequencies |
| **ECE** | Expected Calibration Error - measures probability honesty |
| **Reweighting** | Adjusting sample weights to balance group influence |

---

*End of Chapter 4*

---

*Next: Chapter 5 — Conclusion & Future Directions*
