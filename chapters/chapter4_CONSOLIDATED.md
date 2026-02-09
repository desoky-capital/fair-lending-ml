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

## 4.1 Understanding Algorithmic Fairness

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

**Understanding errors in credit scoring:**

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
threshold = 0.25  # Optimized on validation data
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
    # ... implementation
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

**When reweighting has little effect:**
- Groups already well-represented
- Similar outcome distributions across groups

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

**⚠️ Warning:** Group-specific thresholds are controversial:
- Legally risky (explicit differential treatment)
- Can severely hurt accuracy (as we saw in Table 4.2: 31.4% accuracy)
- May not address root cause

---

### 4.3.3 Post-Processing: Calibration

**Idea:** Adjust probabilities so they're honest across groups.

```python
from sklearn.calibration import CalibratedClassifierCV

# base_model is our tuned XGBoost from Chapter 3
base_model = grid_search.best_estimator_  # Already trained on X_train

# Step 1: Wrap the trained model (no calibration yet)
calibrated_model = CalibratedClassifierCV(
    base_model, 
    method='isotonic',  # or 'sigmoid' for less flexible calibration
    cv='prefit'         # 'prefit' means base_model is already trained
)

# Step 2: Learn the calibration mapping on validation data
# This learns: "When base_model predicts 0.30, what's the actual default rate?"
calibrated_model.fit(X_val, y_val)

# Step 3: Apply the learned mapping to test data
y_prob_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
# Now probabilities are "honest" — if it says 30%, roughly 30% actually default
```

**Why calibration often works best:**
- Improves both accuracy AND fairness
- Doesn't require explicit group treatment
- Makes probabilities meaningful for business decisions

---

### 4.3.4 Comparing Approaches

**Table 4.2: Mitigation Approach Comparison**

*All results evaluated on validation data. "Original" refers to the tuned XGBoost model before any fairness mitigation. Each approach was applied independently to isolate its effect.*

| Approach | DIR | Accuracy | Recommendation |
|----------|-----|----------|----------------|
| Original | 1.03 | 94.8% | Baseline |
| Reweighting (pre-processing) | ~1.03 | ~95% | Minimal effect (data balanced) |
| Fairness-Constrained (in-processing) | ~0.95 | ~93% | Modest improvement, some accuracy cost |
| Group Thresholds (post-processing) | 1.00 | 31.4% | ❌ Destroyed accuracy |
| Calibration (post-processing) | ~1.00 | 96.9% | ✓ Best balance |

> ⚠️ **Important Limitation:** These results are evaluated on validation data only. As Chapter 3 demonstrated, validation performance may not transfer to test data due to distribution shift. In production, you should evaluate all mitigation techniques on held-out test data before deployment.

> 💡 **Key Finding:** Calibration improved both accuracy (94.8% → 96.9%) AND fairness. Group thresholds achieved perfect parity but destroyed accuracy (94.8% → 31.4%).

---

## 4.4 Production Monitoring

### 4.4.1 Why Monitoring Matters

**Critical insight from Chapter 3:** Our model's fairness changed between validation and test sets:
- Validation DIR: 1.030 (passing)
- Test DIR: 0.955 (still passing, but dropped)

Without monitoring, you'd never catch this drift!

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
   • Probability calibration by group
   
   Trade-offs Accepted:
   • Calibration chosen as best balance of fairness and accuracy

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

4. **The 4/5ths rule is the legal baseline** - DIR ≥ 0.80 is required; below triggers investigation

5. **Calibration often wins** - Improves both accuracy and fairness without explicit differential treatment

6. **Group thresholds are dangerous** - Can achieve perfect parity but destroy accuracy

7. **Reweighting helps when data is imbalanced** - Has minimal effect when groups are already balanced

### Process Lessons

8. **Validation fairness ≠ production fairness** - Metrics can drift; continuous monitoring is essential

9. **Documentation matters** - Regulators want to see your reasoning, not just your results

10. **Fairness from day one** - Harder to retrofit than to design in

---

## Common Pitfalls to Avoid

**Table 4.4: Common Fairness Pitfalls**

| Pitfall | Why It Fails | Fix |
|---------|--------------|-----|
| "We don't collect race, so no bias" | Proxies (ZIP code) can discriminate | Audit with external data |
| "Data shows real risk differences" | Historical data may reflect past discrimination | Question your data |
| "We optimized for fairness, done" | Fairness drifts over time | Continuous monitoring |
| "Perfect fairness is impossible, why try?" | Can improve any specific metric | Choose and measure |
| "We'll add fairness later" | Harder to retrofit | Design in from start |

---

## Teaching Notes

### Learning Objectives

By the end of this chapter, learners should be able to:

**LO1: Understand Fairness Definitions**
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

2. **The Calibration Paradox:** Calibration improved both accuracy and fairness. Is this always possible, or did we get lucky? When might calibration hurt fairness?

3. **The Threshold Dilemma:** Group-specific thresholds achieved perfect demographic parity but destroyed accuracy. Is there ever a case where this trade-off is worth it?

4. **The Monitoring Question:** Our DIR dropped from 1.03 (validation) to 0.95 (test). At what point should you retrain vs. adjust thresholds vs. investigate root causes?

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
