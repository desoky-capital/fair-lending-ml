# Section 4.1: Understanding Algorithmic Fairness

Section 2 built a credit default model that achieved strong validation performance (ROC-AUC 0.696) but failed catastrophically on the test set (0% precision, 0% recall). We diagnosed the failure: distribution shift caused the model to assign probabilities 3x lower to test defaults than validation defaults. SHAP analysis revealed the model learned patterns from SMOTE-augmented training data that didn't generalize.

But even if we'd achieved perfect accuracy, a critical question would remain: **Is the model fair?**

This question is harder than it sounds. Unlike accuracy—where higher is clearly better—fairness has multiple competing definitions, each capturing different notions of what "fair" means. A model can be fair by one definition while being deeply unfair by another. Worse, mathematical proofs show that certain combinations of fairness criteria are impossible to satisfy simultaneously.

This section provides the foundation you'll need to:
1. Understand the major definitions of algorithmic fairness
2. Measure fairness using concrete metrics
3. Recognize the trade-offs between different fairness criteria
4. Choose appropriate fairness metrics for credit scoring contexts
5. Navigate the legal and ethical landscape

By the end, you'll be equipped to move beyond vague statements like "our model is fair" to precise, measurable claims like "our model achieves demographic parity within 5 percentage points and maintains calibration across all protected groups."

---

## 3.1.1 What Does "Fair" Mean? The Core Definitions

Imagine three loan applicants with identical credit scores (680), incomes ($60,000), and debt-to-income ratios (0.30). The only difference: Alice is White, Bob is Black, and Carlos is Hispanic. Your model assigns them different default probabilities:
- Alice: 8%
- Bob: 12%
- Carlos: 10%

**Is this fair?**

The answer depends on which definition of fairness you use.

### Definition 1: Fairness Through Blindness (Anti-Classification)

**Principle:** The algorithm should not use protected characteristics (race, gender, age, etc.) as features.

**Applied to our example:** If the model doesn't use race as a feature, it satisfies this definition even if outcomes differ by group.

**Intuition:** You can't discriminate based on race if you never look at race.

**The problem:** As Section 1 discussed, this is legally insufficient. If ZIP code, name patterns, or shopping behavior serve as proxies for race, your model can discriminate without ever seeing a "race" column. Courts have consistently ruled that disparate impact—harm to protected groups regardless of intent—can violate fair lending laws even when protected characteristics aren't explicit features.

**Verdict:** Necessary but not sufficient. Always exclude protected characteristics from features, but don't stop there.

---

### Definition 2: Demographic Parity (Statistical Parity)

**Principle:** The model should approve loans at equal rates across protected groups.

**Mathematical definition:**
```
P(Ŷ = 1 | A = a) = P(Ŷ = 1 | A = b)

Where:
- Ŷ = predicted outcome (1 = default/deny, 0 = no default/approve)
- A = protected attribute (e.g., race, gender)
- a, b = different values of A (e.g., "White", "Black")
```

**In plain English:** If 70% of White applicants are approved, 70% of Black applicants should be approved.

**Applied to our example:** 
- If model approves 75% of White applicants and 65% of Black applicants → Violates demographic parity
- If model approves 70% of both → Satisfies demographic parity

**Intuition:** Equal outcomes reflect equal treatment.

**Strengths:**
- Simple to measure and explain
- Directly addresses outcome disparities
- Aligns with "disparate impact" doctrine in US law (4/5ths rule)

**Weaknesses:**
- Ignores whether groups have different base rates of default
- May require approving higher-risk applicants from one group
- Can conflict with business objectives (profit, risk management)

**When to use:**
- When you believe qualification rates should be similar across groups
- When historical disparities in outcomes need correction
- When legal compliance requires examining approval rates

**Example calculation:**
```python
# Approval rate by race
# Note: In our convention, predict 1 = default (deny), predict 0 = no default (approve)
white_approval_rate = (y_pred[race == 'White'] == 0).mean()
black_approval_rate = (y_pred[race == 'Black'] == 0).mean()

demographic_parity_ratio = black_approval_rate / white_approval_rate

# Legal threshold: 4/5ths rule (0.80)
# If ratio < 0.80, potential disparate impact
```

---

### Definition 3: Equalized Odds (Equal Opportunity + Equal False Positive Rate)

**Principle:** The model should have equal true positive rates AND equal false positive rates across groups.

**Understanding True Positives and False Positives in Credit Scoring:**

In credit scoring, we predict whether someone will **default** (the "positive" class):
- **Y = 1** means the person **actually defaults** (bad outcome)
- **Y = 0** means the person **actually pays back** (good outcome)
- **Ŷ = 1** means we **predict default** → Deny loan
- **Ŷ = 0** means we **predict no default** → Approve loan

| Reality (Y) | Prediction (Ŷ) | Decision | Result | Name |
|-------------|----------------|----------|--------|------|
| 0 (pays back) | 0 (no default) | **APPROVE** | Correct ✓ | **True Negative** |
| 0 (pays back) | 1 (default) | **DENY** | Wrong - missed opportunity ✗ | **False Positive** |
| 1 (defaults) | 0 (no default) | **APPROVE** | Wrong - will default ✗ | **False Negative** |
| 1 (defaults) | 1 (default) | **DENY** | Correct ✓ | **True Positive** |

**Key insight:** 
- **False Positive** = Deny a good borrower (they would have paid back)
- **False Negative** = Approve a bad borrower (they will default)

**Mathematical definition:**
```
P(Ŷ = 1 | Y = 1, A = a) = P(Ŷ = 1 | Y = 1, A = b)  [Equal TPR]
P(Ŷ = 1 | Y = 0, A = a) = P(Ŷ = 1 | Y = 0, A = b)  [Equal FPR]

Where:
- Y = true outcome (1 = actually defaults, 0 = doesn't default)
- Ŷ = predicted outcome (1 = predict default, 0 = predict no default)
- A = protected attribute
```

**In plain English:** 
- Among people who will default, the model should catch them at equal rates regardless of race (Equal TPR)
- Among people who won't default, the model should falsely deny them at equal rates regardless of race (Equal FPR)

**Applied to our example:**
```
White applicants:
  - Of those who default: 60% correctly denied (TPR)
  - Of those who pay back: 10% wrongly denied (FPR)

Black applicants:
  - Of those who default: 60% correctly denied (TPR) ← Equal!
  - Of those who pay back: 10% wrongly denied (FPR) ← Equal!

→ Satisfies equalized odds
```

**Intuition:** The model's errors should be equally distributed across groups.

**Strengths:**
- Considers actual outcomes, not just predictions
- Addresses both harms: missing true defaults AND wrongly denying good borrowers
- More nuanced than demographic parity
- Often compatible with profit maximization

**Weaknesses:**
- Requires knowing true outcomes (not available at decision time)
- More complex to explain to stakeholders
- May still produce different approval rates if base rates differ

**When to use:**
- When you want to ensure errors don't disproportionately burden one group
- When base rates of default genuinely differ across groups
- When you need to balance false positives and false negatives

**Variant: Equal Opportunity (just equal TPR):**
Some applications only care about equal true positive rates:
```
P(Ŷ = 1 | Y = 1, A = a) = P(Ŷ = 1 | Y = 1, A = b)
```

**Applied to lending:** Among people who will actually default, the model catches them at equal rates across race.

---

### Definition 4: Calibration (Predictive Parity)

**Principle:** When the model predicts a certain probability, that prediction should be equally accurate across groups. A prediction of "15% risk" should mean the same thing regardless of race.

**Mathematical definition:**
```
P(Y = 1 | Ŷ = p, A = a) = P(Y = 1 | Ŷ = p, A = b) = p

Where:
- Ŷ = predicted probability
- p = a specific probability value (e.g., 0.15)
```

**In plain English:** If the model predicts 15% default risk, about 15% of those people should actually default—regardless of whether they're White or Black.

**Applied to our example:**
```
White applicants predicted at 15% risk:
  - 14% actually default (well-calibrated ✓)

Black applicants predicted at 15% risk:
  - 28% actually default (under-predicting risk! ✗)

→ Violates calibration
```

**Why miscalibration harms Black borrowers (even though it seems like a "benefit"):**

At first glance, under-predicting risk seems advantageous—Black applicants get approved more easily. But this creates serious harms:

1. **Risky borrowers pushed into unaffordable debt:** Black applicants who actually have 28% risk get approved because the model says 15%. They then default, ruining their credit and financial stability.

2. **Good Black borrowers subsidize risky ones:** When the bank notices Black borrowers default more than predicted, they raise rates for ALL Black borrowers to compensate.

3. **Enables discriminatory pricing:** Same prediction (15%) leads to different treatment—banks add a "risk premium" for Black borrowers because they don't trust the predictions.

**Intuition:** Predictions should mean the same thing regardless of group membership.

**Strengths:**
- Critical for decision-making (you need to trust probabilities)
- Aligns with how stakeholders use model outputs
- Respects individual risk assessment
- Can coexist with different approval rates if base rates differ

**Weaknesses:**
- Can mask disparate impact if one group has systematically higher risk
- Doesn't address outcome disparities
- Requires careful probability estimation (hard with imbalanced data)

**When to use:**
- When decisions are made based on predicted probabilities
- When you need to set different thresholds for different risk tolerances
- When you believe individual risk assessment matters more than group outcomes

**Checking calibration by group:**

```python
import pandas as pd
import numpy as np

# Create results table comparing predicted vs actual by group and bin
results = []

for group in ['White', 'Black']:
    group_mask = (race == group)
    
    for bin_min, bin_max in [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]:
        in_bin = (y_prob >= bin_min) & (y_prob < bin_max) & group_mask
        
        if in_bin.sum() > 0:
            predicted = y_prob[in_bin].mean()   # Mean predicted probability
            actual = y_true[in_bin].mean()      # Actual default rate
            count = in_bin.sum()
            
            results.append({
                'Group': group,
                'Bin': f'{bin_min:.0%}-{bin_max:.0%}',
                'Count': count,
                'Predicted': f'{predicted:.1%}',
                'Actual': f'{actual:.1%}',
                'Error': f'{abs(actual - predicted):.1%}'
            })

df = pd.DataFrame(results)
print(df.to_string(index=False))
```

**Example output:**
```
 Group      Bin  Count Predicted  Actual  Error
 White   0%-10%     45      5.2%    4.8%   0.4%  ← Well calibrated ✓
 White  10%-20%     38     14.3%   15.1%   0.8%  ← Well calibrated ✓
 Black   0%-10%     52      4.9%   13.2%   8.3%  ← Under-predicting risk! ✗
 Black  10%-20%     41     13.8%   27.4%  13.6%  ← Under-predicting risk! ✗
```

**Interpretation:** For White applicants, predictions match reality. For Black applicants, the model says 5% but 13% actually default—model is "dishonest" for Black borrowers.

---

### Definition 5: Individual Fairness

**Principle:** Similar individuals should receive similar predictions.

**Mathematical definition:**
```
If distance(person_i, person_j) < ε, then |prediction_i - prediction_j| < δ

Where distance measures similarity in relevant features
```

**In plain English:** If Alice and Bob have nearly identical credit profiles, they should get nearly identical predictions—regardless of their race, gender, or other protected characteristics.

**Applied to our example:**
```
Alice (White): FICO 680, Income $60k, DTI 0.30 → Predicted 8%
Bob (Black): FICO 680, Income $60k, DTI 0.30 → Predicted 12%

Same features, different predictions → Violates individual fairness
```

**Intuition:** Fairness is about treating individuals consistently based on their relevant characteristics.

**Strengths:**
- Philosophically appealing (treats people as individuals)
- Catches subtle biases that group-level metrics miss
- Aligns with "merit-based" decision-making

**Weaknesses:**
- Requires defining "similarity" (which features matter?)
- Hard to measure at scale
- Doesn't address systemic disparities
- Can be satisfied while still having disparate impact

**When to use:**
- As a diagnostic tool to find anomalies
- When you want to audit specific decisions
- In combination with group fairness metrics

---

## 3.1.2 The Impossibility Results: Why You Must Choose

It would be wonderful if a model could satisfy all fairness definitions simultaneously. Unfortunately, mathematics says otherwise.

### Theorem 1: Calibration vs. Equalized Odds (Kleinberg et al., 2017)

**Result:** If two groups have different base rates of default, a model CANNOT simultaneously achieve:
1. Perfect calibration in both groups, AND
2. Equal false positive rates across groups, AND
3. Equal false negative rates across groups

**Proof sketch:**
Suppose White applicants default at 5% and Black applicants default at 15%.

For calibration: When model predicts 10%, actual default rate should be 10% in both groups.

For equalized odds: False positive rates must be equal. But if White applicants have lower base rate (5%), achieving the same FPR as Black applicants (15% base rate) requires different decision boundaries, which breaks calibration.

**Implication:** You MUST choose between:
- Calibrated predictions (probabilities mean the same thing across groups), OR
- Equal error rates (same TPR and FPR across groups)

You cannot have both when base rates differ.

### Theorem 2: Demographic Parity vs. Predictive Accuracy

**Result:** Enforcing demographic parity when groups have different default rates requires either:
1. Accepting lower overall accuracy, OR
2. Using different decision thresholds for different groups, OR
3. Believing your data's base rates are wrong (measurement bias)

**Example:**
```
White applicants: 10,000 total, 5% default rate (500 risky, 9,500 safe)
Black applicants: 10,000 total, 15% default rate (1,500 risky, 8,500 safe)

Perfect predictor says:
  - Approve 9,500 White applicants (95% approval rate)
  - Approve 8,500 Black applicants (85% approval rate)
  - Approval rate gap: 10 percentage points (violates demographic parity)
  - Disparate impact ratio: 85%/95% = 0.89 (passes 4/5ths rule, but borderline)

To achieve demographic parity (90% approval for both):
  - Deny 500 additional White applicants who are creditworthy (false positives)
  - Approve 500 additional Black applicants who are risky (false negatives)
  - Result: Lower profit, higher defaults, harms to both groups

Consequences:
  - 500 qualified White borrowers wrongly denied (lost opportunity)
  - 500 unqualified Black borrowers pushed into unaffordable debt (will default)
  - Bank loses money on 500 defaults it could have avoided
```

**Implication:** Demographic parity may conflict with accuracy when base rates differ.

### What This Means for You

**You cannot satisfy all fairness criteria.** You must:

1. **Understand the trade-offs** between definitions
2. **Choose** which definition(s) matter most for your context
3. **Justify** your choice based on:
   - Legal requirements
   - Business objectives  
   - Ethical principles
   - Stakeholder values
4. **Document** what you chose and why
5. **Measure** how well you achieve your chosen definition
6. **Monitor** for drift and unintended consequences

The organizations that get into trouble aren't those who make trade-offs—trade-offs are unavoidable. They're the ones who don't realize they're making trade-offs, don't document them, and can't explain them when questioned.

---

## 3.1.3 Fairness Metrics: How to Measure

Now that we understand the definitions, let's make them concrete with metrics you can calculate.

### Metric 1: Disparate Impact Ratio (Demographic Parity)

**What it measures:** Are approval rates equal across groups?

**Formula:**
```
DIR = P(approved | unprivileged) / P(approved | privileged)

Where:
- Unprivileged = historically disadvantaged group (e.g., Black)
- Privileged = historically advantaged group (e.g., White)
```

**Interpretation:**
- DIR = 1.0: Perfect parity
- DIR < 1.0: Unprivileged group approved less often
- DIR < 0.8: Potential legal violation (4/5ths rule)

**Example code:**
```python
import numpy as np

def disparate_impact_ratio(y_pred, protected_attr, unprivileged_value, privileged_value):
    """
    Calculate disparate impact ratio.
    
    In credit scoring: y_pred=0 means "no default" (approve), y_pred=1 means "default" (deny)
    So approval means y_pred == 0
    """
    unprivileged_mask = (protected_attr == unprivileged_value)
    privileged_mask = (protected_attr == privileged_value)
    
    # Approval = predict no default (y_pred == 0)
    unprivileged_approval_rate = (y_pred[unprivileged_mask] == 0).mean()
    privileged_approval_rate = (y_pred[privileged_mask] == 0).mean()
    
    return unprivileged_approval_rate / privileged_approval_rate

# Example usage
y_pred = np.array([0, 1, 0, 1, 0, 1, 1, 0])  # 0=approve, 1=deny
race = np.array(['White', 'White', 'Black', 'Black', 'White', 'Black', 'Black', 'White'])

dir_ratio = disparate_impact_ratio(y_pred, race, 'Black', 'White')
print(f"Disparate Impact Ratio: {dir_ratio:.3f}")

if dir_ratio < 0.8:
    print("⚠️ Potential disparate impact violation (below 4/5ths rule)")
elif dir_ratio < 0.9:
    print("⚠️ Borderline - monitor closely")
else:
    print("✓ Passes 4/5ths rule")
```

---

### Metric 2: Equal Opportunity Difference

**What it measures:** Among people who actually default, is the model equally good at catching them across groups? (Difference in True Positive Rates)

**Formula:**
```
EOD = TPR_unprivileged - TPR_privileged

Where TPR = P(Ŷ = 1 | Y = 1) = True Positives / (True Positives + False Negatives)
```

**Interpretation:**
- EOD = 0: Perfect equal opportunity
- EOD > 0: Model catches more defaults in unprivileged group
- EOD < 0: Model catches fewer defaults in unprivileged group
- |EOD| > 0.1: Significant disparity

**Example code:**
```python
from sklearn.metrics import confusion_matrix

def equal_opportunity_difference(y_true, y_pred, protected_attr, unprivileged_value, privileged_value):
    """
    Calculate difference in true positive rates between groups.
    
    TPR = TP / (TP + FN) = What fraction of actual defaults did we catch?
    """
    def tpr(y_true_group, y_pred_group):
        if len(y_true_group) == 0:
            return 0
        tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group, labels=[0,1]).ravel()
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    
    unprivileged_mask = (protected_attr == unprivileged_value)
    privileged_mask = (protected_attr == privileged_value)
    
    tpr_unprivileged = tpr(y_true[unprivileged_mask], y_pred[unprivileged_mask])
    tpr_privileged = tpr(y_true[privileged_mask], y_pred[privileged_mask])
    
    return tpr_unprivileged - tpr_privileged

# Example
eod = equal_opportunity_difference(y_true, y_pred, race, 'Black', 'White')
print(f"Equal Opportunity Difference: {eod:+.3f}")

if abs(eod) < 0.05:
    print("✓ Approximately equal opportunity")
elif abs(eod) < 0.10:
    print("⚠️ Moderate disparity")
else:
    print("✗ Significant disparity in catching defaults")
```

---

### Metric 3: Average Odds Difference (Full Equalized Odds)

**What it measures:** Average of the TPR difference AND FPR difference. Captures unfairness in BOTH types of errors.

**Why both matter:**
- High TPR difference only: One group's defaults get caught more → other group's risky borrowers slip through
- High FPR difference only: One group's good borrowers get wrongly denied more
- Both high: Model is broken for one group entirely
- Both low: Fair! ✓

**Formula:**
```
AOD = 0.5 × (|TPR_diff| + |FPR_diff|)

Where:
- TPR_diff = TPR_unprivileged - TPR_privileged
- FPR_diff = FPR_unprivileged - FPR_privileged
```

**Interpretation:**
- AOD = 0: Perfect equalized odds
- AOD < 0.05: Good
- AOD 0.05-0.10: Moderate disparity
- AOD > 0.10: Significant disparity

**Example code:**
```python
def average_odds_difference(y_true, y_pred, protected_attr, unprivileged_value, privileged_value):
    """
    Calculate average of |TPR difference| and |FPR difference|.
    """
    def rates(y_true_group, y_pred_group):
        if len(y_true_group) == 0:
            return 0, 0
        tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group, labels=[0,1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        return tpr, fpr
    
    unprivileged_mask = (protected_attr == unprivileged_value)
    privileged_mask = (protected_attr == privileged_value)
    
    tpr_unpriv, fpr_unpriv = rates(y_true[unprivileged_mask], y_pred[unprivileged_mask])
    tpr_priv, fpr_priv = rates(y_true[privileged_mask], y_pred[privileged_mask])
    
    tpr_diff = abs(tpr_unpriv - tpr_priv)
    fpr_diff = abs(fpr_unpriv - fpr_priv)
    
    return 0.5 * (tpr_diff + fpr_diff)

# Example
aod = average_odds_difference(y_true, y_pred, race, 'Black', 'White')
print(f"Average Odds Difference: {aod:.3f}")
```

---

### Metric 4: Expected Calibration Error (ECE)

**What it measures:** How "honest" are the model's probability predictions? Does predicted 15% actually mean 15% default?

**How it works:**
1. Group predictions into bins (0-10%, 10-20%, etc.)
2. For each bin: Compare mean predicted probability to actual default rate
3. Weight each bin's error by how many people are in it
4. Sum up the weighted errors

**Formula:**
```
ECE = Σ (bin_weight × |predicted_rate - actual_rate|)

Where bin_weight = count_in_bin / total_count
```

**Interpretation:**
- ECE = 0: Perfect calibration (predictions match reality)
- ECE < 0.05: Well calibrated
- ECE > 0.10: Poorly calibrated

**Example code:**
```python
def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculate expected calibration error.
    
    Measures how well predicted probabilities match actual outcomes.
    """
    # Create bin edges [0.0, 0.1, 0.2, ..., 1.0]
    bins = np.linspace(0, 1, n_bins + 1)
    
    # Assign each prediction to a bin (0 to n_bins-1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)  # Handle edge cases
    
    ece = 0
    for i in range(n_bins):
        in_bin = (bin_indices == i)
        
        if in_bin.sum() > 0:
            bin_accuracy = y_true[in_bin].mean()     # Actual default rate
            bin_confidence = y_prob[in_bin].mean()   # Mean predicted probability
            bin_weight = in_bin.sum() / len(y_true)  # Fraction of data in this bin
            
            ece += bin_weight * abs(bin_accuracy - bin_confidence)
    
    return ece

# Calculate ECE per group to check calibration fairness
for group in ['White', 'Black']:
    group_mask = (race == group)
    ece = expected_calibration_error(y_true[group_mask], y_prob[group_mask])
    print(f"{group} ECE: {ece:.3f}")
```

**Note:** Both `y_true` and `race` arrays must have the same length as `y_prob`!

---

### Metric 5: Statistical Parity Difference

**What it measures:** Simple difference in approval rates (alternative to ratio).

**Formula:**
```
SPD = P(approved | unprivileged) - P(approved | privileged)
```

**Interpretation:**
- SPD = 0: Perfect parity
- SPD < 0: Unprivileged group approved less often  
- SPD < -0.1: Substantial disparity

**Example code:**
```python
def statistical_parity_difference(y_pred, protected_attr, unprivileged_value, privileged_value):
    """
    Calculate difference in approval rates.
    """
    unprivileged_mask = (protected_attr == unprivileged_value)
    privileged_mask = (protected_attr == privileged_value)
    
    # Approval = predict no default (y_pred == 0)
    unprivileged_approval = (y_pred[unprivileged_mask] == 0).mean()
    privileged_approval = (y_pred[privileged_mask] == 0).mean()
    
    return unprivileged_approval - privileged_approval
```

---

## 3.1.4 Choosing the Right Fairness Metrics for Credit Scoring

Given the impossibility results and multiple definitions, **which metrics should you use for credit modeling?**

### The Legal Perspective

**US fair lending law emphasizes:**

1. **Disparate Impact (Demographic Parity)**
   - Required by ECOA enforcement
   - 4/5ths rule is the standard test
   - **Must measure:** Approval rates by protected group

2. **Individual Treatment (Anti-Classification)**
   - Protected characteristics cannot be explicit features
   - But must still test for proxy discrimination
   - **Must document:** Which features you use and why

3. **Explainability (Adverse Action Notices)**
   - Must explain denials to applicants
   - Implies need for interpretable models or explanation tools
   - **Must provide:** Reasons for denial

**Legal minimum:** Measure disparate impact ratio and ensure DIR ≥ 0.80 for all protected groups.

### The Business Perspective

**Lenders care about:**

1. **Profit:** (Approval rate × Interest earned) - (Default rate × Loss given default)
2. **Risk management:** Keeping default rates within acceptable bounds
3. **Customer satisfaction:** Fair treatment builds loyalty
4. **Regulatory compliance:** Avoiding enforcement actions

**Calibration matters most** because:
- Enables risk-based pricing (charge higher rates for higher risk)
- Allows setting thresholds based on risk tolerance
- Supports portfolio management
- Maintains profitability while expanding access

**Business priority:** Calibration across groups, plus demographic parity within reasonable bounds.

### The Ethical Perspective

**Reasonable people disagree on which fairness definition is most "fair":**

**Demographic parity advocates argue:**
- Historical discrimination means current base rates reflect bias, not true risk
- Equal outcomes are needed to correct systemic injustice
- Individual risk assessment perpetuates structural inequality

**Calibration advocates argue:**
- Individuals should be judged on their actual characteristics, not group membership
- Different outcomes for different risks is fair, not discriminatory
- Forcing equal outcomes hurts both groups (denying credit to qualified applicants, approving risky loans)

**Equal opportunity advocates argue:**
- What matters is catching true positives at equal rates
- False positives are costly but recoverable; false negatives deny opportunity
- Focus on ensuring good borrowers aren't missed disproportionately

**There's no universally "correct" answer.** Your choice depends on:
- Your values
- Your stakeholders' values
- Legal requirements
- Business constraints

### Recommended Approach for Credit Scoring

**Measure multiple metrics, prioritize thoughtfully:**

**Tier 1: Must satisfy (legal compliance)**
- ✅ **Disparate Impact Ratio ≥ 0.80** for all protected groups
- ✅ **Anti-classification:** No protected characteristics as features
- ✅ **Explainability:** Can generate adverse action notices

**Tier 2: Should optimize (business + ethics)**
- ✅ **Calibration:** Similar ECE across groups (< 0.05 difference)
- ✅ **Equal Opportunity:** Similar TPR across groups (< 0.10 difference)
- ✅ **Individual fairness:** Audit similar individuals for consistency

**Tier 3: Monitor (detect issues early)**
- 📊 **Statistical Parity Difference:** Track over time
- 📊 **Average Odds Difference:** Comprehensive error rate view
- 📊 **Precision/Recall by group:** Understand business impact

**Trade-off strategy:**
1. Start with accurate model
2. Check Tier 1 metrics - if violated, MUST fix
3. Optimize Tier 2 metrics - accept small accuracy loss if needed
4. Monitor Tier 3 metrics - investigate anomalies
5. Document everything - your reasoning matters

---

## 3.1.5 Practical Considerations

### Sample Size Matters

**Small groups create measurement problems:**

```python
# Example: Tiny group makes metrics unstable
group_a = np.array([1, 0, 1])  # 3 people, 67% approval
group_b = np.array([1, 0, 0, 0, 1, 0])  # 6 people, 33% approval

# Disparate Impact Ratio = 0.33 / 0.67 = 0.49 (looks terrible!)
# But with only 3 people in Group A, this could be random noise
```

**Recommendation:**
- Require minimum 30-50 samples per group for reliable metrics
- Use confidence intervals, not point estimates
- Bootstrap to assess stability
- Consider aggregating small groups if appropriate

### Multiple Protected Attributes

**You must check fairness across ALL protected attributes:**
- Race (White, Black, Hispanic, Asian, etc.)
- Gender (Male, Female, Non-binary)
- Age (binned: 18-25, 26-35, 36-45, etc.)
- Marital status
- Geographic location (if proxy for race)

**This creates combinatorial explosion:**
- 5 race categories × 3 gender categories × 5 age bins = 75 groups
- Cannot reasonably ensure parity across all 75

**Practical approach:**
1. **Primary analysis:** Major protected groups (race, gender)
2. **Secondary analysis:** Intersectional groups (Black women, Hispanic men)
3. **Threshold:** Focus on groups with ≥50 samples
4. **Document:** Which groups you examined and why

### Fairness Across Time

**Metrics can drift:**
- Population changes
- Economic conditions shift
- Model behavior evolves (especially if adaptive)

**Recommendation:**
- Set up monitoring dashboards
- Re-evaluate fairness metrics quarterly
- Trigger alerts if metrics degrade
- Plan for periodic retraining

### Fairness vs. Privacy

**Measuring fairness requires protected attribute data.** But:
- You shouldn't use it as a feature
- You may not even collect it (privacy regulations)
- Customers may not want to provide it

**Solutions:**
- Collect protected attributes separately from features
- Use survey data or public records for validation
- Test on held-out data with known attributes
- Consider using proxies for analysis only (never as features)

---

## 3.1.6 Common Pitfalls to Avoid

### Pitfall 1: "We Don't Collect Race, So We Can't Be Biased"

**Why this fails:**
- ZIP code, name patterns, shopping behavior can proxy for race
- Disparate impact doesn't require intent
- Regulators will test using public data even if you don't collect it

**Fix:** Use proxy methods or external data to audit your model.

### Pitfall 2: "Our Model Is Unbiased - The Data Shows Real Risk Differences"

**Why this is dangerous:**
- Current data may reflect historical discrimination
- Credit bureau data quality varies by population
- "Real differences" may be artifacts of measurement bias

**Fix:** Question your data. Consider alternative data sources. Test multiple fairness definitions.

### Pitfall 3: "We Optimized for Fairness, So We're Done"

**Why you're not done:**
- Fairness can drift over time
- New populations may emerge
- Adversarial adaptation changes distributions
- Regulations evolve

**Fix:** Continuous monitoring, not one-time fixes.

### Pitfall 4: "Perfect Fairness Is Impossible, So Why Try?"

**Why this is defeatist:**
- Impossibility results show you can't satisfy EVERYTHING
- But you can improve on ANY specific metric you choose
- "Perfect is the enemy of good"

**Fix:** Choose metrics aligned with your values and legal requirements. Measure improvement.

### Pitfall 5: "We'll Add Fairness Later"

**Why this fails:**
- Fairness is harder to retrofit than to design in
- Post-processing can degrade performance more than in-processing
- You may discover your entire approach is flawed

**Fix:** Consider fairness from day one of model development.

---

## Key Takeaways

Before moving to Section 3.2 (measuring bias in your model), ensure you understand:

1. **Multiple fairness definitions exist:**
   - Demographic parity (equal approval rates)
   - Equalized odds (equal TPR and FPR)
   - Calibration (predictions mean same thing across groups)
   - Individual fairness (similar people treated similarly)

2. **These definitions conflict mathematically:**
   - Cannot satisfy all simultaneously when base rates differ
   - Must choose and justify your priorities
   - Document your trade-offs

3. **Concrete metrics exist for each definition:**
   - Disparate Impact Ratio (demographic parity)
   - Equal Opportunity Difference (equalized odds)
   - Expected Calibration Error (calibration)
   - Know how to calculate and interpret each

4. **Error terminology in credit scoring:**
   - Positive class = Default (Y=1)
   - False Positive = Deny a good borrower (predict default when they'd pay back)
   - False Negative = Approve a bad borrower (predict no default when they'll default)

5. **For credit scoring, prioritize:**
   - Tier 1: Disparate impact (≥0.80), anti-classification, explainability
   - Tier 2: Calibration, equal opportunity
   - Tier 3: Monitor everything else

6. **Practical considerations matter:**
   - Sample sizes affect metric reliability
   - Multiple protected attributes create complexity
   - Fairness requires ongoing monitoring
   - Privacy and fairness can conflict

7. **Common pitfalls are avoidable:**
   - Don't assume not collecting race means no bias
   - Don't assume data reflects true risk
   - Don't treat fairness as one-time fix
   - Don't let perfect be the enemy of good
   - Don't defer fairness to later

**You're now equipped with the conceptual foundation to measure and improve fairness in your credit model.**

In Section 3.2, we'll apply these concepts to the XGBoost model from Section 2, measuring exactly how fair (or unfair) it is across multiple protected groups and metrics.

Let's find out what we're working with.

---

*End of Section 3.1*
