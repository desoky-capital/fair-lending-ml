# Section 3.1: Understanding Algorithmic Fairness

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

Imagine three loan applicants with identical credit scores (680), incomes ($60,000), and debt-to-income ratios (0.30). The only difference: Alice is white, Bob is Black, and Carlos is Hispanic. Your model assigns them different default probabilities:
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
- Ŷ = predicted outcome (1 = approve, 0 = deny)
- A = protected attribute (e.g., race, gender)
- a, b = different values of A (e.g., "white", "Black")
```

**In plain English:** If 30% of white applicants are approved, 30% of Black applicants should be approved.

**Applied to our example:** 
- If model approves 25% of white applicants and 15% of Black applicants → Violates demographic parity
- If model approves 25% of both → Satisfies demographic parity

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
white_approval_rate = (predictions[race == 'White'] == 1).mean()
black_approval_rate = (predictions[race == 'Black'] == 1).mean()

demographic_parity_ratio = black_approval_rate / white_approval_rate

# Legal threshold: 4/5ths rule (0.80)
# If ratio < 0.80, potential disparate impact
```

---

### Definition 3: Equalized Odds (Equal Opportunity + Equal False Positive Rate)

**Principle:** The model should have equal true positive rates AND equal false positive rates across groups.

**Mathematical definition:**
```
P(Ŷ = 1 | Y = 1, A = a) = P(Ŷ = 1 | Y = 1, A = b)  [Equal TPR]
P(Ŷ = 1 | Y = 0, A = a) = P(Ŷ = 1 | Y = 0, A = b)  [Equal FPR]

Where:
- Y = true outcome (1 = actually defaults, 0 = doesn't default)
- Ŷ = predicted outcome
- A = protected attribute
```

**In plain English:** 
- Among people who will default, the model should catch them at equal rates regardless of race
- Among people who won't default, the model should falsely flag them at equal rates regardless of race

**Applied to our example:**
```
White applicants:
  - Of those who default: 60% flagged by model
  - Of those who don't default: 10% falsely flagged

Black applicants:
  - Of those who default: 60% flagged by model  ← Equal!
  - Of those who don't default: 10% falsely flagged  ← Equal!

→ Satisfies equalized odds
```

**Intuition:** The model's errors should be equally distributed across groups.

**Strengths:**
- Considers actual outcomes, not just predictions
- Addresses both harms: missing true positives AND false positives
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

**Applied to lending:** Among actual good borrowers, approval rates should be equal across race.

---

### Definition 4: Calibration (Predictive Parity)

**Principle:** When the model predicts a certain probability, that prediction should be equally accurate across groups.

**Mathematical definition:**
```
P(Y = 1 | Ŷ = p, A = a) = P(Y = 1 | Ŷ = p, A = b) = p

Where:
- Ŷ = predicted probability
- p = a specific probability value (e.g., 0.20)
```

**In plain English:** If the model predicts 20% default risk for white applicants, about 20% of them should actually default. Same for Black applicants—if predicted 20%, about 20% should default.

**Applied to our example:**
```
White applicants predicted at 20% risk:
  - 18% actually default  (well-calibrated)

Black applicants predicted at 20% risk:
  - 32% actually default  (under-predicting risk!)

→ Violates calibration
```

**Intuition:** Predictions mean the same thing regardless of group membership.

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

**Example calculation:**
```python
# For each group, bin predictions and calculate actual default rate
for group in ['White', 'Black']:
    group_mask = (race == group)
    
    for bin_min, bin_max in [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), ...]:
        in_bin = (predictions >= bin_min) & (predictions < bin_max) & group_mask
        
         if in_bin.sum() > 0:
            predicted = predictions[in_bin].mean()
            actual = y_true[in_bin].mean()
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
Alice (white): FICO 680, Income $60k, DTI 0.30 → Predicted 8%
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
Suppose Group A defaults at 10% and Group B defaults at 20%.

For calibration: When model predicts 15%, actual default rate should be 15% in both groups.

For equalized odds: False positive rates must be equal. But if Group A has lower base rate (10%), achieving the same FPR as Group B (20% base rate) requires different decision boundaries, which breaks calibration.

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
  - Disparate impact ratio: 85%/95% = 0.89 (below 4/5ths rule of 0.80, but borderline)

To achieve demographic parity (90% approval for both):
  - Deny 500 additional White applicants who are creditworthy (false positives)
  - Approve 500 additional Black applicants who are risky (false negatives)
  - Result: Lower profit, higher defaults, harms to both groups

Consequences:
  - 500 qualified White borrowers wrongly denied (lost opportunity)
  - 500 unqualified Black borrowers pushed into unaffordable debt (will default)
  - Bank loses money on 500 defaults it could have avoided
  - Both interventions cause harm in pursuit of equal approval rates
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

Now that we understand the definitions, let's make them concrete with metrics you can actually calculate.

### Metric 1: Disparate Impact Ratio (Demographic Parity)

**Formula:**
```
DIR = P(Ŷ = 1 | A = unprivileged) / P(Ŷ = 1 | A = privileged)

Where:
- Unprivileged = historically disadvantaged group
- Privileged = historically advantaged group
```

**Interpretation:**
- DIR = 1.0: Perfect parity
- DIR < 1.0: Unprivileged group approved less often
- DIR < 0.8: Potential legal violation (4/5ths rule)

**Example:**
```python
import numpy as np

def disparate_impact_ratio(y_pred, protected_attr, unprivileged_value, privileged_value):
    """
    Calculate disparate impact ratio.
    
    Returns:
        float: Ratio of approval rates (unprivileged / privileged)
    """
    unprivileged_mask = (protected_attr == unprivileged_value)
    privileged_mask = (protected_attr == privileged_value)
    
    unprivileged_approval_rate = y_pred[unprivileged_mask].mean()
    privileged_approval_rate = y_pred[privileged_mask].mean()
    
    return unprivileged_approval_rate / privileged_approval_rate

# Example usage
predictions = np.array([1, 0, 1, 1, 0, 1, 0, 0])
race = np.array(['White', 'White', 'Black', 'Black', 'White', 'Black', 'Black', 'White'])

dir_ratio = disparate_impact_ratio(predictions, race, 'Black', 'White')
print(f"Disparate Impact Ratio: {dir_ratio:.3f}")

if dir_ratio < 0.8:
    print("⚠️ Potential disparate impact violation")
elif dir_ratio < 0.9:
    print("⚠️ Borderline - monitor closely")
else:
    print("✓ Passes 4/5ths rule")
```

---

### Metric 2: Equal Opportunity Difference (Equalized Odds)

**Formula:**
```
EOD = |TPR_unprivileged - TPR_privileged|

Where TPR = P(Ŷ = 1 | Y = 1, A = group)
```

**Interpretation:**
- EOD = 0: Perfect equal opportunity
- EOD > 0.1: Significant disparity in catching true positives
- Positive: Privileged group has advantage
- Negative: Unprivileged group has advantage

**Example:**
```python
from sklearn.metrics import confusion_matrix

def equal_opportunity_difference(y_true, y_pred, protected_attr, unprivileged_value, privileged_value):
    """
    Calculate difference in true positive rates between groups.
    
    Returns:
        float: TPR_unprivileged - TPR_privileged
    """
    def tpr(y_true_group, y_pred_group):
        tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    
    unprivileged_mask = (protected_attr == unprivileged_value)
    privileged_mask = (protected_attr == privileged_value)
    
    tpr_unprivileged = tpr(y_true[unprivileged_mask], y_pred[unprivileged_mask])
    tpr_privileged = tpr(y_true[privileged_mask], y_pred[privileged_mask])
    
    return tpr_unprivileged - tpr_privileged

# Example usage
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 0])
race = np.array(['White', 'White', 'Black', 'Black', 'White', 'Black', 'Black', 'White'])

eod = equal_opportunity_difference(y_true, y_pred, race, 'Black', 'White')
print(f"Equal Opportunity Difference: {eod:+.3f}")

if abs(eod) < 0.05:
    print("✓ Approximately equal opportunity")
elif abs(eod) < 0.10:
    print("⚠️ Moderate disparity")
else:
    print("✗ Significant disparity in true positive rates")
```

---

### Metric 3: Average Odds Difference (Full Equalized Odds)

**Formula:**
```
AOD = 0.5 × (|TPR_unpriv - TPR_priv| + |FPR_unpriv - FPR_priv|)
```

**Interpretation:**
- AOD = 0: Perfect equalized odds
- AOD > 0.1: Significant overall disparity in error rates

**Example:**
```python
def average_odds_difference(y_true, y_pred, protected_attr, unprivileged_value, privileged_value):
    """
    Calculate average of TPR and FPR differences.
    
    Returns:
        float: Average absolute difference in TPR and FPR
    """
    def rates(y_true_group, y_pred_group):
        tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
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

# Example usage
aod = average_odds_difference(y_true, y_pred, race, 'Black', 'White')
print(f"Average Odds Difference: {aod:.3f}")
```

---

### Metric 4: Calibration Metrics

**Expected Calibration Error (ECE):**
```python
def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculate expected calibration error.
    
    Returns:
        float: Weighted average of calibration error across bins
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    
    ece = 0
    for i in range(n_bins):
        in_bin = (bin_indices == i)
        if in_bin.sum() > 0:
            bin_accuracy = y_true[in_bin].mean()
            bin_confidence = y_prob[in_bin].mean()
            bin_weight = in_bin.sum() / len(y_true)
            
            ece += bin_weight * abs(bin_accuracy - bin_confidence)
    
    return ece

# Calculate per group
# Reload real y_true from validation set
y_true = y_val.values  # or y_val if already numpy array

# Reload real y_prob (predictions on validation set)
y_prob = best_model_obj.predict_proba(X_val)[:, 1]

# Create race array to match
np.random.seed(42)
race = np.random.choice(['White', 'Black'], size=len(y_prob))

# Now this will work
for group in ['White', 'Black']:
    group_mask = (race == group)
    ece = expected_calibration_error(y_true[group_mask], y_prob[group_mask])
    print(f"{group} ECE: {ece:.3f}")
```
---

### Metric 5: Statistical Parity Difference

**Formula:**
```
SPD = P(Ŷ = 1 | A = unprivileged) - P(Ŷ = 1 | A = privileged)
```

**Interpretation:**
- SPD = 0: Perfect parity
- SPD < 0: Unprivileged group approved less often  
- SPD < -0.1: Substantial disparity

**Example:**
```python
def statistical_parity_difference(y_pred, protected_attr, unprivileged_value, privileged_value):
    """
    Calculate difference in approval rates.
    
    Returns:
        float: Approval_rate_unprivileged - Approval_rate_privileged
    """
    unprivileged_mask = (protected_attr == unprivileged_value)
    privileged_mask = (protected_attr == privileged_value)
    
    unprivileged_approval = y_pred[unprivileged_mask].mean()
    privileged_approval = y_pred[privileged_mask].mean()
    
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

1. **Profit:** approval rate × interest earned - Default rate × loss given default 
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
   - Calibration (predictions mean same thing)
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

4. **For credit scoring, prioritize:**
   - Tier 1: Disparate impact (≥0.80), anti-classification, explainability
   - Tier 2: Calibration, equal opportunity
   - Tier 3: Monitor everything else

5. **Practical considerations matter:**
   - Sample sizes affect metric reliability
   - Multiple protected attributes create complexity
   - Fairness requires ongoing monitoring
   - Privacy and fairness can conflict

6. **Common pitfalls are avoidable:**
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
