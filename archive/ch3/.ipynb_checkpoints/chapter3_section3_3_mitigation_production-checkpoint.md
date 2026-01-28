# Section 3.3: Bias Mitigation & Production Deployment

Section 3.2 measured fairness across multiple metrics and identified potential disparities. Now we'll explore how to **mitigate** these biases and prepare our model for **production deployment**.

**This section covers:**
1. Pre-processing techniques (reweighting, resampling)
2. Post-processing techniques (threshold adjustment, calibration)
3. Comparing mitigation approaches (accuracy-fairness trade-offs)
4. Production monitoring and documentation

---

## 3.3.1 Overview: Three Approaches to Bias Mitigation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     BIAS MITIGATION STRATEGIES                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PRE-PROCESSING          IN-PROCESSING           POST-PROCESSING        │
│  ───────────────         ─────────────           ───────────────        │
│  Fix the DATA            Fix the MODEL           Fix the OUTPUT         │
│  before training         during training         after prediction       │
│                                                                         │
│  • Reweighting           • Fairness              • Threshold            │
│  • Resampling              constraints             adjustment           │
│  • Feature removal       • Adversarial           • Calibration          │
│                            debiasing               by group             │
│                                                  • Reject option        │
│                                                                         │
│  Pros: Simple            Pros: Integrated        Pros: No retraining    │
│  Cons: May lose info     Cons: Complex           Cons: May hurt accuracy│
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**For this section, we'll focus on the most practical approaches: pre-processing (reweighting) and post-processing (threshold adjustment, calibration).**

---

## 3.3.2 Pre-Processing: Reweighting

**Idea:** Give different weights to samples so that protected groups have equal influence during training.

### Why Reweighting Works

If Black applicants are underrepresented in the training data, or if their outcomes are imbalanced differently than White applicants, the model may learn patterns that disadvantage them. Reweighting corrects this by giving more weight to underrepresented (group, outcome) combinations.

```python
print("="*70)
print("PRE-PROCESSING: REWEIGHTING")
print("="*70)

def calculate_reweighting_weights(y_true, protected_attr, privileged_value):
    """
    Calculate sample weights to balance outcomes across protected groups.
    
    Goal: Make the weighted distribution of outcomes equal across groups.
    
    Weight formula for each sample:
        w = P(Y) * P(A) / P(Y, A)
    
    This ensures equal representation of each (outcome, group) combination.
    """
    n = len(y_true)
    weights = np.ones(n)
    
    # Calculate group sizes
    privileged_mask = (protected_attr == privileged_value)
    unprivileged_mask = ~privileged_mask
    
    n_priv = privileged_mask.sum()
    n_unpriv = unprivileged_mask.sum()
    
    # Calculate outcome rates
    p_pos = y_true.mean()  # P(Y=1)
    p_neg = 1 - p_pos      # P(Y=0)
    
    p_priv = n_priv / n    # P(A=privileged)
    p_unpriv = n_unpriv / n  # P(A=unprivileged)
    
    # Calculate joint probabilities
    p_pos_priv = (y_true[privileged_mask] == 1).sum() / n
    p_neg_priv = (y_true[privileged_mask] == 0).sum() / n
    p_pos_unpriv = (y_true[unprivileged_mask] == 1).sum() / n
    p_neg_unpriv = (y_true[unprivileged_mask] == 0).sum() / n
    
    # Calculate weights for each (group, outcome) combination
    # Privileged, positive outcome (default)
    if p_pos_priv > 0:
        weights[privileged_mask & (y_true == 1)] = (p_pos * p_priv) / p_pos_priv
    # Privileged, negative outcome (no default)
    if p_neg_priv > 0:
        weights[privileged_mask & (y_true == 0)] = (p_neg * p_priv) / p_neg_priv
    # Unprivileged, positive outcome (default)
    if p_pos_unpriv > 0:
        weights[unprivileged_mask & (y_true == 1)] = (p_pos * p_unpriv) / p_pos_unpriv
    # Unprivileged, negative outcome (no default)
    if p_neg_unpriv > 0:
        weights[unprivileged_mask & (y_true == 0)] = (p_neg * p_unpriv) / p_neg_unpriv
    
    return weights

# Generate protected attributes for training data
np.random.seed(42)
race_train = np.random.choice(['White', 'Black'], size=len(y_train), p=[0.7, 0.3])

# Binary: White vs non-White for simplicity
is_white_train = (race_train == 'White')
sample_weights = calculate_reweighting_weights(y_train.values, is_white_train, True)

print(f"Sample weights calculated:")
print(f"  Min weight: {sample_weights.min():.3f}")
print(f"  Max weight: {sample_weights.max():.3f}")
print(f"  Mean weight: {sample_weights.mean():.3f}")

# Show weight distribution by group
print(f"\nWeight distribution:")
print(f"  White, Default:     {sample_weights[is_white_train & (y_train == 1)].mean():.3f}")
print(f"  White, No Default:  {sample_weights[is_white_train & (y_train == 0)].mean():.3f}")
print(f"  Black, Default:     {sample_weights[~is_white_train & (y_train == 1)].mean():.3f}")
print(f"  Black, No Default:  {sample_weights[~is_white_train & (y_train == 0)].mean():.3f}")
```

### Training with Weights

```python
from xgboost import XGBClassifier

##For VALIDATION (used in fairness evaluation)
protected_val = generate_protected_attributes(val, seed=42)
race_val = protected_val['race'].values  # Should be 477 elements!

# Verify
print(f"race_val: {len(race_val)} elements")
print(f"Unique values: {np.unique(race_val)}")

# Train model WITH sample weights
model_reweighted = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# The key: pass sample_weight to fit()
model_reweighted.fit(X_train, y_train, sample_weight=sample_weights)

# Generate predictions
y_prob_reweighted = model_reweighted.predict_proba(X_val)[:, 1]
y_pred_reweighted = (y_prob_reweighted >= threshold).astype(int)

# Evaluate performance
print(f"\nReweighted Model Performance:")
print(f"  ROC-AUC: {roc_auc_score(y_val, y_prob_reweighted):.3f}")
print(f"  Accuracy: {accuracy_score(y_val, y_pred_reweighted):.3f}")

# Check fairness improvement
dir_original = disparate_impact_ratio(y_pred_val, race_val, 'Black', 'White')
dir_reweighted = disparate_impact_ratio(y_pred_reweighted, race_val, 'Black', 'White')

print(f"\nFairness Comparison:")
print(f"  Original DIR:    {dir_original:.3f}")
print(f"  Reweighted DIR:  {dir_reweighted:.3f}")
print(f"  Improvement:     {dir_reweighted - dir_original:+.3f}")
```

```
TL;DR
QuestionAnswerIs this useless?The MODEL is weak, but the LESSON is valuableAre we on track?Yes - this is real-world realityWhat to do?Document it honestly as a teaching moment

```

---

## 3.3.3 Post-Processing: Threshold Adjustment

Post-processing modifies predictions **after** the model is trained. This is useful when you can't retrain or need a quick fix.

### Technique 1: Group-Specific Thresholds

**Idea:** Use different decision thresholds for each group to equalize approval rates.

```python
print("\n" + "="*70)
print("POST-PROCESSING: GROUP-SPECIFIC THRESHOLDS")
print("="*70)

def find_threshold_for_rate(y_prob, target_rate):
    """
    Find threshold that achieves a target approval rate.
    
    Approval = predict 0 (no default) = probability < threshold
    """
    # Try thresholds from 0 to 1
    for t in np.linspace(0.01, 0.99, 99):
        approval_rate = (y_prob < t).mean()
        if approval_rate <= target_rate:
            return t
    return 0.99

def find_group_thresholds_for_parity(y_prob, protected_attr, groups):
    """
    Find thresholds for each group that equalize approval rates.
    
    Strategy: Use the LOWEST group's approval rate as target,
    then find thresholds for other groups to match.
    """
    # Calculate approval rates at default threshold (0.5)
    default_threshold = 0.5
    approval_rates = {}
    
    for group in groups:
        mask = (protected_attr == group)
        approval_rates[group] = (y_prob[mask] < default_threshold).mean()
    
    print("Approval rates at threshold 0.5:")
    for group, rate in approval_rates.items():
        print(f"  {group}: {rate:.1%}")
    
    # Target: match the lowest approval rate (most conservative)
    target_rate = min(approval_rates.values())
    print(f"\nTarget approval rate: {target_rate:.1%}")
    
    # Find threshold for each group
    thresholds = {}
    for group in groups:
        mask = (protected_attr == group)
        thresholds[group] = find_threshold_for_rate(y_prob[mask], target_rate)
    
    print("\nGroup-specific thresholds:")
    for group, t in thresholds.items():
        print(f"  {group}: {t:.3f}")
    
    return thresholds, target_rate

# Find group-specific thresholds
group_thresholds, target_rate = find_group_thresholds_for_parity(
    y_prob_val, race_val, ['White', 'Black', 'Hispanic', 'Asian']
)

# Apply group-specific thresholds
def apply_group_thresholds(y_prob, protected_attr, thresholds):
    """Apply different thresholds to different groups."""
    y_pred = np.zeros(len(y_prob), dtype=int)
    
    for group, threshold in thresholds.items():
        mask = (protected_attr == group)
        y_pred[mask] = (y_prob[mask] >= threshold).astype(int)
    
    return y_pred

y_pred_fair = apply_group_thresholds(y_prob_val, race_val, group_thresholds)

# Verify fairness improvement
print("\nAfter group-specific thresholds:")
for group in ['White', 'Black', 'Hispanic', 'Asian']:
    mask = (race_val == group)
    approval = (y_pred_fair[mask] == 0).mean()
    print(f"  {group} approval rate: {approval:.1%}")

# Check DIR for all groups vs White
print("\nDIR (vs White as privileged group):")
for group in ['Black', 'Hispanic', 'Asian']:
    dir_val = disparate_impact_ratio(y_pred_fair, race_val, group, 'White')
    status = '✓' if dir_val >= 0.8 else '✗'
    print(f"  {group} vs White: {dir_val:.3f} {status}")

# Check accuracy cost
acc_original = accuracy_score(y_val, y_pred_val)
acc_fair = accuracy_score(y_val, y_pred_fair)
print(f"\nAccuracy impact:")
print(f"  Original: {acc_original:.3f}")
print(f"  Fair:     {acc_fair:.3f}")
print(f"  Change:   {acc_fair - acc_original:+.3f}")
```

**Important Consideration:** Group-specific thresholds explicitly treat groups differently. This is legally and ethically debated:
- **Pro:** Corrects for historical discrimination, achieves equal outcomes
- **Con:** Is itself differential treatment based on protected characteristic

Document your reasoning carefully if using this approach.
Note: "Post-processing can achieve fairness metrics, but can't fix a bad model."

### Technique 2: Calibration by Group

**Idea:** Recalibrate probabilities for each group so predictions are "honest" for everyone.

```python
print("\n" + "="*70)
print("POST-PROCESSING: CALIBRATION BY GROUP")
print("="*70)

from sklearn.isotonic import IsotonicRegression

def calibrate_by_group(y_true, y_prob, protected_attr, groups):
    """
    Apply isotonic regression calibration separately for each group.
    
    This ensures "15% predicted" means "15% actual" for ALL groups.
    """
    calibrated_probs = y_prob.copy()
    calibrators = {}
    
    print("Calibrating each group...")
    for group in groups:
        mask = (protected_attr == group)
        n_group = mask.sum()
        
        if n_group < 20:  # Need minimum samples
            print(f"  {group}: Skipped (n={n_group}, need >= 20)")
            continue
        
        # Fit isotonic regression for this group
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(y_prob[mask], y_true[mask])
        calibrators[group] = iso_reg
        
        # Apply calibration
        calibrated_probs[mask] = iso_reg.transform(y_prob[mask])
        
        # Report improvement
        ece_before = expected_calibration_error(y_true[mask], y_prob[mask])
        ece_after = expected_calibration_error(y_true[mask], calibrated_probs[mask])
        print(f"  {group}: ECE {ece_before:.3f} → {ece_after:.3f} (n={n_group})")
    
    return calibrated_probs, calibrators

# Calibrate by race
y_prob_calibrated, calibrators = calibrate_by_group(
    y_true_val, y_prob_val, race_val, ['White', 'Black', 'Hispanic', 'Asian']
)

# Compare calibration
print("\nCalibration comparison (ECE by group):")
print("-"*50)
for group in ['White', 'Black', 'Hispanic', 'Asian']:
    mask = (race_val == group)
    if mask.sum() >= 20:
        ece_orig = expected_calibration_error(y_true_val[mask], y_prob_val[mask])
        ece_cal = expected_calibration_error(y_true_val[mask], y_prob_calibrated[mask])
        print(f"  {group}: {ece_orig:.3f} → {ece_cal:.3f}")

print("\n✓ Calibration makes predictions 'honest' for all groups")
print("  But note: This doesn't change approval rates directly")
```

---

## 3.3.4 Comparing Mitigation Approaches

Let's compare the accuracy-fairness trade-offs:

```python
print("\n" + "="*70)
print("COMPARING MITIGATION APPROACHES")
print("="*70)

def evaluate_approach(name, y_true, y_pred, y_prob, protected_attr):
    """Evaluate both accuracy and fairness."""
    results = {
        'Approach': name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan,
        'DIR (B/W)': disparate_impact_ratio(y_pred, protected_attr, 'Black', 'White'),
        'SPD (B/W)': statistical_parity_difference(y_pred, protected_attr, 'Black', 'White'),
    }
    
    eod = equal_opportunity_difference(y_true, y_pred, protected_attr, 'Black', 'White')
    results['EOD (B/W)'] = eod if not np.isnan(eod) else 0
    
    return results

# Collect results
results = []

# 1. Original model
results.append(evaluate_approach(
    "1. Original", y_true_val, y_pred_val, y_prob_val, race_val
))

# 2. Reweighted model
results.append(evaluate_approach(
    "2. Reweighted", y_true_val, y_pred_reweighted, y_prob_reweighted, race_val
))

# 3. Group-specific thresholds
results.append(evaluate_approach(
    "3. Group Thresholds", y_true_val, y_pred_fair, y_prob_val, race_val
))

# 4. Calibrated (with standard threshold)
y_pred_calibrated = (y_prob_calibrated >= threshold).astype(int)
results.append(evaluate_approach(
    "4. Calibrated", y_true_val, y_pred_calibrated, y_prob_calibrated, race_val
))

# Display comparison
comparison_df = pd.DataFrame(results)
print("\nApproach Comparison:")
print("="*70)
print(comparison_df.to_string(index=False))

# Interpretation
print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print("""
Key Observations:

1. ORIGINAL MODEL
   - Baseline performance and fairness
   - May or may not pass 4/5ths rule (DIR >= 0.80)

2. REWEIGHTED MODEL  
   - Addresses bias at training time
   - Often improves DIR with modest accuracy cost
   - Recommended if you can retrain

3. GROUP THRESHOLDS
   - Guarantees demographic parity
   - May have larger accuracy cost
   - Controversial: explicitly treats groups differently

4. CALIBRATED MODEL
   - Improves prediction honesty
   - Doesn't directly fix approval rate disparity
   - Good for trust, but may need threshold adjustment too

RECOMMENDATION:
- Start with reweighting (if retraining is feasible)
- Add calibration for better probability estimates
- Use group thresholds only if other methods insufficient
- Always document trade-offs and reasoning

TL;DR
Calibration = winner here. Improved accuracy, AUC, AND stayed fair.

""")
```

### Visualizing the Trade-off

```python
# Create accuracy-fairness trade-off plot
fig, ax = plt.subplots(figsize=(10, 6))

approaches = comparison_df['Approach'].values
accuracies = comparison_df['Accuracy'].values
dirs = comparison_df['DIR (B/W)'].values

# Color by compliance
colors = ['red' if d < 0.8 else 'green' for d in dirs]

scatter = ax.scatter(dirs, accuracies, c=colors, s=200, alpha=0.7, edgecolors='black')

# Add labels
for i, approach in enumerate(approaches):
    ax.annotate(approach, (dirs[i], accuracies[i]), 
                xytext=(10, 5), textcoords='offset points', fontsize=9)

# Reference lines
ax.axvline(x=0.8, color='red', linestyle='--', linewidth=2, label='DIR = 0.80 (4/5ths rule)')
ax.axhline(y=accuracies[0], color='gray', linestyle=':', alpha=0.5, label='Baseline accuracy')

ax.set_xlabel('Disparate Impact Ratio (DIR)', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Accuracy vs. Fairness Trade-off\n(Green = Passes 4/5ths rule, Red = Fails)', fontsize=14)
ax.legend(loc='lower right')
ax.set_xlim(0.6, 1.1)

plt.tight_layout()
plt.savefig('accuracy_fairness_tradeoff.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Trade-off plot saved to accuracy_fairness_tradeoff.png")
```

---

## 3.3.5 Production Monitoring

Fairness isn't a one-time fix—it requires ongoing monitoring.

### What to Monitor

```python
print("\n" + "="*70)
print("PRODUCTION MONITORING FRAMEWORK")
print("="*70)

print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    FAIRNESS MONITORING SCHEDULE                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  DAILY                                                              │
│  ─────                                                              │
│  • Approval rates by protected group                                │
│  • Prediction volume by group (detect sampling changes)             │
│  • Mean predicted probability by group                              │
│                                                                     │
│  WEEKLY                                                             │
│  ──────                                                             │
│  • Disparate Impact Ratio (DIR)                                     │
│  • Statistical Parity Difference (SPD)                              │
│  • Equal Opportunity Difference (EOD) - if labels available         │
│                                                                     │
│  MONTHLY                                                            │
│  ───────                                                            │
│  • Full fairness audit (all metrics)                                │
│  • Calibration check by group (ECE)                                 │
│  • Comparison to baseline/deployment metrics                        │
│  • Drift detection                                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       ALERT THRESHOLDS                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Metric      │ Green (OK)   │ Yellow (Watch)  │ Red (Action!)      │
│  ────────────┼──────────────┼─────────────────┼──────────────────  │
│  DIR         │ >= 0.85      │ 0.80 - 0.85     │ < 0.80             │
│  |SPD|       │ < 0.05       │ 0.05 - 0.10     │ > 0.10             │
│  |EOD|       │ < 0.05       │ 0.05 - 0.10     │ > 0.10             │
│  AOD         │ < 0.05       │ 0.05 - 0.10     │ > 0.10             │
│  ECE diff    │ < 0.03       │ 0.03 - 0.05     │ > 0.05             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")
```

### Monitoring Code

```python
def create_fairness_monitoring_snapshot(y_pred, y_prob, protected_attr, 
                                         groups, timestamp=None):
    """
    Create a monitoring snapshot of fairness metrics.
    
    Run this daily/weekly to track fairness over time.
    """
    if timestamp is None:
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    
    snapshot = {
        'timestamp': timestamp,
        'total_predictions': len(y_pred),
        'overall_approval_rate': (y_pred == 0).mean(),
    }
    
    # Metrics by group
    snapshot['by_group'] = {}
    for group in groups:
        mask = (protected_attr == group)
        if mask.sum() == 0:
            continue
        
        snapshot['by_group'][group] = {
            'n': int(mask.sum()),
            'approval_rate': float((y_pred[mask] == 0).mean()),
            'mean_prob': float(y_prob[mask].mean()),
        }
    
    # Fairness metrics (vs first group as reference)
    reference_group = groups[0]
    snapshot['fairness_metrics'] = {}
    
    for group in groups[1:]:
        dir_val = disparate_impact_ratio(y_pred, protected_attr, group, reference_group)
        spd_val = statistical_parity_difference(y_pred, protected_attr, group, reference_group)
        
        snapshot['fairness_metrics'][f'{group}_vs_{reference_group}'] = {
            'DIR': float(dir_val),
            'SPD': float(spd_val),
            'DIR_status': 'GREEN' if dir_val >= 0.85 else 'YELLOW' if dir_val >= 0.80 else 'RED'
        }
    
    return snapshot


def print_snapshot(snapshot, title):
    """Helper function to print snapshot results."""
    print(f"\n{title}")
    print("="*70)
    print(f"Timestamp: {snapshot['timestamp']}")
    print(f"Total predictions: {snapshot['total_predictions']}")
    print(f"Overall approval rate: {snapshot['overall_approval_rate']:.1%}")
    print(f"\nBy Group:")
    for group, metrics in snapshot['by_group'].items():
        print(f"  {group}: n={metrics['n']}, approval={metrics['approval_rate']:.1%}")
    print(f"\nFairness Metrics:")
    for comparison, metrics in snapshot['fairness_metrics'].items():
        status_emoji = {'GREEN': '✓', 'YELLOW': '⚠️', 'RED': '✗'}[metrics['DIR_status']]
        print(f"  {comparison}: DIR={metrics['DIR']:.3f} {status_emoji}")


# ==========================================================================
# VALIDATION SET - Development Check
# ==========================================================================
snapshot_val = create_fairness_monitoring_snapshot(
    y_pred_val, y_prob_val, race_val,
    ['White', 'Black', 'Hispanic', 'Asian']
)
print_snapshot(snapshot_val, "VALIDATION SET MONITORING (Development Check)")


# ==========================================================================
# TEST SET - Simulates Production Monitoring
# ==========================================================================
snapshot_test = create_fairness_monitoring_snapshot(
    y_pred_test, y_prob_test, race_test,
    ['White', 'Black', 'Hispanic', 'Asian']
)
print_snapshot(snapshot_test, "TEST SET MONITORING (Simulates Production)")


# ==========================================================================
# COMPARISON - Do metrics hold on new data?
# ==========================================================================
print("\n" + "="*70)
print("VALIDATION vs TEST COMPARISON")
print("="*70)
print("\nDIR Comparison (Black vs White):")
dir_val = snapshot_val['fairness_metrics']['Black_vs_White']['DIR']
dir_test = snapshot_test['fairness_metrics']['Black_vs_White']['DIR']
print(f"  Validation: {dir_val:.3f}")
print(f"  Test:       {dir_test:.3f}")
print(f"  Difference: {dir_test - dir_val:+.3f}")

print("\nApproval Rate Comparison:")
for group in ['White', 'Black', 'Hispanic', 'Asian']:
    val_rate = snapshot_val['by_group'][group]['approval_rate']
    test_rate = snapshot_test['by_group'][group]['approval_rate']
    print(f"  {group}: Val={val_rate:.1%}, Test={test_rate:.1%}, Diff={test_rate-val_rate:+.1%}")

print("""
NOTE: Differences between validation and test demonstrate why production 
monitoring matters. Fairness metrics can shift on new data due to:
  - Distribution shift
  - Sampling differences  
  - Population changes over time
""")


# Save both snapshots
import json
with open('models/fairness_snapshot_validation.json', 'w') as f:
    json.dump(snapshot_val, f, indent=2)
with open('models/fairness_snapshot_test.json', 'w') as f:
    json.dump(snapshot_test, f, indent=2)
print("✓ Snapshots saved to models/")
```

---

## 3.3.6 Regulatory Documentation

Maintain comprehensive documentation for compliance:

```python
print("\n" + "="*70)
print("REGULATORY DOCUMENTATION")
print("="*70)

fairness_documentation = """
════════════════════════════════════════════════════════════════════════
                    MODEL FAIRNESS DOCUMENTATION
════════════════════════════════════════════════════════════════════════

1. MODEL OVERVIEW
────────────────────────────────────────────────────────────────────────
   Model Name:    Credit Default Prediction Model
   Version:       1.0
   Purpose:       Predict probability of loan default for credit decisions
   
   Protected Characteristics Analyzed:
   • Race/Ethnicity (White, Black, Hispanic, Asian)
   • Gender (Male, Female)
   • Age (where applicable under ECOA)

2. FAIRNESS DEFINITION & METRICS
────────────────────────────────────────────────────────────────────────
   Primary Definition: Disparate Impact (4/5ths rule)
   
   Metrics Measured:
   • Disparate Impact Ratio (DIR) - MUST be >= 0.80
   • Statistical Parity Difference (SPD)
   • Equal Opportunity Difference (EOD)
   • Average Odds Difference (AOD)
   • Expected Calibration Error (ECE) by group
   
   Justification: DIR is required by ECOA enforcement guidelines.
   Additional metrics provide comprehensive fairness view.

3. FAIRNESS TESTING RESULTS
────────────────────────────────────────────────────────────────────────
   Test Date:     [DATE]
   Test Dataset:  Validation set (n=477)
   
   Results (vs White as reference):
   
   | Group    | DIR   | Status | SPD    | EOD    |
   |----------|-------|--------|--------|--------|
   | Black    | X.XXX | PASS   | +X.XXX | +X.XXX |
   | Hispanic | X.XXX | PASS   | +X.XXX | +X.XXX |
   | Asian    | X.XXX | PASS   | +X.XXX | +X.XXX |
   | Female   | X.XXX | PASS   | +X.XXX | +X.XXX |
   
   Overall Compliance: [PASS/FAIL]

4. MITIGATION MEASURES APPLIED
────────────────────────────────────────────────────────────────────────
   Measures Implemented:
   • Sample reweighting during training
   • Probability calibration by group
   • [Other measures if applied]
   
   Measures Considered but Not Implemented:
   • Group-specific thresholds (rejected due to explicit differential treatment)
   • Feature removal (rejected due to significant accuracy impact)
   
   Trade-offs Accepted:
   • [X]% accuracy reduction in exchange for DIR improvement of [Y]

5. ADVERSE ACTION NOTICES
────────────────────────────────────────────────────────────────────────
   Method: SHAP-based feature contribution explanations
   
   Implementation:
   • Top 4 contributing features provided for each denial
   • Reasons translated to consumer-friendly language
   
   Example Reasons:
   • "Debt-to-income ratio exceeds guidelines"
   • "Insufficient credit history length"
   • "Recent payment delinquencies on record"
   • "High credit utilization percentage"

6. ONGOING MONITORING PLAN
────────────────────────────────────────────────────────────────────────
   Frequency:
   • Daily: Approval rates by group
   • Weekly: DIR, SPD metrics
   • Monthly: Full fairness audit
   
   Alert Thresholds:
   • RED (immediate action): DIR < 0.80
   • YELLOW (investigate): DIR 0.80-0.85 or |SPD| > 0.10
   
   Responsible Party: [Model Risk Management Team]
   Escalation Path: [Compliance Officer → Legal → Executive]

7. APPROVAL SIGNATURES
────────────────────────────────────────────────────────────────────────
   Model Developer:     _________________________ Date: __________
   Model Validator:     _________________________ Date: __________
   Compliance Officer:  _________________________ Date: __________
   Business Owner:      _________________________ Date: __________

════════════════════════════════════════════════════════════════════════
"""

print(fairness_documentation)

# Save documentation
with open('models/fairness_documentation.txt', 'w', encoding='utf-8') as f:
    f.write(fairness_documentation)
print("✓ Documentation template saved to models/fairness_documentation.txt")
```

---

## Key Takeaways

### Mitigation Approach Summary

| Approach | When to Use | Pros | Cons |
|----------|-------------|------|------|
| **Reweighting** | Can retrain model | Simple, effective | Needs protected attributes at training |
| **Group Thresholds** | Quick fix needed | Guarantees parity | Explicit differential treatment |
| **Calibration** | Poor calibration by group | Improves trust | Doesn't fix approval disparity |

### Production Checklist

**Before Deployment:**
- [ ] DIR ≥ 0.80 for all protected groups
- [ ] No protected characteristics as features
- [ ] Adverse action notices implemented
- [ ] Trade-offs documented and justified
- [ ] Sign-offs obtained

**After Deployment:**
- [ ] Daily monitoring configured
- [ ] Alert thresholds set
- [ ] Response procedures defined
- [ ] Quarterly audit scheduled

### Final Recommendations

1. **Start with measurement** - You can't fix what you don't measure
2. **Prefer pre-processing** - Reweighting is usually the best first step
3. **Add calibration** - Makes predictions trustworthy for all groups
4. **Use post-processing sparingly** - Group thresholds are controversial
5. **Document everything** - Regulators want to see your reasoning
6. **Monitor continuously** - Fairness can drift over time

---

## Chapter 3 Summary

We've completed a comprehensive journey through fairness in credit scoring:

**Section 3.1: Understanding Algorithmic Fairness**
- Multiple definitions of fairness (demographic parity, equalized odds, calibration)
- Impossibility results showing trade-offs are unavoidable
- Concrete metrics with code implementations

**Section 3.2: Measuring Bias in Our Model**
- Applied all metrics to our XGBoost model
- Analyzed fairness by race and gender
- Identified root causes of potential bias

**Section 3.3: Bias Mitigation & Production**
- Pre-processing (reweighting) and post-processing (thresholds, calibration)
- Compared accuracy-fairness trade-offs
- Production monitoring and documentation

**The key insight:** Fairness isn't a checkbox—it's an ongoing commitment that requires measurement, mitigation, monitoring, and documentation.

---

*End of Section 3.3 and Chapter 3*
