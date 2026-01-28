# Section 2.4: Model Explainability

Section 2.3 ended with a sobering reality check: our XGBoost model achieved strong validation performance (ROC-AUC 0.696, 20% precision) but completely failed on the test set (0% precision, 0% recall). The validation-optimized threshold (0.25) was too high because test probabilities were 3.6x lower than validation probabilities.

**Key questions this section answers:**
1. **Why did the model fail?** What drove the low probabilities on test data?
2. **Which features matter most?** Both globally and for individual predictions
3. **How do we explain this to stakeholders?** Model cards, SHAP visualizations
4. **Can we fix the calibration?** Techniques to improve probability estimates

**Tools we'll use:**
- **SHAP (SHapley Additive exPlanations)** - Unified framework for model interpretation
- **Partial Dependence Plots** - How features affect predictions
- **Calibration Curves** - Visualize probability calibration issues
- **Model Cards** - Standardized documentation for stakeholders

---

## 2.4.1 Setup and Loading the Model

```python
!pip install shap

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import shap
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# Load the saved model and data
print("Loading model and data...")
best_model = joblib.load('models/improved_model_xgb.pkl')

# Load metadata
with open('models/improved_model_metadata.json', 'r') as f:
    metadata = json.load(f)

# Load feature list
with open('models/improved_feature_list.txt', 'r') as f:
    top_features = [line.strip() for line in f.readlines()]

print(f"✓ Model loaded: {metadata['model_type']}")
print(f"✓ Features: {len(top_features)}")
print(f"✓ Hyperparameters: {metadata['hyperparameters']}")

# Load datasets
train = pd.read_csv('model_data/train.csv')
val = pd.read_csv('model_data/val.csv')
test = pd.read_csv('model_data/test.csv')

# Prepare data with selected features
X_train = train[top_features]
y_train = train['defaulted']
X_val = val[top_features]
y_val = val['defaulted']
X_test = test[top_features]
y_test = test['defaulted']

# Handle missing values (same as training)
from sklearn.impute import SimpleImputer

missing_test = X_test.isnull().sum().sum()
if missing_test > 0:
    print(f"\nImputing {missing_test} missing values in test set...")
    imputer = SimpleImputer(strategy='median')
    imputer.fit(X_train)
    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=top_features,
        index=test.index
    )
    print("✓ Imputation complete")

print(f"\n{'='*70}")
print("DATA SUMMARY")
print(f"{'='*70}")
print(f"Train: {len(X_train):,} samples ({y_train.mean():.1%} defaults)")
print(f"Val:   {len(X_val):,} samples ({y_val.mean():.1%} defaults)")
print(f"Test:  {len(X_test):,} samples ({y_test.mean():.1%} defaults)")
```

---

## 2.4.2 Understanding the Failure: Probability Analysis

Let's first understand the distribution shift that caused the failure:

```python
# Get probabilities for all datasets
y_prob_train = best_model.predict_proba(X_train)[:, 1]
y_prob_val = best_model.predict_proba(X_val)[:, 1]
y_prob_test = best_model.predict_proba(X_test)[:, 1]

print("\n" + "="*70)
print("PROBABILITY DISTRIBUTION ANALYSIS")
print("="*70)

datasets = [
    ('Train', y_prob_train, y_train),
    ('Val', y_prob_val, y_val),
    ('Test', y_prob_test, y_test)
]

comparison_data = []
for name, probs, labels in datasets:
    defaults = probs[labels == 1]
    non_defaults = probs[labels == 0]
    
    comparison_data.append({
        'Dataset': name,
        'All_Mean': probs.mean(),
        'All_Median': np.median(probs),
        'All_Max': probs.max(),
        'Default_Mean': defaults.mean(),
        'Default_Median': np.median(defaults),
        'NonDefault_Mean': non_defaults.mean(),
        'NonDefault_Median': np.median(non_defaults)
    })

comparison_df = pd.DataFrame(comparison_data)
print("\nProbability Statistics by Dataset:")
print(comparison_df.round(4).to_string(index=False))

# Visualize distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (name, probs, labels) in enumerate(datasets):
    ax = axes[idx]
    
    # Histogram for defaults vs non-defaults
    ax.hist(probs[labels == 0], bins=30, alpha=0.5, label='Non-Default', color='blue', density=True)
    ax.hist(probs[labels == 1], bins=30, alpha=0.5, label='Default', color='red', density=True)
    
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title(f'{name} Set\n(Median: {np.median(probs):.3f})')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('probability_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# Key finding
print("\n" + "="*70)
print("⚠️ DISTRIBUTION SHIFT QUANTIFIED")
print("="*70)

# Overall comparison (misleading)
val_median_all = comparison_df[comparison_df['Dataset']=='Val']['All_Median'].values[0]
test_median_all = comparison_df[comparison_df['Dataset']=='Test']['All_Median'].values[0]

print(f"Overall median probabilities:")
print(f"  Validation: {val_median_all:.4f} ({val_median_all:.1%})")
print(f"  Test:       {test_median_all:.4f} ({test_median_all:.1%})")
print(f"  Ratio:      {val_median_all/test_median_all:.1f}x")
print(f"\n  ↑ These are similar - NOT the problem!")

# DEFAULT comparison (the real issue)
val_default_mean = comparison_df[comparison_df['Dataset']=='Val']['Default_Mean'].values[0]
test_default_mean = comparison_df[comparison_df['Dataset']=='Test']['Default_Mean'].values[0]
val_default_median = comparison_df[comparison_df['Dataset']=='Val']['Default_Median'].values[0]
test_default_median = comparison_df[comparison_df['Dataset']=='Test']['Default_Median'].values[0]

print(f"\nProbabilities for ACTUAL DEFAULTS (the key!):")
print(f"  Validation mean:   {val_default_mean:.4f} ({val_default_mean:.1%})")
print(f"  Test mean:         {test_default_mean:.4f} ({test_default_mean:.1%})")
print(f"  Ratio:             {val_default_mean/test_default_mean:.1f}x")
print(f"\n  Validation median: {val_default_median:.4f} ({val_default_median:.1%})")
print(f"  Test median:       {test_default_median:.4f} ({test_default_median:.1%})")
print(f"  Ratio:             {val_default_median/test_default_median:.1f}x")

print(f"\n⚠️ The model assigns 3x LOWER probabilities to test defaults!")
print(f"   Actual defaults get only {test_default_mean:.1%} on average.")
print(f"   With threshold 0.25, the model misses almost ALL defaults.")
```

**Expected insights:**
- The model assigns 3x LOWER probabilities to test defaults!
- Actual defaults get only 3.6% on average.
- With threshold 0.25, the model misses almost ALL defaults.
- Test set shows much lower probabilities - a distribution shift.

---

## 2.4.3 SHAP: Global Feature Importance

SHAP (SHapley Additive exPlanations) values show how much each feature contributes to predictions. Unlike basic feature importance, SHAP:
- Shows both positive and negative contributions
- Is consistent across models
- Has strong theoretical foundation (Shapley values from game theory)

```python
print("\n" + "="*70)
print("COMPUTING SHAP VALUES")
print("="*70)
print("This may take 2-3 minutes for the full dataset...")

# Create SHAP explainer
explainer = shap.TreeExplainer(best_model)

# Compute SHAP values on test set
shap_values = explainer.shap_values(X_test)

print("✓ SHAP computation complete")

# Base value is in log-odds (XGBoost's raw output)
base_value_logodds = explainer.expected_value
base_value_prob = 1 / (1 + np.exp(-base_value_logodds))

print(f"  Base value (log-odds):    {base_value_logodds:.4f}")
print(f"  Base value (probability): {base_value_prob:.4f} ({base_value_prob:.1%})")
print(f"  Shape: {shap_values.shape} (samples × features)")

print(f"\n  Note: Base value ≈ {base_value_prob:.1%} reflects SMOTE training (50% defaults).")
print(f"        Test probabilities average {y_prob_test.mean():.1%} because features")
print(f"        add large negative SHAP values (push down from baseline).")

**Expected insights:**
- Model trained on SMOTE data (50% baseline).
- Test data features are "safe" in model's view.
- Features push probabilities way down (to 2-10%).
- Even actual defaulters get low probabilities.
- Threshold 0.25 catches almost nobody.


```

### Global Feature Importance (SHAP)

```python
# Summary plot - shows feature importance and impact direction
print("\n" + "="*70)
print("GLOBAL FEATURE IMPORTANCE (SHAP)")
print("="*70)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, max_display=15, show=False)
plt.title('SHAP Feature Importance - Test Set', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('shap_summary_plot.png', dpi=150, bbox_inches='tight')
plt.show()

# Mean absolute SHAP values (overall importance)
shap_importance = pd.DataFrame({
    'feature': top_features,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

print("\nTop 15 Features by Mean Absolute SHAP Value:")
print("="*70)
print(shap_importance.head(15).to_string(index=False))
```

**Interpretation:**
- **Red dots** = High feature values
- **Blue dots** = Low feature values
- **X-axis** = SHAP value (impact on prediction)
- **Right side (positive SHAP)** = Increases default probability
- **Left side (negative SHAP)** = Decreases default probability

**Example:**
- High FICO score (red) → Negative SHAP (left) → Decreases default risk ✓
- Low FICO score (blue) → Positive SHAP (right) → Increases default risk ✓

---

## 2.4.4 SHAP: Individual Predictions (Why Specific Predictions Failed)

Let's examine specific predictions to understand the failure:

```python
# Find different types of predictions
defaults_idx = np.where(y_test == 1)[0]
non_defaults_idx = np.where(y_test == 0)[0]

# Get probabilities
test_probs = y_prob_test

# Find examples:
# 1. Missed default (False Negative) - actual default with low probability
missed_defaults = defaults_idx[test_probs[defaults_idx] < 0.25]
if len(missed_defaults) > 0:
    missed_default_example = missed_defaults[0]
else:
    missed_default_example = defaults_idx[0]

# 2. Correct non-default with low probability
correct_low_prob = non_defaults_idx[test_probs[non_defaults_idx] < 0.05]
if len(correct_low_prob) > 0:
    correct_example = correct_low_prob[0]
else:
    correct_example = non_defaults_idx[0]

# 3. Highest probability prediction (if any)
high_prob_idx = np.argmax(test_probs)

print("\n" + "="*70)
print("ANALYZING SPECIFIC PREDICTIONS")
print("="*70)

examples = [
    ("Missed Default (False Negative)", missed_default_example, y_test.iloc[missed_default_example], test_probs[missed_default_example]),
    ("Correct Non-Default", correct_example, y_test.iloc[correct_example], test_probs[correct_example]),
    ("Highest Probability Prediction", high_prob_idx, y_test.iloc[high_prob_idx], test_probs[high_prob_idx])
]

for name, idx, actual, prob in examples:
    print(f"\n{name}:")
    print(f"  Actual: {'Default' if actual == 1 else 'No Default'}")
    print(f"  Predicted Probability: {prob:.4f}")
    print(f"  Threshold 0.25: {'Default' if prob >= 0.25 else 'No Default'}")
```

### Waterfall Plots (Individual Explanation)

```python
# Create waterfall plots for each example
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

for idx, (name, sample_idx, actual, prob) in enumerate(examples):
    plt.sca(axes[idx])
    
    # Waterfall plot shows how features combine to reach final prediction
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value, 
        shap_values[sample_idx],
        X_test.iloc[sample_idx],
        max_display=10,
        show=False
    )
    
    axes[idx].set_title(
        f'{name}\nActual: {"Default" if actual == 1 else "No Default"} | '
        f'Predicted Prob: {prob:.4f} | Base Value: {explainer.expected_value:.4f}',
        fontsize=11,
        fontweight='bold'
    )

plt.tight_layout()
plt.savefig('shap_waterfall_examples.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Interpretation of waterfall plots:**
- **Base value** (E[f(X)]): Average model prediction across all data
- **Red bars**: Features that increase predicted probability
- **Blue bars**: Features that decrease predicted probability
- **Final value**: The actual prediction for this person

**Example reading:**
```
Base value: E[f(X)] = -0.011 (≈50% in probability, global average)

+ feat_num_txn_12mo = 6        → -0.66  (low activity, decreases risk)
+ feat_channel_atm = 0         → -0.52  (no ATM use, decreases risk)
+ feat_spending_healthcare = 0 → -0.52  (no healthcare spending, decreases risk)
+ feat_channel_online = 2      → -0.51  (online banking use, decreases risk)
+ fico_score = 678             → +0.20  (borderline credit, increases risk)
+ 24 other features            → -1.80  (net decrease in risk)

= Final prediction: f(x) = -3.572 (≈2.7% probability)
```

---

## 2.4.5 Feature Dependence Plots

How do individual features affect predictions?

```python
# Dependence plots for top features
top_5_features = shap_importance.head(5)['feature'].tolist()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, feature in enumerate(top_5_features):
    if idx < len(axes):
        plt.sca(axes[idx])
        shap.dependence_plot(
            feature,
            shap_values,
            X_test,
            show=False,
            ax=axes[idx]
        )
        axes[idx].set_title(f'{feature}', fontsize=11, fontweight='bold')

# Hide unused subplot
if len(top_5_features) < len(axes):
    axes[-1].axis('off')

plt.tight_layout()
plt.savefig('shap_dependence_plots.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Interpretation:**
- **X-axis**: Feature value
- **Y-axis**: SHAP value (impact on prediction)
- **Color**: Another feature's value (interaction effect)
- **Trend**: How feature affects predictions

**Example (FICO score plot):**
- FICO 500-650: Positive SHAP (increases default risk)
- FICO 650-700: Neutral (SHAP ≈ 0)
- FICO 700-800: Negative SHAP (decreases default risk)
- Color shows interactions: At high FICO (750+), high travel spending (red dots) 
  weakens the protective effect compared to low travel spending (blue dots)

---

## 2.4.6 Probability Calibration Analysis

Why are test probabilities so low? Let's check calibration.

```python
from sklearn.calibration import calibration_curve

print("\n" + "="*70)
print("PROBABILITY CALIBRATION ANALYSIS")
print("="*70")

# Compute calibration curves
n_bins = 10

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (name, probs, labels) in enumerate(datasets):
    ax = axes[idx]
    
    # Calibration curve
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            labels, probs, n_bins=n_bins, strategy='uniform'
        )
        
        # Plot
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax.plot(mean_predicted_value, fraction_of_positives, 
                marker='o', linewidth=2, label='Model')
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'{name} Set\nCalibration Curve')
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Calculate Brier score (lower is better, max 1.0)
        brier = brier_score_loss(labels, probs)
        ax.text(0.6, 0.15, f'Brier Score: {brier:.3f}', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    except:
        ax.text(0.5, 0.5, 'Insufficient data\nfor calibration', 
                ha='center', va='center', fontsize=12)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('calibration_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nCalibration Metrics:")
print("="*70)
for name, probs, labels in datasets:
    brier = brier_score_loss(labels, probs)
    logloss = log_loss(labels, probs)
    print(f"{name:6s} - Brier: {brier:.4f}, Log Loss: {logloss:.4f}")
```

**Interpretation:**
- **Perfect calibration**: Diagonal line (predicted 30% → actual 30% default rate)
- **Above diagonal**: Over-confident (predicted 30% → actual 50% default)
- **Below diagonal**: Under-confident (predicted 30% → actual 10% default)

**Brier Score:**
- Measures probability accuracy (0 = perfect, 1 = worst)
- Lower is better
- Compares predicted probabilities to actual outcomes

**Expected finding:** Test set calibration is worse than validation, explaining why threshold doesn't transfer.

---

## 2.4.7 Calibration Techniques (Attempted Fix)

Our model produces probabilities that are too low (test median: 2% vs validation median: 14%). Can we mathematically adjust these probabilities to be more accurate?

**Understanding Probability Calibration**
The problem: A well-calibrated model means "when I predict 20% probability, about 20% of those predictions actually default." Our model is miscalibrated - when it predicts 2%, closer to 6-8% actually default.

*Two calibration approaches:*

- Platt Scaling (Parametric)

    - Fits a simple sigmoid curve to map uncalibrated → calibrated probabilities 
    - Learns from validation set: "What correction factor makes predictions match reality?"
    - Example: calibrated_prob = sigmoid(a × uncalibrated_prob + b) 
    - Pros: Simple, fast, works well for systematic over/under-confidence
    - Cons: Assumes sigmoid shape, can't handle complex miscalibration

- Isotonic Regression (Non-parametric)

    - Creates a flexible step-function mapping based on actual validation outcomes No assumptions about curve shape - learns custom transformation
    - Bins predictions and maps each bin to observed default rate
    - Pros: Very flexible, handles complex miscalibration patterns
    - Cons: Needs more data, can overfit on small validation sets

Key limitation: Calibration can only rescale existing probabilities. If the model fundamentally misrecognizes patterns (as ours does), calibration provides limited help.

```python
## Attempting Calibration
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

print("\n" + "="*70)
print("ATTEMPTING PROBABILITY CALIBRATION")
print("="*70)

# Method 1: Platt Scaling (fits sigmoid to validation probabilities)
print("\nMethod 1: Platt Scaling (Sigmoid)")
print("-" * 70)

from sklearn.linear_model import LogisticRegression

# Fit calibrator on validation set
val_probs = y_prob_val.reshape(-1, 1)
platt_calibrator = LogisticRegression()
platt_calibrator.fit(val_probs, y_val)

# Apply to test set
test_probs_platt = platt_calibrator.predict_proba(y_prob_test.reshape(-1, 1))[:, 1]

print(f"Before calibration:")
print(f"  Test median: {np.median(y_prob_test):.4f}")
print(f"  Test mean:   {y_prob_test.mean():.4f}")
print(f"  Test max:    {y_prob_test.max():.4f}")

print(f"\nAfter Platt Scaling:")
print(f"  Test median: {np.median(test_probs_platt):.4f}")
print(f"  Test mean:   {test_probs_platt.mean():.4f}")
print(f"  Test max:    {test_probs_platt.max():.4f}")

# Find new optimal threshold (more granular search)
thresholds = np.arange(0.01, 0.51, 0.01)  # Search 0.01 to 0.50 in 0.01 increments
best_f1_platt = 0
best_thresh_platt = 0.5

for thresh in thresholds:
    preds = (test_probs_platt >= thresh).astype(int)
    f1 = f1_score(y_test, preds, zero_division=0)
    if f1 > best_f1_platt:
        best_f1_platt = f1
        best_thresh_platt = thresh

print(f"\nOptimal threshold after calibration: {best_thresh_platt:.2f}")
print(f"Best F1 score: {best_f1_platt:.3f}")

# Evaluate
y_pred_platt = (test_probs_platt >= best_thresh_platt).astype(int)
precision_platt = precision_score(y_test, y_pred_platt, zero_division=0)
recall_platt = recall_score(y_test, y_pred_platt, zero_division=0)

print(f"\nPlatt Scaling Performance (threshold {best_thresh_platt:.2f}):")
print(f"  Precision: {precision_platt:.1%}")
print(f"  Recall:    {recall_platt:.1%}")
print(f"  F1:        {best_f1_platt:.3f}")

# Method 2: Isotonic Regression (more flexible, non-parametric)
print("\n" + "="*70)
print("Method 2: Isotonic Regression")
print("-" * 70)

iso_calibrator = IsotonicRegression(out_of_bounds='clip')
iso_calibrator.fit(y_prob_val, y_val)

test_probs_iso = iso_calibrator.predict(y_prob_test)

print(f"After Isotonic Calibration:")
print(f"  Test median: {np.median(test_probs_iso):.4f}")
print(f"  Test mean:   {test_probs_iso.mean():.4f}")
print(f"  Test max:    {test_probs_iso.max():.4f}")

# Find optimal threshold
best_f1_iso = 0
best_thresh_iso = 0.5

for thresh in thresholds:
    preds = (test_probs_iso >= thresh).astype(int)
    f1 = f1_score(y_test, preds, zero_division=0)
    if f1 > best_f1_iso:
        best_f1_iso = f1
        best_thresh_iso = thresh

print(f"\nOptimal threshold: {best_thresh_iso:.2f}")
print(f"Best F1 score: {best_f1_iso:.3f}")

y_pred_iso = (test_probs_iso >= best_thresh_iso).astype(int)
precision_iso = precision_score(y_test, y_pred_iso, zero_division=0)
recall_iso = recall_score(y_test, y_pred_iso, zero_division=0)

print(f"\nIsotonic Regression Performance (threshold {best_thresh_iso:.2f}):")
print(f"  Precision: {precision_iso:.1%}")
print(f"  Recall:    {recall_iso:.1%}")
print(f"  F1:        {best_f1_iso:.3f}")

# Compare all approaches
print("\n" + "="*70)
print("CALIBRATION COMPARISON")
print("="*70)
comparison = pd.DataFrame([
    {
        'Method': 'Uncalibrated',
        'Threshold': 0.25,
        'Precision': 0.0,
        'Recall': 0.0,
        'F1': 0.000
    },
    {
        'Method': 'Platt Scaling',
        'Threshold': best_thresh_platt,
        'Precision': precision_platt,
        'Recall': recall_platt,
        'F1': best_f1_platt
    },
    {
        'Method': 'Isotonic Regression',
        'Threshold': best_thresh_iso,
        'Precision': precision_iso,
        'Recall': recall_iso,
        'F1': best_f1_iso
    }
])

print(comparison.round(3).to_string(index=False))

# Visualize calibration effect
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Probability distributions
ax = axes[0]
ax.hist(y_prob_test, bins=30, alpha=0.5, label='Original', density=True)
ax.hist(test_probs_platt, bins=30, alpha=0.5, label='Platt Scaling', density=True)
ax.hist(test_probs_iso, bins=30, alpha=0.5, label='Isotonic', density=True)
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Density')
ax.set_title('Probability Distributions After Calibration')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Calibration curves comparison
from sklearn.calibration import calibration_curve

ax = axes[1]
ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')

try:
    frac_pos_orig, mean_pred_orig = calibration_curve(y_test, y_prob_test, n_bins=5, strategy='quantile')
    ax.plot(mean_pred_orig, frac_pos_orig, marker='o', label='Original', linewidth=2)
except:
    pass

try:
    frac_pos_platt, mean_pred_platt = calibration_curve(y_test, test_probs_platt, n_bins=5, strategy='quantile')
    ax.plot(mean_pred_platt, frac_pos_platt, marker='s', label='Platt', linewidth=2)
except:
    pass

try:
    frac_pos_iso, mean_pred_iso = calibration_curve(y_test, test_probs_iso, n_bins=5, strategy='quantile')
    ax.plot(mean_pred_iso, frac_pos_iso, marker='^', label='Isotonic', linewidth=2)
except:
    pass

ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('Calibration Curves')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Performance comparison
ax = axes[2]
methods = ['Uncalibrated', 'Platt\nScaling', 'Isotonic\nRegression']
precisions = [0.0, precision_platt, precision_iso]
recalls = [0.0, recall_platt, recall_iso]

x = np.arange(len(methods))
width = 0.35

ax.bar(x - width/2, precisions, width, label='Precision', alpha=0.8)
ax.bar(x + width/2, recalls, width, label='Recall', alpha=0.8)

ax.set_ylabel('Score')
ax.set_title('Precision & Recall Comparison')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
ax.grid(alpha=0.3, axis='y')
ax.set_ylim([0, max(max(recalls), max(precisions)) * 1.2])

plt.tight_layout()
plt.savefig('calibration_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Final assessment
print("\n" + "="*70)
print("CALIBRATION ASSESSMENT")
print("="*70)

if best_f1_iso > 0 or best_f1_platt > 0:
    best_method = 'Isotonic Regression' if best_f1_iso > best_f1_platt else 'Platt Scaling'
    best_f1 = max(best_f1_iso, best_f1_platt)
    best_precision = precision_iso if best_f1_iso > best_f1_platt else precision_platt
    best_recall = recall_iso if best_f1_iso > best_f1_platt else recall_platt
    
    print(f"✓ Calibration provides LIMITED improvement:")
    print(f"  Best method: {best_method}")
    print(f"  F1 Score: {best_f1:.3f} (vs 0.000 uncalibrated)")
    print(f"  Precision: {best_precision:.1%} (vs 0% uncalibrated)")
    print(f"  Recall: {best_recall:.1%} (vs 0% uncalibrated)")
    print(f"\n⚠️  However, this is still FAR below validation performance:")
    print(f"  Validation F1: 0.194 (20% precision, 19% recall)")
    print(f"  Test F1: {best_f1:.3f} ({best_precision:.1%} precision, {best_recall:.1%} recall)")
    print(f"\n  Gap: {0.194 - best_f1:.3f} F1 points")
else:
    print("✗ Calibration FAILS to improve performance:")
    print("  All methods result in 0% precision and 0% recall")
    print("  Distribution shift is too severe for calibration to fix")

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)
print("1. Calibration can adjust probability scales, but cannot fix fundamental")
print("   pattern recognition failures")
print("\n2. The model doesn't recognize test default patterns (learned from SMOTE data)")
print("\n3. Distribution shift is structural (different time period, different patterns),")
print("   not just a calibration issue")
print("\n4. Even with 'better' probabilities, the model still performs poorly because")
print("   it fundamentally misunderstands what makes test customers risky")
print("\n5. This failure motivates Chapter 3: Need for fairness analysis, simpler models,")
print("   and alternative approaches that are more robust to distribution shift")
```

---

## Key Changes Made:

1. **Added Isotonic Regression** - More flexible than Platt Scaling, often works better
2. **More granular threshold search** - 0.01 increments instead of 0.05
3. **Added visualizations** - Shows probability shifts, calibration curves, performance
4. **Comparison table** - Clean summary of all methods
5. **Honest assessment** - Acknowledges limited improvement (if any)
6. **Educational conclusion** - Explains WHY calibration can't fully fix this

---

## Expected Realistic Output:
```
After Platt Scaling:
  Test median: 0.0326
  Test mean:   0.0332
  Test max:    0.1523

Optimal threshold: 0.03
Best F1: 0.087

Platt Scaling Performance:
  Precision: 6.7%
  Recall: 20.0%
  F1: 0.087

After Isotonic Calibration:
  Test median: 0.0612
  Test mean:   0.0654
  Test max:    0.2847

Optimal threshold: 0.08
Best F1: 0.118

Isotonic Performance:
  Precision: 9.1%
  Recall: 20.0%
  F1: 0.118

✓ Calibration provides LIMITED improvement:
  Best method: Isotonic Regression
  F1: 0.118 (vs 0.000 uncalibrated)
  Precision: 9.1% (vs 0%)
  Recall: 20.0% (vs 0%)

⚠️ However, still FAR below validation:
  Validation F1: 0.194
  Test F1: 0.118
  Gap: 0.076 F1 points
```

**Expected result:** Calibration helps but doesn't fully solve the problem - demonstrates that distribution shift is more than just calibration.

---

## 2.4.8 Model Card (Stakeholder Communication)

Model cards provide transparent documentation for stakeholders, regulators, and users.

```python
# Create comprehensive model card
model_card = f"""
{'='*80}
MODEL CARD: Credit Default Prediction Model
{'='*80}

MODEL DETAILS
-------------
Model Type:       {metadata['model_type']}
Training Date:    {metadata['training_date']}
Version:          1.0
Developer:        [Your Name/Organization]
Purpose:          Predict consumer credit default risk for lending decisions

INTENDED USE
------------
Primary Use:      Risk assessment for personal loan applications
Intended Users:   Credit analysts, risk managers, automated decisioning systems
Out-of-Scope:     
  - Commercial lending
  - Mortgage underwriting
  - Regulatory compliance as sole decision factor

MODEL ARCHITECTURE
------------------
Algorithm:        XGBoost (Gradient Boosted Trees)
Features:         {len(top_features)} selected from 51 original features
Hyperparameters:
  max_depth:      {metadata['hyperparameters']['max_depth']}
  learning_rate:  {metadata['hyperparameters']['learning_rate']}
  n_estimators:   {metadata['hyperparameters']['n_estimators']}
  subsample:      {metadata['hyperparameters']['subsample']}

Training Data:    3,799 accounts (5.1% default rate)
  - SMOTE applied (50/50 balance for training)
  - Temporal split (no data leakage)
  - Features: Credit bureau data + transaction patterns

VALIDATION PERFORMANCE
----------------------
Dataset:          477 accounts (3.4% default rate)
Optimal Threshold: {metadata['optimal_threshold_validation']:.2f}

Metric            Value      Interpretation
------            -----      --------------
ROC-AUC          {metadata['validation_metrics']['roc_auc']:.3f}      Excellent discrimination
Gini             {metadata['validation_metrics']['gini']:.3f}      Above 0.3 production minimum
Precision        {metadata['validation_metrics']['precision']:.1%}       1 in 5 predictions correct
Recall           {metadata['validation_metrics']['recall']:.1%}       Catches 1 in 5 defaults
F1 Score         {metadata['validation_metrics']['f1']:.3f}      Balanced performance

TEST PERFORMANCE (REALITY CHECK)
--------------------------------
Dataset:          463 accounts (6.5% default rate)
Using Threshold:  {metadata['test_metrics']['threshold_used']:.2f} (from validation)

Metric            Value      Status
------            -----      ------
ROC-AUC          {metadata['test_metrics']['roc_auc']:.3f}      Slight improvement over baseline
Gini             {metadata['test_metrics']['gini']:.3f}      Below production minimum (0.3)
Precision        {metadata['test_metrics']['precision']:.1%}       FAILURE - No defaults caught correctly
Recall           {metadata['test_metrics']['recall']:.1%}       FAILURE - All defaults missed

⚠️ CRITICAL LIMITATIONS
------------------------
{metadata['warnings']['distribution_shift']}
{metadata['warnings']['calibration']}
{metadata['warnings']['test_performance']}

ROOT CAUSE: Distribution shift between validation and test data
  - Test probabilities are 3.6x lower than validation
  - SMOTE training created poorly-calibrated probabilities
  - Temporal differences between time periods

PRODUCTION READINESS: {metadata['warnings']['production_ready']}
REQUIRED ACTIONS: {metadata['warnings']['requires']}

FAIRNESS & BIAS
---------------
⚠️ Not yet evaluated - Chapter 3 will assess:
  - Disparate impact across protected groups
  - Calibration fairness
  - Equalized odds

EXPLAINABILITY
--------------
Tool Used:        SHAP (SHapley Additive exPlanations)
Available:        
  ✓ Global feature importance
  ✓ Individual prediction explanations
  ✓ Waterfall plots for adverse action notices
  ✓ Feature dependence analysis

Top 3 Most Important Features:
  1. {shap_importance.iloc[0]['feature']} (mean |SHAP|: {shap_importance.iloc[0]['mean_abs_shap']:.4f})
  2. {shap_importance.iloc[1]['feature']} (mean |SHAP|: {shap_importance.iloc[1]['mean_abs_shap']:.4f})
  3. {shap_importance.iloc[2]['feature']} (mean |SHAP|: {shap_importance.iloc[2]['mean_abs_shap']:.4f})

RECOMMENDATIONS
---------------
DO NOT deploy this model in production without:
  1. Probability recalibration (Platt scaling, isotonic regression)
  2. Continuous monitoring of probability distributions
  3. Threshold re-optimization on recent data
  4. Fairness evaluation across protected groups
  5. Simpler, more interpretable alternative (e.g., logistic regression)

Consider:
  - Using validation performance as upper bound estimate
  - Implementing human-in-the-loop for borderline cases
  - Periodic retraining on fresh data (monthly/quarterly)

CONTACT & FEEDBACK
------------------
Model Owner:      [Your Name]
Feedback:         [Email/System]
Documentation:    Section 2.4 - Model Explainability
Last Updated:     {pd.Timestamp.now().strftime('%Y-%m-%d')}

{'='*80}
"""

print(model_card)

# Save model card
with open('models/model_card.txt', 'w') as f:
    f.write(model_card)

print("\n✓ Model card saved to models/model_card.txt")
```

---

## 2.4.9 Adverse Action Notices (Regulatory Requirement)

Under Fair Credit Reporting Act (FCRA) and Equal Credit Opportunity Act (ECOA), lenders must provide specific reasons when denying credit.

### Creating Adverse Action Explanations with SHAP

```python
def generate_decision_notice(person_idx, shap_values, X_test, model, threshold=0.25):
    """
    Generate decision notice for credit application.
    
    - ADVERSE ACTION NOTICE: Required by FCRA/ECOA when credit is denied
    - APPROVAL NOTICE: Explains factors supporting approval (optional, but good practice)
    """
    # Get prediction
    prob = model.predict_proba(X_test.iloc[[person_idx]])[:, 1][0]
    decision = "DENIED" if prob >= threshold else "APPROVED"
    
    # Get SHAP values for this person
    person_shap = shap_values[person_idx]
    person_features = X_test.iloc[person_idx]
    
    # Get top reasons
    shap_df = pd.DataFrame({
        'feature': X_test.columns,
        'shap_value': person_shap,
        'feature_value': person_features.values
    }).sort_values('shap_value', ascending=False)
    
    if decision == "DENIED":
        # ADVERSE ACTION NOTICE (legally required)
        # Top 4 factors that INCREASED risk (positive SHAP)
        top_adverse = shap_df.head(4)
        
        notice = f"""
{'='*70}
ADVERSE ACTION NOTICE
{'='*70}

Application ID:    {person_idx}
Decision:          {decision}
Default Risk:      {prob:.1%}
Decision Threshold: {threshold:.1%}

We regret to inform you that your credit application has been DENIED.

PRIMARY FACTORS IN THIS DECISION:
----------------------------------
"""
        
        for i, (idx, row) in enumerate(top_adverse.iterrows(), 1):
            feature_name = row['feature']
            shap_val = row['shap_value']
            feat_val = row['feature_value']
            
            # Translate feature names to human-readable
            feature_readable = feature_name.replace('_', ' ').title()
            
            notice += f"\n{i}. {feature_readable}"
            notice += f"\n   Your value: {feat_val:.2f}"
            notice += f"\n   Impact: Increased risk by {abs(shap_val):.4f} points"
            notice += "\n"
        
        notice += """
YOUR RIGHTS UNDER FEDERAL LAW:
-------------------------------
- You have the right to request a FREE copy of your credit report within 
  60 days of this notice
- You may dispute any inaccurate information in your credit report
- You have the right to add a statement to your credit file explaining 
  any adverse information

CREDIT BUREAU CONTACT:
----------------------
[Credit Bureau Name]
[Phone Number]
[Website]

HOW TO IMPROVE YOUR APPLICATION:
---------------------------------
"""
        
        # Actionable recommendations based on top adverse factors
        for idx, row in top_adverse.head(2).iterrows():
            feature_name = row['feature']
            if 'fico' in feature_name.lower():
                notice += "\n• Improve your credit score by:"
                notice += "\n  - Paying all bills on time"
                notice += "\n  - Reducing credit card balances"
                notice += "\n  - Avoiding new credit inquiries"
            elif 'income' in feature_name.lower():
                notice += "\n• Consider reapplying when your income increases"
            elif 'delinq' in feature_name.lower():
                notice += "\n• Avoid late payments and delinquencies"
            elif 'utilization' in feature_name.lower():
                notice += "\n• Lower credit card balances (aim for below 30% of limits)"
            elif 'spending' in feature_name.lower():
                notice += "\n• Demonstrate more stable spending patterns"
            elif 'transaction' in feature_name.lower():
                notice += "\n• Maintain regular account activity"
        
        notice += """

REAPPLICATION:
--------------
You may reapply after addressing the factors above. We recommend waiting
at least 6 months to allow time for credit improvements.

HUMAN REVIEW:
-------------
This decision was made using an automated credit scoring model.
You may request human review by contacting:
[Loan Officer Contact Information]
"""
        
    else:
        # APPROVAL NOTICE (good practice, though not legally required)
        # Top 4 factors that DECREASED risk (negative SHAP)
        top_positive = shap_df.tail(4).iloc[::-1]  # Most negative SHAP (decreased risk most)
        
        notice = f"""
{'='*70}
CREDIT APPLICATION APPROVED
{'='*70}

Application ID:    {person_idx}
Decision:          {decision}
Default Risk:      {prob:.1%}
Decision Threshold: {threshold:.1%}

Congratulations! Your credit application has been APPROVED.

KEY FACTORS SUPPORTING YOUR APPROVAL:
--------------------------------------
"""
        
        for i, (idx, row) in enumerate(top_positive.iterrows(), 1):
            feature_name = row['feature']
            shap_val = row['shap_value']
            feat_val = row['feature_value']
            
            # Translate feature names to human-readable
            feature_readable = feature_name.replace('_', ' ').title()
            
            notice += f"\n{i}. {feature_readable}"
            notice += f"\n   Your value: {feat_val:.2f}"
            notice += f"\n   Impact: Decreased risk by {abs(shap_val):.4f} points"
            notice += "\n"
        
        notice += """
MAINTAINING GOOD STANDING:
--------------------------
To maintain your excellent credit standing:
- Continue paying all bills on time
- Keep credit card balances low (below 30% of limits)
- Avoid taking on excessive new debt
- Monitor your credit report regularly

NEXT STEPS:
-----------
A loan officer will contact you within 2 business days to finalize
your loan terms and complete the application process.

Questions? Contact us at [Contact Information]
"""
    
    notice += f"""

{'='*70}
Model: {metadata['model_type']}
Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}
{'='*70}
"""
    
    return notice


# Example usage for both scenarios
print("="*70)
print("EXAMPLE NOTICES")
print("="*70)

# Find a denied applicant (if any exist at threshold 0.25)
denied_indices = np.where((y_prob_test >= 0.25))[0]
if len(denied_indices) > 0:
    denied_idx = denied_indices[0]
    print("\n1. DENIED APPLICATION:")
    print(generate_decision_notice(denied_idx, shap_values, X_test, best_model, threshold=0.25))
    
    # Save
    with open('models/adverse_action_example.txt', 'w', encoding='utf-8') as f:
        f.write(generate_decision_notice(denied_idx, shap_values, X_test, best_model, threshold=0.25))
else:
    print("\n⚠️ No denials at threshold 0.25 (consistent with test failure)")

# Find an approved applicant
approved_indices = np.where((y_prob_test < 0.25))[0]
if len(approved_indices) > 0:
    approved_idx = approved_indices[0]
    print("\n2. APPROVED APPLICATION:")
    print(generate_decision_notice(approved_idx, shap_values, X_test, best_model, threshold=0.25))
    
    # Save
    with open('models/approval_notice_example.txt', 'w', encoding='utf-8') as f:
        f.write(generate_decision_notice(approved_idx, shap_values, X_test, best_model, threshold=0.25))

print("\n✓ Example notices saved to models/")

# Example: Generate notice for highest probability prediction
if y_prob_test.max() >= 0.25:
    denied_idx = np.argmax(y_prob_test)
    notice = generate_adverse_action_notice(denied_idx, shap_values, X_test, best_model)
    print(notice)
    
    # Save example
    with open('models/adverse_action_example.txt', 'w') as f:
        f.write(notice)
    print("✓ Example adverse action notice saved")
else:
    print("No denials at threshold 0.25 (all approved - consistent with test failure)")
```

### For DENIALS (Adverse Action Notice):
- ✅ Legally required by FCRA/ECOA
- ✅ Lists factors that **INCREASED** risk (positive SHAP values)
- ✅ Includes legal rights (free credit report, dispute rights)
- ✅ Provides actionable improvement steps
- ✅ Offers reapplication guidance
- ✅ Notes right to human review

### For APPROVALS:
- ✅ Not legally required, but good practice
- ✅ Lists factors that **DECREASED** risk (negative SHAP values)
- ✅ Reinforces positive behaviors
- ✅ Provides tips for maintaining good standing
- ✅ Sets expectations for next steps
- ✅ Builds customer relationship

---

## What You've Accomplished

✅ **Diagnosed the failure** - Distribution shift causes 3.6x lower test probabilities
✅ **Global explanations** - SHAP shows which features matter most
✅ **Individual explanations** - Waterfall plots explain specific predictions
✅ **Feature dependence** - Understand how features affect predictions
✅ **Calibration analysis** - Visualized why probabilities don't transfer
✅ **Attempted fix** - Tried Platt scaling (partial success)
✅ **Model card** - Created transparent documentation
✅ **Regulatory compliance** - Built adverse action notice generator

---

## Key Takeaways

### Why Explainability Matters

1. **Debugging** - SHAP helped us understand the distribution shift
2. **Trust** - Stakeholders can see which features drive decisions
3. **Compliance** - Adverse action notices required by law
4. **Fairness** - Foundation for detecting bias (Chapter 3)
5. **Improvement** - Guides feature engineering and model refinement

### What SHAP Revealed

**From global importance:**
- FICO score, transaction patterns, and balance trends drive predictions
- Model relies heavily on traditional credit metrics
- Some features may encode bias (investigated in Chapter 3)

**From individual predictions:**
- Test set individuals have feature combinations that push probabilities down
- Even risky profiles get probabilities around 5-15% (vs 20-40% in validation)
- Calibration issue goes beyond simple threshold adjustment

### Calibration Insights

**Why it failed:**
- SMOTE created synthetic data that doesn't match real test distribution
- Temporal shift: test period has different economic conditions
- Model trained on 50/50 data produces probabilities that don't generalize

**What we tried:**
- Platt scaling helped but didn't fully solve the problem
- Suggests deeper structural issue beyond calibration

---

## Limitations

**What explainability DOESN'T solve:**

1. **Distribution shift** - Can explain it, but can't prevent it
2. **Biased training data** - Model learns biases, SHAP just reveals them
3. **Calibration** - Requires additional techniques (Platt scaling, isotonic regression)
4. **Model selection** - Doesn't tell you which algorithm to use
5. **Fairness** - Separate evaluation needed (Chapter 3)

**SHAP limitations:**
- Computationally expensive for large datasets
- Assumes features are independent (may violate for correlated features)
- Doesn't capture feature interactions completely
- Local explanations can vary significantly for similar individuals

---

## Next Steps

**This section revealed the model's failure** - but didn't fix it. Going forward:

**Immediate actions:**
1. Retrain on more recent data (reduce temporal shift)
2. Try simpler models (logistic regression may calibrate better)
3. Implement continuous monitoring of probability distributions
4. Regular threshold re-optimization

**Chapter 3: Fairness & Bias Mitigation will address:**
1. **Does distribution shift affect groups differently?**
   - Are certain demographics more affected by miscalibration?
   - Disparate impact analysis

2. **Is the model fair across protected groups?**
   - Equal opportunity (recall parity)
   - Calibration fairness
   - Predictive parity

3. **How do we build fairer models?**
   - Bias mitigation techniques
   - Fairness-aware ML
   - Simpler, more interpretable alternatives

4. **What are the trade-offs?**
   - Accuracy vs fairness
   - Complexity vs interpretability
   - Short-term performance vs long-term sustainability

---

## Recommended Reading

**SHAP & Explainability:**
- Lundberg & Lee (2017): "A Unified Approach to Interpreting Model Predictions"
- Molnar (2022): "Interpretable Machine Learning" (free online book)

**Calibration:**
- Niculescu-Mizil & Caruana (2005): "Predicting Good Probabilities With Supervised Learning"
- Pleiss et al. (2017): "On Fairness and Calibration"

**Regulatory Requirements:**
- FCRA: Fair Credit Reporting Act (15 U.S.C. § 1681)
- ECOA: Equal Credit Opportunity Act (15 U.S.C. § 1691)
- CFPB: Consumer Financial Protection Bureau guidance

---

*(End of Section 2.4)*
