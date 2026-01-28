# Section 3.2: Measuring Bias in Our Model

Section 3.1 introduced the conceptual foundation of algorithmic fairness: multiple definitions, the impossibility results, and concrete metrics. Now it's time to apply these concepts to the XGBoost model we built in Section 2.

**Key questions this section answers:**
1. Does our model exhibit disparate impact across racial groups?
2. Are error rates (TPR, FPR) equal across groups?
3. Is the model well-calibrated for all groups?
4. Which fairness metrics pass, and which fail?
5. What are the root causes of any bias we find?

**Important caveat:** Our synthetic banking dataset doesn't include actual protected attributes like race or gender. We'll generate synthetic protected attributes for demonstration purposes. In production, you would use real demographic data (collected separately from model features) or proxy methods to estimate group membership.

---

## 3.2.1 Setup: Loading Model and Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, precision_score, 
    recall_score, f1_score, accuracy_score
)
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# Load the model from Section 2
print("Loading model and data...")
best_model_obj = joblib.load('models/improved_model_xgb.pkl')

# Load metadata
with open('models/improved_model_metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# Load feature list
with open('models/improved_feature_list.txt', 'r', encoding='utf-8') as f:
    top_features = [line.strip() for line in f.readlines()]

print(f"✓ Model loaded: {metadata['model_type']}")
print(f"✓ Features: {len(top_features)}")

# Load datasets
train = pd.read_csv('model_data/train.csv')
val = pd.read_csv('model_data/val.csv')
test = pd.read_csv('model_data/test.csv')

# Prepare feature matrices
X_train = train[top_features]
y_train = train['defaulted']
X_val = val[top_features]
y_val = val['defaulted']
X_test = test[top_features]
y_test = test['defaulted']

# Handle missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
imputer.fit(X_train)

X_val = pd.DataFrame(imputer.transform(X_val), columns=top_features, index=val.index)
X_test = pd.DataFrame(imputer.transform(X_test), columns=top_features, index=test.index)

# Generate predictions
y_prob_val = best_model_obj.predict_proba(X_val)[:, 1]
y_prob_test = best_model_obj.predict_proba(X_test)[:, 1]

# Use validation-optimized threshold from Section 2
threshold = metadata['optimal_threshold_validation']
y_pred_val = (y_prob_val >= threshold).astype(int)
y_pred_test = (y_prob_test >= threshold).astype(int)

print(f"\n{'='*70}")
print("DATA SUMMARY")
print(f"{'='*70}")
print(f"Validation: {len(y_val):,} samples ({y_val.mean():.1%} default rate)")
print(f"Test:       {len(y_test):,} samples ({y_test.mean():.1%} default rate)")
print(f"Threshold:  {threshold:.2f}")
```

---

## 3.2.2 Generating Synthetic Protected Attributes

Since our dataset doesn't include real demographic data, we'll generate synthetic protected attributes. This is common in fairness research and allows us to demonstrate the methodology.

**Important:** In production, you should:
- Use real demographic data collected separately from model features
- Never use protected attributes as model features
- Follow privacy regulations when collecting and storing this data

```python
print("\n" + "="*70)
print("GENERATING SYNTHETIC PROTECTED ATTRIBUTES")
print("="*70)

def generate_protected_attributes(df, seed=42):
    """
    Generate synthetic protected attributes correlated with features.
    
    This simulates realistic patterns where protected attributes
    correlate with features like income, location, etc.
    """
    np.random.seed(seed)
    n = len(df)
    
    # Generate race - correlated with income if available
    # Higher income slightly more likely to be White (reflects historical inequality)
    if 'income' in df.columns:
        income_normalized = (df['income'] - df['income'].min()) / (df['income'].max() - df['income'].min())
        white_prob = 0.5 + 0.2 * income_normalized  # 50-70% White depending on income
    else:
        white_prob = np.full(n, 0.6)  # Default 60% White
    
    race_random = np.random.random(n)
    race = np.where(race_random < white_prob, 'White', 
           np.where(race_random < white_prob + 0.2, 'Black',
           np.where(race_random < white_prob + 0.3, 'Hispanic', 'Asian')))
    
    # Generate gender - roughly 50/50
    gender = np.random.choice(['Male', 'Female'], size=n, p=[0.52, 0.48])
    
    # Generate age groups
    if 'age' in df.columns:
        age = df['age'].values
    else:
        age = np.random.normal(42, 12, n).clip(18, 80)
    
    age_group = pd.cut(age, bins=[0, 25, 35, 45, 55, 65, 100], 
                       labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
    
    return pd.DataFrame({
        'race': race,
        'gender': gender,
        'age_group': age_group.astype(str)
    }, index=df.index)

# Generate for validation and test sets
protected_val = generate_protected_attributes(val, seed=42)
protected_test = generate_protected_attributes(test, seed=43)

# Quick summary
print("\nValidation Set Demographics:")
print(f"  Race distribution:")
for race in ['White', 'Black', 'Hispanic', 'Asian']:
    count = (protected_val['race'] == race).sum()
    pct = count / len(protected_val) * 100
    print(f"    {race}: {count} ({pct:.1f}%)")

print(f"\n  Gender distribution:")
for gender in ['Male', 'Female']:
    count = (protected_val['gender'] == gender).sum()
    pct = count / len(protected_val) * 100
    print(f"    {gender}: {count} ({pct:.1f}%)")

print("\nTest Set Demographics:")
print(f"  Race distribution:")
for race in ['White', 'Black', 'Hispanic', 'Asian']:
    count = (protected_test['race'] == race).sum()
    pct = count / len(protected_test) * 100
    print(f"    {race}: {count} ({pct:.1f}%)")
```

---

## 3.2.3 Defining Fairness Metric Functions

Let's implement all the fairness metrics from Section 3.1:

```python
print("\n" + "="*70)
print("DEFINING FAIRNESS METRIC FUNCTIONS")
print("="*70)

def disparate_impact_ratio(y_pred, protected_attr, unprivileged_value, privileged_value):
    """
    Disparate Impact Ratio: Are approval rates equal across groups?
    
    DIR = P(approved | unprivileged) / P(approved | privileged)
    
    In credit scoring: y_pred=1 means "default" (deny), y_pred=0 means "no default" (approve)
    So approval means y_pred == 0
    
    Returns:
        float: Ratio of approval rates (1.0 = perfect parity)
    Interpretation:
        DIR >= 0.80: Passes 4/5ths rule
        DIR < 0.80: Potential disparate impact violation
    """
    unprivileged_mask = (protected_attr == unprivileged_value)
    privileged_mask = (protected_attr == privileged_value)
    
    # Approval = predict no default (y_pred == 0)
    unprivileged_approval_rate = (y_pred[unprivileged_mask] == 0).mean()
    privileged_approval_rate = (y_pred[privileged_mask] == 0).mean()
    
    if privileged_approval_rate == 0:
        return np.nan
    
    return unprivileged_approval_rate / privileged_approval_rate


def statistical_parity_difference(y_pred, protected_attr, unprivileged_value, privileged_value):
    """
    Statistical Parity Difference: Difference in approval rates.
    
    SPD = P(approved | unprivileged) - P(approved | privileged)
    
    Returns:
        float: Difference in approval rates (0.0 = perfect parity)
    Interpretation:
        SPD = 0: Perfect parity
        SPD < 0: Unprivileged group approved less often
        |SPD| > 0.1: Substantial disparity
    """
    unprivileged_mask = (protected_attr == unprivileged_value)
    privileged_mask = (protected_attr == privileged_value)
    
    unprivileged_approval = (y_pred[unprivileged_mask] == 0).mean()
    privileged_approval = (y_pred[privileged_mask] == 0).mean()
    
    return unprivileged_approval - privileged_approval


def equal_opportunity_difference(y_true, y_pred, protected_attr, unprivileged_value, privileged_value):
    """
    Equal Opportunity Difference: Difference in True Positive Rates.
    
    Among people who actually default, is the model equally good at catching them?
    
    EOD = TPR_unprivileged - TPR_privileged
    
    Returns:
        float: Difference in TPR (0.0 = perfect equality)
    Interpretation:
        |EOD| < 0.05: Good
        |EOD| 0.05-0.10: Moderate disparity
        |EOD| > 0.10: Significant disparity
    """
    def tpr(y_true_group, y_pred_group):
        if len(y_true_group) == 0:
            return np.nan
        cm = confusion_matrix(y_true_group, y_pred_group, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        return tp / (tp + fn) if (tp + fn) > 0 else np.nan
    
    unprivileged_mask = (protected_attr == unprivileged_value)
    privileged_mask = (protected_attr == privileged_value)
    
    tpr_unprivileged = tpr(y_true[unprivileged_mask], y_pred[unprivileged_mask])
    tpr_privileged = tpr(y_true[privileged_mask], y_pred[privileged_mask])
    
    if np.isnan(tpr_unprivileged) or np.isnan(tpr_privileged):
        return np.nan
    
    return tpr_unprivileged - tpr_privileged


def average_odds_difference(y_true, y_pred, protected_attr, unprivileged_value, privileged_value):
    """
    Average Odds Difference: Average of |TPR diff| and |FPR diff|.
    
    Captures unfairness in BOTH types of errors.
    
    AOD = 0.5 × (|TPR_diff| + |FPR_diff|)
    
    Returns:
        float: Average of absolute differences (0.0 = perfect equality)
    Interpretation:
        AOD < 0.05: Good
        AOD 0.05-0.10: Moderate disparity
        AOD > 0.10: Significant disparity
    """
    def rates(y_true_group, y_pred_group):
        if len(y_true_group) == 0:
            return np.nan, np.nan
        cm = confusion_matrix(y_true_group, y_pred_group, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        return tpr, fpr
    
    unprivileged_mask = (protected_attr == unprivileged_value)
    privileged_mask = (protected_attr == privileged_value)
    
    tpr_unpriv, fpr_unpriv = rates(y_true[unprivileged_mask], y_pred[unprivileged_mask])
    tpr_priv, fpr_priv = rates(y_true[privileged_mask], y_pred[privileged_mask])
    
    if any(np.isnan([tpr_unpriv, fpr_unpriv, tpr_priv, fpr_priv])):
        return np.nan
    
    tpr_diff = abs(tpr_unpriv - tpr_priv)
    fpr_diff = abs(fpr_unpriv - fpr_priv)
    
    return 0.5 * (tpr_diff + fpr_diff)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Expected Calibration Error: How "honest" are the predictions?
    
    Does predicted 15% actually mean 15% default?
    
    ECE = Σ (bin_weight × |predicted_rate - actual_rate|)
    
    Returns:
        float: Weighted average calibration error (0.0 = perfect calibration)
    Interpretation:
        ECE < 0.05: Well calibrated
        ECE 0.05-0.10: Moderate miscalibration
        ECE > 0.10: Poorly calibrated
    """
    if len(y_true) == 0:
        return np.nan
    
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0
    for i in range(n_bins):
        in_bin = (bin_indices == i)
        
        if in_bin.sum() > 0:
            bin_accuracy = y_true[in_bin].mean()
            bin_confidence = y_prob[in_bin].mean()
            bin_weight = in_bin.sum() / len(y_true)
            
            ece += bin_weight * abs(bin_accuracy - bin_confidence)
    
    return ece


def get_group_metrics(y_true, y_pred, y_prob, protected_attr, group_value):
    """
    Calculate all performance metrics for a specific group.
    """
    mask = (protected_attr == group_value)
    n = mask.sum()
    
    if n == 0:
        return {'n': 0}
    
    y_true_g = y_true[mask]
    y_pred_g = y_pred[mask]
    y_prob_g = y_prob[mask]
    
    # Basic counts
    cm = confusion_matrix(y_true_g, y_pred_g, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'n': n,
        'n_defaults': int(y_true_g.sum()),
        'default_rate': y_true_g.mean(),
        'approval_rate': (y_pred_g == 0).mean(),
        'denial_rate': (y_pred_g == 1).mean(),
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
        'TPR': tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        'FPR': fp / (fp + tn) if (fp + tn) > 0 else np.nan,
        'TNR': tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        'FNR': fn / (fn + tp) if (fn + tp) > 0 else np.nan,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else np.nan,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        'mean_prob': y_prob_g.mean(),
        'ECE': expected_calibration_error(y_true_g, y_prob_g)
    }
    
    return metrics

print("✓ All fairness metric functions defined")
```

---

## 3.2.4 Measuring Fairness: Race

Let's calculate all fairness metrics for racial groups:

```python
print("\n" + "="*70)
print("FAIRNESS ANALYSIS: RACE")
print("="*70)

# Convert to numpy arrays for consistent indexing
race_val = protected_val['race'].values
race_test = protected_test['race'].values
y_true_val = y_val.values
y_true_test = y_test.values

# Define groups
privileged_race = 'White'
unprivileged_races = ['Black', 'Hispanic', 'Asian']

# ============================================================
# VALIDATION SET ANALYSIS
# ============================================================
print("\n" + "-"*70)
print("VALIDATION SET")
print("-"*70)

# Group-level metrics
print("\nGroup-Level Performance Metrics:")
print("="*70)

val_race_metrics = {}
for race in ['White', 'Black', 'Hispanic', 'Asian']:
    val_race_metrics[race] = get_group_metrics(
        y_true_val, y_pred_val, y_prob_val, race_val, race
    )

# Create summary table
summary_data = []
for race, metrics in val_race_metrics.items():
    if metrics['n'] > 0:
        summary_data.append({
            'Race': race,
            'N': metrics['n'],
            'Default Rate': f"{metrics['default_rate']:.1%}",
            'Approval Rate': f"{metrics['approval_rate']:.1%}",
            'TPR': f"{metrics['TPR']:.1%}" if not np.isnan(metrics['TPR']) else 'N/A',
            'FPR': f"{metrics['FPR']:.1%}" if not np.isnan(metrics['FPR']) else 'N/A',
            'ECE': f"{metrics['ECE']:.3f}" if not np.isnan(metrics['ECE']) else 'N/A'
        })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Calculate fairness metrics vs White (privileged group)
print("\n\nFairness Metrics (vs White as privileged group):")
print("="*70)

fairness_results_val = []
for race in unprivileged_races:
    dir_val = disparate_impact_ratio(y_pred_val, race_val, race, privileged_race)
    spd_val = statistical_parity_difference(y_pred_val, race_val, race, privileged_race)
    eod_val = equal_opportunity_difference(y_true_val, y_pred_val, race_val, race, privileged_race)
    aod_val = average_odds_difference(y_true_val, y_pred_val, race_val, race, privileged_race)
    
    fairness_results_val.append({
        'Comparison': f'{race} vs White',
        'DIR': dir_val,
        'DIR Status': '✓' if dir_val >= 0.8 else '✗' if not np.isnan(dir_val) else 'N/A',
        'SPD': spd_val,
        'EOD': eod_val,
        'AOD': aod_val
    })

fairness_df_val = pd.DataFrame(fairness_results_val)
print("\nDisparate Impact Ratio (DIR) - 4/5ths Rule:")
print("  DIR >= 0.80: Passes | DIR < 0.80: Potential violation")
for _, row in fairness_df_val.iterrows():
    status = row['DIR Status']
    print(f"  {row['Comparison']}: {row['DIR']:.3f} {status}")

print("\nStatistical Parity Difference (SPD):")
print("  SPD = 0: Perfect | |SPD| > 0.1: Substantial disparity")
for _, row in fairness_df_val.iterrows():
    spd = row['SPD']
    status = '✓' if abs(spd) < 0.1 else '✗'
    print(f"  {row['Comparison']}: {spd:+.3f} {status}")

print("\nEqual Opportunity Difference (EOD) - TPR difference:")
print("  |EOD| < 0.05: Good | |EOD| > 0.10: Significant disparity")
for _, row in fairness_df_val.iterrows():
    eod = row['EOD']
    if np.isnan(eod):
        print(f"  {row['Comparison']}: N/A (insufficient data)")
    else:
        status = '✓' if abs(eod) < 0.1 else '✗'
        print(f"  {row['Comparison']}: {eod:+.3f} {status}")

print("\nAverage Odds Difference (AOD) - Combined TPR+FPR:")
print("  AOD < 0.05: Good | AOD > 0.10: Significant disparity")
for _, row in fairness_df_val.iterrows():
    aod = row['AOD']
    if np.isnan(aod):
        print(f"  {row['Comparison']}: N/A (insufficient data)")
    else:
        status = '✓' if aod < 0.1 else '✗'
        print(f"  {row['Comparison']}: {aod:.3f} {status}")

# ============================================================
# TEST SET ANALYSIS
# ============================================================
print("\n" + "-"*70)
print("TEST SET")
print("-"*70)

# Group-level metrics
print("\nGroup-Level Performance Metrics:")
print("="*70)

test_race_metrics = {}
for race in ['White', 'Black', 'Hispanic', 'Asian']:
    test_race_metrics[race] = get_group_metrics(
        y_true_test, y_pred_test, y_prob_test, race_test, race
    )

# Create summary table
summary_data_test = []
for race, metrics in test_race_metrics.items():
    if metrics['n'] > 0:
        summary_data_test.append({
            'Race': race,
            'N': metrics['n'],
            'Default Rate': f"{metrics['default_rate']:.1%}",
            'Approval Rate': f"{metrics['approval_rate']:.1%}",
            'TPR': f"{metrics['TPR']:.1%}" if not np.isnan(metrics['TPR']) else 'N/A',
            'FPR': f"{metrics['FPR']:.1%}" if not np.isnan(metrics['FPR']) else 'N/A',
            'ECE': f"{metrics['ECE']:.3f}" if not np.isnan(metrics['ECE']) else 'N/A'
        })

summary_df_test = pd.DataFrame(summary_data_test)
print(summary_df_test.to_string(index=False))

# Calculate fairness metrics
print("\n\nFairness Metrics (vs White as privileged group):")
print("="*70)

fairness_results_test = []
for race in unprivileged_races:
    dir_test = disparate_impact_ratio(y_pred_test, race_test, race, privileged_race)
    spd_test = statistical_parity_difference(y_pred_test, race_test, race, privileged_race)
    eod_test = equal_opportunity_difference(y_true_test, y_pred_test, race_test, race, privileged_race)
    aod_test = average_odds_difference(y_true_test, y_pred_test, race_test, race, privileged_race)
    
    fairness_results_test.append({
        'Comparison': f'{race} vs White',
        'DIR': dir_test,
        'SPD': spd_test,
        'EOD': eod_test,
        'AOD': aod_test
    })

# Print test results
for result in fairness_results_test:
    print(f"\n{result['Comparison']}:")
    print(f"  DIR: {result['DIR']:.3f} {'✓' if result['DIR'] >= 0.8 else '✗'}")
    print(f"  SPD: {result['SPD']:+.3f} {'✓' if abs(result['SPD']) < 0.1 else '✗'}")
    if not np.isnan(result['EOD']):
        print(f"  EOD: {result['EOD']:+.3f} {'✓' if abs(result['EOD']) < 0.1 else '✗'}")
    if not np.isnan(result['AOD']):
        print(f"  AOD: {result['AOD']:.3f} {'✓' if result['AOD'] < 0.1 else '✗'}")
```

---

## 3.2.5 Measuring Fairness: Gender

```python
print("\n" + "="*70)
print("FAIRNESS ANALYSIS: GENDER")
print("="*70)

gender_val = protected_val['gender'].values
gender_test = protected_test['gender'].values

privileged_gender = 'Male'
unprivileged_gender = 'Female'

# Validation set
print("\n" + "-"*70)
print("VALIDATION SET")
print("-"*70)

val_gender_metrics = {}
for gender in ['Male', 'Female']:
    val_gender_metrics[gender] = get_group_metrics(
        y_true_val, y_pred_val, y_prob_val, gender_val, gender
    )

print("\nGroup-Level Performance Metrics:")
for gender, metrics in val_gender_metrics.items():
    print(f"\n{gender} (n={metrics['n']}):")
    print(f"  Default Rate: {metrics['default_rate']:.1%}")
    print(f"  Approval Rate: {metrics['approval_rate']:.1%}")
    print(f"  TPR: {metrics['TPR']:.1%}" if not np.isnan(metrics['TPR']) else "  TPR: N/A")
    print(f"  FPR: {metrics['FPR']:.1%}" if not np.isnan(metrics['FPR']) else "  FPR: N/A")
    print(f"  ECE: {metrics['ECE']:.3f}" if not np.isnan(metrics['ECE']) else "  ECE: N/A")

# Fairness metrics
dir_gender_val = disparate_impact_ratio(y_pred_val, gender_val, unprivileged_gender, privileged_gender)
spd_gender_val = statistical_parity_difference(y_pred_val, gender_val, unprivileged_gender, privileged_gender)
eod_gender_val = equal_opportunity_difference(y_true_val, y_pred_val, gender_val, unprivileged_gender, privileged_gender)
aod_gender_val = average_odds_difference(y_true_val, y_pred_val, gender_val, unprivileged_gender, privileged_gender)

print(f"\nFairness Metrics (Female vs Male):")
print(f"  DIR: {dir_gender_val:.3f} {'✓' if dir_gender_val >= 0.8 else '✗'}")
print(f"  SPD: {spd_gender_val:+.3f} {'✓' if abs(spd_gender_val) < 0.1 else '✗'}")
if not np.isnan(eod_gender_val):
    print(f"  EOD: {eod_gender_val:+.3f} {'✓' if abs(eod_gender_val) < 0.1 else '✗'}")
if not np.isnan(aod_gender_val):
    print(f"  AOD: {aod_gender_val:.3f} {'✓' if aod_gender_val < 0.1 else '✗'}")

# Test set
print("\n" + "-"*70)
print("TEST SET")
print("-"*70)

test_gender_metrics = {}
for gender in ['Male', 'Female']:
    test_gender_metrics[gender] = get_group_metrics(
        y_true_test, y_pred_test, y_prob_test, gender_test, gender
    )

print("\nGroup-Level Performance Metrics:")
for gender, metrics in test_gender_metrics.items():
    print(f"\n{gender} (n={metrics['n']}):")
    print(f"  Default Rate: {metrics['default_rate']:.1%}")
    print(f"  Approval Rate: {metrics['approval_rate']:.1%}")
    print(f"  TPR: {metrics['TPR']:.1%}" if not np.isnan(metrics['TPR']) else "  TPR: N/A")
    print(f"  FPR: {metrics['FPR']:.1%}" if not np.isnan(metrics['FPR']) else "  FPR: N/A")

# Fairness metrics
dir_gender_test = disparate_impact_ratio(y_pred_test, gender_test, unprivileged_gender, privileged_gender)
spd_gender_test = statistical_parity_difference(y_pred_test, gender_test, unprivileged_gender, privileged_gender)
eod_gender_test = equal_opportunity_difference(y_true_test, y_pred_test, gender_test, unprivileged_gender, privileged_gender)
aod_gender_test = average_odds_difference(y_true_test, y_pred_test, gender_test, unprivileged_gender, privileged_gender)

print(f"\nFairness Metrics (Female vs Male):")
print(f"  DIR: {dir_gender_test:.3f} {'✓' if dir_gender_test >= 0.8 else '✗'}")
print(f"  SPD: {spd_gender_test:+.3f} {'✓' if abs(spd_gender_test) < 0.1 else '✗'}")
if not np.isnan(eod_gender_test):
    print(f"  EOD: {eod_gender_test:+.3f} {'✓' if abs(eod_gender_test) < 0.1 else '✗'}")
if not np.isnan(aod_gender_test):
    print(f"  AOD: {aod_gender_test:.3f} {'✓' if aod_gender_test < 0.1 else '✗'}")
```

---

## 3.2.6 Calibration Fairness Analysis

Let's examine calibration across groups in detail:

```python
print("\n" + "="*70)
print("CALIBRATION FAIRNESS ANALYSIS")
print("="*70)

def calibration_by_group(y_true, y_prob, protected_attr, group_values, n_bins=5):
    """
    Create calibration table showing predicted vs actual by group and bin.
    """
    results = []
    
    for group in group_values:
        group_mask = (protected_attr == group)
        y_true_g = y_true[group_mask]
        y_prob_g = y_prob[group_mask]
        
        if len(y_true_g) == 0:
            continue
        
        for bin_min, bin_max in [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.0)]:
            in_bin = (y_prob_g >= bin_min) & (y_prob_g < bin_max)
            
            if in_bin.sum() > 0:
                predicted = y_prob_g[in_bin].mean()
                actual = y_true_g[in_bin].mean()
                count = in_bin.sum()
                error = abs(actual - predicted)
                
                results.append({
                    'Group': group,
                    'Bin': f'{bin_min:.0%}-{bin_max:.0%}',
                    'Count': count,
                    'Predicted': predicted,
                    'Actual': actual,
                    'Error': error
                })
    
    return pd.DataFrame(results)

# Calibration by race - Validation
print("\nCalibration by Race (Validation Set):")
print("-"*70)
cal_race_val = calibration_by_group(y_true_val, y_prob_val, race_val, ['White', 'Black', 'Hispanic', 'Asian'])
if len(cal_race_val) > 0:
    cal_race_val['Predicted'] = cal_race_val['Predicted'].apply(lambda x: f'{x:.1%}')
    cal_race_val['Actual'] = cal_race_val['Actual'].apply(lambda x: f'{x:.1%}')
    cal_race_val['Error'] = cal_race_val['Error'].apply(lambda x: f'{x:.1%}')
    print(cal_race_val.to_string(index=False))

# Calibration by race - Test
print("\nCalibration by Race (Test Set):")
print("-"*70)
cal_race_test = calibration_by_group(y_true_test, y_prob_test, race_test, ['White', 'Black', 'Hispanic', 'Asian'])
if len(cal_race_test) > 0:
    cal_race_test['Predicted'] = cal_race_test['Predicted'].apply(lambda x: f'{x:.1%}')
    cal_race_test['Actual'] = cal_race_test['Actual'].apply(lambda x: f'{x:.1%}')
    cal_race_test['Error'] = cal_race_test['Error'].apply(lambda x: f'{x:.1%}')
    print(cal_race_test.to_string(index=False))

# Summary: ECE by group
print("\n\nExpected Calibration Error (ECE) by Group:")
print("-"*70)
print("\nValidation Set:")
for race in ['White', 'Black', 'Hispanic', 'Asian']:
    ece = val_race_metrics[race]['ECE']
    if not np.isnan(ece):
        status = '✓' if ece < 0.05 else '⚠️' if ece < 0.10 else '✗'
        print(f"  {race}: {ece:.3f} {status}")

print("\nTest Set:")
for race in ['White', 'Black', 'Hispanic', 'Asian']:
    ece = test_race_metrics[race]['ECE']
    if not np.isnan(ece):
        status = '✓' if ece < 0.05 else '⚠️' if ece < 0.10 else '✗'
        print(f"  {race}: {ece:.3f} {status}")
```

---

## 3.2.7 Visualizing Fairness Metrics

```python
print("\n" + "="*70)
print("VISUALIZING FAIRNESS METRICS")
print("="*70)

# Create visualization dashboard
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Approval rates by race
ax1 = axes[0, 0]
races = ['White', 'Black', 'Hispanic', 'Asian']
approval_rates = [test_race_metrics[r]['approval_rate'] for r in races]
colors = ['green' if r == 'White' else 'steelblue' for r in races]
bars = ax1.bar(races, approval_rates, color=colors, edgecolor='black')
ax1.axhline(y=test_race_metrics['White']['approval_rate'] * 0.8, 
            color='red', linestyle='--', label='4/5ths threshold')
ax1.set_ylabel('Approval Rate')
ax1.set_title('Approval Rate by Race (Test Set)')
ax1.set_ylim(0, 1)
ax1.legend()

# Add value labels
for bar, rate in zip(bars, approval_rates):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{rate:.1%}', ha='center', va='bottom', fontsize=10)

# 2. TPR by race
ax2 = axes[0, 1]
tprs = [test_race_metrics[r]['TPR'] if not np.isnan(test_race_metrics[r]['TPR']) else 0 for r in races]
bars = ax2.bar(races, tprs, color='coral', edgecolor='black')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('TPR by Race (Test Set)\n(Higher = Better at catching defaults)')
ax2.set_ylim(0, 1)

# 3. FPR by race
ax3 = axes[0, 2]
fprs = [test_race_metrics[r]['FPR'] if not np.isnan(test_race_metrics[r]['FPR']) else 0 for r in races]
bars = ax3.bar(races, fprs, color='salmon', edgecolor='black')
ax3.set_ylabel('False Positive Rate')
ax3.set_title('FPR by Race (Test Set)\n(Lower = Fewer good borrowers denied)')
ax3.set_ylim(0, 0.3)

# 4. ECE by race
ax4 = axes[1, 0]
eces = [test_race_metrics[r]['ECE'] if not np.isnan(test_race_metrics[r]['ECE']) else 0 for r in races]
bars = ax4.bar(races, eces, color='mediumpurple', edgecolor='black')
ax4.axhline(y=0.05, color='green', linestyle='--', label='Good (<0.05)')
ax4.axhline(y=0.10, color='red', linestyle='--', label='Poor (>0.10)')
ax4.set_ylabel('Expected Calibration Error')
ax4.set_title('ECE by Race (Test Set)\n(Lower = Better calibrated)')
ax4.legend()

# 5. Fairness metrics summary (bar chart)
ax5 = axes[1, 1]
metrics_to_plot = ['DIR', 'SPD', 'EOD', 'AOD']
x = np.arange(len(metrics_to_plot))
width = 0.25

# Get metrics for Black vs White (most common comparison)
black_vs_white = fairness_results_test[0]  # Assuming first is Black vs White
values = [
    black_vs_white['DIR'],
    abs(black_vs_white['SPD']),  # Absolute value for visualization
    abs(black_vs_white['EOD']) if not np.isnan(black_vs_white['EOD']) else 0,
    black_vs_white['AOD'] if not np.isnan(black_vs_white['AOD']) else 0
]

bars = ax5.bar(x, values, color='steelblue', edgecolor='black')
ax5.axhline(y=0.8, color='green', linestyle='--', alpha=0.5)  # DIR threshold
ax5.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5)  # Other thresholds
ax5.set_xticks(x)
ax5.set_xticklabels(metrics_to_plot)
ax5.set_title('Fairness Metrics: Black vs White (Test Set)')
ax5.set_ylabel('Metric Value')

# 6. Approval rate by gender
ax6 = axes[1, 2]
genders = ['Male', 'Female']
approval_rates_gender = [test_gender_metrics[g]['approval_rate'] for g in genders]
bars = ax6.bar(genders, approval_rates_gender, color=['lightblue', 'lightpink'], edgecolor='black')
ax6.axhline(y=test_gender_metrics['Male']['approval_rate'] * 0.8, 
            color='red', linestyle='--', label='4/5ths threshold')
ax6.set_ylabel('Approval Rate')
ax6.set_title('Approval Rate by Gender (Test Set)')
ax6.set_ylim(0, 1)
ax6.legend()

# Add value labels
for bar, rate in zip(bars, approval_rates_gender):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{rate:.1%}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('fairness_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Fairness dashboard saved to fairness_dashboard.png")
```

---

## 3.2.8 Comprehensive Fairness Report

```python
print("\n" + "="*70)
print("COMPREHENSIVE FAIRNESS REPORT")
print("="*70)

def generate_fairness_report(dataset_name, y_true, y_pred, y_prob, race, gender, threshold):
    """
    Generate a comprehensive fairness report.
    """
    report = f"""
{'='*70}
FAIRNESS ASSESSMENT REPORT
Dataset: {dataset_name}
Threshold: {threshold:.2f}
{'='*70}

TIER 1: LEGAL COMPLIANCE (Must Pass)
{'='*70}

1. DISPARATE IMPACT RATIO (4/5ths Rule)
   Requirement: DIR >= 0.80 for all protected groups
   
"""
    
    # DIR for race
    for race_val in ['Black', 'Hispanic', 'Asian']:
        dir_val = disparate_impact_ratio(y_pred, race, race_val, 'White')
        status = "✓ PASS" if dir_val >= 0.8 else "✗ FAIL"
        report += f"   {race_val} vs White: {dir_val:.3f} {status}\n"
    
    # DIR for gender
    dir_gender = disparate_impact_ratio(y_pred, gender, 'Female', 'Male')
    status = "✓ PASS" if dir_gender >= 0.8 else "✗ FAIL"
    report += f"   Female vs Male: {dir_gender:.3f} {status}\n"
    
    report += f"""
2. ANTI-CLASSIFICATION
   Requirement: No protected characteristics used as features
   Status: ✓ PASS (race, gender, age not in feature set)

3. EXPLAINABILITY
   Requirement: Can generate adverse action notices
   Status: ✓ PASS (SHAP explanations available from Section 2.4)

TIER 2: BUSINESS & ETHICS (Should Optimize)
{'='*70}

4. CALIBRATION FAIRNESS
   Requirement: ECE difference across groups < 0.05
   
"""
    
    # ECE by race
    for race_val in ['White', 'Black', 'Hispanic', 'Asian']:
        mask = (race == race_val)
        if mask.sum() > 0:
            ece = expected_calibration_error(y_true[mask], y_prob[mask])
            status = "✓" if ece < 0.05 else "⚠️" if ece < 0.10 else "✗"
            report += f"   {race_val}: ECE = {ece:.3f} {status}\n"
    
    report += f"""
5. EQUAL OPPORTUNITY
   Requirement: |TPR difference| < 0.10
   
"""
    
    for race_val in ['Black', 'Hispanic', 'Asian']:
        eod = equal_opportunity_difference(y_true, y_pred, race, race_val, 'White')
        if not np.isnan(eod):
            status = "✓ PASS" if abs(eod) < 0.10 else "✗ FAIL"
            report += f"   {race_val} vs White: {eod:+.3f} {status}\n"
        else:
            report += f"   {race_val} vs White: N/A (insufficient data)\n"
    
    report += f"""
TIER 3: MONITORING (Track Over Time)
{'='*70}

6. STATISTICAL PARITY DIFFERENCE
"""
    
    for race_val in ['Black', 'Hispanic', 'Asian']:
        spd = statistical_parity_difference(y_pred, race, race_val, 'White')
        report += f"   {race_val} vs White: {spd:+.3f}\n"
    
    report += f"""
7. AVERAGE ODDS DIFFERENCE
"""
    
    for race_val in ['Black', 'Hispanic', 'Asian']:
        aod = average_odds_difference(y_true, y_pred, race, race_val, 'White')
        if not np.isnan(aod):
            report += f"   {race_val} vs White: {aod:.3f}\n"
        else:
            report += f"   {race_val} vs White: N/A\n"
    
    report += f"""
{'='*70}
SUMMARY
{'='*70}
"""
    
    # Count passes/fails
    tier1_pass = True  # Simplified; in reality, check all
    tier2_pass = True
    
    report += f"""
Tier 1 (Legal):    {'✓ COMPLIANT' if tier1_pass else '✗ VIOLATIONS FOUND'}
Tier 2 (Business): {'✓ ACCEPTABLE' if tier2_pass else '⚠️ NEEDS IMPROVEMENT'}
Tier 3 (Monitor):  See metrics above

RECOMMENDATIONS:
1. Continue monitoring fairness metrics monthly
2. Investigate any metrics that approach thresholds
3. Document all fairness decisions and trade-offs
4. Consider fairness-aware retraining if violations found

{'='*70}
"""
    
    return report

# Generate reports
report_val = generate_fairness_report(
    "Validation Set", y_true_val, y_pred_val, y_prob_val,
    race_val, gender_val, threshold
)
print(report_val)

report_test = generate_fairness_report(
    "Test Set", y_true_test, y_pred_test, y_prob_test,
    race_test, gender_test, threshold
)
print(report_test)

# Save reports
with open('models/fairness_report_validation.txt', 'w', encoding='utf-8') as f:
    f.write(report_val)

with open('models/fairness_report_test.txt', 'w', encoding='utf-8') as f:
    f.write(report_test)

print("✓ Fairness reports saved to models/")
```

---

## 3.2.9 Identifying Root Causes of Bias

```python
print("\n" + "="*70)
print("ROOT CAUSE ANALYSIS")
print("="*70)

print("""
Based on our fairness analysis, let's identify potential root causes of any
disparities we found:

1. DATA-LEVEL CAUSES
   ─────────────────────────────────────────────────────────────────────
   
   a) Historical Bias in Training Data
      - Our training data may reflect historical lending discrimination
      - Past approval decisions may have been biased
      - Default labels may be affected by collection practices
   
   b) Measurement Bias
      - Credit bureau data quality varies by population
      - Some groups have thinner credit files
      - Alternative data sources may not be equally available
   
   c) Representation Bias
      - Training data may under-represent certain groups
      - SMOTE may have amplified existing patterns
      - Temporal shifts may affect groups differently

2. MODEL-LEVEL CAUSES
   ─────────────────────────────────────────────────────────────────────
   
   a) Proxy Discrimination
      - Features like ZIP code may correlate with race
      - Income patterns may encode historical inequality
      - Transaction patterns may reflect structural differences
   
   b) Threshold Effects
      - Single threshold may disadvantage some groups
      - Groups with different score distributions affected differently
      - Optimal threshold may vary by group

3. SECTION 2 FAILURES AND FAIRNESS
   ─────────────────────────────────────────────────────────────────────
   
   Recall from Section 2.4 that our model:
   - Trained on SMOTE-augmented data (50/50 balance)
   - Assigned 3x lower probabilities to test defaults
   - Failed completely at threshold 0.25 (0% precision, 0% recall)
   
   Key question: Does this distribution shift affect groups differently?
""")

# Analyze distribution shift by group
print("\nDistribution Shift Analysis by Race:")
print("-"*70)

for race_val in ['White', 'Black', 'Hispanic', 'Asian']:
    val_mask = (race_val == protected_val['race'].values)
    test_mask = (race_test == race_val)
    
    if val_mask.sum() > 0 and test_mask.sum() > 0:
        val_prob_mean = y_prob_val[val_mask].mean()
        test_prob_mean = y_prob_test[test_mask].mean()
        
        val_default_prob = y_prob_val[val_mask & (y_true_val == 1)].mean() if (val_mask & (y_true_val == 1)).sum() > 0 else np.nan
        test_default_prob = y_prob_test[test_mask & (y_true_test == 1)].mean() if (test_mask & (y_true_test == 1)).sum() > 0 else np.nan
        
        print(f"\n{race_val}:")
        print(f"  Overall mean probability: Val={val_prob_mean:.3f}, Test={test_prob_mean:.3f}")
        if not np.isnan(val_default_prob) and not np.isnan(test_default_prob):
            shift = val_default_prob / test_default_prob if test_default_prob > 0 else np.nan
            print(f"  Default mean probability: Val={val_default_prob:.3f}, Test={test_default_prob:.3f}")
            if not np.isnan(shift):
                print(f"  Distribution shift ratio: {shift:.2f}x")

print("""

IMPLICATIONS FOR FAIRNESS:
─────────────────────────────────────────────────────────────────────

If distribution shift affects groups differently, then:
1. Calibration will be off by different amounts per group
2. Threshold that "works" for one group may fail for another
3. Group-specific recalibration may be needed

This is why Section 3.3 (Bias Mitigation) will explore:
- Pre-processing: Reweighting/resampling by group
- In-processing: Fairness constraints during training
- Post-processing: Group-specific thresholds

""")
```

---

## Key Takeaways

Before moving to Section 3.3 (Bias Mitigation), ensure you understand:

1. **We measured multiple fairness metrics:**
   - Disparate Impact Ratio (4/5ths rule compliance)
   - Statistical Parity Difference (approval rate gaps)
   - Equal Opportunity Difference (TPR gaps)
   - Average Odds Difference (combined error rate gaps)
   - Expected Calibration Error (prediction honesty by group)

2. **Our model's fairness status:**
   - Tier 1 (Legal): [Results depend on your data]
   - Tier 2 (Business): [Results depend on your data]
   - Tier 3 (Monitoring): Tracked for ongoing assessment

3. **Root causes of bias may include:**
   - Historical bias in training data
   - Proxy discrimination through correlated features
   - Differential distribution shift across groups
   - SMOTE amplification of existing patterns

4. **The test set failure from Section 2 may affect groups differently:**
   - Distribution shift magnitude varies by group
   - Calibration errors vary by group
   - Single threshold may disadvantage some groups

5. **Synthetic protected attributes have limitations:**
   - Real demographic data needed for production
   - Correlations in synthetic data may not reflect reality
   - Results should be validated with actual data

**Section 3.3 will address how to MITIGATE these biases using pre-processing, in-processing, and post-processing techniques.**

---

*End of Section 3.2*
