# Appendix C: Code Reference

---

## C.1 Installation Requirements

### Core Dependencies

```bash
# Create virtual environment (recommended)
python -m venv fairness_env
source fairness_env/bin/activate  # Linux/Mac
# or: fairness_env\Scripts\activate  # Windows

# Install core packages
pip install pandas numpy scikit-learn matplotlib seaborn

# Install ML packages
pip install xgboost imbalanced-learn shap

# Install data generation
pip install faker

# Install calibration (included in sklearn)
# No additional install needed
```

### requirements.txt

```
# Core
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0

# ML
xgboost>=1.7.0
imbalanced-learn>=0.10.0
shap>=0.41.0

# Data Generation
faker>=18.0.0

# Optional: Jupyter
jupyter>=1.0.0
ipykernel>=6.0.0
```

### Version Check Script

```python
import pandas as pd
import numpy as np
import sklearn
import xgboost
import shap
import imblearn

print("Package Versions:")
print(f"  pandas:          {pd.__version__}")
print(f"  numpy:           {np.__version__}")
print(f"  scikit-learn:    {sklearn.__version__}")
print(f"  xgboost:         {xgboost.__version__}")
print(f"  shap:            {shap.__version__}")
print(f"  imbalanced-learn: {imblearn.__version__}")
```

---

## C.2 Data Quality Functions (Chapter 2)

### DataQualityLogger

```python
import pandas as pd
from datetime import datetime

class DataQualityLogger:
    """
    Tracks all data quality issues and transformations for audit trail.
    
    Usage:
        logger = DataQualityLogger()
        logger.log_issue('accounts', 'account_id', 'missing', 41, 'drop', 'PK cannot be null')
        report = logger.get_report()
    """
    
    def __init__(self):
        self.issues = []
        
    def log_issue(self, table, column, issue_type, count, action, reason):
        """
        Log a data quality issue and the action taken.
        
        Args:
            table: Name of the table
            column: Column name where issue was found
            issue_type: Type of issue (missing_value, duplicate, etc.)
            count: Number of rows affected
            action: Action taken (drop_record, impute, set_to_null, etc.)
            reason: Business justification for the action
        """
        self.issues.append({
            'timestamp': datetime.now(),
            'table': table,
            'column': column,
            'issue_type': issue_type,
            'rows_affected': count,
            'action_taken': action,
            'reason': reason
        })
        
    def get_report(self):
        """Return DataFrame with all logged issues."""
        return pd.DataFrame(self.issues)
    
    def summary(self):
        """Print summary of issues by type."""
        if not self.issues:
            print("No issues logged.")
            return
        
        df = self.get_report()
        print(f"Total issues: {len(df)}")
        print(f"Total rows affected: {df['rows_affected'].sum():,}")
        print("\nBy issue type:")
        print(df.groupby('issue_type')['rows_affected'].sum())
```

### Schema Validation

```python
import pandas as pd
import numpy as np

def validate_and_coerce_schema(df, schema, table_name, logger=None):
    """
    Validate and coerce data types according to schema.
    
    Args:
        df: Input DataFrame
        schema: Dict mapping column names to expected dtypes
        table_name: Name of table (for logging)
        logger: Optional DataQualityLogger instance
    
    Returns:
        DataFrame with corrected types
    
    Example:
        schema = {'account_id': 'string', 'open_date': 'datetime64[ns]'}
        df_clean = validate_and_coerce_schema(df, schema, 'accounts', logger)
    """
    df_clean = df.copy()
    
    for col, expected_dtype in schema.items():
        if col not in df_clean.columns:
            continue
            
        if expected_dtype == 'datetime64[ns]':
            original_nulls = df_clean[col].isnull().sum()
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
            new_nulls = df_clean[col].isnull().sum()
            failed = new_nulls - original_nulls
            
            if failed > 0 and logger:
                logger.log_issue(table_name, col, 'type_conversion_failure',
                                failed, 'set_to_null', 'Could not parse as datetime')
                
        elif expected_dtype in ['float64', 'int64']:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
        elif expected_dtype == 'string':
            df_clean[col] = df_clean[col].astype(str)
            df_clean[col] = df_clean[col].replace(['nan', 'NaN'], np.nan)
    
    return df_clean
```

### Missing Data Handlers

```python
def handle_missing_accounts(df, logger):
    """
    Handle missing values in accounts table with column-specific strategies.
    
    Strategy by column:
        - account_id: DROP (primary key, cannot be null)
        - customer_id: DROP (required for identification)
        - account_type: IMPUTE from credit_limit
        - open_date: DROP (required for temporal analysis)
        - credit_limit: KEEP NULL (NULL = not applicable)
        - status: IMPUTE as 'active' (most common)
        - branch_code: KEEP NULL (NULL = online-only account)
    """
    print(f"\n[Layer 2: Missing Data] Processing accounts...")
    df_clean = df.copy()
    initial_count = len(df_clean)
    
    # DROP: Missing account_id (primary key)
    missing_id = df_clean['account_id'].isnull()
    if missing_id.any():
        count = missing_id.sum()
        df_clean = df_clean[~missing_id]
        logger.log_issue('accounts', 'account_id', 'missing_value', count, 
                        'drop_record', 'Primary key cannot be null')
        print(f"  Dropped {count} records with missing account_id")
    
    # DROP: Missing customer_id (required for identification)
    missing_cust = df_clean['customer_id'].isnull()
    if missing_cust.any():
        count = missing_cust.sum()
        df_clean = df_clean[~missing_cust]
        logger.log_issue('accounts', 'customer_id', 'missing_value', count,
                        'drop_record', 'Required for identification')
        print(f"  Dropped {count} records with missing customer_id")
    
    # IMPUTE: account_type from credit_limit
    missing_type = df_clean['account_type'].isnull()
    if missing_type.any():
        count = missing_type.sum()
        # If credit_limit exists and > 0, likely credit_card; else checking
        df_clean.loc[missing_type & (df_clean['credit_limit'] > 0), 'account_type'] = 'credit_card'
        df_clean.loc[missing_type & (df_clean['credit_limit'].isnull() | (df_clean['credit_limit'] == 0)), 'account_type'] = 'checking'
        logger.log_issue('accounts', 'account_type', 'missing_value', count,
                        'impute', 'Inferred from credit_limit')
        print(f"  Imputed {count} missing account_type values")
    
    # DROP: Missing open_date (required for temporal analysis)
    missing_date = df_clean['open_date'].isnull()
    if missing_date.any():
        count = missing_date.sum()
        df_clean = df_clean[~missing_date]
        logger.log_issue('accounts', 'open_date', 'missing_value', count,
                        'drop_record', 'Required for temporal analysis')
        print(f"  Dropped {count} records with missing open_date")
    
    # IMPUTE: status as 'active' (most common value)
    missing_status = df_clean['status'].isnull()
    if missing_status.any():
        count = missing_status.sum()
        df_clean.loc[missing_status, 'status'] = 'active'
        logger.log_issue('accounts', 'status', 'missing_value', count,
                        'impute', 'Default to most common value: active')
    
    # KEEP NULL: credit_limit (NULL = not applicable for checking/savings)
    # KEEP NULL: branch_code (NULL = online-only account)
    
    final_count = len(df_clean)
    print(f"  ✓ Accounts: {initial_count} → {final_count} rows")
    
    return df_clean


def handle_missing_transactions(df, logger):
    """
    Handle missing values in transactions table.
    
    Strategy by column:
        - transaction_id: DROP (primary key)
        - account_id: DROP (required for linking)
        - transaction_date: DROP (required for temporal analysis)
        - amount: DROP (core field)
        - transaction_type: IMPUTE as 'other'
        - merchant: KEEP NULL (ATM/transfer = no merchant)
        - channel: IMPUTE as 'unknown'
    """
    print(f"\n[Layer 2: Missing Data] Processing transactions...")
    df_clean = df.copy()
    initial_count = len(df_clean)
    
    # DROP: Missing IDs (transaction_id or account_id)
    missing_ids = df_clean['transaction_id'].isnull() | df_clean['account_id'].isnull()
    if missing_ids.any():
        count = missing_ids.sum()
        df_clean = df_clean[~missing_ids]
        logger.log_issue('transactions', 'id_fields', 'missing_value', count,
                        'drop_record', 'Primary/foreign key cannot be null')
        print(f"  Dropped {count} records with missing IDs")
    
    # DROP: Missing transaction_date
    missing_date = df_clean['transaction_date'].isnull()
    if missing_date.any():
        count = missing_date.sum()
        df_clean = df_clean[~missing_date]
        logger.log_issue('transactions', 'transaction_date', 'missing_value', count,
                        'drop_record', 'Required for temporal analysis')
        print(f"  Dropped {count} records with missing transaction_date")
    
    # IMPUTE: transaction_type
    missing_type = df_clean['transaction_type'].isnull()
    if missing_type.any():
        count = missing_type.sum()
        df_clean.loc[missing_type, 'transaction_type'] = 'other'
        logger.log_issue('transactions', 'transaction_type', 'missing_value', count,
                        'impute', 'Default to other')
    
    # KEEP NULL: merchant (legitimate for ATM withdrawals, transfers)
    
    final_count = len(df_clean)
    print(f"  ✓ Transactions: {initial_count} → {final_count} rows")
    
    return df_clean


def handle_missing_balances(df, logger):
    """
    Handle missing values in balances table.
    
    Strategy by column:
        - account_id: DROP (required for linking)
        - balance_date: DROP (required for temporal analysis)
        - available_balance: DROP (core field)
        - ledger_balance: DROP (core field)
        - overdraft_count: IMPUTE as 0
    """
    print(f"\n[Layer 2: Missing Data] Processing balances...")
    df_clean = df.copy()
    initial_count = len(df_clean)
    
    # DROP: Missing account_id or balance_date
    missing_critical = (df_clean['account_id'].isnull() | 
                       df_clean['balance_date'].isnull() |
                       df_clean['available_balance'].isnull() |
                       df_clean['ledger_balance'].isnull())
    if missing_critical.any():
        count = missing_critical.sum()
        df_clean = df_clean[~missing_critical]
        logger.log_issue('balances', 'critical_fields', 'missing_value', count,
                        'drop_record', 'Missing ID, date, or balance fields')
        print(f"  Dropped {count} records with missing IDs or dates")
    
    # IMPUTE: overdraft_count as 0
    missing_od = df_clean['overdraft_count'].isnull()
    if missing_od.any():
        count = missing_od.sum()
        df_clean.loc[missing_od, 'overdraft_count'] = 0
        logger.log_issue('balances', 'overdraft_count', 'missing_value', count,
                        'impute', 'Default to 0')
    
    final_count = len(df_clean)
    print(f"  ✓ Balances: {initial_count} → {final_count} rows")
    
    return df_clean
```

### Deduplication & Consistency

```python
def deduplicate_and_standardize_accounts(df, logger):
    """Remove duplicates and standardize text in accounts table."""
    print(f"\n[Layer 3: Deduplication & Consistency] Processing accounts...")
    df_clean = df.copy()
    
    # Remove duplicate account_ids (keep first occurrence)
    duplicates = df_clean['account_id'].duplicated()
    if duplicates.any():
        count = duplicates.sum()
        df_clean = df_clean[~duplicates]
        logger.log_issue('accounts', 'account_id', 'duplicate', count,
                        'drop_record', 'Kept first occurrence of duplicate')
        print(f"  Removed {count} duplicate account_ids")
    
    # Standardize account_type (fix typos and casing)
    type_mapping = {
        'chekcing': 'checking',
        'CHECKING': 'checking',
        'SAVINGS': 'savings',
        'Checking': 'checking',
        'Savings': 'savings'
    }
    for bad_val, good_val in type_mapping.items():
        mask = df_clean['account_type'] == bad_val
        if mask.any():
            df_clean.loc[mask, 'account_type'] = good_val
            logger.log_issue('accounts', 'account_type', 'typo', mask.sum(),
                            'standardize', f'Fixed: {bad_val} → {good_val}')
    
    # Standardize status
    status_mapping = {'ACTIVE': 'active', 'CLOSED': 'closed', 'Active': 'active'}
    for bad_val, good_val in status_mapping.items():
        mask = df_clean['status'] == bad_val
        if mask.any():
            df_clean.loc[mask, 'status'] = good_val
            logger.log_issue('accounts', 'status', 'inconsistent_case', mask.sum(),
                            'standardize', f'Fixed: {bad_val} → {good_val}')
    
    return df_clean


def deduplicate_and_standardize_transactions(df, logger):
    """Remove duplicates and standardize text in transactions table."""
    print(f"\n[Layer 3: Deduplication & Consistency] Processing transactions...")
    df_clean = df.copy()
    
    # Remove duplicate transaction_ids
    duplicates = df_clean['transaction_id'].duplicated()
    if duplicates.any():
        count = duplicates.sum()
        df_clean = df_clean[~duplicates]
        logger.log_issue('transactions', 'transaction_id', 'duplicate', count,
                        'drop_record', 'Kept first occurrence of duplicate')
        print(f"  Removed {count} duplicate transaction_ids")
    
    # Standardize transaction_type
    type_mapping = {
        'DEBIT': 'debit', 'CREDIT': 'credit',
        'Debit': 'debit', 'Credit': 'credit'
    }
    for bad_val, good_val in type_mapping.items():
        mask = df_clean['transaction_type'] == bad_val
        if mask.any():
            df_clean.loc[mask, 'transaction_type'] = good_val
            logger.log_issue('transactions', 'transaction_type', 'inconsistent_case', 
                            mask.sum(), 'standardize', f'Fixed: {bad_val} → {good_val}')
    
    return df_clean


def deduplicate_and_standardize_balances(df, logger):
    """Remove duplicates and enforce business rules in balances table."""
    print(f"\n[Layer 3: Deduplication & Consistency] Processing balances...")
    df_clean = df.copy()
    
    # Remove exact duplicate rows
    duplicates = df_clean.duplicated(subset=['account_id', 'balance_date'])
    if duplicates.any():
        count = duplicates.sum()
        df_clean = df_clean[~duplicates]
        logger.log_issue('balances', 'account_id+balance_date', 'duplicate', count,
                        'drop_record', 'Removed duplicate balance snapshots')
        print(f"  Removed {count} duplicate balance snapshots")
    
    # Business rule: available_balance cannot exceed ledger_balance
    violation = df_clean['available_balance'] > df_clean['ledger_balance']
    if violation.any():
        count = violation.sum()
        # Fix by capping available_balance at ledger_balance
        df_clean.loc[violation, 'available_balance'] = df_clean.loc[violation, 'ledger_balance']
        logger.log_issue('balances', 'available_balance', 'business_rule_violation', count,
                        'correct', 'Capped available_balance at ledger_balance')
        print(f"  Fixed {count} records where available_balance > ledger_balance")
    
    return df_clean
```

---

## C.3 Fairness Metrics Functions (Chapter 4)

### Disparate Impact Ratio

```python
import numpy as np

def disparate_impact_ratio(y_pred, protected_attr, unprivileged_value, privileged_value):
    """
    Calculate Disparate Impact Ratio.
    
    DIR = P(approve | unprivileged) / P(approve | privileged)
    
    Args:
        y_pred: Predicted labels (1 = default/deny, 0 = approve)
        protected_attr: Array of protected attribute values
        unprivileged_value: Value identifying unprivileged group
        privileged_value: Value identifying privileged group
    
    Returns:
        float: DIR (1.0 = perfect parity, ≥0.80 passes 4/5ths rule)
    """
    unpriv_mask = (protected_attr == unprivileged_value)
    priv_mask = (protected_attr == privileged_value)
    
    unpriv_approval = (y_pred[unpriv_mask] == 0).mean()
    priv_approval = (y_pred[priv_mask] == 0).mean()
    
    if priv_approval == 0:
        return np.nan
    
    return unpriv_approval / priv_approval
```

### Statistical Parity Difference

```python
def statistical_parity_difference(y_pred, protected_attr, unprivileged_value, privileged_value):
    """
    Calculate Statistical Parity Difference.
    
    SPD = P(approve | unprivileged) - P(approve | privileged)
    
    Returns:
        float: SPD (0.0 = perfect parity)
    """
    unpriv_mask = (protected_attr == unprivileged_value)
    priv_mask = (protected_attr == privileged_value)
    
    unpriv_approval = (y_pred[unpriv_mask] == 0).mean()
    priv_approval = (y_pred[priv_mask] == 0).mean()
    
    return unpriv_approval - priv_approval
```

### Equal Opportunity Difference

```python
from sklearn.metrics import recall_score

def equal_opportunity_difference(y_true, y_pred, protected_attr, unprivileged_value, privileged_value):
    """
    Calculate Equal Opportunity Difference (TPR gap).
    
    EOD = TPR(unprivileged) - TPR(privileged)
    
    Returns:
        float: EOD (0.0 = equal opportunity)
    """
    unpriv_mask = (protected_attr == unprivileged_value)
    priv_mask = (protected_attr == privileged_value)
    
    # Check for sufficient positive samples
    if (y_true[unpriv_mask] == 1).sum() == 0 or (y_true[priv_mask] == 1).sum() == 0:
        return np.nan
    
    tpr_unpriv = recall_score(y_true[unpriv_mask], y_pred[unpriv_mask], zero_division=0)
    tpr_priv = recall_score(y_true[priv_mask], y_pred[priv_mask], zero_division=0)
    
    return tpr_unpriv - tpr_priv
```

### Average Odds Difference

```python
def average_odds_difference(y_true, y_pred, protected_attr, unprivileged_value, privileged_value):
    """
    Calculate Average Odds Difference.
    
    AOD = 0.5 * [(FPR_unpriv - FPR_priv) + (TPR_unpriv - TPR_priv)]
    
    Returns:
        float: AOD (0.0 = equalized odds)
    """
    unpriv_mask = (protected_attr == unprivileged_value)
    priv_mask = (protected_attr == privileged_value)
    
    # TPR
    tpr_unpriv = recall_score(y_true[unpriv_mask], y_pred[unpriv_mask], zero_division=0)
    tpr_priv = recall_score(y_true[priv_mask], y_pred[priv_mask], zero_division=0)
    
    # FPR
    neg_unpriv = (y_true[unpriv_mask] == 0).sum()
    neg_priv = (y_true[priv_mask] == 0).sum()
    
    if neg_unpriv == 0 or neg_priv == 0:
        return np.nan
    
    fpr_unpriv = ((y_pred[unpriv_mask] == 1) & (y_true[unpriv_mask] == 0)).sum() / neg_unpriv
    fpr_priv = ((y_pred[priv_mask] == 1) & (y_true[priv_mask] == 0)).sum() / neg_priv
    
    return 0.5 * ((fpr_unpriv - fpr_priv) + (tpr_unpriv - tpr_priv))
```

### Expected Calibration Error

```python
def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculate Expected Calibration Error.
    
    ECE = Σ |bin_accuracy - bin_confidence| × bin_weight
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins (default 10)
    
    Returns:
        float: ECE (0.0 = perfect calibration)
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

### Complete Fairness Report

```python
def generate_fairness_metrics(y_true, y_pred, y_prob, protected_attr, 
                               unprivileged_value, privileged_value):
    """
    Generate all fairness metrics for a single protected attribute comparison.
    
    Returns:
        dict: All metrics
    """
    return {
        'disparate_impact_ratio': disparate_impact_ratio(
            y_pred, protected_attr, unprivileged_value, privileged_value),
        'statistical_parity_difference': statistical_parity_difference(
            y_pred, protected_attr, unprivileged_value, privileged_value),
        'equal_opportunity_difference': equal_opportunity_difference(
            y_true, y_pred, protected_attr, unprivileged_value, privileged_value),
        'average_odds_difference': average_odds_difference(
            y_true, y_pred, protected_attr, unprivileged_value, privileged_value),
        'ece_unprivileged': expected_calibration_error(
            y_true[protected_attr == unprivileged_value], 
            y_prob[protected_attr == unprivileged_value]),
        'ece_privileged': expected_calibration_error(
            y_true[protected_attr == privileged_value], 
            y_prob[protected_attr == privileged_value])
    }
```

---

## C.4 Bias Mitigation Functions (Chapter 4)

### Reweighting

```python
def calculate_reweighting_weights(y_true, protected_attr, privileged_value):
    """
    Calculate sample weights for fairness-aware training.
    
    Weight = P(Y) × P(A) / P(Y, A)
    
    Args:
        y_true: True labels
        protected_attr: Protected attribute values
        privileged_value: Value identifying privileged group
    
    Returns:
        np.array: Sample weights
    """
    n = len(y_true)
    weights = np.ones(n)
    
    priv_mask = (protected_attr == privileged_value)
    unpriv_mask = ~priv_mask
    
    n_priv = priv_mask.sum()
    n_unpriv = unpriv_mask.sum()
    
    p_pos = y_true.mean()
    p_neg = 1 - p_pos
    p_priv = n_priv / n
    p_unpriv = n_unpriv / n
    
    # Joint probabilities
    p_pos_priv = ((y_true == 1) & priv_mask).sum() / n
    p_neg_priv = ((y_true == 0) & priv_mask).sum() / n
    p_pos_unpriv = ((y_true == 1) & unpriv_mask).sum() / n
    p_neg_unpriv = ((y_true == 0) & unpriv_mask).sum() / n
    
    # Calculate weights
    if p_pos_priv > 0:
        weights[priv_mask & (y_true == 1)] = (p_pos * p_priv) / p_pos_priv
    if p_neg_priv > 0:
        weights[priv_mask & (y_true == 0)] = (p_neg * p_priv) / p_neg_priv
    if p_pos_unpriv > 0:
        weights[unpriv_mask & (y_true == 1)] = (p_pos * p_unpriv) / p_pos_unpriv
    if p_neg_unpriv > 0:
        weights[unpriv_mask & (y_true == 0)] = (p_neg * p_unpriv) / p_neg_unpriv
    
    return weights
```

### Threshold Optimization

```python
def find_optimal_threshold(y_true, y_prob, metric='f1'):
    """
    Find optimal decision threshold for given metric.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        metric: 'f1', 'precision', 'recall', or 'balanced_accuracy'
    
    Returns:
        tuple: (optimal_threshold, metric_value)
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score
    
    metric_funcs = {
        'f1': f1_score,
        'precision': precision_score,
        'recall': recall_score,
        'balanced_accuracy': balanced_accuracy_score
    }
    
    best_threshold = 0.5
    best_score = 0
    
    for thresh in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_prob >= thresh).astype(int)
        score = metric_funcs[metric](y_true, y_pred, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold, best_score
```

### Calibration

```python
from sklearn.calibration import CalibratedClassifierCV

def calibrate_model(model, X_cal, y_cal, method='isotonic'):
    """
    Calibrate a trained model's probability estimates.
    
    Args:
        model: Trained classifier
        X_cal: Calibration features
        y_cal: Calibration labels
        method: 'isotonic' or 'sigmoid'
    
    Returns:
        CalibratedClassifierCV: Calibrated model
    """
    calibrated = CalibratedClassifierCV(model, method=method, cv='prefit')
    calibrated.fit(X_cal, y_cal)
    return calibrated
```

---

## C.5 SHAP Explanation Functions (Chapter 3)

### Generate SHAP Values

```python
import shap

def get_shap_explanations(model, X, sample_size=100):
    """
    Generate SHAP values for model explanations.
    
    Args:
        model: Trained model (tree-based)
        X: Feature DataFrame
        sample_size: Number of samples to explain (for speed)
    
    Returns:
        tuple: (shap_values, explainer)
    """
    explainer = shap.TreeExplainer(model)
    
    if len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=42)
    else:
        X_sample = X
    
    shap_values = explainer.shap_values(X_sample)
    
    return shap_values, explainer, X_sample
```

### Adverse Action Notice

```python
def generate_adverse_action_reasons(shap_values, feature_names, applicant_idx, top_n=4):
    """
    Generate top reasons for denial based on SHAP values.
    
    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
        applicant_idx: Index of applicant
        top_n: Number of reasons to return
    
    Returns:
        list: Top contributing factors
    """
    applicant_shap = shap_values[applicant_idx]
    
    # Get features that increased default probability (positive SHAP)
    feature_contributions = list(zip(feature_names, applicant_shap))
    feature_contributions.sort(key=lambda x: x[1], reverse=True)
    
    top_factors = feature_contributions[:top_n]
    
    reasons = []
    for feature, shap_val in top_factors:
        if shap_val > 0:  # Only include factors that increased risk
            reasons.append({
                'factor': feature,
                'contribution': shap_val
            })
    
    return reasons
```

---

## C.6 Monitoring Functions (Chapter 4)

### Fairness Snapshot

```python
from datetime import datetime
import json

def create_fairness_snapshot(y_true, y_pred, y_prob, protected_attr, 
                              threshold, dataset_name, privileged_value='White'):
    """
    Create a snapshot of fairness metrics for monitoring.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        protected_attr: Protected attribute values
        threshold: Decision threshold used
        dataset_name: Name for this snapshot
        privileged_value: Reference group
    
    Returns:
        dict: Snapshot with all metrics
    """
    snapshot = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'threshold': threshold,
        'n_samples': len(y_true),
        'overall_approval_rate': float((y_pred == 0).mean()),
        'overall_default_rate': float(y_true.mean()),
        'groups': {}
    }
    
    for group in np.unique(protected_attr):
        mask = (protected_attr == group)
        if mask.sum() > 0:
            snapshot['groups'][str(group)] = {
                'n_samples': int(mask.sum()),
                'approval_rate': float((y_pred[mask] == 0).mean()),
                'default_rate': float(y_true[mask].mean()),
                'avg_probability': float(y_prob[mask].mean())
            }
    
    # Calculate DIR for each non-privileged group
    snapshot['fairness_metrics'] = {}
    for group in np.unique(protected_attr):
        if group != privileged_value:
            dir_val = disparate_impact_ratio(y_pred, protected_attr, group, privileged_value)
            snapshot['fairness_metrics'][f'DIR_{group}_vs_{privileged_value}'] = float(dir_val) if not np.isnan(dir_val) else None
    
    return snapshot


def save_snapshot(snapshot, filepath):
    """Save snapshot to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(snapshot, f, indent=2)


def compare_snapshots(snapshot1, snapshot2):
    """
    Compare two snapshots and identify drift.
    
    Returns:
        dict: Comparison results with alerts
    """
    comparison = {
        'snapshot1': snapshot1['dataset'],
        'snapshot2': snapshot2['dataset'],
        'alerts': []
    }
    
    # Compare DIR
    for metric_key in snapshot1.get('fairness_metrics', {}):
        val1 = snapshot1['fairness_metrics'].get(metric_key)
        val2 = snapshot2['fairness_metrics'].get(metric_key)
        
        if val1 is not None and val2 is not None:
            change = val2 - val1
            comparison[metric_key] = {
                'before': val1,
                'after': val2,
                'change': change
            }
            
            if val2 < 0.80:
                comparison['alerts'].append(f"⚠️ {metric_key} below 0.80 threshold")
            elif abs(change) > 0.05:
                comparison['alerts'].append(f"📊 {metric_key} changed by {change:+.3f}")
    
    return comparison
```

---

## C.7 Complete Example Script

```python
"""
Complete Credit Fairness Pipeline
=================================
Run this script to execute the full pipeline from data to fairness evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
print("Loading data...")
data = pd.read_csv('your_credit_data.csv')

# 2. Prepare Features and Target
feature_cols = [col for col in data.columns if col.startswith('feat_')]
X = data[feature_cols]
y = data['defaulted']
protected = data['race']  # or whatever protected attribute

# 3. Split Data
X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(
    X, y, protected, test_size=0.2, random_state=42, stratify=y
)

# 4. Handle Missing Values
imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols)
X_test = pd.DataFrame(imputer.transform(X_test), columns=feature_cols)

# 5. Calculate Reweighting (optional)
weights = calculate_reweighting_weights(y_train.values, prot_train.values, 'White')

# 6. Train Model
print("Training model...")
model = XGBClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train, sample_weight=weights)

# 7. Get Predictions
y_prob = model.predict_proba(X_test)[:, 1]
threshold, _ = find_optimal_threshold(y_test, y_prob, metric='f1')
y_pred = (y_prob >= threshold).astype(int)

# 8. Calculate Fairness Metrics
print("\nFairness Metrics:")
metrics = generate_fairness_metrics(
    y_test.values, y_pred, y_prob, prot_test.values, 'Black', 'White'
)
for metric, value in metrics.items():
    print(f"  {metric}: {value:.3f}" if value else f"  {metric}: N/A")

# 9. Check Compliance
dir_val = metrics['disparate_impact_ratio']
if dir_val and dir_val >= 0.80:
    print("\n✓ Model passes 4/5ths rule")
else:
    print("\n✗ Model fails 4/5ths rule - mitigation needed")

# 10. Create Monitoring Snapshot
snapshot = create_fairness_snapshot(
    y_test.values, y_pred, y_prob, prot_test.values, 
    threshold, 'production_v1'
)
save_snapshot(snapshot, 'fairness_snapshot.json')
print("\n✓ Fairness snapshot saved")
```

---

*End of Appendix C*
