"""
Complete Data Cleaning Pipeline - Chapter 2 Solution
=====================================================

This is the complete, working implementation of the data cleaning pipeline
from Chapter 2, Section 2 of "Code, Cash, and Conviction: Building Ethical
Fintech Systems for Industry and the Classroom."

Purpose:
    - Reference implementation for students
    - Can be run as-is to clean the synthetic banking data
    - Demonstrates all four layers of the cleaning pipeline

Usage:
    python cleaning_pipeline_complete.py
    
    OR in Jupyter:
    from generate_banking_data import BankingDataGenerator
    
Requirements:
    - pandas
    - numpy
    - matplotlib
    - Synthetic banking data in: synthetic_banking_data/raw/

Author: [Your Name]
Course: Master of FinTech & Analytics
Chapter: 2 - Data Foundations
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DATA QUALITY LOGGER
# ============================================================================

class DataQualityLogger:
    """
    Tracks all data quality issues and transformations for audit trail.
    
    Every cleaning operation should be logged to maintain complete lineage.
    """
    
    def __init__(self):
        self.issues = []
        
    def log_issue(self, table, column, issue_type, count, action, reason):
        """
        Log a data quality issue and the action taken.
        
        Args:
            table: Name of the table (accounts, transactions, balances)
            column: Column name where issue was found
            issue_type: Type of issue (missing_value, duplicate, etc.)
            count: Number of rows affected
            action: Action taken (drop_record, impute, flag, etc.)
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
    
    def save_report(self, filepath):
        """Save lineage report to CSV."""
        report = self.get_report()
        report.to_csv(filepath, index=False)
        print(f"✓ Lineage report saved to {filepath}")


# ============================================================================
# LAYER 1: SCHEMA VALIDATION & TYPE COERCION
# ============================================================================

# Define expected schemas
ACCOUNT_SCHEMA = {
    'account_id': 'string',
    'customer_id': 'string', 
    'account_type': 'string',
    'open_date': 'datetime64[ns]',
    'credit_limit': 'float64',
    'status': 'string',
    'branch_code': 'string'
}

TRANSACTION_SCHEMA = {
    'transaction_id': 'string',
    'account_id': 'string',
    'transaction_date': 'datetime64[ns]',
    'transaction_time': 'string',
    'amount': 'float64',
    'transaction_type': 'string',
    'merchant_category': 'string',
    'merchant_name': 'string',
    'channel': 'string'
}

BALANCE_SCHEMA = {
    'account_id': 'string',
    'balance_date': 'datetime64[ns]',
    'available_balance': 'float64',
    'ledger_balance': 'float64',
    'overdraft_count': 'int64'
}


def validate_and_coerce_schema(df, schema, table_name, logger):
    """
    Validate and coerce data types according to schema.
    
    Args:
        df: Input dataframe
        schema: Dictionary mapping column names to expected dtypes
        table_name: Name of the table for logging
        logger: DataQualityLogger instance
    
    Returns:
        DataFrame with corrected types
    """
    print(f"\n[Layer 1: Schema Validation] Processing {table_name}...")
    df_clean = df.copy()
    
    for col, expected_dtype in schema.items():
        if col not in df_clean.columns:
            print(f"  ⚠️  Column '{col}' missing from {table_name}")
            continue
            
        current_dtype = str(df_clean[col].dtype)
        
        # Skip if already correct type
        if current_dtype == expected_dtype:
            print(f"  ✓ {col}: Already {expected_dtype}")
            continue
        
        print(f"  → {col}: Converting {current_dtype} → {expected_dtype}...", end=' ')
        
        # Handle datetime conversion
        if expected_dtype == 'datetime64[ns]':
            # Replace string 'NaN' with actual NaN
            df_clean[col] = df_clean[col].replace('NaN', np.nan)
            
            # Convert to datetime, coercing errors to NaT
            original_nulls = df_clean[col].isnull().sum()
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
            new_nulls = df_clean[col].isnull().sum()
            failed_conversions = new_nulls - original_nulls
            
            if failed_conversions > 0:
                logger.log_issue(
                    table=table_name,
                    column=col,
                    issue_type='type_conversion_failure',
                    count=failed_conversions,
                    action='set_to_null',
                    reason=f'Could not parse as datetime, set to NaT'
                )
                print(f"{failed_conversions} invalid dates → NaT")
            else:
                print("Success (no failures)")
        
        # Handle numeric conversion
        elif expected_dtype in ['float64', 'int64']:
            df_clean[col] = df_clean[col].replace('NaN', np.nan)
            original_nulls = df_clean[col].isnull().sum()
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            new_nulls = df_clean[col].isnull().sum()
            failed_conversions = new_nulls - original_nulls
            
            if failed_conversions > 0:
                logger.log_issue(
                    table=table_name,
                    column=col,
                    issue_type='type_conversion_failure',
                    count=failed_conversions,
                    action='set_to_null',
                    reason=f'Could not parse as {expected_dtype}, set to NaN'
                )
                print(f"{failed_conversions} invalid numbers → NaN")
            else:
                print("Success (no failures)")
            
            # Convert to int if specified (after handling nulls)
            if expected_dtype == 'int64' and df_clean[col].notna().any():
                df_clean[col] = df_clean[col].astype('Int64')  # Nullable integer
        
        # Handle string conversion
        elif expected_dtype == 'string':
            df_clean[col] = df_clean[col].astype(str)
            df_clean[col] = df_clean[col].replace('nan', np.nan)
            df_clean[col] = df_clean[col].replace('NaN', np.nan)
            print("Success")
    
    print(f"  ✓ Schema validation complete for {table_name}")
    return df_clean


# ============================================================================
# LAYER 2: HANDLING MISSING DATA
# ============================================================================

def handle_missing_accounts(df, logger):
    """
    Handle missing values in accounts table with column-specific strategies.
    
    Strategy by column:
        - account_id: DROP (primary key, cannot be null)
        - customer_id: DROP (required for identification)
        - account_type: IMPUTE from credit_limit (has type → is credit/loan)
        - open_date: DROP (required for temporal analysis)
        - credit_limit: KEEP NULL (0 means no credit, NULL means not applicable)
        - status: IMPUTE as 'active' (most common, safe assumption)
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
    
    # DROP: Missing customer_id
    missing_cust = df_clean['customer_id'].isnull()
    if missing_cust.any():
        count = missing_cust.sum()
        df_clean = df_clean[~missing_cust]
        logger.log_issue('accounts', 'customer_id', 'missing_value', count,
                        'drop_record', 'Customer ID required for account identification')
        print(f"  Dropped {count} records with missing customer_id")
    
    # IMPUTE: Missing account_type (infer from credit_limit)
    missing_type = df_clean['account_type'].isnull()
    if missing_type.any():
        count = missing_type.sum()
        # If has credit_limit > 0, assume credit_card, else checking
        df_clean.loc[missing_type & (df_clean['credit_limit'] > 0), 'account_type'] = 'credit_card'
        df_clean.loc[missing_type & (df_clean['credit_limit'].isnull() | (df_clean['credit_limit'] == 0)), 
                     'account_type'] = 'checking'
        logger.log_issue('accounts', 'account_type', 'missing_value', count,
                        'impute', 'Inferred from credit_limit: >0 = credit_card, else checking')
        print(f"  Imputed {count} missing account_type values")
    
    # DROP: Missing open_date
    missing_date = df_clean['open_date'].isnull()
    if missing_date.any():
        count = missing_date.sum()
        df_clean = df_clean[~missing_date]
        logger.log_issue('accounts', 'open_date', 'missing_value', count,
                        'drop_record', 'Open date required for temporal analysis')
        print(f"  Dropped {count} records with missing open_date")
    
    # IMPUTE: Missing status
    missing_status = df_clean['status'].isnull()
    if missing_status.any():
        count = missing_status.sum()
        df_clean.loc[missing_status, 'status'] = 'active'
        logger.log_issue('accounts', 'status', 'missing_value', count,
                        'impute', 'Assumed active (most common status)')
        print(f"  Imputed {count} missing status values as 'active'")
    
    # KEEP NULL: credit_limit and branch_code
    # These have valid business meaning when null
    
    final_count = len(df_clean)
    print(f"  ✓ Accounts: {initial_count} → {final_count} rows ({initial_count - final_count} dropped)")
    
    return df_clean


def handle_missing_transactions(df, logger):
    """Handle missing values in transactions table."""
    print(f"\n[Layer 2: Missing Data] Processing transactions...")
    df_clean = df.copy()
    initial_count = len(df_clean)
    
    # DROP: Missing transaction_id or account_id
    missing_id = df_clean['transaction_id'].isnull() | df_clean['account_id'].isnull()
    if missing_id.any():
        count = missing_id.sum()
        df_clean = df_clean[~missing_id]
        logger.log_issue('transactions', 'transaction_id/account_id', 'missing_value', count,
                        'drop_record', 'Cannot identify transaction without IDs')
        print(f"  Dropped {count} records with missing IDs")
    
    # DROP: Missing transaction_date
    missing_date = df_clean['transaction_date'].isnull()
    if missing_date.any():
        count = missing_date.sum()
        df_clean = df_clean[~missing_date]
        logger.log_issue('transactions', 'transaction_date', 'missing_value', count,
                        'drop_record', 'Transaction date required for temporal analysis')
        print(f"  Dropped {count} records with missing transaction_date")
    
    # IMPUTE: Missing transaction_time
    missing_time = df_clean['transaction_time'].isnull()
    if missing_time.any():
        count = missing_time.sum()
        df_clean.loc[missing_time, 'transaction_time'] = '00:00:00'
        logger.log_issue('transactions', 'transaction_time', 'missing_value', count,
                        'impute', 'Set to midnight when time unknown')
        print(f"  Imputed {count} missing transaction_time as '00:00:00'")
    
    # DROP: Missing amount
    missing_amount = df_clean['amount'].isnull()
    if missing_amount.any():
        count = missing_amount.sum()
        df_clean = df_clean[~missing_amount]
        logger.log_issue('transactions', 'amount', 'missing_value', count,
                        'drop_record', 'Transaction amount is critical financial data, cannot impute')
        print(f"  Dropped {count} records with missing amount")
    
    # KEEP NULL: merchant_category, merchant_name (not all transactions have merchants)
    # KEEP NULL: transaction_type, channel (if missing, indicates data quality issue but not critical)
    
    final_count = len(df_clean)
    print(f"  ✓ Transactions: {initial_count} → {final_count} rows ({initial_count - final_count} dropped)")
    
    return df_clean


def handle_missing_balances(df, logger):
    """Handle missing values in balances table."""
    print(f"\n[Layer 2: Missing Data] Processing balances...")
    df_clean = df.copy()
    initial_count = len(df_clean)
    
    # DROP: Missing account_id or balance_date
    missing_id_date = df_clean['account_id'].isnull() | df_clean['balance_date'].isnull()
    if missing_id_date.any():
        count = missing_id_date.sum()
        df_clean = df_clean[~missing_id_date]
        logger.log_issue('balances', 'account_id/balance_date', 'missing_value', count,
                        'drop_record', 'Cannot identify balance record without account ID and date')
        print(f"  Dropped {count} records with missing IDs or dates")
    
    # DROP: Missing balance amounts
    missing_balance = df_clean['available_balance'].isnull() | df_clean['ledger_balance'].isnull()
    if missing_balance.any():
        count = missing_balance.sum()
        df_clean = df_clean[~missing_balance]
        logger.log_issue('balances', 'available_balance/ledger_balance', 'missing_value', count,
                        'drop_record', 'Balance amounts are critical financial data')
        print(f"  Dropped {count} records with missing balance amounts")
    
    # IMPUTE: Missing overdraft_count
    missing_overdraft = df_clean['overdraft_count'].isnull()
    if missing_overdraft.any():
        count = missing_overdraft.sum()
        df_clean.loc[missing_overdraft, 'overdraft_count'] = 0
        logger.log_issue('balances', 'overdraft_count', 'missing_value', count,
                        'impute', 'Assumed zero overdrafts when not specified')
        print(f"  Imputed {count} missing overdraft_count as 0")
    
    final_count = len(df_clean)
    print(f"  ✓ Balances: {initial_count} → {final_count} rows ({initial_count - final_count} dropped)")
    
    return df_clean


# ============================================================================
# LAYER 3: DEDUPLICATION & CONSISTENCY
# ============================================================================

def deduplicate_and_standardize_accounts(df, logger):
    """Remove duplicates and standardize text in accounts table."""
    print(f"\n[Layer 3: Deduplication & Consistency] Processing accounts...")
    df_clean = df.copy()
    initial_count = len(df_clean)
    
    # Remove duplicate account_ids (keep first occurrence)
    duplicates = df_clean['account_id'].duplicated()
    if duplicates.any():
        count = duplicates.sum()
        df_clean = df_clean[~duplicates]
        logger.log_issue('accounts', 'account_id', 'duplicate', count,
                        'drop_record', 'Kept first occurrence of duplicate account_id')
        print(f"  Removed {count} duplicate account_ids")
    
    # Standardize account_type
    type_mapping = {
        'chekcing': 'checking',  # Fix typo
        'CHECKING': 'checking',
        'SAVINGS': 'savings',
        'CREDIT_CARD': 'credit_card',
        'LOAN': 'loan'
    }
    
    for bad_val, good_val in type_mapping.items():
        mask = df_clean['account_type'] == bad_val
        if mask.any():
            count = mask.sum()
            df_clean.loc[mask, 'account_type'] = good_val
            logger.log_issue('accounts', 'account_type', 'typo', count,
                            'standardize', f'Fixed: {bad_val} → {good_val}')
            print(f"  Fixed {count} account_type: {bad_val} → {good_val}")
    
    # Standardize status (strip whitespace, lowercase)
    df_clean['status'] = df_clean['status'].str.strip().str.lower()
    
    # Fix sentinel values in credit_limit
    sentinel_mask = df_clean['credit_limit'] == -999
    if sentinel_mask.any():
        count = sentinel_mask.sum()
        df_clean.loc[sentinel_mask, 'credit_limit'] = 0
        logger.log_issue('accounts', 'credit_limit', 'sentinel_value', count,
                        'replace', 'Replaced -999 sentinel with 0')
        print(f"  Fixed {count} sentinel values in credit_limit (-999 → 0)")
    
    # Validate branch_code format (should be BR-XXX)
    import re
    branch_pattern = r'^BR-\d{3}$'
    # Only validate non-null branch codes
    has_branch = df_clean['branch_code'].notna()
    if has_branch.any():
        matches_pattern = df_clean.loc[has_branch, 'branch_code'].str.match(branch_pattern)
        invalid_branch = has_branch & ~matches_pattern.reindex(df_clean.index, fill_value=True)
        if invalid_branch.any():
            count = invalid_branch.sum()
            df_clean.loc[invalid_branch, 'branch_code'] = np.nan
            logger.log_issue('accounts', 'branch_code', 'invalid_format', count,
                            'set_to_null', 'Branch code must match BR-XXX format')
            print(f"  Cleared {count} invalid branch codes")
    
    final_count = len(df_clean)
    print(f"  ✓ Accounts: {initial_count} → {final_count} rows ({initial_count - final_count} dropped)")
    
    return df_clean


def deduplicate_and_standardize_transactions(df, logger):
    """Remove duplicates and standardize text in transactions table."""
    print(f"\n[Layer 3: Deduplication & Consistency] Processing transactions...")
    df_clean = df.copy()
    initial_count = len(df_clean)
    
    # Remove duplicate transaction_ids
    duplicates = df_clean['transaction_id'].duplicated()
    if duplicates.any():
        count = duplicates.sum()
        df_clean = df_clean[~duplicates]
        logger.log_issue('transactions', 'transaction_id', 'duplicate', count,
                        'drop_record', 'Kept first occurrence of duplicate transaction_id')
        print(f"  Removed {count} duplicate transaction_ids")
    
    # Remove completely duplicate rows
    full_duplicates = df_clean.duplicated()
    if full_duplicates.any():
        count = full_duplicates.sum()
        df_clean = df_clean[~full_duplicates]
        logger.log_issue('transactions', 'all_columns', 'duplicate', count,
                        'drop_record', 'Removed completely duplicate records')
        print(f"  Removed {count} fully duplicate records")
    
    # Standardize transaction_type
    df_clean['transaction_type'] = df_clean['transaction_type'].str.strip().str.lower()
    
    # Standardize channel
    df_clean['channel'] = df_clean['channel'].str.strip().str.lower()
    
    # Fix outlier amounts (obvious errors like 999999.99)
    outlier_mask = df_clean['amount'].abs() > 50000
    if outlier_mask.any():
        count = outlier_mask.sum()
        # These are likely data entry errors, drop them
        df_clean = df_clean[~outlier_mask]
        logger.log_issue('transactions', 'amount', 'outlier', count,
                        'drop_record', 'Amounts > $50,000 are likely data entry errors')
        print(f"  Removed {count} transactions with outlier amounts (>$50k)")
    
    # Remove future-dated transactions
    today = pd.Timestamp.now().normalize()
    future_dates = df_clean['transaction_date'] > today
    if future_dates.any():
        count = future_dates.sum()
        df_clean = df_clean[~future_dates]
        logger.log_issue('transactions', 'transaction_date', 'future_date', count,
                        'drop_record', 'Transactions cannot occur in the future')
        print(f"  Removed {count} future-dated transactions")
    
    final_count = len(df_clean)
    print(f"  ✓ Transactions: {initial_count} → {final_count} rows ({initial_count - final_count} dropped)")
    
    return df_clean


def deduplicate_and_standardize_balances(df, logger):
    """Remove duplicates and enforce business rules in balances table."""
    print(f"\n[Layer 3: Deduplication & Consistency] Processing balances...")
    df_clean = df.copy()
    initial_count = len(df_clean)
    
    # Remove duplicate (account_id, balance_date) pairs
    duplicates = df_clean.duplicated(subset=['account_id', 'balance_date'])
    if duplicates.any():
        count = duplicates.sum()
        df_clean = df_clean[~duplicates]
        logger.log_issue('balances', 'account_id+balance_date', 'duplicate', count,
                        'drop_record', 'One balance per account per date')
        print(f"  Removed {count} duplicate balance records")
    
    # Enforce business rule: ledger_balance >= available_balance
    violation_mask = df_clean['ledger_balance'] < df_clean['available_balance']
    if violation_mask.any():
        count = violation_mask.sum()
        # Swap them (likely data entry error where columns were reversed)
        temp = df_clean.loc[violation_mask, 'available_balance'].copy()
        df_clean.loc[violation_mask, 'available_balance'] = df_clean.loc[violation_mask, 'ledger_balance']
        df_clean.loc[violation_mask, 'ledger_balance'] = temp
        logger.log_issue('balances', 'ledger_balance', 'business_rule_violation', count,
                        'swap_values', 'Ledger must be >= available; swapped values')
        print(f"  Fixed {count} records where ledger < available (swapped values)")
    
    # Fix negative overdraft counts
    negative_overdraft = df_clean['overdraft_count'] < 0
    if negative_overdraft.any():
        count = negative_overdraft.sum()
        df_clean.loc[negative_overdraft, 'overdraft_count'] = 0
        logger.log_issue('balances', 'overdraft_count', 'invalid_value', count,
                        'replace', 'Overdraft count cannot be negative; set to 0')
        print(f"  Fixed {count} negative overdraft counts")
    
    # Remove future-dated balances
    today = pd.Timestamp.now().normalize()
    future_dates = df_clean['balance_date'] > today
    if future_dates.any():
        count = future_dates.sum()
        df_clean = df_clean[~future_dates]
        logger.log_issue('balances', 'balance_date', 'future_date', count,
                        'drop_record', 'Balance dates cannot be in the future')
        print(f"  Removed {count} future-dated balances")
    
    final_count = len(df_clean)
    print(f"  ✓ Balances: {initial_count} → {final_count} rows ({initial_count - final_count} dropped)")
    
    return df_clean


# ============================================================================
# LAYER 4: CROSS-TABLE VALIDATION
# ============================================================================

def validate_referential_integrity(accounts, transactions, balances, logger):
    """
    Ensure referential integrity across tables.
    
    Rules:
        - All transaction.account_id must exist in accounts.account_id
        - All balance.account_id must exist in accounts.account_id
        - Transaction dates must be >= account open_date
    """
    print(f"\n[Layer 4: Cross-Table Validation] Enforcing referential integrity...")
    
    valid_account_ids = set(accounts['account_id'].unique())
    
    # Validate transactions
    trans_initial = len(transactions)
    orphaned_trans = ~transactions['account_id'].isin(valid_account_ids)
    if orphaned_trans.any():
        count = orphaned_trans.sum()
        transactions = transactions[~orphaned_trans]
        logger.log_issue('transactions', 'account_id', 'orphaned_foreign_key', count,
                        'drop_record', 'Transaction references non-existent account')
        print(f"  Removed {count} orphaned transactions")
    
    # Validate transaction dates vs account open dates
    transactions_with_dates = transactions.merge(
        accounts[['account_id', 'open_date']], 
        on='account_id', 
        how='left'
    )
    
    before_open = transactions_with_dates['transaction_date'] < transactions_with_dates['open_date']
    if before_open.any():
        count = before_open.sum()
        invalid_trans_ids = transactions_with_dates.loc[before_open, 'transaction_id']
        transactions = transactions[~transactions['transaction_id'].isin(invalid_trans_ids)]
        logger.log_issue('transactions', 'transaction_date', 'temporal_violation', count,
                        'drop_record', 'Transaction date before account open date')
        print(f"  Removed {count} transactions dated before account opening")
    
    trans_final = len(transactions)
    print(f"  ✓ Transactions: {trans_initial} → {trans_final} rows")
    
    # Validate balances
    bal_initial = len(balances)
    orphaned_bal = ~balances['account_id'].isin(valid_account_ids)
    if orphaned_bal.any():
        count = orphaned_bal.sum()
        balances = balances[~orphaned_bal]
        logger.log_issue('balances', 'account_id', 'orphaned_foreign_key', count,
                        'drop_record', 'Balance references non-existent account')
        print(f"  Removed {count} orphaned balances")
    
    # Validate balance dates vs account open dates
    balances_with_dates = balances.merge(
        accounts[['account_id', 'open_date']], 
        on='account_id', 
        how='left'
    )
    
    before_open_bal = balances_with_dates['balance_date'] < balances_with_dates['open_date']
    if before_open_bal.any():
        count = before_open_bal.sum()
        invalid_bal_mask = balances['account_id'].isin(
            balances_with_dates.loc[before_open_bal, 'account_id']
        ) & balances['balance_date'].isin(
            balances_with_dates.loc[before_open_bal, 'balance_date']
        )
        balances = balances[~invalid_bal_mask]
        logger.log_issue('balances', 'balance_date', 'temporal_violation', count,
                        'drop_record', 'Balance date before account open date')
        print(f"  Removed {count} balances dated before account opening")
    
    bal_final = len(balances)
    print(f"  ✓ Balances: {bal_initial} → {bal_final} rows")
    print(f"  ✓ Referential integrity enforced!")
    
    return accounts, transactions, balances


# ============================================================================
# QUALITY METRICS & REPORTING
# ============================================================================

def generate_quality_report(accounts_raw, transactions_raw, balances_raw,
                            accounts_clean, transactions_clean, balances_clean):
    """Generate before/after quality metrics."""
    
    report = {
        'accounts': {
            'before': {
                'row_count': len(accounts_raw),
                'null_count': accounts_raw.isnull().sum().sum(),
                'duplicate_ids': accounts_raw['account_id'].duplicated().sum()
            },
            'after': {
                'row_count': len(accounts_clean),
                'null_count': accounts_clean.isnull().sum().sum(),
                'duplicate_ids': accounts_clean['account_id'].duplicated().sum()
            }
        },
        'transactions': {
            'before': {
                'row_count': len(transactions_raw),
                'null_count': transactions_raw.isnull().sum().sum(),
                'duplicate_ids': transactions_raw['transaction_id'].duplicated().sum()
            },
            'after': {
                'row_count': len(transactions_clean),
                'null_count': transactions_clean.isnull().sum().sum(),
                'duplicate_ids': transactions_clean['transaction_id'].duplicated().sum()
            }
        },
        'balances': {
            'before': {
                'row_count': len(balances_raw),
                'null_count': balances_raw.isnull().sum().sum(),
                'duplicates': balances_raw.duplicated().sum()
            },
            'after': {
                'row_count': len(balances_clean),
                'null_count': balances_clean.isnull().sum().sum(),
                'duplicates': balances_clean.duplicated().sum()
            }
        }
    }
    
    return report


def create_quality_dashboard(report, lineage_report):
    """Create visual dashboard of data quality improvements."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Data Quality Pipeline Results', fontsize=16, fontweight='bold')
    
    # Row 1, Col 1: Row counts before/after
    tables = ['accounts', 'transactions', 'balances']
    before_counts = [report[t]['before']['row_count'] for t in tables]
    after_counts = [report[t]['after']['row_count'] for t in tables]
    
    x = np.arange(len(tables))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, before_counts, width, label='Before', color='#e74c3c')
    axes[0, 0].bar(x + width/2, after_counts, width, label='After', color='#2ecc71')
    axes[0, 0].set_ylabel('Row Count')
    axes[0, 0].set_title('Records: Before vs After')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(tables)
    axes[0, 0].legend()
    
    # Row 1, Col 2: Null counts
    before_nulls = [report[t]['before']['null_count'] for t in tables]
    after_nulls = [report[t]['after']['null_count'] for t in tables]
    
    axes[0, 1].bar(x - width/2, before_nulls, width, label='Before', color='#e74c3c')
    axes[0, 1].bar(x + width/2, after_nulls, width, label='After', color='#2ecc71')
    axes[0, 1].set_ylabel('Null Count')
    axes[0, 1].set_title('Missing Values: Before vs After')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(tables)
    axes[0, 1].legend()
    
    # Row 1, Col 3: Issues by type
    issue_counts = lineage_report['issue_type'].value_counts()
    axes[0, 2].barh(issue_counts.index, issue_counts.values, color='#3498db')
    axes[0, 2].set_xlabel('Count')
    axes[0, 2].set_title('Issues by Type')
    
    # Row 2, Col 1: Issues by table
    table_counts = lineage_report['table'].value_counts()
    axes[1, 0].pie(table_counts.values, labels=table_counts.index, autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title('Issues by Table')
    
    # Row 2, Col 2: Actions taken
    action_counts = lineage_report['action_taken'].value_counts()
    axes[1, 1].pie(action_counts.values, labels=action_counts.index, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Actions Taken')
    
    # Row 2, Col 3: Total rows affected
    total_affected = lineage_report['rows_affected'].sum()
    total_original = sum(before_counts)
    pct_affected = (total_affected / total_original) * 100
    
    axes[1, 2].text(0.5, 0.6, f"{total_affected:,}", ha='center', va='center', fontsize=36, fontweight='bold')
    axes[1, 2].text(0.5, 0.4, f"rows affected", ha='center', va='center', fontsize=14)
    axes[1, 2].text(0.5, 0.25, f"({pct_affected:.1f}% of original data)", ha='center', va='center', fontsize=12, style='italic')
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Main data cleaning pipeline.
    
    Runs all four layers of cleaning and generates output files.
    """
    print("="*70)
    print("DATA CLEANING PIPELINE - Chapter 2 Complete Solution")
    print("="*70)
    
    # Initialize logger
    logger = DataQualityLogger()
    
    # ========================================================================
    # LOAD RAW DATA
    # ========================================================================
    
    print("\n" + "="*70)
    print("LOADING RAW DATA")
    print("="*70)
    
    # Check if data exists
    if not os.path.exists('synthetic_banking_data/raw/accounts.csv'):
        print("\n❌ ERROR: Synthetic banking data not found!")
        print("\nPlease generate the data first:")
        print("  from generate_banking_data import BankingDataGenerator")
        print("  generator = BankingDataGenerator()")
        print("  data = generator.generate_all()")
        return
    
    accounts_raw = pd.read_csv('synthetic_banking_data/raw/accounts.csv')
    transactions_raw = pd.read_csv('synthetic_banking_data/raw/transactions.csv')
    balances_raw = pd.read_csv('synthetic_banking_data/raw/balances.csv')
    
    print(f"\n✓ Loaded {len(accounts_raw):,} accounts")
    print(f"✓ Loaded {len(transactions_raw):,} transactions")
    print(f"✓ Loaded {len(balances_raw):,} balance records")
    
    # ========================================================================
    # LAYER 1: SCHEMA VALIDATION
    # ========================================================================
    
    print("\n" + "="*70)
    print("LAYER 1: SCHEMA VALIDATION & TYPE COERCION")
    print("="*70)
    
    accounts = validate_and_coerce_schema(accounts_raw, ACCOUNT_SCHEMA, 'accounts', logger)
    transactions = validate_and_coerce_schema(transactions_raw, TRANSACTION_SCHEMA, 'transactions', logger)
    balances = validate_and_coerce_schema(balances_raw, BALANCE_SCHEMA, 'balances', logger)
    
    # ========================================================================
    # LAYER 2: MISSING DATA
    # ========================================================================
    
    print("\n" + "="*70)
    print("LAYER 2: HANDLING MISSING DATA")
    print("="*70)
    
    accounts = handle_missing_accounts(accounts, logger)
    transactions = handle_missing_transactions(transactions, logger)
    balances = handle_missing_balances(balances, logger)
    
    # ========================================================================
    # LAYER 3: DEDUPLICATION & CONSISTENCY
    # ========================================================================
    
    print("\n" + "="*70)
    print("LAYER 3: DEDUPLICATION & CONSISTENCY")
    print("="*70)
    
    accounts = deduplicate_and_standardize_accounts(accounts, logger)
    transactions = deduplicate_and_standardize_transactions(transactions, logger)
    balances = deduplicate_and_standardize_balances(balances, logger)
    
    # ========================================================================
    # LAYER 4: CROSS-TABLE VALIDATION
    # ========================================================================
    
    print("\n" + "="*70)
    print("LAYER 4: CROSS-TABLE VALIDATION")
    print("="*70)
    
    accounts_clean, transactions_clean, balances_clean = validate_referential_integrity(
        accounts, transactions, balances, logger
    )
    
    # ========================================================================
    # GENERATE REPORTS
    # ========================================================================
    
    print("\n" + "="*70)
    print("GENERATING QUALITY REPORTS")
    print("="*70)
    
    # Get lineage report
    lineage_report = logger.get_report()
    
    # Generate quality metrics
    quality_report = generate_quality_report(
        accounts_raw, transactions_raw, balances_raw,
        accounts_clean, transactions_clean, balances_clean
    )
    
    # Create output directory
    os.makedirs('data_mart_clean', exist_ok=True)
    os.makedirs('data_mart_clean/data', exist_ok=True)
    os.makedirs('data_mart_clean/documentation', exist_ok=True)
    os.makedirs('data_mart_clean/logs', exist_ok=True)
    
    # Save cleaned data
    accounts_clean.to_csv('data_mart_clean/data/accounts_clean.csv', index=False)
    transactions_clean.to_csv('data_mart_clean/data/transactions_clean.csv', index=False)
    balances_clean.to_csv('data_mart_clean/data/balances_clean.csv', index=False)
    print(f"\n✓ Saved cleaned data to data_mart_clean/data/")
    
    # Save lineage report
    logger.save_report('data_mart_clean/logs/data_lineage_report.csv')
    
    # Save quality report
    with open('data_mart_clean/documentation/quality_report.json', 'w') as f:
        json.dump(quality_report, f, indent=2, default=str)
    print(f"✓ Saved quality report to data_mart_clean/documentation/quality_report.json")
    
    # Create and save dashboard
    fig = create_quality_dashboard(quality_report, lineage_report)
    fig.savefig('data_mart_clean/documentation/quality_dashboard.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved quality dashboard to data_mart_clean/documentation/quality_dashboard.png")
    plt.close()
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    
    print(f"\n📊 SUMMARY:")
    print(f"\nOriginal Data:")
    print(f"  Accounts:     {len(accounts_raw):,} records")
    print(f"  Transactions: {len(transactions_raw):,} records")
    print(f"  Balances:     {len(balances_raw):,} records")
    
    print(f"\nCleaned Data:")
    print(f"  Accounts:     {len(accounts_clean):,} records ({len(accounts_raw) - len(accounts_clean):,} dropped)")
    print(f"  Transactions: {len(transactions_clean):,} records ({len(transactions_raw) - len(transactions_clean):,} dropped)")
    print(f"  Balances:     {len(balances_clean):,} records ({len(balances_raw) - len(balances_clean):,} dropped)")
    
    print(f"\n✓ {len(lineage_report)} data quality issues identified and resolved")
    print(f"✓ Complete audit trail saved to data_mart_clean/logs/")
    print(f"✓ All outputs saved to data_mart_clean/")
    
    print("\n" + "="*70)
    

if __name__ == "__main__":
    main()
