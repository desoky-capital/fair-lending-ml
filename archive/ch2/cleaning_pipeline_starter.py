"""
Data Cleaning Pipeline - Starter Template
==========================================

HOW TO USE THIS FILE:

Option 1 - Complete this file:
    1. Fill in all TODO sections
    2. Run: from generate_banking_data import BankingDataGenerator
    
Option 2 - Copy to your notebook:
    1. Create new notebook
    2. Copy one function at a time
    3. Fill in TODOs
    4. Test each function as you go
    
Option 3 - Use as reference:
    Follow Section 2 in the book, refer to this for structure
    
Choose the approach that matches your learning style!

==========================================

This template provides the structure for the data cleaning pipeline from
Chapter 2, Section 2 of "Code, Cash, and Conviction: Building Ethical
Fintech Systems for Industry and the Classroom."

Your task: Fill in the TODO sections to complete the pipeline.

Follow along with the book (Section 2) for guidance on each function.

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

# TODO: Define expected schemas for each table
# Hint: Map column names to expected data types (string, float64, datetime64[ns], etc.)

ACCOUNT_SCHEMA = {
    # TODO: Fill in expected data types for each column
    # Example: 'account_id': 'string',
}

TRANSACTION_SCHEMA = {
    # TODO: Fill in expected data types
}

BALANCE_SCHEMA = {
    # TODO: Fill in expected data types
}


def validate_and_coerce_schema(df, schema, table_name, logger):
    """
    Validate and coerce data types according to schema.
    
    TODO:
    1. Loop through each column in the schema
    2. Check if column exists in dataframe
    3. If data type doesn't match schema:
       - For datetime: use pd.to_datetime() with errors='coerce'
       - For numeric: use pd.to_numeric() with errors='coerce'
       - For string: use .astype(str)
    4. Log any conversion failures using logger.log_issue()
    5. Return the cleaned dataframe
    
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
    
    # TODO: Implement schema validation logic
    # Your code here...
    
    print(f"  ✓ Schema validation complete for {table_name}")
    return df_clean


# ============================================================================
# LAYER 2: HANDLING MISSING DATA
# ============================================================================

def handle_missing_accounts(df, logger):
    """
    Handle missing values in accounts table with column-specific strategies.
    
    TODO: Implement missing data strategies:
    
    For each column, decide:
    - DROP: If the field is critical (primary key, required for analysis)
    - IMPUTE: If you can infer a reasonable value
    - KEEP NULL: If NULL has business meaning
    
    Strategies by column:
    - account_id: DROP (primary key, cannot be null)
    - customer_id: DROP (required for identification)
    - account_type: IMPUTE from credit_limit or set to default
    - open_date: DROP (required for temporal analysis)
    - credit_limit: KEEP NULL (NULL = not applicable for this account type)
    - status: IMPUTE as 'active' (most common)
    - branch_code: KEEP NULL (NULL = online-only account)
    
    Remember to log each decision with logger.log_issue()
    """
    print(f"\n[Layer 2: Missing Data] Processing accounts...")
    df_clean = df.copy()
    initial_count = len(df_clean)
    
    # TODO: Handle missing account_id
    # Hint: Drop rows where account_id is null
    # Don't forget to log the issue!
    
    # TODO: Handle missing customer_id
    
    # TODO: Handle missing account_type
    # Hint: Can you infer from credit_limit?
    
    # TODO: Handle missing open_date
    
    # TODO: Handle missing status
    # Hint: What's a safe default?
    
    # credit_limit and branch_code: Keep NULL (has business meaning)
    
    final_count = len(df_clean)
    print(f"  ✓ Accounts: {initial_count} → {final_count} rows ({initial_count - final_count} dropped)")
    
    return df_clean


def handle_missing_transactions(df, logger):
    """
    Handle missing values in transactions table.
    
    TODO: Implement missing data strategies for transactions
    
    Key decisions:
    - transaction_id, account_id: Must have (DROP if missing)
    - transaction_date: Required (DROP if missing)
    - transaction_time: Can impute (e.g., '00:00:00' if unknown)
    - amount: Critical financial data (DROP if missing - never impute money!)
    - merchant_* fields: Can be NULL (not all transactions have merchants)
    """
    print(f"\n[Layer 2: Missing Data] Processing transactions...")
    df_clean = df.copy()
    initial_count = len(df_clean)
    
    # TODO: Implement missing data handling for transactions
    # Your code here...
    
    final_count = len(df_clean)
    print(f"  ✓ Transactions: {initial_count} → {final_count} rows ({initial_count - final_count} dropped)")
    
    return df_clean


def handle_missing_balances(df, logger):
    """
    Handle missing values in balances table.
    
    TODO: Implement missing data strategies for balances
    
    Key decisions:
    - account_id, balance_date: Must have (composite key)
    - available_balance, ledger_balance: Critical financial data (DROP if missing)
    - overdraft_count: Can impute as 0
    """
    print(f"\n[Layer 2: Missing Data] Processing balances...")
    df_clean = df.copy()
    initial_count = len(df_clean)
    
    # TODO: Implement missing data handling for balances
    # Your code here...
    
    final_count = len(df_clean)
    print(f"  ✓ Balances: {initial_count} → {final_count} rows ({initial_count - final_count} dropped)")
    
    return df_clean


# ============================================================================
# LAYER 3: DEDUPLICATION & CONSISTENCY
# ============================================================================

def deduplicate_and_standardize_accounts(df, logger):
    """
    Remove duplicates and standardize text in accounts table.
    
    TODO: 
    1. Remove duplicate account_ids (keep='first')
    2. Fix typos in account_type (e.g., 'chekcing' → 'checking')
    3. Standardize case (convert to lowercase)
    4. Strip whitespace from text fields
    5. Fix sentinel values (e.g., -999 → 0 or NULL)
    6. Validate branch_code format (should be BR-XXX)
    
    Log each fix with logger.log_issue()
    """
    print(f"\n[Layer 3: Deduplication & Consistency] Processing accounts...")
    df_clean = df.copy()
    initial_count = len(df_clean)
    
    # TODO: Remove duplicate account_ids
    
    # TODO: Fix typos and standardize account_type
    # Hint: Create a mapping dictionary like {'chekcing': 'checking', 'CHECKING': 'checking'}
    
    # TODO: Standardize status (strip whitespace, lowercase)
    
    # TODO: Fix sentinel values in credit_limit (-999 → 0)
    
    # TODO: Validate branch_code format using regex
    # Pattern: BR-XXX where X is a digit
    
    final_count = len(df_clean)
    print(f"  ✓ Accounts: {initial_count} → {final_count} rows ({initial_count - final_count} dropped)")
    
    return df_clean


def deduplicate_and_standardize_transactions(df, logger):
    """
    Remove duplicates and standardize text in transactions table.
    
    TODO:
    1. Remove duplicate transaction_ids
    2. Remove fully duplicate rows
    3. Standardize transaction_type and channel (lowercase, strip)
    4. Remove outlier amounts (e.g., > $50,000 = likely error)
    5. Remove future-dated transactions
    """
    print(f"\n[Layer 3: Deduplication & Consistency] Processing transactions...")
    df_clean = df.copy()
    initial_count = len(df_clean)
    
    # TODO: Implement deduplication and standardization
    # Your code here...
    
    final_count = len(df_clean)
    print(f"  ✓ Transactions: {initial_count} → {final_count} rows ({initial_count - final_count} dropped)")
    
    return df_clean


def deduplicate_and_standardize_balances(df, logger):
    """
    Remove duplicates and enforce business rules in balances table.
    
    TODO:
    1. Remove duplicate (account_id, balance_date) pairs
    2. Enforce business rule: ledger_balance >= available_balance
       - If violated, swap the values (likely data entry error)
    3. Fix negative overdraft counts (set to 0)
    4. Remove future-dated balances
    """
    print(f"\n[Layer 3: Deduplication & Consistency] Processing balances...")
    df_clean = df.copy()
    initial_count = len(df_clean)
    
    # TODO: Implement deduplication and business rule enforcement
    # Your code here...
    
    final_count = len(df_clean)
    print(f"  ✓ Balances: {initial_count} → {final_count} rows ({initial_count - final_count} dropped)")
    
    return df_clean


# ============================================================================
# LAYER 4: CROSS-TABLE VALIDATION
# ============================================================================

def validate_referential_integrity(accounts, transactions, balances, logger):
    """
    Ensure referential integrity across tables.
    
    TODO:
    1. Get list of valid account_ids from accounts table
    2. Remove transactions with account_ids not in valid list (orphaned)
    3. Remove balances with account_ids not in valid list (orphaned)
    4. Check temporal constraints:
       - transaction_date must be >= account open_date
       - balance_date must be >= account open_date
    5. Log all referential integrity violations
    
    Returns:
        Tuple of (accounts, transactions, balances) with integrity enforced
    """
    print(f"\n[Layer 4: Cross-Table Validation] Enforcing referential integrity...")
    
    # TODO: Get valid account IDs
    # Hint: valid_account_ids = set(accounts['account_id'].unique())
    
    # TODO: Remove orphaned transactions
    # Hint: Use .isin() to check if account_id is in valid set
    
    # TODO: Remove orphaned balances
    
    # TODO: Validate temporal constraints
    # Hint: Merge with accounts table to get open_date, then compare
    
    # TODO: Log all violations
    
    print(f"  ✓ Referential integrity enforced!")
    
    return accounts, transactions, balances


# ============================================================================
# QUALITY METRICS & REPORTING
# ============================================================================

def generate_quality_report(accounts_raw, transactions_raw, balances_raw,
                            accounts_clean, transactions_clean, balances_clean):
    """
    Generate before/after quality metrics.
    
    TODO:
    1. Count rows before/after for each table
    2. Count nulls before/after
    3. Count duplicates before/after
    4. Return as structured dictionary
    """
    
    report = {
        'accounts': {
            'before': {
                # TODO: Add metrics
            },
            'after': {
                # TODO: Add metrics
            }
        },
        'transactions': {
            # TODO: Add before/after metrics
        },
        'balances': {
            # TODO: Add before/after metrics
        }
    }
    
    return report


def create_quality_dashboard(report, lineage_report):
    """
    Create visual dashboard of data quality improvements.
    
    TODO:
    1. Create 2x3 subplot grid
    2. Plot before/after row counts (bar chart)
    3. Plot before/after null counts (bar chart)
    4. Plot issues by type (horizontal bar)
    5. Plot issues by table (pie chart)
    6. Plot actions taken (pie chart)
    7. Show total rows affected (text)
    
    Hint: Use matplotlib's plt.subplots() and various plot types
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Data Quality Pipeline Results', fontsize=16, fontweight='bold')
    
    # TODO: Create visualizations
    # Your code here...
    
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Main data cleaning pipeline.
    
    TODO: 
    1. Load raw data from synthetic_banking_data/raw/
    2. Run Layer 1: Schema validation
    3. Run Layer 2: Missing data handling
    4. Run Layer 3: Deduplication & consistency
    5. Run Layer 4: Cross-table validation
    6. Generate quality reports
    7. Save cleaned data to data_mart_clean/
    8. Save lineage report and dashboard
    """
    print("="*70)
    print("DATA CLEANING PIPELINE - Starter Template")
    print("="*70)
    
    # Initialize logger
    logger = DataQualityLogger()
    
    # TODO: Load raw data
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
    
    # TODO: Load the three CSV files
    # accounts_raw = pd.read_csv(...)
    # transactions_raw = pd.read_csv(...)
    # balances_raw = pd.read_csv(...)
    
    # TODO: Run Layer 1 - Schema Validation
    print("\n" + "="*70)
    print("LAYER 1: SCHEMA VALIDATION & TYPE COERCION")
    print("="*70)
    
    # accounts = validate_and_coerce_schema(...)
    # transactions = validate_and_coerce_schema(...)
    # balances = validate_and_coerce_schema(...)
    
    # TODO: Run Layer 2 - Missing Data
    print("\n" + "="*70)
    print("LAYER 2: HANDLING MISSING DATA")
    print("="*70)
    
    # accounts = handle_missing_accounts(...)
    # transactions = handle_missing_transactions(...)
    # balances = handle_missing_balances(...)
    
    # TODO: Run Layer 3 - Deduplication & Consistency
    print("\n" + "="*70)
    print("LAYER 3: DEDUPLICATION & CONSISTENCY")
    print("="*70)
    
    # accounts = deduplicate_and_standardize_accounts(...)
    # transactions = deduplicate_and_standardize_transactions(...)
    # balances = deduplicate_and_standardize_balances(...)
    
    # TODO: Run Layer 4 - Cross-Table Validation
    print("\n" + "="*70)
    print("LAYER 4: CROSS-TABLE VALIDATION")
    print("="*70)
    
    # accounts_clean, transactions_clean, balances_clean = validate_referential_integrity(...)
    
    # TODO: Generate and save reports
    print("\n" + "="*70)
    print("GENERATING QUALITY REPORTS")
    print("="*70)
    
    # lineage_report = logger.get_report()
    # quality_report = generate_quality_report(...)
    
    # TODO: Create output directories and save files
    
    # TODO: Print final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    
    print("\n📝 TODO: Complete the pipeline by filling in all TODO sections!")
    print("Refer to Chapter 2, Section 2 in the book for detailed guidance.\n")


if __name__ == "__main__":
    main()
