"""
Synthetic Banking Data Generator - Jupyter Notebook Version
============================================================

This version is designed to be run in Jupyter notebooks.
For command-line usage, use generate_banking_data.py instead.

Usage in Jupyter:
    from generate_banking_data_notebook import BankingDataGenerator
    
    # Create generator with desired parameters
    generator = BankingDataGenerator(
        n_accounts=1000,
        seed=42,
        start_date='2019-01-01'
    )
    
    # Generate all data
    data = generator.generate_all(output_dir='synthetic_banking_data')
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path

try:
    from faker import Faker
except ImportError:
    print("Error: faker library not installed. Run: pip install faker")
    raise


class BankingDataGenerator:
    """Generates synthetic banking data with configurable parameters."""
    
    def __init__(self, n_accounts=1000, seed=42, start_date='2019-01-01'):
        """
        Initialize the data generator.
        
        Args:
            n_accounts: Number of accounts to generate
            seed: Random seed for reproducibility
            start_date: Earliest possible account open date
        """
        self.n_accounts = n_accounts
        self.seed = seed
        self.start_date = pd.to_datetime(start_date)
        
        # Set random seeds
        np.random.seed(seed)
        self.faker = Faker()
        Faker.seed(seed)
        
        # Distribution parameters
        self.account_type_probs = [0.40, 0.30, 0.20, 0.10]  # checking, savings, credit, loan
        self.status_probs = [0.80, 0.10, 0.05, 0.05]  # active, closed, suspended, dormant
        self.txn_type_deposit_probs = [0.50, 0.20, 0.15, 0.10, 0.05]  # purchase, withdrawal, deposit, transfer, fee
        self.txn_type_credit_probs = [0.70, 0.20, 0.05, 0.05]  # purchase, payment, fee, interest
        self.channel_probs = [0.35, 0.35, 0.15, 0.10, 0.05]  # online, mobile, atm, branch, phone
        
    def generate_accounts(self):
        """Generate clean account data."""
        print(f"Generating {self.n_accounts} accounts...")
        
        account_ids = [f"ACC-{str(i).zfill(8)}" for i in range(1, self.n_accounts + 1)]
        customer_ids = [f"CUST-{str(i).zfill(7)}" for i in range(1000001, 1000001 + self.n_accounts)]
        
        accounts = pd.DataFrame({
            'account_id': account_ids,
            'customer_id': customer_ids,
            'account_type': np.random.choice(
                ['checking', 'savings', 'credit_card', 'loan'],
                self.n_accounts,
                p=self.account_type_probs
            ),
            'open_date': [
                self.faker.date_between(start_date=self.start_date, end_date='today')
                for _ in range(self.n_accounts)
            ],
            'credit_limit': 0,  # Will fill conditionally
            'status': np.random.choice(
                ['active', 'closed', 'suspended', 'dormant'],
                self.n_accounts,
                p=self.status_probs
            ),
            'branch_code': [
                f"BR-{np.random.randint(100, 200)}" if np.random.random() > 0.1 
                else None  # 10% online-only accounts
                for _ in range(self.n_accounts)
            ]
        })
        
        # Set credit limits for credit accounts
        credit_mask = accounts['account_type'].isin(['credit_card', 'loan'])
        accounts.loc[credit_mask, 'credit_limit'] = np.random.randint(
            1000, 50000, 
            size=credit_mask.sum()
        )
        
        return accounts
    
    def generate_transactions(self, accounts):
        """Generate clean transaction data based on accounts."""
        print("Generating transactions...")
        
        transactions = []
        txn_id_counter = 1
        
        # Generate transactions for each account
        for _, account in accounts.iterrows():
            account_id = account['account_id']
            open_date = account['open_date']
            account_type = account['account_type']
            account_status = account['status']
            
            # Closed accounts have fewer transactions
            if account_status == 'closed':
                n_txns = np.random.poisson(lam=5)
            elif account_status == 'dormant':
                n_txns = np.random.poisson(lam=2)
            else:  # active or suspended
                n_txns = np.random.poisson(lam=10)
            
            for _ in range(n_txns):
                # Transaction date between open date and today
                days_range = (datetime.now().date() - open_date).days
                if days_range <= 0:
                    continue
                    
                txn_date = open_date + timedelta(days=np.random.randint(0, days_range))
                
                # Transaction type depends on account type
                if account_type in ['checking', 'savings']:
                    txn_type = np.random.choice(
                        ['purchase', 'withdrawal', 'deposit', 'transfer', 'fee'],
                        p=self.txn_type_deposit_probs
                    )
                else:  # credit_card, loan
                    txn_type = np.random.choice(
                        ['purchase', 'payment', 'fee', 'interest'],
                        p=self.txn_type_credit_probs
                    )
                
                # Amount logic (lognormal distribution for realistic skew)
                if txn_type in ['purchase', 'withdrawal', 'fee', 'interest']:
                    amount = -abs(np.random.lognormal(3, 1.5))  # Negative for debits
                else:  # deposit, payment, transfer
                    amount = abs(np.random.lognormal(4, 1.5))
                
                # Merchant info only for purchases
                if txn_type == 'purchase':
                    merchant_category = np.random.choice([
                        'grocery', 'restaurant', 'fuel', 'retail', 
                        'entertainment', 'healthcare', 'travel', 'utilities'
                    ])
                    merchant_name = self.faker.company()
                else:
                    merchant_category = None
                    merchant_name = None
                
                transactions.append({
                    'transaction_id': f"TXN-{str(txn_id_counter).zfill(10)}",
                    'account_id': account_id,
                    'transaction_date': txn_date,
                    'transaction_time': self.faker.time(),
                    'amount': round(amount, 2),
                    'transaction_type': txn_type,
                    'merchant_category': merchant_category,
                    'merchant_name': merchant_name,
                    'channel': np.random.choice(
                        ['online', 'mobile', 'atm', 'branch', 'phone'],
                        p=self.channel_probs
                    )
                })
                txn_id_counter += 1
        
        return pd.DataFrame(transactions)
    
    def generate_balances(self, accounts):
        """Generate clean balance snapshot data."""
        print("Generating balance snapshots...")
        
        balances = []
        
        for _, account in accounts.iterrows():
            account_id = account['account_id']
            open_date = account['open_date']
            account_type = account['account_type']
            
            # Starting balance depends on account type
            if account_type in ['checking', 'savings']:
                current_balance = np.random.uniform(100, 10000)
            elif account_type == 'credit_card':
                current_balance = -np.random.uniform(0, 5000)  # Negative = owed
            else:  # loan
                current_balance = -np.random.uniform(5000, 40000)
            
            # Generate monthly snapshots from open date to now
            current_date = open_date
            end_date = datetime.now().date()
            
            while current_date <= end_date:
                # Random walk for balance changes
                change = np.random.normal(0, 500)
                current_balance += change
                
                # Keep reasonable bounds
                if account_type in ['checking', 'savings']:
                    current_balance = max(-1000, current_balance)  # Can go slightly negative
                elif account_type == 'credit_card':
                    credit_limit = account['credit_limit']
                    current_balance = max(-credit_limit, current_balance)
                
                # Overdraft count (for deposit accounts that go negative)
                if account_type in ['checking', 'savings'] and current_balance < 0:
                    overdraft_count = np.random.randint(1, 5)
                else:
                    overdraft_count = 0
                
                # Ledger balance slightly different from available (holds, pending)
                ledger_balance = current_balance + np.random.uniform(-50, 100)
                
                balances.append({
                    'account_id': account_id,
                    'balance_date': current_date,
                    'available_balance': round(current_balance, 2),
                    'ledger_balance': round(ledger_balance, 2),
                    'overdraft_count': overdraft_count
                })
                
                # Move to end of next month
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1, day=1)
                
                # Go to last day of month
                if current_date.month == 12:
                    current_date = current_date.replace(day=31)
                else:
                    next_month = current_date.replace(month=current_date.month + 1, day=1)
                    current_date = next_month - timedelta(days=1)
        
        return pd.DataFrame(balances)
    
    def inject_data_quality_issues(self, accounts, transactions, balances):
        """
        Inject realistic data quality problems into clean datasets.
        
        This simulates the kinds of issues found in real-world banking data:
        - Missing values
        - Duplicates
        - Format inconsistencies
        - Referential integrity violations
        - Business rule violations
        - Data entry errors
        """
        print("Injecting data quality issues...")
        
        accounts_messy = accounts.copy()
        transactions_messy = transactions.copy()
        balances_messy = balances.copy()
        
        # ============================================
        # ACCOUNTS TABLE PROBLEMS
        # ============================================
        
        # Problem 1: Missing account_ids (5%)
        missing_mask = np.random.random(len(accounts_messy)) < 0.05
        accounts_messy.loc[missing_mask, 'account_id'] = None
        
        # Problem 2: Duplicate account_ids (2%)
        if len(accounts_messy) > 10:
            n_duplicates = max(1, int(len(accounts_messy) * 0.02))
            dup_indices = np.random.choice(
                accounts_messy[accounts_messy['account_id'].notna()].index,
                size=n_duplicates,
                replace=False
            )
            reference_id = accounts_messy.loc[dup_indices[0], 'account_id']
            accounts_messy.loc[dup_indices, 'account_id'] = reference_id
        
        # Problem 3: Account type typos and case inconsistencies
        typo_mask = np.random.random(len(accounts_messy)) < 0.03
        accounts_messy.loc[
            typo_mask & (accounts_messy['account_type'] == 'checking'), 
            'account_type'
        ] = 'chekcing'
        
        upper_mask = np.random.random(len(accounts_messy)) < 0.03
        accounts_messy.loc[upper_mask, 'account_type'] = \
            accounts_messy.loc[upper_mask, 'account_type'].str.upper()
        
        null_type_mask = np.random.random(len(accounts_messy)) < 0.02
        accounts_messy.loc[null_type_mask, 'account_type'] = None
        
        # Problem 4: Mixed date formats
        accounts_messy['open_date'] = accounts_messy['open_date'].astype(str)
        
        format1_mask = np.random.random(len(accounts_messy)) < 0.50
        accounts_messy.loc[format1_mask, 'open_date'] = pd.to_datetime(
            accounts_messy.loc[format1_mask, 'open_date']
        ).dt.strftime('%m/%d/%Y')
        
        format2_mask = ~format1_mask & (np.random.random(len(accounts_messy)) > 0.01)
        accounts_messy.loc[format2_mask, 'open_date'] = pd.to_datetime(
            accounts_messy.loc[format2_mask, 'open_date']
        ).dt.strftime('%Y-%m-%d')
        
        invalid_date_mask = ~format1_mask & ~format2_mask
        accounts_messy.loc[invalid_date_mask, 'open_date'] = '99-99-9999'
        
        # Problem 5: Sentinel values for credit_limit (-999 instead of NULL)
        sentinel_mask = (accounts_messy['credit_limit'] == 0) & \
                       (np.random.random(len(accounts_messy)) < 0.05)
        accounts_messy.loc[sentinel_mask, 'credit_limit'] = -999
        
        # Problem 6: Status with whitespace and unknown values
        ws_mask = np.random.random(len(accounts_messy)) < 0.05
        accounts_messy.loc[ws_mask, 'status'] = \
            accounts_messy.loc[ws_mask, 'status'] + ' '
        
        unknown_status_mask = np.random.random(len(accounts_messy)) < 0.02
        accounts_messy.loc[unknown_status_mask, 'status'] = 'pending'
        
        # Problem 7: Invalid branch codes
        invalid_branch_mask = accounts_messy['branch_code'].notna() & \
                             (np.random.random(len(accounts_messy)) < 0.05)
        accounts_messy.loc[invalid_branch_mask, 'branch_code'] = 'BR-INVALID'
        
        # ============================================
        # TRANSACTIONS TABLE PROBLEMS
        # ============================================
        
        # Similar issues for transactions...
        # (keeping implementation brief for notebook version)
        # See full version in generate_banking_data.py for complete implementation
        
        # Duplicate transaction_ids
        if len(transactions_messy) > 10:
            n_dup_txns = max(1, int(len(transactions_messy) * 0.01))
            dup_txn_indices = np.random.choice(
                transactions_messy[transactions_messy['transaction_id'].notna()].index,
                size=n_dup_txns,
                replace=False
            )
            reference_txn_id = transactions_messy.loc[dup_txn_indices[0], 'transaction_id']
            transactions_messy.loc[dup_txn_indices, 'transaction_id'] = reference_txn_id
        
        # Missing transaction_ids
        missing_txn_mask = np.random.random(len(transactions_messy)) < 0.02
        transactions_messy.loc[missing_txn_mask, 'transaction_id'] = None
        
        # Orphaned transactions
        orphan_mask = np.random.random(len(transactions_messy)) < 0.03
        transactions_messy.loc[orphan_mask, 'account_id'] = 'ACC-99999999'
        
        # Mixed date formats
        transactions_messy['transaction_date'] = transactions_messy['transaction_date'].astype(str)
        format1_mask_txn = np.random.random(len(transactions_messy)) < 0.50
        transactions_messy.loc[format1_mask_txn, 'transaction_date'] = pd.to_datetime(
            transactions_messy.loc[format1_mask_txn, 'transaction_date']
        ).dt.strftime('%m/%d/%Y')
        
        # ... (see full script for complete implementation)
        
        # ============================================
        # BALANCES TABLE PROBLEMS
        # ============================================
        
        # Orphaned balances
        orphan_bal_mask = np.random.random(len(balances_messy)) < 0.02
        balances_messy.loc[orphan_bal_mask, 'account_id'] = 'ACC-99999999'
        
        # Mixed date formats
        balances_messy['balance_date'] = balances_messy['balance_date'].astype(str)
        format1_mask_bal = np.random.random(len(balances_messy)) < 0.50
        balances_messy.loc[format1_mask_bal, 'balance_date'] = pd.to_datetime(
            balances_messy.loc[format1_mask_bal, 'balance_date']
        ).dt.strftime('%m/%d/%Y')
        
        # ... (see full script for complete implementation)
        
        return accounts_messy, transactions_messy, balances_messy
    
    def generate_all(self, output_dir='synthetic_banking_data'):
        """
        Generate all datasets (clean and messy versions) and save to disk.
        
        Args:
            output_dir: Directory to save the generated files
            
        Returns:
            dict: Dictionary containing clean and raw dataframes
        """
        print(f"\n{'='*60}")
        print("Synthetic Banking Data Generator")
        print(f"{'='*60}\n")
        print(f"Configuration:")
        print(f"  - Number of accounts: {self.n_accounts}")
        print(f"  - Random seed: {self.seed}")
        print(f"  - Start date: {self.start_date.date()}")
        print(f"  - Output directory: {output_dir}\n")
        
        # Create output directories
        output_path = Path(output_dir)
        clean_path = output_path / 'clean'
        raw_path = output_path / 'raw'
        
        clean_path.mkdir(parents=True, exist_ok=True)
        raw_path.mkdir(parents=True, exist_ok=True)
        
        # Generate clean data
        print("\n[1/4] Generating clean datasets...")
        accounts_clean = self.generate_accounts()
        transactions_clean = self.generate_transactions(accounts_clean)
        balances_clean = self.generate_balances(accounts_clean)
        
        print(f"  ✓ Accounts: {len(accounts_clean)} records")
        print(f"  ✓ Transactions: {len(transactions_clean)} records")
        print(f"  ✓ Balances: {len(balances_clean)} records")
        
        # Save clean data
        print("\n[2/4] Saving clean datasets...")
        accounts_clean.to_csv(clean_path / 'accounts_clean.csv', index=False)
        transactions_clean.to_csv(clean_path / 'transactions_clean.csv', index=False)
        balances_clean.to_csv(clean_path / 'balances_clean.csv', index=False)
        print(f"  ✓ Saved to {clean_path}/")
        
        # Generate messy data
        print("\n[3/4] Creating messy versions with data quality issues...")
        accounts_raw, transactions_raw, balances_raw = self.inject_data_quality_issues(
            accounts_clean, transactions_clean, balances_clean
        )
        
        # Save raw/messy data
        print("\n[4/4] Saving raw/messy datasets...")
        accounts_raw.to_csv(raw_path / 'accounts.csv', index=False)
        transactions_raw.to_csv(raw_path / 'transactions.csv', index=False)
        balances_raw.to_csv(raw_path / 'balances.csv', index=False)
        print(f"  ✓ Saved to {raw_path}/")
        
        # Summary
        print(f"\n{'='*60}")
        print("Generation Complete!")
        print(f"{'='*60}\n")
        print("Summary Statistics:")
        print(f"  Clean data:  {clean_path}/")
        print(f"    - accounts_clean.csv:      {len(accounts_clean):,} rows")
        print(f"    - transactions_clean.csv:  {len(transactions_clean):,} rows")
        print(f"    - balances_clean.csv:      {len(balances_clean):,} rows")
        print(f"\n  Raw/messy data:  {raw_path}/")
        print(f"    - accounts.csv:      {len(accounts_raw):,} rows")
        print(f"    - transactions.csv:  {len(transactions_raw):,} rows")
        print(f"    - balances.csv:      {len(balances_raw):,} rows")
        print(f"\nData quality issues have been injected into the raw data.")
        print(f"Use the clean data as a reference for validation.\n")
        
        return {
            'clean': {
                'accounts': accounts_clean,
                'transactions': transactions_clean,
                'balances': balances_clean
            },
            'raw': {
                'accounts': accounts_raw,
                'transactions': transactions_raw,
                'balances': balances_raw
            }
        }


# ============================================
# JUPYTER NOTEBOOK USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    print("✓ Data generator loaded successfully!\n")
    print("Usage:")
    print("  generator = BankingDataGenerator(n_accounts=1000, seed=42)")
    print("  data = generator.generate_all()")