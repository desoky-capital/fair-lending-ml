"""
Credit Data Generator - Standalone Version
===========================================

Generates synthetic credit data without requiring Chapter 2 dependencies.
All functionality is self-contained in this file.

Usage:
    from generate_credit_data import CreditDataGenerator
    
    generator = CreditDataGenerator(
        n_accounts=1000,
        seed=42,
        start_date='2019-01-01',
        prediction_date='2024-01-01'
    )
    
    data = generator.generate_all()
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


class CreditDataGenerator:
    """
    Generates synthetic credit data with realistic patterns and quality issues.
    Self-contained - no external dependencies beyond standard libraries.
    """
    
    def __init__(self, n_accounts=1000, seed=42, start_date='2019-01-01', 
                 prediction_date='2024-01-01', default_rate=0.05):
        """
        Initialize the credit data generator.
        
        Args:
            n_accounts: Number of accounts to generate
            seed: Random seed for reproducibility
            start_date: Earliest possible account open date
            prediction_date: Date at which we make predictions (train/test split)
            default_rate: Overall default rate in population (5% is realistic)
        """
        self.n_accounts = n_accounts
        self.seed = seed
        self.start_date = pd.to_datetime(start_date)
        self.prediction_date = pd.to_datetime(prediction_date)
        self.default_rate = default_rate
        
        # Set random seeds
        np.random.seed(seed)
        self.faker = Faker()
        Faker.seed(seed)
        
        # Distribution parameters
        self.account_type_probs = [0.40, 0.30, 0.20, 0.10]  # checking, savings, credit, loan
        self.status_probs = [0.80, 0.10, 0.05, 0.05]  # active, closed, suspended, dormant
        
    def generate_accounts(self):
        """Generate account master data with credit attributes."""
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
                self.faker.date_between(start_date=self.start_date, end_date=self.prediction_date)
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
        
        # Add credit bureau attributes
        print("  Adding credit bureau attributes...")
        accounts = self._add_credit_attributes(accounts)
        
        # Add application data
        print("  Adding application data...")
        accounts = self._add_application_data(accounts)
        
        # Add demographic proxies
        print("  Adding demographic proxies...")
        accounts = self._add_demographic_proxies(accounts)
        
        # Generate default labels
        print("  Generating default labels...")
        accounts = self._generate_default_labels(accounts)
        
        return accounts
    
    def _add_credit_attributes(self, accounts):
        """Add credit bureau attributes."""
        n = len(accounts)
        
        # FICO score (300-850)
        fico_scores = np.clip(
            np.random.normal(loc=690, scale=80, size=n),
            300, 850
        ).astype(int)
        
        # Credit history length (months)
        history_length = np.random.gamma(shape=3, scale=20, size=n).astype(int)
        history_length = np.clip(history_length, 3, 360)
        
        # Number of inquiries
        inquiries = np.random.poisson(lam=1.5, size=n)
        inquiries = np.clip(inquiries, 0, 15)
        
        # Delinquencies
        delinquencies = np.random.poisson(lam=0.3, size=n)
        
        # Credit utilization
        utilization = np.random.beta(a=2, b=5, size=n)
        utilization = np.clip(utilization * 100, 0, 150)
        
        # Number of accounts
        num_accounts = np.random.poisson(lam=5, size=n)
        num_accounts = np.clip(num_accounts, 0, 30)
        
        accounts['fico_score'] = fico_scores
        accounts['credit_history_months'] = history_length
        accounts['num_inquiries_6mo'] = inquiries
        accounts['num_delinquencies_24mo'] = delinquencies
        accounts['credit_utilization_pct'] = np.round(utilization, 1)
        accounts['num_active_credit_lines'] = num_accounts
        
        return accounts
    
    def _add_application_data(self, accounts):
        """Add application-time data."""
        n = len(accounts)
        
        # Annual income
        income = np.random.lognormal(mean=10.8, sigma=0.6, size=n)
        income = np.clip(income, 15000, 500000)
        
        # Employment status
        employment = np.random.choice(
            ['employed', 'self_employed', 'unemployed', 'retired', 'student'],
            size=n,
            p=[0.70, 0.15, 0.05, 0.07, 0.03]
        )
        
        # Employment tenure
        tenure = np.random.gamma(shape=2, scale=18, size=n).astype(int)
        tenure = np.clip(tenure, 0, 480)
        tenure[np.isin(employment, ['unemployed', 'student'])] = 0
        
        # Debt-to-income ratio
        base_dti = np.random.beta(a=2, b=5, size=n) * 50
        income_factor = np.clip(50000 / income, 0.5, 2.0)
        dti = base_dti * income_factor
        dti = np.clip(dti, 0, 80)
        
        accounts['annual_income'] = np.round(income, 0).astype(int)
        accounts['employment_status'] = employment
        accounts['employment_tenure_months'] = tenure
        accounts['debt_to_income_ratio'] = np.round(dti, 1)
        
        return accounts
    
    def _add_demographic_proxies(self, accounts):
        """Add demographic proxy variables."""
        n = len(accounts)
        
        # Age
        age = np.random.gamma(shape=3, scale=12, size=n) + 18
        age = np.clip(age, 18, 80).astype(int)
        
        # Region (proxy for race/SES)
        regions = np.random.choice(['A', 'B', 'C'], size=n, p=[0.4, 0.4, 0.2])
        
        # ZIP codes by region
        zip_codes = []
        for region in regions:
            if region == 'A':
                zip_code = np.random.randint(90000, 90100)
            elif region == 'B':
                zip_code = np.random.randint(80000, 80100)
            else:
                zip_code = np.random.randint(70000, 70100)
            zip_codes.append(str(zip_code))
        
        accounts['age'] = age
        accounts['zip_code'] = zip_codes
        accounts['region'] = regions
        
        # Account tenure
        accounts['account_tenure_months'] = (
            (self.prediction_date - pd.to_datetime(accounts['open_date'])).dt.days / 30
        ).astype(int)
        
        return accounts
    
    def _generate_default_labels(self, accounts):
        """Generate default labels with realistic risk factors."""
        n = len(accounts)
        
        # Base probability from FICO
        fico_factor = (850 - accounts['fico_score']) / 550
        base_prob = 0.005 + fico_factor * 0.395
        
        # Adjust for DTI
        dti_factor = accounts['debt_to_income_ratio'] / 100
        base_prob *= (1 + dti_factor)
        
        # Adjust for delinquencies
        delinq_factor = 1 + (accounts['num_delinquencies_24mo'] * 0.5)
        base_prob *= delinq_factor
        
        # Adjust for utilization
        util_factor = accounts['credit_utilization_pct'] / 100
        base_prob *= (1 + util_factor * 0.3)
        
        # Adjust for age
        age_factor = np.clip((40 - accounts['age']) / 40, 0, 1)
        base_prob *= (1 + age_factor * 0.2)
        
        # Regional effect (creates fairness challenge)
        region_multiplier = accounts['region'].map({'A': 0.8, 'B': 1.0, 'C': 1.5})
        base_prob *= region_multiplier
        
        # Cap and scale
        default_prob = np.clip(base_prob, 0, 0.5)
        current_mean = default_prob.mean()
        default_prob *= (self.default_rate / current_mean)
        default_prob = np.clip(default_prob, 0, 0.5)
        
        # Generate defaults
        defaults = np.random.binomial(1, default_prob)
        
        # Add default dates
        default_dates = []
        for i, defaulted in enumerate(defaults):
            if defaulted == 1:
                days_until = np.random.randint(30, 365)
                default_date = self.prediction_date + timedelta(days=days_until)
            else:
                default_date = None
            default_dates.append(default_date)
        
        accounts['default_probability'] = np.round(default_prob, 4)
        accounts['defaulted'] = defaults
        accounts['default_date'] = default_dates
        
        print(f"    Generated {defaults.sum()} defaults ({defaults.sum()/n*100:.1f}% default rate)")
        print(f"    Default rate by region: A={defaults[accounts['region']=='A'].mean():.1%}, "
              f"B={defaults[accounts['region']=='B'].mean():.1%}, "
              f"C={defaults[accounts['region']=='C'].mean():.1%}")
        
        return accounts
    
    def generate_transactions(self, accounts):
        """Generate transaction data."""
        print("Generating transactions...")
        
        transactions = []
        txn_id_counter = 1
        
        for _, account in accounts.iterrows():
            account_id = account['account_id']
            open_date = account['open_date']
            account_type = account['account_type']
            account_status = account['status']
            
            # Transaction count based on status
            if account_status == 'closed':
                n_txns = np.random.poisson(lam=5)
            elif account_status == 'dormant':
                n_txns = np.random.poisson(lam=2)
            else:
                n_txns = np.random.poisson(lam=10)
            
            for _ in range(n_txns):
                days_range = (self.prediction_date.date() - open_date).days
                if days_range <= 0:
                    continue
                    
                txn_date = open_date + timedelta(days=np.random.randint(0, days_range))
                
                # Transaction type
                if account_type in ['checking', 'savings']:
                    txn_type = np.random.choice(
                        ['purchase', 'withdrawal', 'deposit', 'transfer', 'fee'],
                        p=[0.50, 0.20, 0.15, 0.10, 0.05]
                    )
                else:
                    txn_type = np.random.choice(
                        ['purchase', 'payment', 'fee', 'interest'],
                        p=[0.70, 0.20, 0.05, 0.05]
                    )
                
                # Amount
                if txn_type in ['purchase', 'withdrawal', 'fee', 'interest']:
                    amount = -abs(np.random.lognormal(3, 1.5))
                else:
                    amount = abs(np.random.lognormal(4, 1.5))
                
                # Merchant info
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
                        p=[0.35, 0.35, 0.15, 0.10, 0.05]
                    )
                })
                txn_id_counter += 1
        
        return pd.DataFrame(transactions)
    
    def generate_balances(self, accounts):
        """Generate balance snapshots."""
        print("Generating balance snapshots...")
        
        balances = []
        
        for _, account in accounts.iterrows():
            account_id = account['account_id']
            open_date = account['open_date']
            account_type = account['account_type']
            
            # Starting balance
            if account_type in ['checking', 'savings']:
                current_balance = np.random.uniform(100, 10000)
            elif account_type == 'credit_card':
                current_balance = -np.random.uniform(0, 5000)
            else:
                current_balance = -np.random.uniform(5000, 40000)
            
            # Monthly snapshots
            current_date = open_date
            
            while current_date <= self.prediction_date.date():
                change = np.random.normal(0, 500)
                current_balance += change
                
                # Bounds
                if account_type in ['checking', 'savings']:
                    current_balance = max(-1000, current_balance)
                elif account_type == 'credit_card':
                    credit_limit = account['credit_limit']
                    current_balance = max(-credit_limit, current_balance)
                
                # Overdrafts
                if account_type in ['checking', 'savings'] and current_balance < 0:
                    overdraft_count = np.random.randint(1, 5)
                else:
                    overdraft_count = 0
                
                ledger_balance = current_balance + np.random.uniform(-50, 100)
                
                balances.append({
                    'account_id': account_id,
                    'balance_date': current_date,
                    'available_balance': round(current_balance, 2),
                    'ledger_balance': round(ledger_balance, 2),
                    'overdraft_count': overdraft_count
                })
                
                # Next month
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1, day=1)
                
                if current_date.month == 12:
                    current_date = current_date.replace(day=31)
                else:
                    next_month = current_date.replace(month=current_date.month + 1, day=1)
                    current_date = next_month - timedelta(days=1)
        
        return pd.DataFrame(balances)
    
    def inject_data_quality_issues(self, accounts, transactions, balances):
        """Inject realistic data quality problems."""
        print("Injecting data quality issues...")
        
        accounts = accounts.copy()
        transactions = transactions.copy()
        balances = balances.copy()
        
        # Missing values (5%)
        for col in ['fico_score', 'annual_income', 'branch_code']:
            if col in accounts.columns:
                mask = np.random.random(len(accounts)) < 0.05
                accounts.loc[mask, col] = np.nan
        
        # Duplicate records (2%)
        n_dups = int(len(accounts) * 0.02)
        dup_indices = np.random.choice(accounts.index, n_dups, replace=False)
        accounts = pd.concat([accounts, accounts.loc[dup_indices]], ignore_index=True)
        
        # Format inconsistencies in dates
        date_mask = np.random.random(len(accounts)) < 0.05
        accounts.loc[date_mask, 'open_date'] = accounts.loc[date_mask, 'open_date'].apply(
            lambda x: x.strftime('%m/%d/%Y') if pd.notnull(x) else x
        )
        
        # Typos in categorical fields
        typo_mask = np.random.random(len(accounts)) < 0.02
        accounts.loc[typo_mask, 'account_type'] = accounts.loc[typo_mask, 'account_type'].replace({
            'checking': 'chekcing',
            'savings': 'savngs'
        })
        
        return accounts, transactions, balances
    
    def generate_all(self, output_dir='synthetic_credit_data'):
        """Generate complete dataset."""
        print(f"\n{'='*60}")
        print(f"Credit Data Generator - Starting")
        print(f"{'='*60}\n")
        
        # Generate data
        accounts_clean = self.generate_accounts()
        transactions_clean = self.generate_transactions(accounts_clean)
        balances_clean = self.generate_balances(accounts_clean)
        
        # Inject issues
        accounts_messy, transactions_messy, balances_messy = self.inject_data_quality_issues(
            accounts_clean.copy(),
            transactions_clean.copy(),
            balances_clean.copy()
        )
        
        # Save
        print(f"\nSaving data to {output_dir}/...")
        self._save_data(output_dir, accounts_clean, transactions_clean, balances_clean,
                       accounts_messy, transactions_messy, balances_messy)
        
        print(f"\n{'='*60}")
        print(f"Credit Data Generation Complete!")
        print(f"{'='*60}\n")
        print(f"📂 Output directory: {output_dir}/")
        print(f"📊 Data summary:")
        print(f"   - Accounts: {len(accounts_clean):,}")
        print(f"   - Transactions: {len(transactions_clean):,}")
        print(f"   - Balance snapshots: {len(balances_clean):,}")
        print(f"   - Default rate: {accounts_clean['defaulted'].mean():.1%}")
        
        return {
            'accounts_clean': accounts_clean,
            'transactions_clean': transactions_clean,
            'balances_clean': balances_clean,
            'accounts_messy': accounts_messy,
            'transactions_messy': transactions_messy,
            'balances_messy': balances_messy
        }
    
    def _save_data(self, output_dir, accounts_clean, transactions_clean, balances_clean,
                   accounts_messy, transactions_messy, balances_messy):
        """Save datasets to files."""
        raw_dir = Path(output_dir) / 'raw'
        clean_dir = Path(output_dir) / 'clean'
        raw_dir.mkdir(parents=True, exist_ok=True)
        clean_dir.mkdir(parents=True, exist_ok=True)
        
        # Save messy (for exercises)
        accounts_messy.to_csv(raw_dir / 'accounts.csv', index=False)
        transactions_messy.to_csv(raw_dir / 'transactions.csv', index=False)
        balances_messy.to_csv(raw_dir / 'balances.csv', index=False)
        
        # Save clean (for validation)
        accounts_clean.to_csv(clean_dir / 'accounts_clean.csv', index=False)
        transactions_clean.to_csv(clean_dir / 'transactions_clean.csv', index=False)
        balances_clean.to_csv(clean_dir / 'balances_clean.csv', index=False)


# Example usage
if __name__ == "__main__":
    generator = CreditDataGenerator(
        n_accounts=1000,
        seed=42,
        start_date='2019-01-01',
        prediction_date='2024-01-01',
        default_rate=0.05
    )
    
    data = generator.generate_all(output_dir='synthetic_credit_data')
    
    print("\nSample account data:")
    print(data['accounts_clean'][['account_id', 'fico_score', 'annual_income', 
                                    'debt_to_income_ratio', 'region', 'defaulted']].head(10))
