"""
Credit Data Generator for Chapter 3
====================================

Generates synthetic credit data with realistic patterns and quality issues.
Self-contained - no external dependencies beyond standard libraries.

Usage:
    from generate_credit_data import CreditDataGenerator
    
    generator = CreditDataGenerator(n_accounts=1000, seed=42)
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
    print("Error: faker library not installed.")
    print("Please run: pip install faker")
    raise


class CreditDataGenerator:
    """
    Generates synthetic credit data for learning credit risk modeling.
    
    Creates three datasets:
    - accounts: Customer accounts with credit attributes and default labels
    - transactions: Transaction history (similar to Chapter 2)
    - balances: Monthly balance snapshots (similar to Chapter 2)
    
    All datasets include realistic data quality issues for cleaning practice.
    """
    
    def __init__(self, n_accounts=1000, seed=42, start_date='2019-01-01', 
                 prediction_date='2024-01-01', default_rate=0.05):
        """
        Initialize the credit data generator.
        
        Args:
            n_accounts: Number of accounts to generate (default: 1000)
            seed: Random seed for reproducibility (default: 42)
            start_date: Earliest possible account open date
            prediction_date: Date at which we make predictions (for train/test split)
            default_rate: Overall default rate in population (5% is realistic)
        """
        self.n_accounts = n_accounts
        self.seed = seed
        self.start_date = pd.to_datetime(start_date)
        self.prediction_date = pd.to_datetime(prediction_date)
        self.default_rate = default_rate
        
        # Set random seeds for reproducibility
        np.random.seed(seed)
        self.faker = Faker()
        Faker.seed(seed)
        
        # Distribution parameters (realistic proportions)
        self.account_type_probs = [0.40, 0.30, 0.20, 0.10]  # checking, savings, credit, loan
        self.status_probs = [0.80, 0.10, 0.05, 0.05]  # active, closed, suspended, dormant
    
    def generate_accounts(self):
        """
        Generate account master data with credit attributes.
        
        This is the main dataset - combines:
        - Basic account info (ID, type, open date)
        - Credit bureau attributes (FICO, delinquencies, inquiries)
        - Application data (income, employment, DTI)
        - Demographic proxies (age, region) for fairness testing
        - Default labels (target variable for modeling)
        """
        print(f"Generating {self.n_accounts} accounts...")
        
        # Generate IDs
        account_ids = [f"ACC-{str(i).zfill(8)}" for i in range(1, self.n_accounts + 1)]
        customer_ids = [f"CUST-{str(i).zfill(7)}" for i in range(1000001, 1000001 + self.n_accounts)]
        
        # Basic account attributes
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
            'credit_limit': 0,  # Will fill conditionally below
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
        
        # Add credit-specific attributes
        print("  Adding credit bureau attributes...")
        accounts = self._add_credit_attributes(accounts)
        
        print("  Adding application data...")
        accounts = self._add_application_data(accounts)
        
        print("  Adding demographic proxies...")
        accounts = self._add_demographic_proxies(accounts)
        
        print("  Generating default labels...")
        accounts = self._generate_default_labels(accounts)
        
        return accounts
    
    def _add_credit_attributes(self, accounts):
        """
        Add credit bureau attributes to accounts.
        
        These are features you'd get from Equifax, Experian, or TransUnion.
        In production, you'd pull these via API. Here, we generate realistic values.
        """
        n = len(accounts)
        
        # FICO score (300-850 range)
        # Distribution: roughly normal around 690 with some left skew
        fico_scores = np.clip(
            np.random.normal(loc=690, scale=80, size=n),
            300, 850
        ).astype(int)
        
        # Credit history length (months since oldest account)
        # Longer history → generally better credit
        history_length = np.random.gamma(shape=3, scale=20, size=n).astype(int)
        history_length = np.clip(history_length, 3, 360)  # 3 months to 30 years
        
        # Number of credit inquiries in last 6 months
        # Multiple inquiries suggest credit shopping (or desperation)
        inquiries = np.random.poisson(lam=1.5, size=n)
        inquiries = np.clip(inquiries, 0, 15)
        
        # Number of delinquencies (30+ days past due) in last 24 months
        # Most people: 0, some: 1-2, problem borrowers: 3+
        delinquencies = np.random.poisson(lam=0.3, size=n)
        
        # Total credit utilization (balance / limit across all revolving accounts)
        # Good: <30%, acceptable: 30-50%, concerning: >50%
        utilization = np.random.beta(a=2, b=5, size=n)  # Skewed toward low
        utilization = np.clip(utilization * 100, 0, 150)  # Can exceed 100%
        
        # Number of active credit lines
        num_accounts = np.random.poisson(lam=5, size=n)
        num_accounts = np.clip(num_accounts, 0, 30)
        
        # Add to dataframe
        accounts['fico_score'] = fico_scores
        accounts['credit_history_months'] = history_length
        accounts['num_inquiries_6mo'] = inquiries
        accounts['num_delinquencies_24mo'] = delinquencies
        accounts['credit_utilization_pct'] = np.round(utilization, 1)
        accounts['num_active_credit_lines'] = num_accounts
        
        return accounts
    
    def _add_application_data(self, accounts):
        """
        Add application-time data (income, employment, purpose).
        
        In production, this comes from the loan application form.
        Note: Income is often self-reported and may be inflated.
        """
        n = len(accounts)
        
        # Annual income (log-normal distribution, realistic for US)
        # Median ~$50k, mean ~$60k, some high earners
        income = np.random.lognormal(mean=10.8, sigma=0.6, size=n)
        income = np.clip(income, 15000, 500000)  # $15k - $500k
        
        # Employment status
        employment = np.random.choice(
            ['employed', 'self_employed', 'unemployed', 'retired', 'student'],
            size=n,
            p=[0.70, 0.15, 0.05, 0.07, 0.03]
        )
        
        # Employment tenure (months at current job)
        tenure = np.random.gamma(shape=2, scale=18, size=n).astype(int)
        tenure = np.clip(tenure, 0, 480)  # 0 to 40 years
        
        # For unemployed/students, set tenure to 0
        tenure[np.isin(employment, ['unemployed', 'student'])] = 0
        
        # Debt-to-income ratio (DTI)
        # Total monthly debt payments / gross monthly income
        # Good: <20%, acceptable: 20-36%, risky: >43%
        base_dti = np.random.beta(a=2, b=5, size=n) * 50  # 0-50%
        
        # Adjust DTI based on income (lower income → higher DTI pressure)
        income_factor = np.clip(50000 / income, 0.5, 2.0)
        dti = base_dti * income_factor
        dti = np.clip(dti, 0, 80)
        
        # Add to dataframe
        accounts['annual_income'] = np.round(income, 0).astype(int)
        accounts['employment_status'] = employment
        accounts['employment_tenure_months'] = tenure
        accounts['debt_to_income_ratio'] = np.round(dti, 1)
        
        return accounts
    
    def _add_demographic_proxies(self, accounts):
        """
        Add demographic proxy variables for fairness testing.
        
        ⚠️ CRITICAL ETHICS NOTE:
        ========================
        In production, you would NEVER generate synthetic demographic data.
        You MUST collect REAL demographic information to test for bias.
        
        We're using proxies here (age, geography) because:
        1. This is educational synthetic data
        2. We need attributes that exhibit fairness challenges
        3. Age and location correlate with protected characteristics
        4. This teaches pattern recognition skills for real bias detection
        
        In a real system:
        - Collect race, gender, ethnicity through voluntary self-identification
        - Store securely and separately from modeling data
        - Use ONLY for fairness testing, never as model features
        - Follow EEOC/CFPB guidance on data collection
        
        For this chapter, our proxies are:
        - Age (legal to consider, but disparate impact concerns exist)
        - ZIP code/Region (proxy for race/SES due to residential segregation)
        - Account tenure (proxy for financial stability)
        """
        n = len(accounts)
        
        # Age (18-80, skewed toward younger)
        age = np.random.gamma(shape=3, scale=12, size=n) + 18
        age = np.clip(age, 18, 80).astype(int)
        
        # Region (proxy for segregated neighborhoods)
        # In reality, ZIP codes correlate with race/income due to historical redlining
        # We create 3 "regions" with different socioeconomic conditions:
        # - Region A: Low-default area (simulates affluent neighborhoods)
        # - Region B: Medium-default area (simulates mixed demographics)
        # - Region C: High-default area (simulates historically divested areas)
        regions = np.random.choice(['A', 'B', 'C'], size=n, p=[0.4, 0.4, 0.2])
        
        # Generate ZIP codes by region
        zip_codes = []
        for region in regions:
            if region == 'A':
                zip_code = np.random.randint(90000, 90100)  # Fictional high-income
            elif region == 'B':
                zip_code = np.random.randint(80000, 80100)  # Fictional middle-income
            else:  # C
                zip_code = np.random.randint(70000, 70100)  # Fictional low-income
            zip_codes.append(str(zip_code))
        
        # Add to dataframe
        accounts['age'] = age
        accounts['zip_code'] = zip_codes
        accounts['region'] = regions  # For analysis; wouldn't have this label in real data
        
        # Calculate account tenure (months since account opened)
        accounts['account_tenure_months'] = (
            (self.prediction_date - pd.to_datetime(accounts['open_date'])).dt.days / 30
        ).astype(int)
        
        return accounts
    
    def _generate_default_labels(self, accounts):
        """
        Generate default labels (target variable).
        
        This is the Y in our supervised learning problem.
        A "default" means the borrower failed to repay (typically 90+ days past due).
        
        We create realistic default probabilities based on credit characteristics:
        - Lower FICO → higher default risk
        - Higher DTI → higher default risk
        - More delinquencies → higher default risk
        - Younger age → slightly higher risk (less established)
        - Region C → higher risk (simulates structural disadvantage)
        
        ⚠️ THE FAIRNESS CHALLENGE:
        ==========================
        Notice that Region C has a 1.5x default rate multiplier.
        
        Question: Is this capturing real risk, or encoding discrimination?
        
        - One view: "Region C genuinely has higher defaults due to economic factors"
        - Another view: "Region C has higher defaults BECAUSE of historical discrimination;
                        using it perpetuates inequality"
        
        This is the central tension in fair lending. Section 3 will teach you
        how to measure and address this disparate impact.
        """
        n = len(accounts)
        
        # Base default probability from FICO score
        # FICO 300 → ~40% default prob, FICO 850 → ~0.5% default prob
        fico_factor = (850 - accounts['fico_score']) / 550  # 0 to 1
        base_prob = 0.005 + fico_factor * 0.395  # 0.5% to 40%
        
        # Adjust for DTI
        dti_factor = accounts['debt_to_income_ratio'] / 100
        base_prob *= (1 + dti_factor)
        
        # Adjust for delinquency history (strong predictor)
        delinq_factor = 1 + (accounts['num_delinquencies_24mo'] * 0.5)
        base_prob *= delinq_factor
        
        # Adjust for credit utilization
        util_factor = accounts['credit_utilization_pct'] / 100
        base_prob *= (1 + util_factor * 0.3)
        
        # Adjust for age (younger → slightly higher risk)
        age_factor = np.clip((40 - accounts['age']) / 40, 0, 1)
        base_prob *= (1 + age_factor * 0.2)
        
        # ⚠️ CRITICAL: Regional effect (creates fairness challenge)
        # This is where disparate impact emerges
        # Region C has 1.5x default rate even after controlling for credit factors
        region_multiplier = accounts['region'].map({'A': 0.8, 'B': 1.0, 'C': 1.5})
        base_prob *= region_multiplier
        
        # Cap probabilities and scale to target default rate
        default_prob = np.clip(base_prob, 0, 0.5)
        current_mean = default_prob.mean()
        default_prob *= (self.default_rate / current_mean)
        default_prob = np.clip(default_prob, 0, 0.5)
        
        # Generate actual defaults (Bernoulli trials)
        defaults = np.random.binomial(1, default_prob)
        
        # Add default dates for those who defaulted
        # Defaults occur within 12 months of prediction date
        default_dates = []
        for i, defaulted in enumerate(defaults):
            if defaulted == 1:
                days_until = np.random.randint(30, 365)
                default_date = self.prediction_date + timedelta(days=days_until)
            else:
                default_date = None
            default_dates.append(default_date)
        
        # Add to dataframe
        accounts['default_probability'] = np.round(default_prob, 4)  # For analysis only
        accounts['defaulted'] = defaults  # TARGET VARIABLE
        accounts['default_date'] = default_dates
        
        # Print summary statistics
        print(f"    Generated {defaults.sum()} defaults ({defaults.sum()/n*100:.1f}% default rate)")
        print(f"    Default rate by region:")
        print(f"      Region A: {defaults[accounts['region']=='A'].mean():.1%}")
        print(f"      Region B: {defaults[accounts['region']=='B'].mean():.1%}")
        print(f"      Region C: {defaults[accounts['region']=='C'].mean():.1%} ⚠️")
        
        return accounts
    
    def generate_transactions(self, accounts):
        """
        Generate transaction history for accounts.
        
        Similar to Chapter 2's transaction generation, but adapted for credit data.
        Transactions provide behavioral signals for feature engineering.
        """
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
            else:  # active or suspended
                n_txns = np.random.poisson(lam=10)
            
            for _ in range(n_txns):
                # Transaction date between open date and prediction date
                days_range = (self.prediction_date.date() - open_date).days
                if days_range <= 0:
                    continue
                    
                txn_date = open_date + timedelta(days=np.random.randint(0, days_range))
                
                # Transaction type depends on account type
                if account_type in ['checking', 'savings']:
                    txn_type = np.random.choice(
                        ['purchase', 'withdrawal', 'deposit', 'transfer', 'fee'],
                        p=[0.50, 0.20, 0.15, 0.10, 0.05]
                    )
                else:  # credit_card, loan
                    txn_type = np.random.choice(
                        ['purchase', 'payment', 'fee', 'interest'],
                        p=[0.70, 0.20, 0.05, 0.05]
                    )
                
                # Amount (log-normal for realistic distribution)
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
                        p=[0.35, 0.35, 0.15, 0.10, 0.05]
                    )
                })
                txn_id_counter += 1
        
        return pd.DataFrame(transactions)
    
    def generate_balances(self, accounts):
        """
        Generate monthly balance snapshots for accounts.
        
        Similar to Chapter 2's balance generation.
        Balance history captures financial stability patterns.
        """
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
            
            # Generate monthly snapshots from open date to prediction date
            current_date = open_date
            
            while current_date <= self.prediction_date.date():
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
        Inject realistic data quality problems.
        
        Applies the same types of issues from Chapter 2:
        - Missing values
        - Duplicate records
        - Format inconsistencies
        - Invalid values
        
        These issues test your cleaning pipeline!
        """
        print("Injecting data quality issues...")
        
        accounts = accounts.copy()
        transactions = transactions.copy()
        balances = balances.copy()
        
        # Missing values (~5% in key fields)
        for col in ['fico_score', 'annual_income', 'branch_code']:
            if col in accounts.columns:
                mask = np.random.random(len(accounts)) < 0.05
                accounts.loc[mask, col] = np.nan
        
        # Duplicate records (~2%)
        n_dups = int(len(accounts) * 0.02)
        if n_dups > 0:
            dup_indices = np.random.choice(accounts.index, n_dups, replace=False)
            accounts = pd.concat([accounts, accounts.loc[dup_indices]], ignore_index=True)
        
        # Format inconsistencies in dates (~5%)
        date_mask = np.random.random(len(accounts)) < 0.05
        accounts.loc[date_mask, 'open_date'] = accounts.loc[date_mask, 'open_date'].apply(
            lambda x: x.strftime('%m/%d/%Y') if pd.notnull(x) else x
        )
        
        # Typos in categorical fields (~2%)
        typo_mask = np.random.random(len(accounts)) < 0.02
        accounts.loc[typo_mask, 'account_type'] = accounts.loc[typo_mask, 'account_type'].replace({
            'checking': 'chekcing',
            'savings': 'savngs'
        })
        
        # Invalid values
        # Some FICO scores out of range
        invalid_mask = np.random.random(len(accounts)) < 0.01
        accounts.loc[invalid_mask, 'fico_score'] = np.random.choice([250, 900])
        
        # Some negative incomes (data entry error)
        invalid_mask = np.random.random(len(accounts)) < 0.01
        accounts.loc[invalid_mask, 'annual_income'] = -abs(accounts.loc[invalid_mask, 'annual_income'])
        
        # DTI over 100% (mathematically possible but suspicious)
        invalid_mask = np.random.random(len(accounts)) < 0.01
        accounts.loc[invalid_mask, 'debt_to_income_ratio'] = np.random.uniform(100, 200, invalid_mask.sum())
        
        return accounts, transactions, balances
    
    def generate_all(self, output_dir='synthetic_credit_data'):
        """
        Generate complete credit dataset.
        
        This is the main entry point. It:
        1. Generates clean data
        2. Injects quality issues
        3. Saves both versions (clean for validation, messy for practice)
        
        Returns:
            dict with all datasets (both clean and messy versions)
        """
        print(f"\n{'='*70}")
        print(f"Credit Data Generator - Starting")
        print(f"{'='*70}\n")
        print(f"Configuration:")
        print(f"  Accounts: {self.n_accounts:,}")
        print(f"  Seed: {self.seed}")
        print(f"  Date range: {self.start_date.date()} to {self.prediction_date.date()}")
        print(f"  Target default rate: {self.default_rate:.1%}\n")
        
        # Step 1: Generate clean data
        print("Step 1: Generating accounts with credit attributes...")
        accounts_clean = self.generate_accounts()
        
        print("\nStep 2: Generating transaction history...")
        transactions_clean = self.generate_transactions(accounts_clean)
        
        print("\nStep 3: Generating balance snapshots...")
        balances_clean = self.generate_balances(accounts_clean)
        
        # Step 2: Inject data quality issues
        print("\nStep 4: Injecting realistic data quality issues...")
        accounts_messy, transactions_messy, balances_messy = self.inject_data_quality_issues(
            accounts_clean.copy(),
            transactions_clean.copy(),
            balances_clean.copy()
        )
        
        # Step 3: Save everything
        print(f"\nStep 5: Saving data to {output_dir}/...")
        self._save_data(output_dir, accounts_clean, transactions_clean, balances_clean,
                       accounts_messy, transactions_messy, balances_messy)
        
        # Summary
        print(f"\n{'='*70}")
        print(f"✅ Credit Data Generation Complete!")
        print(f"{'='*70}\n")
        print(f"📂 Output directory: {output_dir}/")
        print(f"\n📊 Data Summary:")
        print(f"   Accounts:     {len(accounts_clean):,}")
        print(f"   Transactions: {len(transactions_clean):,}")
        print(f"   Balances:     {len(balances_clean):,}")
        print(f"   Default rate: {accounts_clean['defaulted'].mean():.1%}")
        print(f"\n📁 File Structure:")
        print(f"   {output_dir}/")
        print(f"   ├── raw/              ← Use this for exercises (has quality issues)")
        print(f"   │   ├── accounts.csv")
        print(f"   │   ├── transactions.csv")
        print(f"   │   └── balances.csv")
        print(f"   └── clean/            ← Reference data (for validation)")
        print(f"       ├── accounts_clean.csv")
        print(f"       ├── transactions_clean.csv")
        print(f"       └── balances_clean.csv")
        print(f"\n💡 Next Steps:")
        print(f"   1. Load the messy data from {output_dir}/raw/")
        print(f"   2. Clean it using Chapter 2 techniques")
        print(f"   3. Engineer features (Section 2.1 continues)")
        print(f"   4. Build your credit model (Section 2.2)")
        
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
        """Save all datasets to appropriate directories."""
        # Create directory structure
        raw_dir = Path(output_dir) / 'raw'
        clean_dir = Path(output_dir) / 'clean'
        raw_dir.mkdir(parents=True, exist_ok=True)
        clean_dir.mkdir(parents=True, exist_ok=True)
        
        # Save messy data (for exercises - students work with this)
        accounts_messy.to_csv(raw_dir / 'accounts.csv', index=False)
        transactions_messy.to_csv(raw_dir / 'transactions.csv', index=False)
        balances_messy.to_csv(raw_dir / 'balances.csv', index=False)
        
        # Save clean data (for validation - students check against this)
        accounts_clean.to_csv(clean_dir / 'accounts_clean.csv', index=False)
        transactions_clean.to_csv(clean_dir / 'transactions_clean.csv', index=False)
        balances_clean.to_csv(clean_dir / 'balances_clean.csv', index=False)

