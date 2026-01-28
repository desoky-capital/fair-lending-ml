"""
Credit Feature Engineering
==========================

Engineers behavioral features from transaction and balance history.
Maintains point-in-time correctness and logs all transformations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineeringLogger:
    """
    Extends Chapter 2's DataQualityLogger pattern for feature engineering.
    Logs every feature calculation for audit trail.
    """
    
    def __init__(self):
        self.log = []
        
    def log_feature(self, feature_name, calculation_logic, n_missing, n_total):
        """Record a feature calculation."""
        self.log.append({
            'timestamp': datetime.now(),
            'feature_name': feature_name,
            'calculation_logic': calculation_logic,
            'missing_values': n_missing,
            'total_records': n_total,
            'missing_rate': n_missing / n_total if n_total > 0 else 0
        })
        
    def get_report(self):
        """Return feature engineering report as DataFrame."""
        return pd.DataFrame(self.log)
    
    def save_report(self, filepath):
        """Save report to CSV."""
        self.get_report().to_csv(filepath, index=False)
        print(f"Feature engineering report saved to {filepath}")


class CreditFeatureEngineer:
    """
    Engineers features from cleaned banking data.
    
    Takes as input:
    - accounts (with credit attributes from data generator)
    - transactions (from data generator)
    - balances (from data generator)
    
    Returns:
    - Enriched accounts DataFrame with engineered features
    """
    
    def __init__(self, prediction_date='2024-01-01'):
        """
        Initialize feature engineer.
        
        Args:
            prediction_date: Date at which features are calculated (no data after this!)
        """
        self.prediction_date = pd.to_datetime(prediction_date)
        self.logger = FeatureEngineeringLogger()
        
    def engineer_all_features(self, accounts, transactions, balances):
        """
        Main entry point: engineer all features.
        
        Args:
            accounts: Account master data (from data generator)
            transactions: Transaction history (from data generator)
            balances: Balance snapshots (from data generator)
            
        Returns:
            accounts DataFrame with engineered features added
        """
        print("Engineering features from transaction and balance history...")
        print(f"Prediction date: {self.prediction_date.date()}")
        print(f"Accounts: {len(accounts):,}")
        print(f"Transactions: {len(transactions):,}")
        print(f"Balances: {len(balances):,}\n")
        
        # Filter data to point-in-time (no data leakage!)
        transactions_pit = self._filter_to_prediction_date(transactions, 'transaction_date')
        balances_pit = self._filter_to_prediction_date(balances, 'balance_date')
        
        print(f"After point-in-time filter:")
        print(f"Transactions: {len(transactions_pit):,}")
        print(f"Balances: {len(balances_pit):,}\n")
        
        # Engineer features by category
        accounts = self._engineer_balance_features(accounts, balances_pit)
        accounts = self._engineer_transaction_features(accounts, transactions_pit)
        accounts = self._engineer_spending_patterns(accounts, transactions_pit)
        accounts = self._engineer_temporal_features(accounts, transactions_pit, balances_pit)
        
        print(f"\nFeature engineering complete!")
        print(f"Total features: {len(accounts.columns)}")
        print(f"New features added: {len(accounts.columns) - len(accounts.columns[accounts.columns.str.startswith('feat_')])}")
        
        return accounts
    
    def _filter_to_prediction_date(self, df, date_column):
        """
        Filter DataFrame to only include records before prediction_date.
        This ensures no data leakage.
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        filtered = df[df[date_column] <= self.prediction_date]
        
        print(f"  {date_column}: Filtered {len(df) - len(filtered):,} future records")
        return filtered
    
    def _engineer_balance_features(self, accounts, balances):
        """
        Engineer features from balance history.
        
        Features capture:
        - Average balance levels
        - Balance volatility (std dev)
        - Trend (increasing or decreasing?)
        - Overdraft frequency
        """
        print("Engineering balance features...")
        
        # Group by account
        balance_agg = balances.groupby('account_id').agg({
            'available_balance': ['mean', 'std', 'min', 'max'],
            'ledger_balance': ['mean'],
            'overdraft_count': ['sum', 'mean']
        }).reset_index()
        
        # Flatten column names
        balance_agg.columns = ['account_id', 
                                'feat_avg_balance', 'feat_balance_std', 
                                'feat_min_balance', 'feat_max_balance',
                                'feat_avg_ledger_balance',
                                'feat_total_overdrafts', 'feat_avg_monthly_overdrafts']
        
        # Calculate balance trend (recent vs. historical)
        recent_balances = balances[
            balances['balance_date'] >= self.prediction_date - timedelta(days=90)
        ].groupby('account_id')['available_balance'].mean()
        
        historical_balances = balances[
            balances['balance_date'] < self.prediction_date - timedelta(days=90)
        ].groupby('account_id')['available_balance'].mean()
        
        balance_trend = (recent_balances - historical_balances).rename('feat_balance_trend')
        
        # Merge features
        accounts = accounts.merge(balance_agg, on='account_id', how='left')
        accounts = accounts.merge(balance_trend, on='account_id', how='left')
        
        # Fill missing (for accounts with no balance history)
        balance_features = [c for c in accounts.columns if c.startswith('feat_')]
        for feat in balance_features:
            n_missing = accounts[feat].isna().sum()
            accounts[feat] = accounts[feat].fillna(0)
            self.logger.log_feature(
                feature_name=feat,
                calculation_logic="Aggregated from balance snapshots, 0 if missing",
                n_missing=n_missing,
                n_total=len(accounts)
            )
        
        print(f"  Added {len(balance_features)} balance features")
        return accounts
    
    def _engineer_transaction_features(self, accounts, transactions):
        """
        Engineer features from transaction volume and patterns.
        
        Features capture:
        - Transaction frequency
        - Average transaction size
        - Deposit vs. withdrawal patterns
        """
        print("Engineering transaction features...")
        
        # Overall transaction stats
        txn_agg = transactions.groupby('account_id').agg({
            'transaction_id': 'count',  # Total transactions
            'amount': ['mean', 'std', 'sum'],
        }).reset_index()
        
        txn_agg.columns = ['account_id', 
                           'feat_num_transactions',
                           'feat_avg_transaction_amount',
                           'feat_transaction_amount_std',
                           'feat_total_transaction_volume']
        
        # Separate debits (negative) and credits (positive)
        debits = transactions[transactions['amount'] < 0].groupby('account_id').agg({
            'transaction_id': 'count',
            'amount': 'sum'
        }).rename(columns={'transaction_id': 'feat_num_debits', 
                           'amount': 'feat_total_debits'})
        
        credits = transactions[transactions['amount'] > 0].groupby('account_id').agg({
            'transaction_id': 'count',
            'amount': 'sum'
        }).rename(columns={'transaction_id': 'feat_num_credits',
                           'amount': 'feat_total_credits'})
        
        # Merge
        accounts = accounts.merge(txn_agg, on='account_id', how='left')
        accounts = accounts.merge(debits, on='account_id', how='left')
        accounts = accounts.merge(credits, on='account_id', how='left')
        
        # Create ratio features
        accounts['feat_credit_to_debit_ratio'] = (
            accounts['feat_total_credits'] / accounts['feat_total_debits'].abs()
        ).replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Transaction frequency (per month)
        account_tenure_months = accounts['account_tenure_months'].replace(0, 1)  # Avoid division by zero
        accounts['feat_transactions_per_month'] = (
            accounts['feat_num_transactions'] / account_tenure_months
        ).fillna(0)
        
        # Fill missing
        txn_features = [c for c in accounts.columns if c.startswith('feat_') 
                        and c not in accounts.columns[:accounts.columns.get_loc('feat_avg_balance')]]
        for feat in txn_features:
            n_missing = accounts[feat].isna().sum()
            accounts[feat] = accounts[feat].fillna(0)
            self.logger.log_feature(
                feature_name=feat,
                calculation_logic="Aggregated from transactions, 0 if missing",
                n_missing=n_missing,
                n_total=len(accounts)
            )
        
        print(f"  Added {len(txn_features)} transaction features")
        return accounts
    
    def _engineer_spending_patterns(self, accounts, transactions):
        """
        Engineer features from spending behavior (purchases only).
        
        Features capture:
        - Spending by category
        - Discretionary vs. essential spending
        - Channel preferences
        """
        print("Engineering spending pattern features...")
        
        # Filter to purchases only
        purchases = transactions[transactions['transaction_type'] == 'purchase'].copy()
        
        if len(purchases) == 0:
            print("  No purchase transactions found, skipping spending features")
            return accounts
        
        # Total spending by category
        category_spending = purchases.groupby(['account_id', 'merchant_category'])['amount'].sum().abs()
        category_spending = category_spending.unstack(fill_value=0)
        category_spending.columns = ['feat_spending_' + c for c in category_spending.columns]
        
        # Merge
        accounts = accounts.merge(category_spending, on='account_id', how='left')
        
        # Discretionary vs. essential spending
        discretionary_categories = ['restaurant', 'entertainment', 'travel', 'retail']
        essential_categories = ['grocery', 'healthcare', 'utilities']
        
        spending_cols = [c for c in accounts.columns if c.startswith('feat_spending_')]
        for col in spending_cols:
            accounts[col] = accounts[col].fillna(0)
        
        accounts['feat_discretionary_spending'] = accounts[
            [f'feat_spending_{c}' for c in discretionary_categories if f'feat_spending_{c}' in accounts.columns]
        ].sum(axis=1)
        
        accounts['feat_essential_spending'] = accounts[
            [f'feat_spending_{c}' for c in essential_categories if f'feat_spending_{c}' in accounts.columns]
        ].sum(axis=1)
        
        accounts['feat_discretionary_ratio'] = (
            accounts['feat_discretionary_spending'] / 
            (accounts['feat_discretionary_spending'] + accounts['feat_essential_spending'])
        ).fillna(0)
        
        # Channel usage (online/mobile vs. branch)
        channel_usage = purchases.groupby(['account_id', 'channel'])['transaction_id'].count()
        channel_usage = channel_usage.unstack(fill_value=0)
        channel_usage.columns = ['feat_channel_' + c for c in channel_usage.columns]
        
        accounts = accounts.merge(channel_usage, on='account_id', how='left')
        
        channel_cols = [c for c in accounts.columns if c.startswith('feat_channel_')]
        for col in channel_cols:
            n_missing = accounts[col].isna().sum()
            accounts[col] = accounts[col].fillna(0)
            self.logger.log_feature(
                feature_name=col,
                calculation_logic="Purchase count by channel, 0 if missing",
                n_missing=n_missing,
                n_total=len(accounts)
            )
        
        # Digital channel preference (online + mobile)
        if 'feat_channel_online' in accounts.columns and 'feat_channel_mobile' in accounts.columns:
            total_channels = accounts[[c for c in accounts.columns if c.startswith('feat_channel_')]].sum(axis=1)
            accounts['feat_digital_channel_pct'] = (
                (accounts['feat_channel_online'] + accounts['feat_channel_mobile']) / total_channels
            ).fillna(0)
        
        print(f"  Added spending pattern features")
        return accounts
    
    def _engineer_temporal_features(self, accounts, transactions, balances):
        """
        Engineer features that capture trends over time.
        
        Features capture:
        - Recent vs. historical behavior (improving or deteriorating?)
        - Seasonality (spending spikes?)
        - Recency (last transaction date)
        """
        print("Engineering temporal features...")
        
        # Calculate features for multiple time windows
        windows = {
            '3mo': 90,
            '6mo': 180,
            '12mo': 365
        }
        
        for window_name, window_days in windows.items():
            window_start = self.prediction_date - timedelta(days=window_days)
            
            # Transaction count in window
            txn_in_window = transactions[
                transactions['transaction_date'] >= window_start
            ].groupby('account_id')['transaction_id'].count()
            
            accounts[f'feat_num_txn_{window_name}'] = accounts['account_id'].map(
                txn_in_window
            ).fillna(0)
            
            # Average balance in window
            bal_in_window = balances[
                balances['balance_date'] >= window_start
            ].groupby('account_id')['available_balance'].mean()
            
            accounts[f'feat_avg_balance_{window_name}'] = accounts['account_id'].map(
                bal_in_window
            ).fillna(0)
        
        # Trend indicators: compare recent (3mo) to historical (12mo)
        accounts['feat_txn_trend'] = (
            accounts['feat_num_txn_3mo'] - accounts['feat_num_txn_12mo'] / 4
        ) / (accounts['feat_num_txn_12mo'] / 4 + 1)  # Avoid division by zero
        
        accounts['feat_balance_trend_3mo_vs_12mo'] = (
            accounts['feat_avg_balance_3mo'] - accounts['feat_avg_balance_12mo']
        ) / (accounts['feat_avg_balance_12mo'].abs() + 1)
        
        # Days since last transaction
        last_txn_date = transactions.groupby('account_id')['transaction_date'].max()
        accounts['feat_days_since_last_txn'] = (
            self.prediction_date - accounts['account_id'].map(last_txn_date)
        ).dt.days.fillna(999)  # 999 = no transactions
        
        # Log temporal features
        temporal_features = [c for c in accounts.columns if any(
            x in c for x in ['_3mo', '_6mo', '_12mo', '_trend', '_since_']
        )]
        for feat in temporal_features:
            n_missing = accounts[feat].isna().sum()
            if n_missing > 0:
                accounts[feat] = accounts[feat].fillna(0)
            self.logger.log_feature(
                feature_name=feat,
                calculation_logic=f"Calculated over time window, 0 if missing",
                n_missing=n_missing,
                n_total=len(accounts)
            )
        
        print(f"  Added {len(temporal_features)} temporal features")
        return accounts
    
    def get_feature_list(self, accounts):
        """
        Return list of all engineered features.
        Useful for model training.
        """
        return [c for c in accounts.columns if c.startswith('feat_')]
    
    def save_feature_documentation(self, accounts, filepath):
        """
        Save comprehensive feature documentation.
        Includes: feature name, data type, missing rate, basic stats.
        """
        feature_cols = self.get_feature_list(accounts)
        
        doc = []
        for feat in feature_cols:
            doc.append({
                'feature_name': feat,
                'data_type': str(accounts[feat].dtype),
                'missing_rate': accounts[feat].isna().mean(),
                'mean': accounts[feat].mean() if pd.api.types.is_numeric_dtype(accounts[feat]) else None,
                'std': accounts[feat].std() if pd.api.types.is_numeric_dtype(accounts[feat]) else None,
                'min': accounts[feat].min() if pd.api.types.is_numeric_dtype(accounts[feat]) else None,
                'max': accounts[feat].max() if pd.api.types.is_numeric_dtype(accounts[feat]) else None,
                'unique_values': accounts[feat].nunique()
            })
        
        doc_df = pd.DataFrame(doc)
        doc_df.to_csv(filepath, index=False)
        print(f"\nFeature documentation saved to {filepath}")
        return doc_df