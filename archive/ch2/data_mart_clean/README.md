# Atlas Bank Clean Data Mart

## Overview
This data mart contains cleaned and validated banking data from Atlas Bank's legacy system.
All data has been processed through a rigorous 4-layer data quality pipeline with full audit trail.

## Contents

### Data Files (`data/`)
- `accounts_clean.csv` - 457 customer accounts
- `transactions_clean.csv` - 1,886 transactions
- `balances_clean.csv` - 9,333 monthly balance snapshots

### Documentation (`documentation/`)
- `data_dictionary_clean.json` - Complete schema and metadata
- `quality_improvement_report.json` - Before/after quality metrics

### Audit Logs (`logs/`)
- `data_lineage_report.csv` - Full lineage of all transformations

## Data Quality Summary

**Original Data:**
- Accounts: 1,000 records
- Transactions: 8,755 records  
- Balances: 42,289 records

**Cleaned Data:**
- Accounts: 457 records (543 dropped, 54.3%)
- Transactions: 1,886 records (6,869 dropped, 78.5%)
- Balances: 9,333 records (32,956 dropped, 77.9%)

**Key Improvements:**
- Removed 40,368 total problematic records
- Resolved 26 distinct data quality issues
- Enforced referential integrity across all tables
- Standardized all date, text, and numeric formats
- Documented every transformation decision

## Quality Assurance

This data mart is production-ready and regulator-compliant:
- ✅ Complete data lineage tracking
- ✅ Every decision documented and justified
- ✅ Referential integrity enforced
- ✅ Business rules validated
- ✅ All transformations reproducible

## Usage

Load the cleaned data:
```python
import pandas as pd

accounts = pd.read_csv('data_mart_clean/data/accounts_clean.csv')
transactions = pd.read_csv('data_mart_clean/data/transactions_clean.csv')
balances = pd.read_csv('data_mart_clean/data/balances_clean.csv')
```

Review the audit trail:
```python
lineage = pd.read_csv('data_mart_clean/logs/data_lineage_report.csv')
```

## Generated

**Date:** 2026-01-27 14:26:01
**Pipeline:** Chapter 2 Data Cleaning Pipeline
**Source:** Code, Cash, and Conviction - Building Ethical Fintech Systems
