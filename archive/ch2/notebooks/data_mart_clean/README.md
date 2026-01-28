# Atlas Bank Clean Data Mart

## Overview
This data mart contains cleaned and validated banking data from Atlas Bank's legacy system.
All data has been processed through a rigorous 4-layer data quality pipeline with full audit trail.

## Contents

### Data Files (`data/`)
- `accounts_clean.csv` - 443 customer accounts
- `transactions_clean.csv` - 1,768 transactions
- `balances_clean.csv` - 9,183 monthly balance snapshots

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
- Accounts: 443 records (557 dropped, 55.7%)
- Transactions: 1,768 records (6,987 dropped, 79.8%)
- Balances: 9,183 records (33,106 dropped, 78.3%)

**Key Improvements:**
- Removed 40,650 total problematic records
- Resolved 24 distinct data quality issues
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

**Date:** 2026-01-08 12:27:00
**Pipeline:** Chapter 2 Data Cleaning Pipeline
**Source:** Code, Cash, and Conviction - Building Ethical Fintech Systems
