# Synthetic Banking Dataset

**For: Code, Cash, and Conviction - Chapter 2: Data Foundations**

## Overview

This dataset represents 5 years of banking data from a fictional retail bank called "Atlas Bank." The data includes customer accounts, transaction history, and balance snapshots.

**⚠️ Important**: This dataset contains **intentional data quality issues** for educational purposes. Part of your learning objective is to identify and resolve these issues.
    
## Business Context

Atlas Bank is a mid-sized retail bank serving ~1,000 customers across multiple branches. The bank offers:
- Checking and savings accounts (deposit accounts)
- Credit cards
- Personal loans

This dataset was exported from Atlas Bank's legacy core banking system, which has known data quality problems that have accumulated over the years. Your task is to clean and prepare this data for regulatory reporting and analytical use.

## Dataset Files

### `accounts.csv` (~1,000 rows)

Customer account master data. Each row represents one account.

| Column | Description | Expected Values |
|--------|-------------|-----------------|
| `account_id` | Unique account identifier | ACC-{8 digits}, e.g., ACC-00000001 |
| `customer_id` | Customer identifier | CUST-{7 digits}, e.g., CUST-1000001 |
| `account_type` | Type of account | checking, savings, credit_card, loan |
| `open_date` | Date account was opened | YYYY-MM-DD |
| `credit_limit` | Credit limit (for credit accounts) | 0 for deposit accounts, >0 for credit |
| `status` | Current account status | active, closed, suspended, dormant |
| `branch_code` | Branch where account was opened | BR-{3 digits}, or NULL for online |

**Sample row:**
```csv
ACC-00000001,CUST-1000001,checking,2020-01-15,0,active,BR-101
```

### `transactions.csv` (~10,000 rows)

Transaction history for all accounts. Each row represents one transaction.

| Column | Description | Expected Values |
|--------|-------------|-----------------|
| `transaction_id` | Unique transaction identifier | TXN-{10 digits} |
| `account_id` | Foreign key to accounts | Must reference existing account_id |
| `transaction_date` | Date transaction occurred | YYYY-MM-DD |
| `transaction_time` | Time transaction occurred | HH:MM:SS |
| `amount` | Transaction amount | Negative for debits, positive for credits |
| `transaction_type` | Type of transaction | purchase, withdrawal, deposit, transfer, fee, payment, interest |
| `merchant_category` | Category of merchant | grocery, restaurant, fuel, retail, etc. |
| `merchant_name` | Merchant name | Free text, NULL for non-purchase transactions |
| `channel` | Transaction channel | online, mobile, atm, branch, phone |

**Sample row:**
```csv
TXN-0000000001,ACC-00000001,2020-01-16,09:23:45,-45.50,purchase,grocery,Whole Foods Market,mobile
```

### `balances.csv` (~5,000 rows)

Monthly balance snapshots for all accounts. Each row represents the end-of-month balance for one account.

| Column | Description | Expected Values |
|--------|-------------|-----------------|
| `account_id` | Foreign key to accounts | Must reference existing account_id |
| `balance_date` | Date of balance snapshot | YYYY-MM-DD (typically last day of month) |
| `available_balance` | Balance available for immediate use | Any numeric value |
| `ledger_balance` | Ledger (book) balance | Should be >= available_balance |
| `overdraft_count` | Number of overdrafts in the month | 0 or positive integer |

**Sample row:**
```csv
ACC-00000001,2020-01-31,1250.50,1250.50,0
```

## Known Data Quality Issues

⚠️ **Discovery Exercise**: This list is intentionally incomplete. Part of your learning objective is to systematically explore the data and document ALL issues you find, including those not mentioned here.

The following issues are known to exist in this dataset:
- **Missing values** in required fields
- **Inconsistent formatting** (dates, text fields)
- **Duplicate records**
- **Invalid values** that violate business rules
- **Referential integrity violations** (foreign key mismatches)
- **Data entry errors** (typos, incorrect values)

Your data cleaning pipeline should identify and handle all of these issues appropriately.

## Business Rules

When cleaning this data, you must enforce these business rules:

### Accounts
1. Every account must have a unique, non-null `account_id`
2. `account_type` must be one of: checking, savings, credit_card, loan
3. `open_date` must be a valid date (not in the future)
4. `credit_limit` should be 0 for checking/savings, >0 for credit_card/loan
5. `status` must be one of: active, closed, suspended, dormant

### Transactions
1. Every transaction must have a unique, non-null `transaction_id`
2. `account_id` must reference an existing account in `accounts.csv`
3. `transaction_date` must be on or after the account `open_date`
4. `transaction_date` must not be in the future
5. Debits (purchases, withdrawals, fees) should be negative amounts
6. Credits (deposits, payments) should be positive amounts
7. `merchant_category` and `merchant_name` are required for purchases, NULL otherwise

### Balances
1. `account_id` must reference an existing account
2. `balance_date` should be a valid date
3. `ledger_balance` should be >= `available_balance` (ledger includes pending holds)
4. `overdraft_count` must be a non-negative integer

## Learning Objectives

By working with this dataset, you should be able to:

1. **Identify** data quality issues systematically (not just by eyeballing)
2. **Document** issues in a structured way (what, how many, why it matters)
3. **Design** appropriate handling strategies for each type of issue
4. **Implement** a reproducible data cleaning pipeline
5. **Validate** that your cleaned data satisfies all business rules
6. **Track lineage** of all transformations for audit purposes

## Getting Started

### Step 1: Initial Exploration
Load each CSV file and perform basic exploratory data analysis:
```python
import pandas as pd

accounts = pd.read_csv('accounts.csv')
transactions = pd.read_csv('transactions.csv')
balances = pd.read_csv('balances.csv')

# Examine structure
print(accounts.info())
print(accounts.describe())

# Look for obvious issues
print(accounts.head(20))
print(accounts.isnull().sum())
```

### Step 2: Systematic Quality Assessment
Create a data quality report that documents:
- Completeness (% missing for each field)
- Uniqueness (duplicates in fields that should be unique)
- Validity (values that violate business rules)
- Consistency (format variations, typos)
- Referential integrity (orphaned foreign keys)

### Step 3: Design Your Pipeline
For each issue you find, decide:
- Should this record be **dropped**?
- Should the value be **corrected** (and how)?
- Should it be **flagged** for manual review?
- Should it be **imputed** (and using what logic)?

Document your rationale for each decision.

### Step 4: Implement and Validate
Build your cleaning pipeline, then validate:
- Do all records satisfy business rules?
- Is the data internally consistent?
- Can you trace every transformation?
- Can someone else reproduce your results?

## Expected Outcomes

After cleaning this data, you should have:

1. **Three clean CSV files** that satisfy all business rules
2. **A data quality report** documenting all issues found
3. **A data dictionary** explaining each field and any assumptions
4. **A lineage log** showing all transformations applied
5. **A validation report** proving the cleaned data is correct

## Tips for Success

✅ **DO:**
- Start with a copy of the raw data (never modify the original)
- Document every assumption and decision
- Create automated tests for business rules
- Keep your pipeline code clean and commented
- Think about how this would work at scale (even though this dataset is small)

❌ **DON'T:**
- Manually edit CSV files in Excel (not reproducible!)
- Delete records without understanding why they're problematic
- Assume missing values mean "zero" or "unknown"
- Silently coerce types without checking for errors
- Skip documentation ("I'll remember what I did...")

## Validation Dataset

A **clean reference version** of this dataset exists but is not provided to you initially. Your instructor may use it to validate your work. Your cleaned data should closely match the reference data, though there may be multiple valid approaches to handling some ambiguous cases.

## Use in Later Chapters

This dataset will be reused throughout the book:
- **Chapter 3**: Credit risk modeling using account and transaction features
- **Chapter 4**: Fraud detection using transaction patterns
- **Chapter 5**: Fairness analysis using customer demographics (to be added)
- **Chapter 6**: Model governance and monitoring

The quality of your Chapter 2 work will directly impact your results in later chapters - garbage in, garbage out!

## Questions to Consider

As you work through this exercise, reflect on:

1. **Regulatory perspective**: If an auditor reviewed your data pipeline, what would they want to see?
2. **Business impact**: How might each data quality issue harm customers or the business?
3. **Automation**: Which cleaning steps could be automated vs. require human judgment?
4. **Monitoring**: In production, how would you detect when new data quality issues arise?
5. **Trade-offs**: When is it acceptable to lose some data (drop records) to ensure quality?

## Getting Help

- Review **Chapter 2** of the textbook for detailed guidance
- Examine the code examples in the chapter's Jupyter notebooks
- Discuss ambiguous cases with your instructor or study group
- Remember: there's often more than one "right" answer for handling messy data

## Citation

If you use this dataset in academic work, please cite:

```
Synthetic Banking Dataset for "Code, Cash, and Conviction: Building Ethical Fintech Systems"
Chapter 2: Data Foundations
[Author names], 2026
```

## License

This synthetic dataset is provided for educational purposes under the MIT License. The data is entirely fictional and does not represent any real individuals or institutions.

---

**Ready to start?** Load the data and begin your systematic exploration. Good luck! 🚀
