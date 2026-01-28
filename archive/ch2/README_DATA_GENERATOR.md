# Synthetic Banking Data Generator

## 🚀 Quick Start (For Jupyter Notebook Users)

**The easiest way to generate data is directly in your Jupyter notebook.**

### Step 1: Install Required Package
In a Jupyter notebook cell, run:
```python
!pip install faker
```

### Step 2: Run the Generator
In the next cell:
```python
from generate_banking_data import BankingDataGenerator

# Create generator and generate data
generator = BankingDataGenerator(n_accounts=1000, seed=42)
data = generator.generate_all()
```

### Step 3: Load Your Data
```python
import pandas as pd

# Load the messy data for cleaning exercises
accounts = pd.read_csv('synthetic_banking_data/raw/accounts.csv')
transactions = pd.read_csv('synthetic_banking_data/raw/transactions.csv')
balances = pd.read_csv('synthetic_banking_data/raw/balances.csv')

print(f"✓ Loaded {len(accounts)} accounts")
print(f"✓ Loaded {len(transactions)} transactions")  
print(f"✓ Loaded {len(balances)} balance records")
```

**That's it!** You now have data ready for Chapter 2 exercises.

---

## 📁 What Gets Created

After running the generator, you'll have this folder structure:

```
ch2/
└── synthetic_banking_data/
    ├── raw/                    ← Use this for exercises
    │   ├── accounts.csv        (messy data with quality issues)
    │   ├── transactions.csv    (messy data with quality issues)
    │   └── balances.csv        (messy data with quality issues)
    └── clean/                  ← Reference data for validation
        ├── accounts_clean.csv
        ├── transactions_clean.csv
        └── balances_clean.csv
```

---

## 🎓 For Students

### Which Files Should I Use?

**For Chapter 2 exercises, use the files in `raw/`:**
- These contain intentional data quality problems
- Your job is to clean them!

**The files in `clean/` are reference data:**
- These show what the data should look like after cleaning
- Use them to validate your cleaning pipeline

### What Kind of Problems Will I Find?

The raw data contains realistic issues you'd find in real banking systems:

**In accounts.csv:**
- ~5% missing account IDs
- ~2% duplicate records
- Typos like "chekcing" instead of "checking"
- Mixed date formats (01/15/2020 vs 2020-01-15)
- Invalid branch codes

**In transactions.csv:**
- Missing transaction IDs
- Orphaned records (transactions for accounts that don't exist)
- Future dates (data entry errors)
- Amount outliers (999999.99 - clearly wrong)
- Typos in transaction types

**In balances.csv:**
- Business rule violations (ledger_balance < available_balance)
- Negative overdraft counts
- Missing dates
- String 'NaN' instead of actual null values

---

## ⚙️ Customization Options

Want to generate different data? You can customize:

### Generate More Accounts
```python
generator = BankingDataGenerator(n_accounts=5000, seed=42)
data = generator.generate_all()
```

### Change the Random Seed (for different data)
```python
generator = BankingDataGenerator(n_accounts=1000, seed=999)
data = generator.generate_all()
```

### Change the Start Date
```python
generator = BankingDataGenerator(
    n_accounts=1000, 
    seed=42,
    start_date='2018-01-01'
)
data = generator.generate_all()
```

### Save to a Different Folder
```python
data = generator.generate_all(output_dir='my_custom_data')
```

---

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'faker'"
**Solution:** Install faker in your notebook:
```python
!pip install faker
```

### "ModuleNotFoundError: No module named 'generate_banking_data'"
**Solution:** Use the `%run` command instead of import if your notebook is in a subfolder of the working directory:
```python
from generate_banking_data import BankingDataGenerator
```

### "FileNotFoundError: [Errno 2] No such file or directory: 'accounts.csv'"
**Solution:** Include the full path:
```python
accounts = pd.read_csv('synthetic_banking_data/raw/accounts.csv')
```

### Not Sure What Directory You're In?
**Check with:**
```python
import os
print(os.getcwd())  
```

If you're in the wrong directory:
```python
os.chdir('target-directory')  # target-directory is your desired folder name
```

---

## 💻 For Advanced Users (Command Line)

If you prefer using the terminal/command line:

### Installation
```bash
pip install pandas numpy faker
```

### Generate Data
```bash
python generate_banking_data.py
```

### With Options
```bash
# Generate 2000 accounts
python generate_banking_data.py --n_accounts 2000

# Use different seed
python generate_banking_data.py --seed 123

# Different start date
python generate_banking_data.py --start_date 2018-01-01

# Custom output directory
python generate_banking_data.py --output my_data/

# Combine options
python generate_banking_data.py --n_accounts 500 --seed 999
```

---

## 📊 Dataset Details

### Size (with default settings)
- **Accounts:** ~1,000 records
- **Transactions:** ~10,000 records (avg 10 per account)
- **Balances:** ~40,000 records (monthly snapshots for 3-5 years)

### Distributions

**Account Types:**
- Checking: 40%
- Savings: 30%
- Credit Card: 20%
- Loan: 10%

**Account Status:**
- Active: 80%
- Closed: 10%
- Suspended: 5%
- Dormant: 5%

**Transaction Channels:**
- Online: 35%
- Mobile: 35%
- ATM: 15%
- Branch: 10%
- Phone: 5%

### Reproducibility

Using the same seed always produces identical data:
```python
generator = BankingDataGenerator(seed=42)
data1 = generator.generate_all()

generator = BankingDataGenerator(seed=42)
data2 = generator.generate_all()

# data1 and data2 are identical!
```

---

## 👨‍🏫 For Instructors

### Using in Class

**Option 1: Students generate their own data**
- Walk through the Quick Start steps in class
- Students can customize (different seeds = different data)
- Good for understanding the generation process

**Option 2: Provide pre-generated data**
- Include the CSV files in your course materials
- Students skip generation, go straight to cleaning
- Faster, everyone has identical data

**Option 3: Hybrid approach**
- Provide pre-generated data as backup
- Encourage students to generate their own
- Those who struggle can use the backup

### Dataset Sizes for Different Uses

- **100-500 accounts:** Quick demos, in-class exercises (runs in seconds)
- **1,000-2,000 accounts:** Homework assignments, labs (runs in 10-30 seconds)
- **5,000+ accounts:** Capstone projects, realistic scenarios (may take 1-2 minutes)

### Reusing Across Chapters

This dataset works for:
- **Chapter 2:** Data cleaning and validation
- **Chapter 3:** Credit risk modeling (add credit scores, default labels)
- **Chapter 4:** Fraud detection (add fraud labels)
- **Chapter 5:** Fairness analysis (add demographic data)
- **Chapter 6:** Model governance examples

---

## 📚 Related Files

- **`generate_banking_data.py`** - Jupyter-friendly generator (use with import)
- **`generate_banking_data.py`** - Command-line version
- **`README_DATASET.md`** - Detailed data dictionary and business context
- **`data_dictionary.json`** - Machine-readable schema documentation

---

## ❓ Questions?

**For help with:**
- **Technical issues:** See Troubleshooting section above
- **Data dictionary:** Check `README_DATASET.md`
- **Course questions:** Contact your instructor
- **Chapter content:** Refer to Chapter 2 of "Code, Cash, and Conviction"

---

## 📄 License

MIT License - Free to use for educational purposes

**Attribution:** "Code, Cash, and Conviction: Building Ethical Fintech Systems"

---

## 🎯 Next Steps

Once you have your data generated:

1. **Verify it worked:** Check that `synthetic_banking_data/raw/` has 3 CSV files
2. **Open Chapter 2:** Follow along with Section 2 (Code Walkthrough)
3. **Start cleaning:** Build your data cleaning pipeline!

**Good luck! 🚀**
