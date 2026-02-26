# Code, Capital, and Conscience

## Building Fair Machine Learning Systems in Financial Services

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## 📖 About This Book

This repository contains the complete manuscript, code, and supporting materials for *Code, Capital, and Conscience: Building Fair Machine Learning Systems in Financial Services*.

**The book teaches you to build ML systems that are accurate, fair, and compliant with financial regulations.**

Whether you're a data scientist building models, a compliance officer validating them, or an executive overseeing AI strategy—this book meets you where you are.

---

## 🎯 What You'll Learn

- **Build a complete credit risk model** from raw data to production deployment
- **Measure fairness** using industry-standard metrics (DIR, SPD, EOD, calibration)
- **Mitigate bias** with practical techniques that actually work
- **Explain predictions** using SHAP for regulatory compliance
- **Monitor production systems** for fairness drift
- **Navigate regulations** including ECOA, FCRA, SR 11-7, and the EU AI Act

---

## 📁 Repository Structure

```
building-fair-ml/
│
├── front_matter/
│   ├── preface.md
│   ├── table_of_contents.md
│   ├── list_of_figures.md
│   └── list_of_tables.md
│
├── chapters/
│   ├── chapter1_CONSOLIDATED.md    # Introduction
│   ├── chapter2_CONSOLIDATED.md    # Data Foundations
│   ├── chapter3_CONSOLIDATED.md    # Building the Credit Model
│   ├── chapter4_CONSOLIDATED.md    # Fairness & Compliance
│   └── chapter5_CONSOLIDATED.md    # Conclusion & Future Directions
│
├── appendices/
│   ├── appendix_A_fairness_metrics.md      # Metrics Quick Reference
│   ├── appendix_B_regulatory_reference.md  # Regulatory Quick Reference
│   ├── appendix_C_code_reference.md        # Code Reference
│   └── appendix_D_documentation_templates.md # Templates
│
├── figures/
│   ├── chapter2/                   # 1 figure
│   ├── chapter3/                   # 17 figures
│   └── chapter4/                   # 2 figures
│
├── notebooks/
│   ├── generate_credit_data.py             # Data generator
│   ├── engineer_credit_features.py         # Feature engineering
│   ├── notebook_audit_report.md            # Notebook QA documentation
│   ├── notebook_code_notes.md              # Code review notes
│   ├── ch2_data_pipeline.ipynb     # 25 cells — data quality & features
│   ├── ch3_modeling.ipynb          # 90 cells — modeling & explainability
│   └── ch4_fairness.ipynb          # 64 cells — fairness & monitoring
│
└── archive/                        # Historical drafts (not needed to run)
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/building-fair-ml.git
cd building-fair-ml
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Generate the Data

```bash
cd notebooks
python generate_credit_data.py
```

This creates:
- `synthetic_credit_data/raw/` — Messy data for Chapter 2 exercises
- `synthetic_credit_data/clean/` — Clean data for Chapter 3+ modeling

### 4. Run the Notebooks

```bash
jupyter notebook
```

Open in order: `ch2_data_pipeline.ipynb` → `ch3_modeling.ipynb` → `ch4_fairness.ipynb`

### 5. Start Reading!

- **Hands-on path:** Run the code as you read (15-20 hours)
- **Conceptual path:** Read explanations, skip code (6-8 hours)

---

## 📋 Requirements

```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.6.0
shap>=0.41.0
matplotlib>=3.5.0
seaborn>=0.11.0
faker>=18.0.0
imbalanced-learn>=0.10.0
```

---

## 📊 Book Statistics

| Component | Count |
|-----------|-------|
| Chapters | 5 |
| Appendices | 4 |
| Figures | 20 |
| Tables | 62 |
| Code examples | 80+ |

---

## 🛤️ Reading Paths

### Path A: Hands-On Practitioner
**Time:** 15-20 hours

```
Chapter 1 → Chapter 2 → Chapter 3 → Chapter 4 → Chapter 5
   ↓           ↓           ↓           ↓           ↓
 Setup      Build       Model      Fairness     Apply
           Pipeline   + Calibrate   Testing
```

Run all code, complete exercises, build the full pipeline.

### Path B: Technical Leader
**Time:** 6-8 hours

```
Chapter 1 → Chapter 3 (Key Takeaways) → Chapter 4 → Chapter 5
```

Focus on concepts, skim code, grasp trade-offs.

### Path C: Compliance/Risk Professional
**Time:** 3-4 hours

```
Chapter 1 → Chapter 4 → Chapter 5 → Appendix B → Appendix D
```

Focus on regulatory requirements, documentation, governance.

---

## 📚 Chapter Overview

| Chapter | Title | Focus |
|---------|-------|-------|
| 1 | Introduction | Why fairness matters, book structure |
| 2 | Data Foundations | Data quality, cleaning, documentation |
| 3 | Building the Credit Model | Model development, calibration, SHAP explainability |
| 4 | Fairness & Compliance | Measuring and mitigating bias, monitoring |
| 5 | Conclusion | Key lessons, emerging regulations, culture |

---

## 🔧 Key Code Components

### Data Generation
```python
from generate_credit_data import CreditDataGenerator

generator = CreditDataGenerator(n_accounts=1000, seed=42)
data = generator.generate_all()
```

### Fairness Metrics
```python
# Disparate Impact Ratio
dir_score = (y_pred[unprivileged] == 0).mean() / (y_pred[privileged] == 0).mean()

# Must be >= 0.80 to pass 4/5ths rule
```

### SHAP Explanations
```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Mohamed Desoky, PhD, MBA**

Academic Director, Wake Forest University School of Professional Studies

- Background in investment banking, engineering, and academia
- Focus on fintech, AI ethics, and responsible ML
- US Army veteran

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

## 📬 Contact

- **Issues:** Use GitHub Issues for bug reports and feature requests
- **Email:** [your email]
- **LinkedIn:** [your LinkedIn]

---

## ⭐ Citation

If you use this book or code in your research or teaching, please cite:

```bibtex
@book{desoky2026codecapital,
  title={Code, Capital, and Conscience: Building Fair Machine Learning Systems in Financial Services},
  author={Desoky, Mohamed},
  year={2026},
  publisher={Self-published}
}
```

---

*Built with ❤️ for fairer financial systems*
