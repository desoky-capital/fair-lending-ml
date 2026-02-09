# Chapter 3: Building the Credit Model

---

## Executive Summary

**What this chapter covers:**
- The unique challenges of credit modeling: legal constraints, explainability requirements, adversarial environments
- Real-world case studies: Apple Card, ZestFinance, UK Mortgage Bias
- Building a complete credit risk model: data generation, feature engineering, baseline model, improvements
- Understanding class imbalance and why naive models fail
- SHAP-based explainability and adverse action notices
- The critical lesson: validation performance ≠ test performance (distribution shift)

**Key takeaways:**
- Credit modeling is legally constrained—fairness is mandated, not optional
- Removing protected characteristics from features doesn't ensure fairness
- Class imbalance requires special handling (SMOTE, class weights, threshold adjustment)
- SMOTE can improve validation metrics but create poorly-calibrated probabilities
- Strong validation performance doesn't guarantee test performance
- Explainability isn't optional—it's a regulatory requirement

**Time estimate:**
- Path A (Hands-On): 6-8 hours (reading + coding)
- Path B (Conceptual): 3-4 hours (reading only)

**What you'll build:**
- Synthetic credit data generator with realistic patterns
- Feature engineering pipeline (68+ features)
- Baseline logistic regression model
- Improved XGBoost model with hyperparameter tuning
- SHAP-based explanation system
- Adverse action notice generator

---

## 3.1 Problem Framing: Credit Risk in Context

### The Apple Card Wake-Up Call

On November 7, 2019, Danish programmer David Heinemeier Hansson posted a thread on Twitter that would ignite a national conversation about algorithmic fairness in lending. His wife, with a higher credit score than his own, had been approved for an Apple Card with a credit limit 1/20th the size of his. When he questioned Goldman Sachs, he was told the algorithm had made the decision—and that even the bank couldn't fully explain why.

Within days, the New York Department of Financial Services launched an investigation. Goldman Sachs insisted their algorithms contained no explicit gender bias. The investigation found no evidence of intentional discrimination. Yet the fundamental questions remained: Was the model actually fair? Could anyone tell?

For those building AI systems in financial services, this case crystallizes a central challenge: **Creating models that are not just accurate and compliant, but demonstrably fair and explainable.**

> 💡 **Key Insight:** The Apple Card case shows that good intentions aren't enough. Even without explicit bias, algorithms can produce discriminatory outcomes—and if you can't explain why decisions were made, you can't prove they were fair.

---

### 3.1.1 From Rules to Algorithms

For most of banking history, credit decisions were made by loan officers using rules of thumb, personal judgment, and sometimes explicit discrimination. The FICO score, introduced in 1989, represented a revolution: a statistical model that could predict creditworthiness more accurately and consistently than human judgment.

The promise was compelling:
- **More consistent decisions** - No more "who you know" determining credit access
- **Better risk prediction** - Lower default rates, lower prices for good borrowers
- **Expanded access** - Previously "unscoreable" populations could be evaluated
- **Reduced discrimination** - Objective algorithms would replace subjective bias

### The Machine Learning Era

The 2010s brought machine learning models that promised even better predictions. But this new power brought new problems:

**Black box opacity.** While a logistic regression might have 20 coefficients you could inspect, a gradient boosted tree ensemble might have thousands of decision rules.

**Proxy discrimination.** Even if you don't include protected characteristics as features, correlated variables—ZIP code, shopping patterns, social connections—can serve as proxies.

**Regulatory uncertainty.** Regulations like ECOA were written for simple scoring models. How do they apply when even creators can't fully explain decisions?

### Where We Are Now

The current consensus: **Better predictions aren't enough. Models must be accurate, fair, and explainable.**

> **Note:** While this chapter focuses on the US regulatory environment (ECOA, Fair Lending laws, CFPB guidance), similar concerns have emerged globally. The EU's GDPR Article 22 establishes rights around automated decision-making, and regulators from the UK to Singapore are developing frameworks for responsible AI in finance.

---

### 3.1.2 What Makes Credit Modeling Different

**Table 3.1: What Makes Credit Models Different**

| Dimension | Credit Models | Fraud Detection | Recommender Systems |
|-----------|---------------|-----------------|---------------------|
| **Stakes** | High (economic access) | High (financial loss) | Low (ad relevance) |
| **Fairness** | Legally mandated | Important | Optional |
| **Explainability** | Required by law | Helpful | Rarely needed |
| **Regulation** | Heavy (ECOA, FCRA) | Moderate | Light |

#### High Stakes, Asymmetric Errors

- **False positive (deny a good borrower):** Someone wrongly excluded from economic opportunity
- **False negative (approve a bad borrower):** Lender loses money, borrower pushed into unsustainable debt

#### Legally Mandated Fairness

The Equal Credit Opportunity Act (ECOA) prohibits discrimination based on:
- Race, color, national origin
- Sex, gender identity
- Religion, marital status
- Age (with exceptions)
- Receipt of public assistance

**Critically, discrimination can be illegal even if unintentional.** The doctrine of "disparate impact" means that if your model systematically disadvantages protected groups—even if race and gender aren't features—you may be violating the law.

#### The Explainability Requirement

US law requires lenders to provide "adverse action notices" to rejected applicants, including the principal reasons for denial (ECOA Section 701). Black box models, no matter how accurate, may be legally unusable if you can't explain their decisions.

> 🎓 **Teaching Note:** This is unique to financial services. Most ML practitioners don't face Department of Justice investigations if their model has differential error rates across demographic groups.

---

### 3.1.3 Real-World Case Studies

#### Case 1: Apple Card and Goldman Sachs (2019)

The problem: Algorithm couldn't explain why identical-seeming applicants got vastly different limits.

The result: NY DFS investigation, Congressional hearings, new CFPB guidance.

**Lesson:** Explainability isn't optional. If you can't explain decisions, your model is legally risky.

#### Case 2: ZestFinance and CFPB (2023) - Proxy Discrimination

The problem: Algorithm used ZIP code, shopping patterns, and device type—proxies that correlated with race.

The result: CFPB enforcement action, multi-million dollar settlement.

**Lesson:** Removing protected characteristics doesn't ensure fairness. You must actively test for disparate impact.

#### Case 3: UK Mortgage Bias (2022) - Measurement Bias

The problem: Credit bureau data quality varied by population. Immigrants had thinner files, which the algorithm treated as negative signals.

**Lesson:** Bias isn't just in algorithms—it's in the data.

#### Common Threads

1. **Good intentions aren't enough**
2. **Opacity creates liability**
3. **Impact matters more than intent**
4. **Documentation is critical**
5. **The bar is rising**

---

## 3.2 Data Preparation

### 3.2.1 Loading Clean Data from Chapter 2

In Chapter 2, we cleaned and validated our banking data. Now we load those clean files as our starting point:

```python
import pandas as pd
import numpy as np

# Load the clean data produced in Chapter 2
accounts = pd.read_csv('data_mart_clean/data/accounts_clean.csv')
transactions = pd.read_csv('data_mart_clean/data/transactions_clean.csv')
balances = pd.read_csv('data_mart_clean/data/balances_clean.csv')

# Parse dates
accounts['open_date'] = pd.to_datetime(accounts['open_date'])
transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
balances['balance_date'] = pd.to_datetime(balances['balance_date'])

print(f"Loaded {len(accounts):,} accounts")
print(f"Loaded {len(transactions):,} transactions")
print(f"Loaded {len(balances):,} balance records")
```

The clean accounts file includes:
- **Credit features:** FICO score, delinquencies, inquiries, income, DTI
- **Demographic attributes:** age, region (for fairness testing)
- **Target variable:** `defaulted` (0 or 1)
- **Built-in fairness challenge:** Region C has ~2.3x the default rate of Region A

### 3.2.2 Feature Engineering

From raw transaction and balance data, we engineer behavioral features. This aggregation happens **before** the train/test split to ensure all datasets have the same columns.

```python
# Engineer balance features per account
balance_features = balances.groupby('account_id').agg(
    avg_balance_3mo=('available_balance', lambda x: x.tail(3).mean()),
    avg_balance_6mo=('available_balance', lambda x: x.tail(6).mean()),
    avg_balance_12mo=('available_balance', 'mean'),
    balance_volatility=('available_balance', 'std'),
    min_balance=('available_balance', 'min'),
    max_balance=('available_balance', 'max')
).reset_index()

# Engineer transaction features per account
txn_features = transactions.groupby('account_id').agg(
    txn_count=('transaction_id', 'count'),
    avg_txn_amount=('amount', 'mean'),
    total_spent=('amount', lambda x: x[x > 0].sum()),
    total_received=('amount', lambda x: abs(x[x < 0].sum())),
    txn_volatility=('amount', 'std')
).reset_index()

# Merge everything into one modeling-ready dataset
data = accounts.merge(balance_features, on='account_id', how='left')
data = data.merge(txn_features, on='account_id', how='left')

# Fill NaN for accounts with no transactions/balances
data = data.fillna(0)

print(f"Final dataset: {data.shape[1]} features, {len(data):,} accounts")
```

**Balance Features:**
- Average balance (3mo, 6mo, 12mo)
- Balance volatility and trend
- Minimum balance periods

**Transaction Features:**
- Transaction frequency and amounts
- Spending patterns by category
- Recent vs. historical behavior

**Final dataset:** 68 features (24 original + 44 engineered)

### 3.2.3 Train/Validation/Test Split

**Critical:** We use temporal splits, not random splits.

```python
# Temporal split based on account open date
train = data[data['open_date'] < '2022-07-01']  # ~80%
val = data[(data['open_date'] >= '2022-07-01') & 
           (data['open_date'] < '2023-04-01')]   # ~10%
test = data[data['open_date'] >= '2023-04-01']  # ~10%
```

**Why temporal?** Random splits leak future information. In production, you only have past data to predict future outcomes.

---

## 3.3 Baseline Model

### 3.3.1 The Class Imbalance Problem

```python
print(f"Default rates:")
print(f"  Train: {train['defaulted'].mean():.1%}")  # ~5%
print(f"  Val:   {val['defaulted'].mean():.1%}")    # ~3%
print(f"  Test:  {test['defaulted'].mean():.1%}")   # ~6%
```

**The problem:** Only ~5% of accounts default. A naive model can get 95% accuracy by predicting "no default" for everyone—but catches 0% of actual defaults!

> 💡 **Key Insight:** In credit, failing to detect defaults is catastrophic. Accuracy is misleading. A 96% accurate model that catches no defaults is useless.

### 3.3.2 Handling Class Imbalance

**Three approaches:**

**Table 3.2: Class Imbalance Handling Approaches**

| Approach | How It Works | Trade-off |
|----------|--------------|-----------|
| **Class Weights** | Penalize default misclassifications more | Simple but limited |
| **SMOTE** | Generate synthetic minority examples | Better recall, calibration issues |
| **Threshold Adjustment** | Lower decision threshold | Precision-recall trade-off |

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Before: 5% defaults
# After: 50% defaults (balanced)
```

> 🎓 **Teaching Note:** SMOTE creates synthetic minority examples by interpolating between existing samples. This gives the model more default examples to learn from.

### 3.3.3 Training the Baseline

Before we train our first model, let's define the key metrics we'll use throughout the book:

**Key Metrics:**
- **Accuracy:** Of all predictions, what percentage were correct? Formula: (TP + TN) / total. *Caution:* In imbalanced data, accuracy is misleading—a model predicting "no default" for everyone achieves 95% accuracy but catches zero defaults.
- **Recall (True Positive Rate):** Of all actual defaults, what percentage did the model catch? Formula: TP / (TP + FN). Higher is better for risk detection.
- **Precision:** Of all accounts the model predicted would default, what percentage actually did? Formula: TP / (TP + FP). Higher means fewer false alarms.
- **ROC-AUC:** Measures the model's ability to distinguish between defaulters and non-defaulters across all possible thresholds. Ranges from 0.5 (random guessing) to 1.0 (perfect separation).

Now let's train our baseline logistic regression on the SMOTE-balanced data:

```python
from sklearn.linear_model import LogisticRegression

model_smote = LogisticRegression(max_iter=1000, random_state=42)
model_smote.fit(X_train_smote, y_train_smote)
```

**Baseline performance (evaluated on validation data—NOT the SMOTE training data):**
- Recall: 31% (catches 1/3 of defaults)
- Precision: 3% (97 false alarms per correct detection)
- ROC-AUC: 0.524 (barely better than random)

> 💡 **Why evaluate on validation, not training?** SMOTE balanced the training data to 50/50, but the real world (and our validation set) still has ~5% defaults. Evaluating on validation shows how the model performs on realistic data.

### 3.3.4 Initial Fairness Assessment

```python
# Check default rates by region
fairness_stats = val.groupby('region').agg({
    'defaulted': 'mean',
    'predicted_default': 'mean'
})
```

**Red flag:** Region C gets predicted to default at 2x the rate of Region A. This disparate impact will be addressed in Chapter 4.

---

## 3.4 Model Improvement

### 3.4.1 Better Algorithms

Our baseline logistic regression with SMOTE achieved only 0.524 ROC-AUC—barely better than guessing. Logistic regression fits a linear decision boundary, but credit risk is rarely that simple. Let's try more powerful algorithms.

**Random Forest** builds hundreds of decision trees, each trained on a random subset of data and features, then averages their predictions:

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',  # Handle imbalance by penalizing minority class errors more
    random_state=42
)
rf_model.fit(X_train, y_train)
```

**XGBoost (Extreme Gradient Boosting)** builds trees sequentially, where each new tree specifically focuses on correcting the errors of previous trees:

```python
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=19,  # Handle imbalance: ratio of non-defaults to defaults
    random_state=42
)
xgb_model.fit(X_train, y_train)
```

> 💡 **Different Imbalance Strategies:** Notice we use three different approaches across our models:
> - **Logistic Regression:** SMOTE (creates synthetic default examples *before* training)
> - **Random Forest:** `class_weight='balanced'` (penalizes errors on defaults more *during* training)
> - **XGBoost:** `scale_pos_weight=19` (same idea as class_weight, different syntax)
>
> Class weights and scale_pos_weight avoid SMOTE's calibration issues because they don't change the data—they just make each default error hurt 19x more.

XGBoost delivered the best validation performance, so we selected it as our final model.

### 3.4.2 Feature Selection

Not all 68 features help. We select top 30 by importance. Here, `feature_cols` refers to the list of all feature column names used during training:

```python
# Get feature importance from XGBoost
feature_cols = [col for col in data.columns if col not in ['account_id', 'defaulted', 'region']]

importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

top_features = importance.head(30)['feature'].tolist()
```

### 3.4.3 Hyperparameter Tuning

**Hyperparameter tuning** finds the best model settings. Unlike model parameters (learned during training), hyperparameters are choices we make *before* training:
- `max_depth`: How deep each tree can grow (deeper = more complex patterns, higher overfit risk)
- `learning_rate`: How fast the model learns from errors (slower = more stable, needs more trees)
- `n_estimators`: How many trees to build (more = better fit, slower training)

GridSearchCV tries every combination and picks the one with the best ROC-AUC:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 200]
}

grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='roc_auc')
grid_search.fit(X_train, y_train)  # Note: using original data, not SMOTE

# Best model
best_model = grid_search.best_estimator_
```

### 3.4.4 Validation Performance

**Improved performance (Validation):**
- Precision: 20% (vs 3% baseline) - **6.7x improvement**
- Recall: 19% (vs 31% baseline) - **Different trade-off**
- ROC-AUC: 0.696 (vs 0.524 baseline) - **33% improvement**
- Gini: 0.391 (crosses 0.3 production threshold!)

> 💡 **What is Gini?** The Gini coefficient (calculated as 2 × ROC-AUC − 1) scales ROC-AUC to a 0-to-1 range where 0 means random and 1 means perfect discrimination. Credit industry typically requires Gini ≥ 0.30 for production models. Our 0.391 clears this threshold—on validation data.

### 3.4.5 The Test Set Reality Check

**CRITICAL FINDING:** Model fails on test set.

```
Test Set Performance:
  Precision: 0%
  Recall: 0%
  ROC-AUC: 0.579

Root Cause: Distribution shift
  Validation probabilities: median 0.14
  Test probabilities: median 0.02 (7x lower!)
```

**What is distribution shift?** The test data comes from a later time period with different borrower characteristics—the *feature distributions* (income levels, credit patterns, economic conditions) shifted between validation and test periods. This caused the model's predicted probabilities to drop dramatically. The 0.25 threshold that worked on validation no longer separates defaulters from non-defaulters on test data.

> 💡 **Key Insight:** This isn't a coding error—it's a fundamental challenge in ML. Models learn patterns from training data; when the real world changes, those patterns may not hold. This is why monitoring (Chapter 4) and temporal splits are essential.

### 3.4.6 Key Lessons from Model Improvement

1. **Validation ≠ Test** - Strong validation performance doesn't guarantee test performance
2. **SMOTE has trade-offs** - Improves validation metrics but may create poorly-calibrated probabilities
3. **Distribution shift is real** - Temporal differences require continuous monitoring
4. **ROC-AUC can be misleading** - Good AUC with bad calibration = unusable model
5. **Honest evaluation matters** - Reporting failures teaches more than hiding them

---

## 3.5 Explainability

### 3.5.1 Why Explainability Matters

1. **Debugging** - Understand why the model failed
2. **Trust** - Stakeholders can see which features drive decisions
3. **Compliance** - Adverse action notices required by law
4. **Fairness** - Foundation for detecting bias
5. **Improvement** - Guides feature engineering

### 3.5.2 SHAP: Global Feature Importance

**SHAP (SHapley Additive exPlanations)** decomposes predictions into feature contributions:

```python
import shap

# best_model is the tuned XGBoost from Section 3.4.3
best_model = grid_search.best_estimator_

# Note: We explain this model despite its test set failure. 
# SHAP isn't just for celebrating good models—it's often MORE valuable
# for understanding WHY a model failed.

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Summary plot shows feature importance
shap.summary_plot(shap_values, X_test)
```

> 💡 **Why X_test, not X_validate?** We generate SHAP explanations on the test set because it represents truly unseen data—the closest approximation to production. The validation set was used during hyperparameter tuning, so the model has been indirectly optimized for it. SHAP on X_test gives us the most honest view of how the model explains itself on new data.

**Top features by SHAP importance:**
1. FICO score
2. Transaction frequency (recent)
3. Channel diversity
4. Average balance (3mo)
5. Credit utilization

### 3.5.3 Understanding the Failure

```python
# Compare probability distributions
print("Probabilities for ACTUAL DEFAULTS:")
print(f"  Validation mean: {val_default_probs.mean():.4f}")
print(f"  Test mean:       {test_default_probs.mean():.4f}")
print(f"  Ratio:           {val_default_probs.mean()/test_default_probs.mean():.1f}x")
```

**The model assigns 3x LOWER probabilities to test defaults!**

The problem isn't overall probability distribution—it's specifically how the model scores actual defaulters.

### 3.5.4 Adverse Action Notices

US law requires lenders to explain denials. SHAP enables this:

```python
def generate_adverse_action_notice(applicant_idx, shap_values, X_test):
    """Generate ECOA-compliant adverse action notice."""
    
    # Get SHAP values for this applicant
    applicant_shap = shap_values[applicant_idx]
    
    # Find top factors contributing to denial
    shap_df = pd.DataFrame({
        'feature': X_test.columns,
        'shap_value': applicant_shap,
        'feature_value': X_test.iloc[applicant_idx].values
    })
    
    top_negative = shap_df.nlargest(4, 'shap_value')
    
    notice = """
    ADVERSE ACTION NOTICE
    
    Your credit application has been DENIED.
    
    PRINCIPAL REASONS FOR THIS DECISION:
    """
    
    for i, row in top_negative.iterrows():
        notice += f"\n{i+1}. {row['feature'].replace('_', ' ').title()}"
        notice += f"\n   Your value: {row['feature_value']:.2f}"
    
    return notice
```

### 3.5.5 Model Card

A model card documents everything stakeholders need to know:

```markdown
## Model Card: Credit Risk Prediction

### Model Details
- Type: XGBoost Classifier
- Features: 30 (from 68 engineered)
- Training: SMOTE-balanced data

### Intended Use
- Credit decisioning for consumer loans
- NOT for final approval (human review required)

### Performance Metrics

**Table 3.3: Model Performance Across Datasets**

| Dataset | ROC-AUC | Precision | Recall |
|---------|---------|-----------|--------|
| Train   | 0.89    | 35%       | 42%    |
| Val     | 0.70    | 20%       | 19%    |
| Test    | 0.58    | 0%        | 0%     |

### Known Limitations
⚠️ CRITICAL: Model fails on test set due to distribution shift
⚠️ Requires probability recalibration before production use
⚠️ Disparate impact on Region C (2x higher denial rate)

### Ethical Considerations
- Fairness testing incomplete (see Chapter 4)
- Proxy discrimination risk not fully evaluated
```

---

## Key Takeaways

### Technical Lessons

1. **Class imbalance requires special handling** - Naive models achieve high accuracy but 0% recall

2. **SMOTE improves validation but may hurt generalization** - Creates synthetic data that doesn't match real distribution

3. **Validation ≠ Test** - Always evaluate on held-out test set; validation performance can be misleading

4. **Distribution shift breaks models** - Temporal differences require monitoring and recalibration

5. **ROC-AUC can be misleading** - Good AUC with bad calibration = unusable model

### Process Lessons

6. **Use temporal splits, not random** - Prevents future information leakage

7. **Test set is sacred** - Evaluate only once at the end

8. **Document everything** - Model cards, metadata, warnings

### Conceptual Lessons

9. **Credit modeling is legally constrained** - Fairness mandated by law, not optional

10. **Explainability is required** - ECOA demands adverse action notices

11. **Good intentions aren't enough** - Must actively test for disparate impact

12. **Honest evaluation matters** - Reporting failures teaches more than hiding them

---

## Connecting to Chapter 4

This chapter built models but revealed critical problems:

- **Distribution shift** causes test failure
- **Disparate impact** affects Region C
- **Calibration issues** make probabilities unusable

**Chapter 4 (Fairness & Compliance) will address:**
- Formal fairness metrics (DIR, SPD, EOD)
- Bias mitigation techniques
- How to balance accuracy and fairness
- Production monitoring for fairness

---

## Teaching Notes

### Learning Objectives

By the end of this chapter, learners should be able to:

**LO1: Understand Credit Context**
- Explain why credit modeling differs from typical ML
- Identify regulatory requirements (ECOA, FCRA, SR 11-7)
- Articulate the fairness-accuracy trade-off

**LO2: Handle Class Imbalance**
- Implement SMOTE, class weights, and threshold adjustment
- Evaluate trade-offs between approaches
- Understand why accuracy is misleading

**LO3: Build and Evaluate Models**
- Engineer features from banking data
- Train and tune classification models
- Use credit-specific metrics (ROC-AUC, Gini, KS)

**LO4: Explain Predictions**
- Apply SHAP for global and local explanations
- Generate regulatory-compliant adverse action notices
- Create model documentation (model cards)

**LO5: Recognize Limitations**
- Identify distribution shift and its consequences
- Understand why validation performance doesn't guarantee test performance
- Document model limitations honestly

### Discussion Questions

1. **The Apple Card Question:** Goldman Sachs claimed their algorithm didn't use gender. How could it still produce gender-biased outcomes?

2. **The SMOTE Dilemma:** SMOTE improved our validation metrics but hurt test performance. When should you use SMOTE? When should you avoid it?

3. **Threshold Selection:** We optimized the threshold on validation data, but it failed on test data. How would you handle threshold selection in production?

4. **Explainability Trade-offs:** More complex models (XGBoost) often perform better but are harder to explain. How do you balance accuracy and explainability?

5. **Documentation Ethics:** Is it ethical to deploy a model with known limitations if you document them? Where's the line?

### Suggested Exercises

**Exercise 1: Alternative Imbalance Handling (Intermediate)**

Try class weights instead of SMOTE. Compare validation and test performance. Which approach generalizes better?

**Exercise 2: Feature Engineering (Intermediate)**

Add new features from transaction data (e.g., weekend spending ratio, late-night transactions). Do they improve model performance?

**Exercise 3: Calibration (Advanced)**

Apply Platt scaling or isotonic regression to calibrate probabilities. Does it fix the test set failure?

**Exercise 4: Adverse Action Notices (Applied)**

Generate adverse action notices for 10 denied applicants. Are the explanations reasonable? Would a customer understand them?

### Key Terms Introduced

**Table 3.4: Key Terms - Credit Modeling**

| Term | Definition |
|------|------------|
| **ECOA** | Equal Credit Opportunity Act - prohibits credit discrimination |
| **Disparate impact** | Disproportionate harm to protected groups, even without intent |
| **SMOTE** | Synthetic Minority Oversampling Technique - creates synthetic examples |
| **Class imbalance** | When one class is much rarer than others (e.g., 5% defaults) |
| **Distribution shift** | When test data differs from training data |
| **Calibration** | Whether predicted probabilities match actual frequencies |
| **SHAP** | SHapley Additive exPlanations - decomposes predictions into feature contributions |
| **Adverse action notice** | Legal requirement to explain credit denials |
| **Gini coefficient** | Model discrimination metric (2 × AUC - 1); production minimum ~0.3 |
| **KS statistic** | Maximum separation between cumulative distributions |

---

*End of Chapter 3*

---

*Next: Chapter 4 — Fairness & Compliance*
