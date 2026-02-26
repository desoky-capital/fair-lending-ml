# Chapter 3: Building the Credit Model

---

## Executive Summary

**What this chapter covers:**
- The unique challenges of credit modeling: legal constraints, explainability requirements, adversarial environments
- Real-world case studies: Apple Card, ZestFinance, UK Mortgage Bias
- Building a complete credit risk model: data generation, feature engineering, baseline model, improvements
- Recognizing class imbalance and why naive models fail
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
- Feature engineering pipeline (51 features from credit bureau + transaction data)
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

In Chapter 2, we cleaned and validated our credit data. Now we load those clean files as our starting point:

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

> ⚠️ **Key Insight — Watch Your Window Sizes:** The `.tail(3)` and `.tail(6)` calls above assume each account has at least 3 or 6 months of balance data. But our generator creates accounts with open dates ranging from 2019 to 2024 — an account opened in October 2023 only has ~3 monthly snapshots. For that account, `avg_balance_3mo`, `avg_balance_6mo`, and `avg_balance_12mo` all return the same value, because `.tail(6)` on a 3-row Series silently returns all 3 rows. No error, no warning — just a misleading feature.
>
> In production, you'd want to either (a) filter to accounts with sufficient history, (b) add a `balance_months_available` feature so the model knows how much history it's working with, or (c) set the feature to `NaN` when there's insufficient data. We proceed here for simplicity, but this is exactly the kind of silent data issue Chapter 2 warned us about.

**Balance Features:**
- Average balance (3mo, 6mo, 12mo)
- Balance volatility and trend
- Minimum balance periods

**Transaction Features:**
- Transaction frequency and amounts
- Spending patterns by category
- Recent vs. historical behavior

**Final dataset:** 51 features (credit bureau attributes + engineered transaction and balance features)

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

> 💡 **How One Model Produces a Curve:** A classification model doesn't just output "default" or "no default" — it outputs a *probability* (e.g., 0.73). We then choose a decision threshold: above it, we predict default. By sweeping that threshold from 0.0 to 1.0, we get a different precision/recall pair at each step — and that series of points traces the Precision-Recall curve. The ROC curve works the same way, plotting True Positive Rate vs. False Positive Rate at each threshold. A single model, many thresholds, one curve.

Now let's train our baseline logistic regression on the SMOTE-balanced data:

```python
from sklearn.linear_model import LogisticRegression

model_smote = LogisticRegression(max_iter=1000, random_state=42)
model_smote.fit(X_train_smote, y_train_smote)
```

**Baseline performance (evaluated on validation data—NOT the SMOTE training data):**
- Recall: 31% (catches 1/3 of defaults)
- Precision: 2.7% (97 false alarms per correct detection)
- ROC-AUC: 0.524 (barely better than random)

We also train a class-weighted logistic regression for comparison — it uses `class_weight='balanced'` to penalize default misclassifications more heavily, without creating synthetic data:

```python
model_weighted = LogisticRegression(
    class_weight='balanced', max_iter=1000, random_state=42
)
model_weighted.fit(X_train, y_train)
```

**Class-weighted performance (Validation, threshold 0.40):**
- Recall: 75% (catches 3 of every 4 defaults)
- Precision: 4.3% (still mostly false alarms)
- ROC-AUC: 0.549

> 📌 **Teaching Note:** The class-weighted model's 75% recall looks impressive, but at 4.3% precision it's flagging nearly everyone — that recall is bought with false alarms. We carry both approaches forward for comparison.

> 💡 **Why evaluate on validation, not training?** SMOTE balanced the training data to 50/50, but the real world (and our validation set) still has ~5% defaults. Evaluating on validation shows how the model performs on realistic data.

Let's visualize the baseline model's performance to understand exactly where it's failing. First, the confusion matrix — a table showing what the model got right and wrong:

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate predictions on validation set
y_val_pred = model_smote.predict(X_val)

# Plot confusion matrix
cm = confusion_matrix(y_val, y_val_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Default', 'Default'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix — SMOTE Logistic Regression (Validation Set)')
plt.tight_layout()
plt.savefig('fig_3_1_confusion_matrix.png', dpi=150)
```

*Figure 3.1: Confusion Matrix — SMOTE Logistic Regression (Validation Set)*

The confusion matrix confirms the problem: the model flags many non-defaulters as defaults (high false positives) while still missing most actual defaults (low true positives). With only 3% precision, 97 out of every 100 "default" predictions are wrong.

Now let's examine which features the logistic regression considers most important. Since logistic regression is a linear model, the coefficients directly tell us each feature's influence:

```python
# Extract logistic regression coefficients
import pandas as pd

coef_df = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': model_smote.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

# Plot top 10 features
top_10 = coef_df.head(10).sort_values('coefficient')
colors = ['red' if c < 0 else 'green' for c in top_10['coefficient']]
plt.barh(top_10['feature'], top_10['coefficient'], color=colors)
plt.xlabel('Coefficient (Negative = Lower Default Risk)')
plt.title('Top 10 Most Important Features')
plt.tight_layout()
plt.savefig('fig_3_2_lr_feature_importance.png', dpi=150)
```

*Figure 3.2: Top 10 Most Important Features (Logistic Regression)*

The green bars (positive coefficients) push toward predicting default, while red bars (negative) push away from it. Notice that `credit_history_months` and `credit_utilization_pct` are the strongest predictors — this aligns with domain knowledge. Interestingly, `feat_avg_balance` has a negative coefficient (higher balances reduce default risk), but `feat_avg_balance_3mo` also appears negative, suggesting some multicollinearity between our engineered features.

Next, the ROC and Precision-Recall curves show the model's discrimination ability across all possible decision thresholds:

```python
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Get predicted probabilities
y_val_proba = model_smote.predict_proba(X_val)[:, 1]

# ROC Curve
fpr, tpr, _ = roc_curve(y_val, y_val_proba)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'Model (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — SMOTE Logistic Regression (Validation Set)')
plt.legend()
plt.savefig('fig_3_3_roc_curve.png', dpi=150)
```

*Figure 3.3: ROC Curve — SMOTE Logistic Regression (Validation Set)*

The ROC curve barely lifts above the random baseline diagonal — confirming the 0.524 AUC. The model has almost no ability to distinguish defaulters from non-defaulters.

```python
# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_val, y_val_proba)
ap = auc(recall, precision)
baseline_rate = y_val.mean()
plt.plot(recall, precision, label=f'PR curve (AP = {ap:.3f})')
plt.axhline(y=baseline_rate, color='navy', linestyle='--', 
            label=f'Baseline (random) = {baseline_rate:.3f}')
plt.xlabel('Recall (True Positive Rate)')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve — SMOTE Logistic Regression (Validation Set)')
plt.legend()
plt.savefig('fig_3_4_pr_curve.png', dpi=150)
```

*Figure 3.4: Precision-Recall Curve — SMOTE Logistic Regression (Validation Set)*

The PR curve hugs the baseline — the model's average precision (0.036) is barely above the random baseline (0.034, the default rate in the dataset). For imbalanced problems like ours, the PR curve is often more informative than the ROC curve because it focuses on the minority class.

**Verdict:** Our SMOTE logistic regression baseline is essentially useless. Time to try more powerful models.

### 3.3.4 Initial Fairness Assessment

Before moving on, let's check whether the baseline model treats all regions equally:

```python
# Check default rates by region
fairness_stats = val.groupby('region').agg({
    'defaulted': 'mean',
    'predicted_default': 'mean'
})
print(fairness_stats)
```

**Expected Output:**
```
              defaulted  predicted_default
region                                    
Region A         0.032           0.021
Region B         0.051           0.038
Region C         0.068           0.052
```

**Red flag:** Region C gets predicted to default at higher rates. The Disparate Impact Ratio (Region C / Region A) is 1.16 — technically passing the 4/5ths rule threshold of 0.80.

> 📌 **Teaching Note:** A passing DIR doesn't mean the model is fair. At this point the model is flagging ~30% of accounts per region vs 2–4% actual defaults — it's over-denying everyone equally. Fairness metrics can look acceptable when a model is uniformly bad. A better model with more targeted predictions may reveal larger disparities — which is exactly what Chapter 4 investigates.

---

## 3.4 Model Improvement

### 3.4.1 Better Algorithms

Our baseline logistic regression with SMOTE achieved only 0.524 ROC-AUC—barely better than guessing. Logistic regression fits a linear decision boundary, but credit risk is rarely that simple. We train all improved models using SMOTE-balanced data to give them more default examples to learn from:

**Random Forest** builds hundreds of decision trees, each trained on a random subset of data and features, then averages their predictions:

```python
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Balance training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=42
)
rf_model.fit(X_train_smote, y_train_smote)
```

**XGBoost (Extreme Gradient Boosting)** builds trees sequentially, where each new tree specifically focuses on correcting the errors of previous trees:

```python
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=100, max_depth=5,
    learning_rate=0.1, random_state=42
)
xgb_model.fit(X_train_smote, y_train_smote)
```

> 💡 **Why SMOTE for all models?** We use SMOTE rather than class weights here because with only ~120 default examples out of 3,600 training accounts, even tree-based models struggle to learn default patterns. SMOTE creates enough synthetic defaults for the model to identify feature combinations that signal risk. The trade-off: SMOTE distorts the model's probability scale (calibrated for a 50/50 world, not the real 3.4% default rate), which we'll address with calibration in §3.5.

XGBoost delivered the best validation performance, so we selected it as our final model. Let's compare the three models' feature importance to understand what each one learned:

```python
# Random Forest feature importance
rf_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

rf_importance.head(10).plot.barh(x='feature', y='importance')
plt.title('Top 10 Features — Random Forest')
plt.tight_layout()
plt.savefig('fig_3_5_rf_feature_importance.png', dpi=150)
```

*Figure 3.5: Top 10 Most Important Features (Random Forest)*

Random Forest identifies similar top features to logistic regression (FICO score, credit history) but the relative importance shifts — tree-based models can capture non-linear relationships that logistic regression misses.

Let's see how all three models compare head-to-head on validation data:

```python
from sklearn.metrics import roc_auc_score, precision_score, recall_score

models = {
    'SMOTE LR': (model_smote, X_val, y_val),
    'Random Forest': (rf_model, X_val, y_val),
    'XGBoost': (xgb_model, X_val, y_val)
}

results = []
for name, (model, X, y) in models.items():
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.20).astype(int)  # Threshold optimized on validation
    results.append({
        'model': name,
        'roc_auc': roc_auc_score(y, y_proba),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred)
    })

comparison = pd.DataFrame(results)
print(comparison.to_string(index=False))
```

*Figure 3.6: Model Comparison (Validation Set)*

XGBoost clearly outperforms — its AUC is substantially higher than both the baseline logistic regression and Random Forest. This is why we select it as our model for tuning.

### 3.4.2 Feature Selection

Not all 51 features help. We select the top 30 by importance. Here, `feature_cols` refers to the list of all feature column names used during training:

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

Let's visualize the tuned XGBoost's feature importance:

```python
# Feature importance from tuned XGBoost
tuned_importance = pd.DataFrame({
    'feature': top_features,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

tuned_importance.head(15).plot.barh(x='feature', y='importance')
plt.title('Top 15 Features — Tuned XGBoost')
plt.xlabel('Feature Importance (Gain)')
plt.tight_layout()
plt.savefig('fig_3_7_xgb_feature_importance.png', dpi=150)
```

*Figure 3.7: Top 15 Most Important Features (Tuned XGBoost)*

With the tuned model, we can also optimize the decision threshold. The default 0.50 threshold isn't appropriate for imbalanced data — we need to find where precision and recall are best balanced for our use case:

```python
from sklearn.metrics import precision_recall_curve

y_val_proba = best_model.predict_proba(X_val)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)

# Find threshold that maximizes F1 score
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
best_threshold = thresholds[f1_scores.argmax()]
print(f"Optimal threshold: {best_threshold:.2f}")

# Plot precision and recall vs. threshold
plt.plot(thresholds, precisions[:-1], label='Precision')
plt.plot(thresholds, recalls[:-1], label='Recall')
plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'Best threshold = {best_threshold:.2f}')
plt.xlabel('Decision Threshold')
plt.ylabel('Score')
plt.title('Threshold Optimization — Tuned XGBoost')
plt.legend()
plt.savefig('fig_3_8_threshold_optimization.png', dpi=150)
```

*Figure 3.8: Threshold Optimization — Tuned XGBoost*

As the threshold drops, recall increases (we catch more defaults) but precision decreases (more false alarms). The optimal threshold of ~0.20 balances these trade-offs for our credit risk use case, where missing a default is costlier than a false alarm.

### 3.4.4 Validation Performance

Using the optimized threshold of 0.20:

**Improved performance (Tuned XGBoost, Validation):**
- Precision: 9.5% (vs 2.7% baseline) — still low, but 3.5x improvement
- Recall: 25% (catches 1 in 4 defaults)
- F1: 0.138 (best among all variants)
- ROC-AUC: 0.658 (vs 0.524 baseline) — 26% improvement
- Gini: 0.316 (crosses 0.3 production threshold!)

> 💡 **What is Gini?** The Gini coefficient (calculated as 2 × ROC-AUC − 1) scales ROC-AUC to a 0-to-1 range where 0 means random and 1 means perfect discrimination. Credit industry typically requires Gini ≥ 0.30 for production models. Our 0.316 barely clears this threshold — on validation data.

### 3.4.5 The Test Set Reality Check

Now for the moment of truth. We apply our best model to the held-out test set using the same 0.20 threshold that worked on validation:

```python
from sklearn.metrics import precision_score, recall_score, roc_auc_score

# Apply best model to test set
y_test_proba = best_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= 0.20).astype(int)  # Same threshold from validation

# The moment of truth
print("Test Set Performance:")
print(f"  Precision: {precision_score(y_test, y_test_pred, zero_division=0):.1%}")
print(f"  Recall: {recall_score(y_test, y_test_pred):.1%}")
print(f"  ROC-AUC: {roc_auc_score(y_test, y_test_proba):.3f}")
```

**Test performance (threshold 0.20):**
- Precision: 14.3% — 1 in 7 flagged accounts actually defaults
- Recall: 10% — catches 3 of 30 actual defaults
- ROC-AUC: 0.683 — genuine improvement over baseline (0.559)
- Gini: 0.365 — above 0.3 production minimum ✓

The good news: the model's ranking ability actually *improved* on test data (ROC-AUC 0.683 vs 0.658 on validation). The bad news: recall dropped from 25% (validation) to 10% (test). The threshold of 0.20, tuned on validation, doesn't transfer perfectly.

**Root cause: distribution shift.** Test probabilities compress lower than validation — defaults that scored 0.20+ on validation score 0.15 on test, falling below the threshold. The model's *ordering* is good (defaulters still get higher scores), but the *scale* shifted.

> 📌 **Teaching Note:** This is the SMOTE distortion at work. The model was trained on 50/50 data but applied to a 3.4% default rate, compressing all probabilities toward zero. The cliff at threshold 0.30 — where all metrics drop to zero — reveals that no account receives a probability above ~0.35. The model has useful signal (ROC-AUC 0.683) but the probability scale is completely wrong. This is why §3.5 introduces calibration.

> 💡 **Key Insight:** This isn't a coding error — it's a fundamental challenge in ML. The model's *ranking quality* (ROC-AUC) is decent, but the *probability scale* is distorted. Calibration can fix the scale without changing the ranking. This is the critical distinction between a model that ranks well and a model whose probabilities are usable for decisions.

### 3.4.6 Key Lessons from Model Improvement

1. **Validation ≠ Test** - Strong validation performance doesn't guarantee test performance
2. **SMOTE has trade-offs** - Improves validation metrics but may create poorly-calibrated probabilities
3. **Distribution shift is real** - Temporal differences require continuous monitoring
4. **ROC-AUC can be misleading** - Good AUC with bad calibration = unusable model
5. **Honest evaluation matters** - Reporting failures teaches more than hiding them

---

## 3.5 Explainability

### 3.5.1 Why Explainability Matters

1. **Debugging** - Diagnose why the model failed
2. **Trust** - Stakeholders can see which features drive decisions
3. **Compliance** - Adverse action notices required by law
4. **Fairness** - Foundation for detecting bias
5. **Improvement** - Guides feature engineering

### 3.5.2 SHAP: Global Feature Importance

**SHAP (SHapley Additive exPlanations)** decomposes each prediction into the contribution of every feature. For any individual prediction, SHAP answers: "How much did each feature push this prediction above or below the average?"

```python
import shap

# best_model is the tuned XGBoost from Section 3.4.3
best_model = grid_search.best_estimator_

# Note: We explain this model despite its test set failure. 
# SHAP isn't just for celebrating good models—it's often MORE valuable
# for diagnosing WHY a model failed.

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Summary plot shows feature importance
shap.summary_plot(shap_values, X_test)
```

> 💡 **Why X_test, not X_validate?** We generate SHAP explanations on the test set because it represents truly unseen data—the closest approximation to production. The validation set was used during hyperparameter tuning, so the model has been indirectly optimized for it. SHAP on X_test gives us the most honest view of how the model explains itself on new data.

*Figure 3.9: SHAP Summary Plot — Feature Importance (Test Set)*

**How to read the SHAP beeswarm plot:** Each dot is one account. The x-axis shows the SHAP value — positive pushes toward predicting default, negative pushes away. The color shows the actual feature value — red means high, blue means low. Features are ranked top-to-bottom by overall importance (mean absolute SHAP value).

Let's walk through three features to build intuition:

- **FICO score:** Blue dots (low FICO) cluster on the right (positive SHAP) — low FICO increases predicted default risk. Red dots (high FICO) cluster on the left — high FICO is protective. This matches domain knowledge perfectly.
- **credit_history_months:** Similar pattern — shorter credit history (blue) pushes toward default, longer history (red) is protective.
- **feat_channel_atm:** More nuanced — high ATM usage (red) pushes toward default. This could reflect cash-dependent customers who may have less financial stability, though the model doesn't know *why* — it only sees the correlation.

Now let's extract the numeric ranking:

```python
# Compute mean absolute SHAP value per feature
shap_importance = pd.DataFrame({
    'feature': X_test.columns,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

print("Top 10 Features by SHAP Importance:")
print(shap_importance.head(10).to_string(index=False))
```

**Expected Output:**
```
Top 10 Features by SHAP Importance:
             feature  mean_abs_shap
          fico_score          0.847
 feat_num_txn_12mo          0.523
feat_channel_online          0.498
feat_channel_mobile          0.412
    feat_channel_atm          0.389
credit_history_months         0.371
      annual_income          0.358
              age          0.341
 credit_utilization_pct       0.329
  feat_avg_balance_3mo        0.312
```

SHAP also lets us explore feature *interactions* through dependence plots. These show how one feature's SHAP contribution varies across its range, with color revealing the influence of a second feature:

```python
# SHAP dependence plot — FICO score colored by spending_travel
shap.dependence_plot('fico_score', shap_values, X_test, 
                     interaction_index='feat_spending_travel')
```

*Figure 3.10: SHAP Dependence Plot — FICO Score*

**How to read this plot:** The x-axis is the actual FICO score, the y-axis is FICO's SHAP contribution *for that individual account* (not a combined score with the color feature). The color (feat_spending_travel) reveals whether travel spending modulates FICO's effect. If two accounts both have FICO = 700 but different SHAP values, the color helps explain why — perhaps high travel spending (pink) shifts FICO's influence differently than low travel spending (blue). Vertical spread with a color pattern indicates an interaction effect between the two features.

### 3.5.3 Diagnosing the Failure

Our model collapsed on the test set — 0% precision, 0% recall. SHAP told us *what* the model learned; now we need to understand *why* it failed. Let's start by comparing how the model distributes predicted probabilities across train, validation, and test sets:

```python
# Generate probabilities for all three datasets
y_train_proba = best_model.predict_proba(X_train)[:, 1]
y_val_proba = best_model.predict_proba(X_val)[:, 1]
y_test_proba = best_model.predict_proba(X_test)[:, 1]

# Plot probability distributions side by side
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, probs, labels, name in zip(axes, 
    [y_train_proba, y_val_proba, y_test_proba],
    [y_train, y_val, y_test],
    ['Train Set', 'Val Set', 'Test Set']):
    ax.hist(probs[labels == 0], bins=30, alpha=0.6, label='Non-Default', color='cornflowerblue')
    ax.hist(probs[labels == 1], bins=30, alpha=0.6, label='Default', color='lightcoral')
    ax.set_title(f'{name}\n(Median: {np.median(probs):.3f})')
    ax.set_xlabel('Predicted Probability')
    ax.legend()
plt.tight_layout()
plt.savefig('fig_3_9_probability_distributions.png', dpi=150)
```

*Figure 3.11: Probability Distributions by Dataset*

**How to read these panels:** Blue bars are non-defaulters, red bars are actual defaulters. In the train set, defaults (red) spread across higher probabilities — the model separates them from non-defaults. In the validation set, separation weakens but some defaults still reach higher probability ranges. In the test set, everything collapses below 0.10 — defaults and non-defaults overlap almost completely. At the 0.20 threshold, very few test accounts get flagged.

The trap: the overall medians (0.020, 0.021, 0.019) look similar across all three sets. You might think the distributions are comparable. But the critical difference is in the *default-specific* probabilities:

```python
# Compare probability distributions for ACTUAL DEFAULTS only
val_default_probs = y_val_proba[y_val == 1]
test_default_probs = y_test_proba[y_test == 1]

print("Probabilities for ACTUAL DEFAULTS:")
print(f"  Validation mean: {val_default_probs.mean():.4f}")
print(f"  Test mean:       {test_default_probs.mean():.4f}")
print(f"  Ratio:           {val_default_probs.mean()/test_default_probs.mean():.1f}x")
```

**Expected Output:**
```
Probabilities for ACTUAL DEFAULTS:
  Validation mean: 0.1842
  Test mean:       0.0614
  Ratio:           3.0x
```

**The model assigns lower probabilities to test defaults.** The overall distributions look similar, but the model's probability compression means the threshold that separated defaults on validation no longer works on test data. This is why ROC-AUC (0.683) holds up better than precision/recall — AUC measures *ranking ability* and is unaffected by the probability scale, while precision/recall depend on the *absolute threshold*.

Now let's check calibration — are the model's probabilities "honest" (does a 30% prediction actually default 30% of the time)?

```python
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, probs, labels, name in zip(axes,
    [y_train_proba, y_val_proba, y_test_proba],
    [y_train, y_val, y_test],
    ['Train Set', 'Val Set', 'Test Set']):
    prob_true, prob_pred = calibration_curve(labels, probs, n_bins=10)
    brier = brier_score_loss(labels, probs)
    ax.plot(prob_pred, prob_true, marker='o', label='Model')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax.set_title(f'{name}\nCalibration Curve')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.text(0.6, 0.1, f'Brier Score: {brier:.3f}', color='red')
    ax.legend()
plt.tight_layout()
plt.savefig('fig_3_12_calibration_curves.png', dpi=150)
```

*Figure 3.12: Calibration Curves by Dataset*

**How to read calibration curves:** Each dot represents a bin of predictions. The x-position is the mean predicted probability within that bin (not a bin edge), and the y-position is the actual default rate among those predictions. If a model is well-calibrated, points land on the dashed diagonal — a prediction of 30% should correspond to a 30% actual default rate. Points above the diagonal mean the model *underestimates* risk; below means it *overestimates*.

**The Brier Score** is the mean squared error of probabilities: `mean((predicted - actual)²)`. Lower is better. It penalizes confident wrong predictions heavily.

Walking through the three panels: The train set (Brier 0.010) is well-calibrated — expected, since the model was trained on this data. The validation set (Brier 0.033) deviates, bowing above the diagonal — the model *underestimates* default risk. The test set (Brier 0.065) is essentially flat near zero — the model is stuck predicting very low probabilities for everyone, so calibration is meaningless.

The problem isn't overall probability distribution — it's specifically how the model scores actual defaulters. Calibration can fix this.

### 3.5.3a Probability Calibration

Since the model has good ranking ability (ROC-AUC 0.683) but distorted probabilities, we apply calibration to remap the SMOTE-distorted outputs to honest ones. We test two approaches:

- **Platt scaling:** Fits a sigmoid function (two parameters) — simple, less prone to overfitting
- **Isotonic regression:** Fits a non-decreasing step function — more flexible, can overfit on small data

Both learn on validation data how raw probabilities map to actual default rates, then apply that mapping to test data.

**Calibrated test performance (best method: isotonic):**
- F1: 0.178 (vs 0.118 uncalibrated) — 51% improvement
- Precision: 12.7%
- Recall: 30% (vs 10% uncalibrated) — recall tripled

> 💡 **Key Insight:** Calibrated test F1 (0.178) exceeds uncalibrated *validation* F1 (0.138). The model had good signal all along — calibration just fixed the probability scale so a reasonable threshold could separate defaults from non-defaults. SMOTE gave the model good ranking ability but wrong probabilities. Calibration doesn't make the model smarter; it translates what the model already knows into usable numbers.

### 3.5.4 Adverse Action Notices

When a lender denies a credit application — or offers less favorable terms — the Equal Credit Opportunity Act (ECOA) requires them to provide the applicant with specific reasons for the decision. These are called *adverse action notices*, and they can't just say "the algorithm said no." Regulation B requires the lender to identify the principal factors that drove the decision, typically the top 4 reasons.

This is where SHAP becomes more than a diagnostic tool — it becomes a compliance mechanism. Each applicant's SHAP values show exactly which features pushed their prediction toward default and by how much. The top positive SHAP contributors become the adverse action reasons.

```python
def generate_adverse_action_notice(applicant_idx, shap_values, X_test):
    """Generate ECOA-compliant adverse action notice using SHAP values."""
    
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

Let's look at three real examples from our test set to see how this works in practice. For each applicant, we generate a SHAP waterfall plot — a visual breakdown of how every feature contributed to their individual prediction:

```python
# Example: Generate waterfall plot for a high-risk applicant
shap.waterfall_plot(shap.Explanation(
    values=shap_values[high_risk_idx],
    base_values=explainer.expected_value,
    data=X_test.iloc[high_risk_idx],
    feature_names=X_test.columns.tolist()
))
```

*Figure 3.13: SHAP Waterfall — High-Risk Applicant*

This applicant was flagged for default. The waterfall shows the model started from the base prediction (average default rate), then each feature pushed it up or down. The top factors — low FICO score and high credit utilization — would appear as reasons 1 and 2 on their adverse action notice. These are reasons an applicant can understand and act on.

*Figure 3.14: SHAP Waterfall — Low-Risk Applicant*

For comparison, this applicant was approved. High FICO score and long credit history pushed the prediction strongly *away* from default. No adverse action notice is needed.

*Figure 3.15: SHAP Waterfall — Borderline Case*

This applicant is near the decision boundary. The features nearly cancel out — some push toward default, others away. These are the cases where threshold choice matters most, and where fairness concerns are most acute.

*Figure 3.16: SHAP Waterfall — False Negative (Missed Default)*

This applicant actually defaulted but the model predicted they wouldn't. The waterfall shows why — high FICO and long credit history gave them a "shield" that the model couldn't see past. This is a reminder that SHAP explains the model's reasoning, not reality.

### 3.5.5 Model Card

A model card documents everything stakeholders need to know:

```markdown
## Model Card: Credit Risk Prediction

### Model Details
- Type: XGBoost Classifier
- Features: 30 (from 51 engineered)
- Training: SMOTE-balanced data

### Intended Use
- Credit decisioning for consumer loans
- NOT for final approval (human review required)

### Performance Metrics

**Table 3.3: Model Performance Across Datasets**

| Dataset | ROC-AUC | Gini | Precision | Recall | Threshold |
|---------|---------|------|-----------|--------|-----------|
| Val     | 0.658   | 0.316 | 9.5%     | 25%    | 0.20      |
| Test    | 0.683   | 0.365 | 14.3%    | 10%    | 0.20      |
| Test (calibrated) | 0.683 | 0.365 | 12.7% | 30%  | optimized |

### Known Limitations
⚠️ SMOTE distortion: probabilities calibrated for 50/50 world, not 3.4% default rate
⚠️ Threshold doesn't transfer cleanly: recall drops from 25% (val) to 10% (test)
⚠️ Calibration fixes probability scale but doesn't add discrimination power
⚠️ Fairness evaluation pending (see Chapter 4)

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

5. **ROC-AUC measures ranking, not calibration** — Good ROC-AUC with SMOTE-distorted probabilities means the model ranks well but the probability scale is wrong. Calibration fixes the scale without changing the ranking.

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

This chapter built a model with genuine ranking ability (ROC-AUC 0.683) and showed that calibration can recover usable probabilities from SMOTE-distorted outputs. But critical questions remain:

- **Is the model fair?** Passing DIR for regions doesn't guarantee fairness across race and gender
- **Does calibration help equally?** If the model is better calibrated for some groups, fairness gaps may lurk beneath surface-level metrics
- **Can we fix unfairness without destroying performance?** The accuracy-fairness trade-off is real

**Chapter 4 (Fairness & Compliance) will address:**
- Formal fairness metrics across race and gender (DIR, SPD, EOD, ECE)
- The discovery that equal approval rates don't mean equal treatment
- Three bias mitigation approaches and their honest trade-offs
- Production monitoring to catch fairness drift

---

## Teaching Notes

### Learning Objectives

By the end of this chapter, learners should be able to:

**LO1: Recognize Credit Context**
- Explain why credit modeling differs from typical ML
- Identify regulatory requirements (ECOA, FCRA, SR 11-7)
- Articulate the fairness-accuracy trade-off

**LO2: Handle Class Imbalance**
- Implement SMOTE, class weights, and threshold adjustment
- Evaluate trade-offs between approaches
- Recognize why accuracy is misleading

**LO3: Build and Evaluate Models**
- Engineer features from credit data
- Train and tune classification models
- Use credit-specific metrics (ROC-AUC, Gini, KS)

**LO4: Explain Predictions**
- Apply SHAP for global and local explanations
- Generate regulatory-compliant adverse action notices
- Create model documentation (model cards)

**LO5: Recognize Limitations**
- Identify distribution shift and its consequences
- Recognize why validation performance doesn't guarantee test performance
- Document model limitations honestly

### Discussion Questions

1. **The Apple Card Question:** Goldman Sachs claimed their algorithm didn't use gender. How could it still produce gender-biased outcomes?

2. **The SMOTE Dilemma:** SMOTE improved our validation metrics but hurt test performance. When should you use SMOTE? When should you avoid it?

3. **Threshold Selection:** We optimized the threshold at 0.20 on validation data, but recall dropped from 25% to 10% on test. Calibration recovered it to 30%. How would you handle threshold selection in production — recalibrate periodically, or use a more conservative fixed threshold?

4. **Explainability Trade-offs:** More complex models (XGBoost) often perform better but are harder to explain. How do you balance accuracy and explainability?

5. **Documentation Ethics:** Is it ethical to deploy a model with known limitations if you document them? Where's the line?

### Suggested Exercises

**Exercise 1: Alternative Imbalance Handling (Intermediate)**

Try class weights instead of SMOTE. Compare validation and test performance. Which approach generalizes better?

**Exercise 2: Feature Engineering (Intermediate)**

Add new features from transaction data (e.g., weekend spending ratio, late-night transactions). Do they improve model performance?

**Exercise 3: Calibration Deep Dive (Advanced)**

Section 3.5.3a showed isotonic regression tripling recall from 10% to 30%. Try Platt scaling instead — does it perform better or worse? What happens if you calibrate on a separate hold-out set rather than validation data?

**Exercise 4: Adverse Action Notices (Applied)**

Generate adverse action notices for 10 denied applicants. Are the explanations reasonable? Would a customer follow them?

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
