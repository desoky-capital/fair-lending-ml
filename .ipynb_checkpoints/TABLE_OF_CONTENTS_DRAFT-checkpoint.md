# CODE, CAPITAL, AND CONSCIENCE

## Building Fair and Compliant Machine Learning Systems for Financial Services

---

# Table of Contents

---

## Preface
- Why We Wrote This Book
- Two Ways to Read This Book
- Acknowledgments

---

## Chapter 1: Introduction .......................................... 1

### 1.1 Why This Book Exists ........................................ 2
- The $80 Million Wake-Up Call
- The Gap This Book Fills
- Code, Capital, and Conscience

### 1.2 Who This Book Is For ........................................ 6
- Two Ways to Read This Book (Path A: Hands-On, Path B: Conceptual)
- Path A: Hands-On Readers
- Path B: Conceptual Readers
- What Makes This Book Accessible

### 1.3 How to Use This Book ....................................... 10
- The Journey
- Reading Recommendations
- Key Features to Look For

### 1.4 What You'll Build .......................................... 13
- The Credit Default Model
- The Lessons You'll Learn
- A Preview of Results

### 1.5 Let's Begin ................................................ 16

---

## Chapter 2: Building the Credit Model ........................... 17

### 2.1 Data Preparation ........................................... 18
- 2.1.1 Loading and Exploring the Data
- 2.1.2 Data Quality Assessment
- 2.1.3 Handling Missing Values
- 2.1.4 Feature Engineering
- 2.1.5 Train/Validation/Test Split
- Key Takeaways

### 2.2 Baseline Model ............................................. 35
- 2.2.1 Why Start with a Baseline?
- 2.2.2 Logistic Regression Implementation
- 2.2.3 Evaluation Metrics
- 2.2.4 Interpreting Results
- 2.2.5 Baseline Performance Summary
- Key Takeaways

### 2.3 Model Improvement .......................................... 52
- 2.3.1 Addressing Class Imbalance
- 2.3.2 XGBoost Implementation
- 2.3.3 Hyperparameter Tuning
- 2.3.4 Threshold Optimization
- 2.3.5 The SMOTE Lesson: When Improvement Backfires
- 2.3.6 Distribution Shift and Model Failure
- Key Takeaways

### 2.4 Explainability ............................................. 72
- 2.4.1 Why Explainability Matters
- 2.4.2 Introduction to SHAP
- 2.4.3 Global Explanations: Feature Importance
- 2.4.4 Local Explanations: Individual Predictions
- 2.4.5 Generating Adverse Action Notices
- 2.4.6 Explanation Best Practices
- Key Takeaways

---

## Chapter 3: Fairness & Compliance ............................... 92

### 3.1 Understanding Algorithmic Fairness ......................... 93
- 3.1.1 What Is Fairness?
- 3.1.2 Protected Classes in Credit
- 3.1.3 Fairness Definitions
  - Demographic Parity
  - Equalized Odds
  - Equal Opportunity
  - Calibration
- 3.1.4 The Impossibility Theorems
- 3.1.5 Choosing a Fairness Definition
- Key Takeaways

### 3.2 Measuring Bias in Our Model ............................... 115
- 3.2.1 Setup: Loading the Model and Data
- 3.2.2 Generating Synthetic Protected Attributes
- 3.2.3 Implementing Fairness Metrics
  - Disparate Impact Ratio (DIR)
  - Statistical Parity Difference (SPD)
  - Equal Opportunity Difference (EOD)
  - Average Odds Difference (AOD)
  - Expected Calibration Error (ECE)
- 3.2.4 Measuring Fairness by Race
- 3.2.5 Measuring Fairness by Gender
- 3.2.6 Calibration Analysis by Group
- 3.2.7 Fairness Visualization Dashboard
- 3.2.8 Comprehensive Fairness Report
- 3.2.9 Root Cause Analysis
- Key Takeaways

### 3.3 Bias Mitigation & Production Deployment ................... 145
- 3.3.1 Overview: Three Approaches to Bias Mitigation
- 3.3.2 Pre-Processing: Reweighting
  - How Weights Affect Training
  - Training with Weights
  - Interpreting Results
- 3.3.3 Post-Processing: Threshold Adjustment
  - Group-Specific Thresholds
  - Calibration by Group
- 3.3.4 Comparing Mitigation Approaches
- 3.3.5 Production Monitoring
  - What to Monitor
  - Alert Thresholds
  - Monitoring Code
  - Validation vs. Test Comparison
- 3.3.6 Regulatory Documentation
- Key Takeaways

---

## Chapter 4: Conclusion & Future Directions ..................... 178

### 4.1 Key Lessons Learned ....................................... 179
- 4.1.1 Technical Lessons
  - Lesson 1: SMOTE Can Create Problems It Claims to Solve
  - Lesson 2: Distribution Shift Will Break Your Model
  - Lesson 3: Calibration Often Beats Other Mitigation Techniques
  - Lesson 4: Group-Specific Thresholds Are Dangerous
  - Lesson 5: Reweighting Only Helps If Data Is Actually Imbalanced
  - Lesson 6: Know When Each Mitigation Technique Acts
- 4.1.2 Process Lessons
  - Lesson 7: The Train/Validate/Test Split Has Distinct Purposes
  - Lesson 8: Fairness Must Be Checked on Multiple Datasets
  - Lesson 9: Document Trade-offs, Not Just Decisions
  - Lesson 10: Monitor Predictions, Not Just Outcomes
- 4.1.3 Conceptual Lessons
  - Lesson 11: Fairness Definitions Conflict—Choose Deliberately
  - Lesson 12: A Fair But Useless Model Is Still Useless
  - Lesson 13: Fairness Is Ongoing, Not One-Time
  - Lesson 14: Technical Fairness ≠ Actual Fairness
- 4.1.4 Summary: The Top 14 Lessons

### 4.2 Emerging Regulations & Trends ............................. 195
- 4.2.1 The Current Regulatory Baseline
- 4.2.2 The EU AI Act: A New Paradigm
  - Risk-Based Classification
  - High-Risk System Requirements
  - Conformity Assessment
- 4.2.3 US Regulatory Evolution
  - CFPB Guidance on AI
  - State-Level AI Laws
  - Federal Proposals
- 4.2.4 Emerging Technical Standards
  - NIST AI Risk Management Framework
  - ISO/IEC Standards
- 4.2.5 Key Trends to Watch
- 4.2.6 Preparing for the Future
- 4.2.7 Summary

### 4.3 Building a Fairness-First Culture ......................... 212
- 4.3.1 Why Culture Matters
- 4.3.2 Organizational Roles: Who Owns Fairness?
  - The Fairness Responsibility Matrix
  - The Fairness Champion Role
- 4.3.3 Embedding Fairness in Workflows
  - Stage Gates with Fairness Criteria
  - Pre-Deployment Checklist
  - Code Review Standards
- 4.3.4 Training & Awareness
- 4.3.5 Continuous Improvement
  - Learning from Incidents
  - Metrics That Matter
  - Regular Reviews
- 4.3.6 Final Thoughts
  - What We've Learned Together
  - The Three Pillars
  - A Call to Action

---

## Appendices

### Appendix A: Fairness Metrics Quick Reference .................. 228
- Metric Definitions
- Formulas
- Interpretation Guide
- Thresholds and Standards

### Appendix B: Regulatory Quick Reference ........................ 232
- ECOA Summary
- FCRA Summary
- SR 11-7 Summary
- EU AI Act Summary

### Appendix C: Code Reference .................................... 236
- Complete Function Library
- Installation Requirements
- Data Sources

### Appendix D: Documentation Templates ........................... 240
- Model Card Template
- Fairness Testing Report Template
- Production Monitoring Checklist

---

## Glossary ..................................................... 244

## References ................................................... 248

## Index ........................................................ 252

---

## List of Figures

- Figure 2.1: Data Distribution Overview
- Figure 2.2: Feature Correlation Matrix
- Figure 2.3: Model Performance Comparison
- Figure 2.4: SHAP Summary Plot
- Figure 2.5: SHAP Waterfall Plot (Individual Prediction)
- Figure 3.1: Fairness Definitions Comparison
- Figure 3.2: Impossibility Theorem Visualization
- Figure 3.3: Fairness Dashboard (6-Panel)
- Figure 3.4: Calibration Curves by Group
- Figure 3.5: Accuracy vs. Fairness Trade-off
- Figure 4.1: Mitigation Techniques Timeline
- Figure 4.2: EU AI Act Risk Tiers
- Figure 4.3: The Three Pillars of Sustainable Fairness

---

## List of Tables

- Table 2.1: Dataset Summary Statistics
- Table 2.2: Feature Descriptions
- Table 2.3: Model Performance Metrics
- Table 3.1: Protected Classes Under ECOA
- Table 3.2: Fairness Metrics Summary
- Table 3.3: Fairness Results by Race
- Table 3.4: Fairness Results by Gender
- Table 3.5: Mitigation Approach Comparison
- Table 3.6: Monitoring Alert Thresholds
- Table 4.1: The 14 Key Lessons
- Table 4.2: Emerging Regulations Timeline
- Table 4.3: Fairness Responsibility Matrix

---

*Page numbers are estimates and will be finalized during layout.*
