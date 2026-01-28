# List of Tables

---

## Chapter 2: Data Foundations

| Table | Title | Page |
|-------|-------|------|
| **2.1** | The Five Dimensions of Data Quality | ___ |
| **2.2** | Regulatory Requirements for Data Management | ___ |
| **2.3** | Data Quality Standards by Use Case | ___ |
| **2.4** | Chapter 2 Self-Assessment Rubric | ___ |
| **2.5** | Key Terms: Data Foundations | ___ |

---

## Chapter 3: Building the Credit Model

| Table | Title | Page |
|-------|-------|------|
| **3.1** | What Makes Credit Models Different | ___ |
| **3.2** | Class Imbalance Handling Approaches | ___ |
| **3.3** | Model Performance Across Datasets (The Failure) | ___ |
| **3.4** | Key Terms: Credit Modeling | ___ |

---

## Chapter 4: Fairness & Compliance

| Table | Title | Page |
|-------|-------|------|
| **4.1** | Error Types in Credit Scoring | ___ |
| **4.2** | Mitigation Approach Comparison | ___ |
| **4.3** | Monitoring Alert Thresholds | ___ |
| **4.4** | Common Fairness Pitfalls | ___ |
| **4.5** | Key Terms: Fairness & Compliance | ___ |

---

## Chapter 5: Conclusion & Future Directions

| Table | Title | Page |
|-------|-------|------|
| **5.1** | Mitigation Results Comparison | ___ |
| **5.2** | Mitigation Techniques: When They Act | ___ |
| **5.3** | Train/Validate/Test Purposes | ___ |
| **5.4** | Fairness Definitions and Conflicts | ___ |
| **5.5** | State-Level AI Regulations | ___ |
| **5.6** | Key Regulatory Trends | ___ |
| **5.7** | Fairness Responsibility Matrix | ___ |
| **5.8** | ML Lifecycle Stage Gates | ___ |
| **5.9** | Organizational Health Metrics | ___ |
| **5.10** | Key Terms: Conclusion | ___ |

---

## Appendix Tables

| Table | Title | Location |
|-------|-------|----------|
| **A.1** | Fairness Metrics at a Glance | Appendix A |
| **A.2** | Disparate Impact Ratio Interpretation | Appendix A |
| **A.3** | Statistical Parity Difference Interpretation | Appendix A |
| **A.4** | Equal Opportunity Difference Interpretation | Appendix A |
| **A.5** | Expected Calibration Error Interpretation | Appendix A |
| **A.6** | Legal/Regulatory Thresholds | Appendix A |
| **A.7** | Industry Best Practices Thresholds | Appendix A |
| **A.8** | Error Types Reference | Appendix A |
| **A.9** | Rate Formulas Reference | Appendix A |
| **A.10** | Sample Size Guidelines | Appendix A |
| **B.1** | Key Regulations Overview | Appendix B |
| **B.2** | ECOA Protected Characteristics | Appendix B |
| **B.3** | EU AI Act Risk Classification | Appendix B |
| **B.4** | EU AI Act High-Risk Requirements | Appendix B |
| **B.5** | EU AI Act Timeline | Appendix B |
| **B.6** | EU AI Act Penalties | Appendix B |
| **B.7** | State-Level AI Laws | Appendix B |
| **B.8** | Pre-Deployment Compliance Checklist | Appendix B |
| **B.9** | Post-Deployment Compliance Checklist | Appendix B |
| **B.10** | Key Regulatory Contacts | Appendix B |

---

## Table Details

### Chapter 2 Tables

**Table 2.1: The Five Dimensions of Data Quality**
Defines the five key dimensions for assessing data quality: Accuracy (does data correctly represent reality?), Completeness (are all required fields populated?), Consistency (do related elements agree?), Timeliness (is data fresh enough?), and Validity (do values conform to business rules?).

**Table 2.2: Regulatory Requirements for Data Management**
Maps key regulatory concepts to their practical meaning: Reproducibility (version control), Justification (documented rationale), Auditability (complete lineage), and Validation (automated tests).

**Table 2.3: Data Quality Standards by Use Case**
Shows how data quality standards vary by application: exploratory analysis (lower bar), model training (high bar), regulatory reporting (very high bar), and production decisioning (highest bar).

**Table 2.4: Chapter 2 Self-Assessment Rubric**
Scoring rubric for evaluating data pipeline quality across five dimensions: Data Quality (25 points), Code Quality (20 points), Documentation (20 points), Business Context (20 points), and Validation (15 points).

**Table 2.5: Key Terms: Data Foundations**
Definitions for BCBS 239, SR 11-7, data lineage, referential integrity, fit-for-purpose, and type coercion.

---

### Chapter 3 Tables

**Table 3.1: What Makes Credit Models Different**
Compares credit models to fraud detection and recommender systems across four dimensions: Stakes (high for credit), Fairness (legally mandated), Explainability (required by law), and Regulation (heavy - ECOA, FCRA).

**Table 3.2: Class Imbalance Handling Approaches**
Compares three approaches to handling class imbalance: Class Weights (simple but limited), SMOTE (better recall but calibration issues), and Threshold Adjustment (precision-recall trade-off).

**Table 3.3: Model Performance Across Datasets (The Failure)**
Shows the critical validation-to-test performance drop: Training (AUC 0.89, 35% precision, 42% recall), Validation (AUC 0.70, 20% precision, 19% recall), Test (AUC 0.58, 0% precision, 0% recall). This table documents the honest failure that is central to the book's teaching approach.

**Table 3.4: Key Terms: Credit Modeling**
Definitions for ECOA, disparate impact, SMOTE, class imbalance, distribution shift, calibration, SHAP, adverse action notice, Gini coefficient, and KS statistic.

---

### Chapter 4 Tables

**Table 4.1: Error Types in Credit Scoring**
Defines the four outcomes in credit prediction: True Negative (correct approval), False Positive (wrongly denied good borrower), False Negative (wrongly approved bad borrower), and True Positive (correct denial).

**Table 4.2: Mitigation Approach Comparison**
Compares four mitigation approaches with their Disparate Impact Ratio and Accuracy: Original (DIR 1.03, 94.8%), Reweighting (DIR ~1.03, ~95%), Group Thresholds (DIR 1.00, 31.4% - destroyed accuracy), Calibration (DIR ~1.00, 96.9% - best balance).

**Table 4.3: Monitoring Alert Thresholds**
Defines the traffic-light alert system: GREEN (DIR ≥ 0.90, normal monitoring), YELLOW (DIR 0.80-0.90, investigate within 7 days), RED (DIR < 0.80, immediate escalation).

**Table 4.4: Common Fairness Pitfalls**
Lists five common mistakes and their fixes: not collecting race doesn't prevent bias, data may reflect past discrimination, fairness drifts over time, perfect isn't enemy of good, and fairness should be designed in from start.

**Table 4.5: Key Terms: Fairness & Compliance**
Definitions for disparate impact, 4/5ths rule, demographic parity, equalized odds, calibration, ECE, and reweighting.

---

### Chapter 5 Tables

**Table 5.1: Mitigation Results Comparison**
Full comparison of four mitigation approaches showing Accuracy, ROC-AUC, and DIR. Calibration emerges as clear winner with 96.9% accuracy and 0.848 ROC-AUC.

**Table 5.2: Mitigation Techniques: When They Act**
Shows when each technique applies: Reweighting (before/during training, requires retraining), Calibration (after training, no retraining), Group Thresholds (after training, no retraining).

**Table 5.3: Train/Validate/Test Purposes**
Clarifies the distinct purposes: Train (fit parameters), Validate (choose model/tune - YES you can tune), Test (final evaluation only - NEVER tune).

**Table 5.4: Fairness Definitions and Conflicts**
Shows how demographic parity, equal opportunity, and calibration conflict with each other and cannot be simultaneously satisfied when base rates differ.

**Table 5.5: State-Level AI Regulations**
Lists key state-level provisions: Colorado (insurance AI testing), California (algorithmic accountability proposals), New York (bias audits for employment AI).

**Table 5.6: Key Regulatory Trends**
Projects five trends from current to future state: Explainability (nice to have → legal requirement), Monitoring (pre-deployment → real-time), Third-Party Audits (internal → mandatory external), Individual Rights (adverse action → full explanation rights), Liability (unclear → defined frameworks).

**Table 5.7: Fairness Responsibility Matrix**
Assigns responsibilities across six organizational roles: Executive Leadership, Product/Business Owners, Data Science/ML, Model Risk/Compliance, Legal, and Operations.

**Table 5.8: ML Lifecycle Stage Gates**
Defines fairness requirements at five gates: Project Initiation, Data Preparation, Model Development, Pre-Deployment, and Post-Deployment.

**Table 5.9: Organizational Health Metrics**
Defines five organizational metrics with targets: Time to detect (< 24 hours), Time to remediate (< 1 week), Pre-deployment catch rate (> 90%), Training completion (100%), Checklist compliance (100%).

**Table 5.10: Key Terms: Conclusion**
Definitions for EU AI Act, High-Risk AI, Conformity Assessment, Stage Gate, Fairness Champion, and Blameless Postmortem.

---

## Summary Statistics

| Chapter | Table Count |
|---------|-------------|
| Chapter 2 | 5 |
| Chapter 3 | 4 |
| Chapter 4 | 5 |
| Chapter 5 | 10 |
| **Chapters Total** | **24** |
| Appendix A | 10 |
| Appendix B | 10 |
| **Appendices Total** | **20** |
| **Grand Total** | **44 tables** |

---

*End of List of Tables*
