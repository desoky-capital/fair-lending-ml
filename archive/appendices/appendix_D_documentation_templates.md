# Appendix D: Documentation Templates

---

## D.1 Model Card Template

```
════════════════════════════════════════════════════════════════════════════════
                              MODEL CARD
════════════════════════════════════════════════════════════════════════════════

MODEL DETAILS
────────────────────────────────────────────────────────────────────────────────
Model Name:         _________________________________________________
Version:            _________________________________________________
Date:               _________________________________________________
Type:               [ ] Logistic Regression  [ ] Random Forest  [ ] XGBoost
                    [ ] Neural Network       [ ] Other: _______________

Developer:          _________________________________________________
Owner:              _________________________________________________
Contact:            _________________________________________________

INTENDED USE
────────────────────────────────────────────────────────────────────────────────
Primary Use Case:   _________________________________________________
                    _________________________________________________

Target Users:       [ ] Loan Officers  [ ] Automated System  [ ] Risk Analysts
                    [ ] Other: _______________

Out-of-Scope Uses:  _________________________________________________
                    _________________________________________________

TRAINING DATA
────────────────────────────────────────────────────────────────────────────────
Data Source:        _________________________________________________
Date Range:         From: ______________ To: ______________
Sample Size:        ______________ records
Default Rate:       ______________%

Data Quality:
  [ ] Completeness check passed
  [ ] Duplicate removal completed
  [ ] Schema validation completed
  [ ] Referential integrity verified

Protected Attributes Analyzed:
  [ ] Race/Ethnicity    [ ] Gender    [ ] Age    [ ] Other: _______________

FEATURES
────────────────────────────────────────────────────────────────────────────────
Number of Features: ______________

Top 5 Features by Importance:
  1. _________________________________ (Importance: ________)
  2. _________________________________ (Importance: ________)
  3. _________________________________ (Importance: ________)
  4. _________________________________ (Importance: ________)
  5. _________________________________ (Importance: ________)

Protected Characteristics Used as Features:  [ ] Yes  [ ] No
  If Yes, justify: __________________________________________________

Proxy Variables Identified:  [ ] Yes  [ ] No
  If Yes, list: ____________________________________________________

PERFORMANCE METRICS
────────────────────────────────────────────────────────────────────────────────
                        Training        Validation      Test
ROC-AUC:                __________      __________      __________
Gini:                   __________      __________      __________
Precision:              __________      __________      __________
Recall:                 __________      __________      __________
F1 Score:               __________      __________      __________

Decision Threshold:     __________
Threshold Selection Method: _________________________________________

FAIRNESS METRICS
────────────────────────────────────────────────────────────────────────────────
Reference Group: _____________________

                                    Group A         Group B         Group C
                                    ___________     ___________     ___________
Disparate Impact Ratio:             __________      __________      __________
Statistical Parity Difference:      __________      __________      __________
Equal Opportunity Difference:       __________      __________      __________
Expected Calibration Error:         __________      __________      __________

4/5ths Rule Compliance:  [ ] PASS  [ ] FAIL
  If FAIL, mitigation applied: ______________________________________

BIAS MITIGATION
────────────────────────────────────────────────────────────────────────────────
Mitigation Techniques Evaluated:
  [ ] None required (baseline fair)
  [ ] Reweighting - Result: ________________________________________
  [ ] Calibration - Result: ________________________________________
  [ ] Threshold Adjustment - Result: ________________________________
  [ ] Other: _______________________________________________________

Final Mitigation Applied: __________________________________________
Justification: _____________________________________________________

LIMITATIONS AND RISKS
────────────────────────────────────────────────────────────────────────────────
Known Limitations:
  1. _______________________________________________________________
  2. _______________________________________________________________
  3. _______________________________________________________________

Potential Risks:
  1. _______________________________________________________________
  2. _______________________________________________________________

Populations Where Model May Underperform:
  _________________________________________________________________

EXPLAINABILITY
────────────────────────────────────────────────────────────────────────────────
Explanation Method:  [ ] SHAP  [ ] LIME  [ ] Coefficients  [ ] Other: _______

Adverse Action Notice Generation:  [ ] Automated  [ ] Manual  [ ] N/A
Sample Adverse Action Reasons Provided: _______________________________

MONITORING PLAN
────────────────────────────────────────────────────────────────────────────────
Monitoring Frequency:
  [ ] Real-time  [ ] Daily  [ ] Weekly  [ ] Monthly

Metrics Monitored:
  [ ] Approval rates by group
  [ ] Disparate Impact Ratio
  [ ] Model performance (AUC, accuracy)
  [ ] Prediction distribution
  [ ] Other: _______________________

Alert Thresholds:
  RED (Immediate Action):   DIR < ________  or  AUC drop > ________
  YELLOW (Investigate):     DIR < ________  or  AUC drop > ________

Retraining Trigger: ________________________________________________

APPROVALS
────────────────────────────────────────────────────────────────────────────────
Model Developer:        _________________________ Date: ___________
Model Validator:        _________________________ Date: ___________
Business Owner:         _________________________ Date: ___________
Compliance Officer:     _________________________ Date: ___________
Legal Review:           _________________________ Date: ___________

════════════════════════════════════════════════════════════════════════════════
```

---

## D.2 Fairness Testing Report Template

```
════════════════════════════════════════════════════════════════════════════════
                        FAIRNESS TESTING REPORT
════════════════════════════════════════════════════════════════════════════════

REPORT INFORMATION
────────────────────────────────────────────────────────────────────────────────
Model Name:         _________________________________________________
Model Version:      _________________________________________________
Report Date:        _________________________________________________
Report Author:      _________________________________________________
Review Status:      [ ] Draft  [ ] Under Review  [ ] Approved

EXECUTIVE SUMMARY
────────────────────────────────────────────────────────────────────────────────
Overall Fairness Status:  [ ] PASS  [ ] CONDITIONAL PASS  [ ] FAIL

Key Findings:
  1. _______________________________________________________________
  2. _______________________________________________________________
  3. _______________________________________________________________

Recommended Actions:
  1. _______________________________________________________________
  2. _______________________________________________________________

TEST METHODOLOGY
────────────────────────────────────────────────────────────────────────────────
Testing Framework:  _________________________________________________

Protected Attributes Tested:
  [ ] Race/Ethnicity    Groups: _____________________________________
  [ ] Gender            Groups: _____________________________________
  [ ] Age               Groups: _____________________________________
  [ ] Other: __________ Groups: _____________________________________

Reference Group for Comparisons: ____________________________________

Fairness Definitions Applied:
  [ ] Demographic Parity (4/5ths rule)
  [ ] Equal Opportunity
  [ ] Equalized Odds
  [ ] Calibration
  [ ] Individual Fairness

Thresholds Used:
  DIR Minimum:                    __________
  SPD Maximum Absolute:           __________
  EOD Maximum Absolute:           __________
  ECE Maximum:                    __________

DATA SUMMARY
────────────────────────────────────────────────────────────────────────────────
Test Dataset:       _________________________________________________
Sample Size:        ______________ records
Date Range:         From: ______________ To: ______________

Group Distribution:
  Group                 Count           Percentage
  ___________________   __________      __________
  ___________________   __________      __________
  ___________________   __________      __________
  ___________________   __________      __________

Minimum Sample Size Requirement:  __________ per group
Groups Below Minimum:  [ ] None  [ ] List: ___________________________

DETAILED RESULTS
────────────────────────────────────────────────────────────────────────────────

TIER 1: LEGAL COMPLIANCE
═══════════════════════════════════════════════════════════════════════════════

1. Disparate Impact Ratio (4/5ths Rule)
   Requirement: DIR ≥ 0.80

   Comparison               DIR         Status
   ─────────────────────────────────────────────
   __________________ vs Reference    _______    [ ] PASS [ ] FAIL
   __________________ vs Reference    _______    [ ] PASS [ ] FAIL
   __________________ vs Reference    _______    [ ] PASS [ ] FAIL

2. Anti-Classification Check
   Protected attributes used as features:  [ ] Yes  [ ] No
   Status: [ ] PASS  [ ] FAIL

3. Explainability Check
   Adverse action notices available:  [ ] Yes  [ ] No
   Status: [ ] PASS  [ ] FAIL

TIER 1 OVERALL: [ ] COMPLIANT  [ ] NON-COMPLIANT

TIER 2: BUSINESS FAIRNESS
═══════════════════════════════════════════════════════════════════════════════

4. Calibration by Group
   Requirement: ECE difference < 0.05

   Group                   ECE         Status
   ─────────────────────────────────────────────
   Reference              _______      
   ____________________   _______      Diff: _______ [ ] PASS [ ] FAIL
   ____________________   _______      Diff: _______ [ ] PASS [ ] FAIL
   ____________________   _______      Diff: _______ [ ] PASS [ ] FAIL

5. Equal Opportunity
   Requirement: |EOD| < 0.10

   Comparison               EOD         Status
   ─────────────────────────────────────────────
   __________________ vs Reference    _______    [ ] PASS [ ] FAIL
   __________________ vs Reference    _______    [ ] PASS [ ] FAIL
   __________________ vs Reference    _______    [ ] PASS [ ] FAIL

TIER 2 OVERALL: [ ] ACCEPTABLE  [ ] NEEDS IMPROVEMENT

TIER 3: MONITORING METRICS
═══════════════════════════════════════════════════════════════════════════════

6. Statistical Parity Difference

   Comparison                          SPD
   ─────────────────────────────────────────────
   __________________ vs Reference    _______
   __________________ vs Reference    _______
   __________________ vs Reference    _______

7. Average Odds Difference

   Comparison                          AOD
   ─────────────────────────────────────────────
   __________________ vs Reference    _______
   __________________ vs Reference    _______
   __________________ vs Reference    _______

ROOT CAUSE ANALYSIS (If Issues Found)
────────────────────────────────────────────────────────────────────────────────
Issue Identified: ___________________________________________________

Potential Causes:
  [ ] Historical bias in training data
  [ ] Proxy discrimination
  [ ] Measurement bias (data quality varies by group)
  [ ] Sample size imbalance
  [ ] Other: _______________________________________________________

Investigation Performed: ____________________________________________
___________________________________________________________________

Findings: __________________________________________________________
___________________________________________________________________

MITIGATION RECOMMENDATIONS
────────────────────────────────────────────────────────────────────────────────

Recommended Mitigations:
  [ ] Reweighting during training
  [ ] Probability calibration
  [ ] Threshold adjustment
  [ ] Feature removal: _______________________________________________
  [ ] Additional data collection
  [ ] Model architecture change
  [ ] No mitigation required

Priority:  [ ] Critical (before deployment)  [ ] High  [ ] Medium  [ ] Low

Expected Impact on Accuracy: ________________________________________

SIGN-OFF
────────────────────────────────────────────────────────────────────────────────
Prepared by:         _________________________ Date: ___________
Reviewed by:         _________________________ Date: ___________
Approved by:         _________________________ Date: ___________

APPENDIX: Raw Metrics Data
────────────────────────────────────────────────────────────────────────────────
[Attach detailed metric calculations and supporting data]

════════════════════════════════════════════════════════════════════════════════
```

---

## D.3 Production Monitoring Checklist

```
════════════════════════════════════════════════════════════════════════════════
                    PRODUCTION MONITORING CHECKLIST
════════════════════════════════════════════════════════════════════════════════

MODEL INFORMATION
────────────────────────────────────────────────────────────────────────────────
Model Name:         _________________________________________________
Model Version:      _________________________________________________
Deployment Date:    _________________________________________________
Review Period:      From: ______________ To: ______________
Reviewer:           _________________________________________________

DAILY CHECKS
────────────────────────────────────────────────────────────────────────────────
Date: ______________

[ ] Model serving without errors
    Error count: __________  Expected: < __________

[ ] Prediction volume within expected range
    Volume: __________  Expected range: __________ to __________

[ ] Approval rates by group checked
    Group A: __________%   Group B: __________%   Group C: __________%
    Any group > 10% different from baseline? [ ] Yes [ ] No
    If Yes, escalate: ________________________________________________

[ ] Average prediction confidence checked
    Mean probability: __________  Expected range: __________ to __________

WEEKLY CHECKS
────────────────────────────────────────────────────────────────────────────────
Week of: ______________

[ ] Disparate Impact Ratio calculated
    DIR (Group B vs A): __________  Threshold: ≥ 0.80  [ ] PASS [ ] FAIL
    DIR (Group C vs A): __________  Threshold: ≥ 0.80  [ ] PASS [ ] FAIL
    
    If any FAIL → Escalate immediately

[ ] Statistical Parity Difference calculated
    SPD (Group B vs A): __________  
    SPD (Group C vs A): __________  
    
    If |SPD| > 0.10 → Investigate

[ ] Prediction distribution compared to baseline
    KS statistic vs baseline: __________
    
    If KS > 0.10 → Investigate drift

[ ] Feature distribution spot check
    Features checked: ________________________________________________
    Any anomalies? [ ] Yes [ ] No
    If Yes, describe: ________________________________________________

MONTHLY CHECKS
────────────────────────────────────────────────────────────────────────────────
Month: ______________

[ ] Full fairness metrics calculated
    
    Metric                      Value       Threshold   Status
    ────────────────────────────────────────────────────────────
    Disparate Impact Ratio      _______     ≥ 0.80      [ ] OK [ ] ALERT
    Statistical Parity Diff     _______     < 0.10      [ ] OK [ ] ALERT
    Equal Opportunity Diff      _______     < 0.10      [ ] OK [ ] ALERT
    Average Odds Diff           _______     < 0.10      [ ] OK [ ] ALERT
    
[ ] Calibration check performed
    ECE (Overall): __________
    ECE (Group A): __________
    ECE (Group B): __________
    ECE (Group C): __________
    
    If ECE > 0.10 or group difference > 0.05 → Recalibration needed

[ ] Model performance vs validation baseline
    
    Metric          Validation      Current         Change
    ────────────────────────────────────────────────────────────
    ROC-AUC         __________      __________      __________
    Precision       __________      __________      __________
    Recall          __________      __________      __________
    
    If AUC dropped > 0.05 → Investigate

[ ] Adverse action notice audit (sample of 10)
    
    Notice #    Reasons Clear?    Factors Accurate?    Customer Friendly?
    ─────────────────────────────────────────────────────────────────────
    1           [ ] Yes [ ] No    [ ] Yes [ ] No       [ ] Yes [ ] No
    2           [ ] Yes [ ] No    [ ] Yes [ ] No       [ ] Yes [ ] No
    ...
    10          [ ] Yes [ ] No    [ ] Yes [ ] No       [ ] Yes [ ] No
    
    Pass rate: ____/10  (Minimum: 9/10)

[ ] Customer complaint review
    Total complaints: __________
    Fairness-related: __________
    Patterns identified: ___________________________________________

QUARTERLY CHECKS
────────────────────────────────────────────────────────────────────────────────
Quarter: ______________

[ ] Comprehensive fairness audit completed
    Report attached: [ ] Yes  Document ID: ___________________________

[ ] Comparison to original validation metrics
    
    Original vs Current performance within acceptable range? [ ] Yes [ ] No
    Original vs Current fairness within acceptable range?    [ ] Yes [ ] No

[ ] Regulatory update review
    New regulations identified: [ ] Yes [ ] No
    If Yes, impact assessment: _______________________________________

[ ] Retraining assessment
    Retraining recommended? [ ] Yes [ ] No
    Rationale: ______________________________________________________

[ ] Documentation update
    Model card updated: [ ] Yes [ ] N/A
    Fairness report updated: [ ] Yes [ ] N/A

ALERT LOG
────────────────────────────────────────────────────────────────────────────────
Date        Alert Type      Description                     Resolution
──────────────────────────────────────────────────────────────────────────────
__________  ____________    ____________________________    ________________
__________  ____________    ____________________________    ________________
__________  ____________    ____________________________    ________________

SIGN-OFF
────────────────────────────────────────────────────────────────────────────────
Daily checks completed by:     _________________________ Date: ___________
Weekly checks completed by:    _________________________ Date: ___________
Monthly checks completed by:   _________________________ Date: ___________
Quarterly review approved by:  _________________________ Date: ___________

════════════════════════════════════════════════════════════════════════════════
```

---

## D.4 Pre-Deployment Sign-Off Form

```
════════════════════════════════════════════════════════════════════════════════
                      PRE-DEPLOYMENT SIGN-OFF FORM
════════════════════════════════════════════════════════════════════════════════

MODEL INFORMATION
────────────────────────────────────────────────────────────────────────────────
Model Name:              _____________________________________________
Version:                 _____________________________________________
Proposed Deploy Date:    _____________________________________________
Target Environment:      [ ] Production  [ ] Staging  [ ] A/B Test

TECHNICAL READINESS
────────────────────────────────────────────────────────────────────────────────
[ ] Code reviewed and approved
    Reviewer: _________________________ Date: ___________

[ ] Unit tests passing
    Test coverage: __________%

[ ] Integration tests passing
    
[ ] Performance benchmarks met
    Latency: __________ ms (Requirement: < __________ ms)
    Throughput: __________ /sec (Requirement: > __________ /sec)

[ ] Rollback procedure documented and tested

FAIRNESS READINESS
────────────────────────────────────────────────────────────────────────────────
[ ] Fairness testing report completed and approved
    Report ID: _________________________

[ ] Disparate Impact Ratio ≥ 0.80 for all groups
    DIR values: _________________________________________________

[ ] Protected characteristics NOT used as features
    Verification method: _________________________________________

[ ] Adverse action notice system operational
    Sample notice reviewed: [ ] Yes

[ ] Fairness monitoring configured
    Dashboard URL: ______________________________________________
    Alert recipients: ___________________________________________

DOCUMENTATION READINESS
────────────────────────────────────────────────────────────────────────────────
[ ] Model card completed
    Document location: ___________________________________________

[ ] Training data documented
    Document location: ___________________________________________

[ ] Feature definitions documented
    Document location: ___________________________________________

[ ] Known limitations documented

OPERATIONAL READINESS
────────────────────────────────────────────────────────────────────────────────
[ ] Monitoring dashboards configured
    
[ ] Alert thresholds set
    RED threshold (DIR): __________
    YELLOW threshold (DIR): __________

[ ] On-call rotation assigned
    Primary: _________________________
    Secondary: _______________________

[ ] Incident response procedure documented

[ ] Customer support trained on new model
    Training date: ___________

APPROVALS
────────────────────────────────────────────────────────────────────────────────
All items above must be checked before obtaining approvals.

Model Developer
  Name: ___________________________ 
  Signature: ______________________ Date: ___________
  Confirmation: "I confirm the model meets technical requirements."

Model Validator (Independent)
  Name: ___________________________ 
  Signature: ______________________ Date: ___________
  Confirmation: "I confirm independent validation was performed."

Business Owner
  Name: ___________________________ 
  Signature: ______________________ Date: ___________
  Confirmation: "I accept accountability for business outcomes."

Compliance Officer
  Name: ___________________________ 
  Signature: ______________________ Date: ___________
  Confirmation: "I confirm regulatory requirements are met."

Legal (if required)
  Name: ___________________________ 
  Signature: ______________________ Date: ___________
  Confirmation: "I confirm legal review is complete."

Final Approval Authority
  Name: ___________________________ 
  Signature: ______________________ Date: ___________
  
  Decision: [ ] APPROVED FOR DEPLOYMENT
            [ ] APPROVED WITH CONDITIONS: _____________________________
            [ ] NOT APPROVED - Reason: ________________________________

════════════════════════════════════════════════════════════════════════════════
```

---

## D.5 Incident Response Template

```
════════════════════════════════════════════════════════════════════════════════
                        FAIRNESS INCIDENT REPORT
════════════════════════════════════════════════════════════════════════════════

INCIDENT IDENTIFICATION
────────────────────────────────────────────────────────────────────────────────
Incident ID:         _________________________________________________
Model Affected:      _________________________________________________
Date Discovered:     _________________________________________________
Discovered By:       _________________________________________________
Discovery Method:    [ ] Monitoring Alert  [ ] Customer Complaint  
                     [ ] Audit Finding     [ ] Other: ________________

SEVERITY ASSESSMENT
────────────────────────────────────────────────────────────────────────────────
Severity Level:  [ ] Critical  [ ] High  [ ] Medium  [ ] Low

Criteria:
  [ ] DIR below 0.80 (Critical)
  [ ] DIR between 0.80-0.85 (High)
  [ ] Fairness drift > 5% from baseline (Medium)
  [ ] Customer complaints indicating pattern (Medium)
  [ ] Minor metric degradation (Low)

Estimated Impact:
  Affected population: __________ customers
  Affected time period: From __________ To __________
  Potential harm: ________________________________________________

INCIDENT DESCRIPTION
────────────────────────────────────────────────────────────────────────────────
Summary:
___________________________________________________________________________
___________________________________________________________________________
___________________________________________________________________________

Metrics at Discovery:
  DIR: __________  (Threshold: 0.80)
  SPD: __________  (Baseline: __________)
  Other: _____________________________________________________________

IMMEDIATE RESPONSE
────────────────────────────────────────────────────────────────────────────────
Response Initiated: Date: __________ Time: __________
Responder: _________________________

Actions Taken:
[ ] Model suspended
[ ] Threshold adjusted to: __________
[ ] Manual review implemented
[ ] Stakeholders notified: ____________________________________________
[ ] Other: ___________________________________________________________

ROOT CAUSE ANALYSIS
────────────────────────────────────────────────────────────────────────────────
Root Cause Category:
  [ ] Data drift
  [ ] Population shift
  [ ] Feature distribution change
  [ ] Model degradation
  [ ] Implementation error
  [ ] External factor: ________________________________________________
  [ ] Unknown

Detailed Analysis:
___________________________________________________________________________
___________________________________________________________________________
___________________________________________________________________________

Contributing Factors:
  1. ___________________________________________________________________
  2. ___________________________________________________________________
  3. ___________________________________________________________________

REMEDIATION
────────────────────────────────────────────────────────────────────────────────
Short-term Fix:
  Action: _____________________________________________________________
  Implemented by: _________________________ Date: __________
  Effectiveness verified: [ ] Yes [ ] No

Long-term Solution:
  Action: _____________________________________________________________
  Owner: _________________________
  Target completion: __________

Fairness Metrics After Remediation:
  DIR: __________  (Target: ≥ 0.80)
  SPD: __________  (Target: < 0.10)

PREVENTION
────────────────────────────────────────────────────────────────────────────────
Process Changes:
  1. ___________________________________________________________________
  2. ___________________________________________________________________

Monitoring Enhancements:
  1. ___________________________________________________________________
  2. ___________________________________________________________________

Documentation Updates:
  [ ] Model card updated
  [ ] Monitoring checklist updated
  [ ] Incident playbook updated

LESSONS LEARNED
────────────────────────────────────────────────────────────────────────────────
What went well:
___________________________________________________________________________

What could be improved:
___________________________________________________________________________

Key takeaways:
  1. ___________________________________________________________________
  2. ___________________________________________________________________

CLOSURE
────────────────────────────────────────────────────────────────────────────────
Incident Resolved: Date: __________
Closure Approved by: _________________________ Date: __________

Post-Incident Review Completed: [ ] Yes  Date: __________

════════════════════════════════════════════════════════════════════════════════
```

---

*End of Appendix D*
