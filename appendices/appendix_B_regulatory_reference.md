# Appendix B: Regulatory Quick Reference

---

## B.1 Overview: Key Regulations for Credit AI

**Table B.1: Key Regulations Overview**

| Regulation | Jurisdiction | Focus | Key Requirement |
|------------|--------------|-------|-----------------|
| **ECOA** | US Federal | Fair lending | No discrimination; adverse action notices |
| **FCRA** | US Federal | Credit reporting | Accuracy; dispute resolution |
| **SR 11-7** | US Federal | Model risk | Validation; documentation; monitoring |
| **Regulation B** | US Federal | Credit decisions | Specific denial reasons required |
| **EU AI Act** | European Union | AI systems | Risk classification; conformity assessment |

---

## B.2 Equal Credit Opportunity Act (ECOA)

### Overview

**Table B.2: ECOA Overview**

| Aspect | Detail |
|--------|--------|
| **Enforced by** | CFPB, DOJ, FTC |
| **Applies to** | Any creditor (banks, fintechs, lenders) |
| **Purpose** | Prohibit credit discrimination |

### Protected Characteristics

ECOA prohibits discrimination based on:

**Table B.3: ECOA Protected Characteristics**

| Protected Class | Notes |
|-----------------|-------|
| Race | Includes ethnicity |
| Color | Skin color |
| Religion | Any religious affiliation |
| National origin | Country of birth/ancestry |
| Sex | Includes gender identity (per CFPB guidance) |
| Marital status | Single, married, divorced, widowed |
| Age | With limited exceptions for elderly |
| Public assistance | Receipt of welfare, Social Security |

### Key Requirements

**Table B.4: ECOA Key Requirements**

| Requirement | What It Means | How We Address It |
|-------------|---------------|-------------------|
| **No disparate treatment** | Cannot intentionally discriminate | Exclude protected attributes from features |
| **No disparate impact** | Cannot have discriminatory effect | Test with DIR ≥ 0.80 |
| **Adverse action notice** | Must explain denials | SHAP-based explanations |
| **Record retention** | Keep records 25 months | Documentation system |

### The 4/5ths Rule

```
If: Approval_rate(Group B) / Approval_rate(Group A) < 0.80

Then: Potential disparate impact violation
      Burden shifts to creditor to prove business necessity
```

### Penalties

**Table B.5: ECOA Penalties**

| Violation Type | Potential Penalty |
|----------------|-------------------|
| Individual | Actual damages + punitive (up to $10,000) |
| Class action | Up to $500,000 or 1% of net worth |
| Pattern/practice | DOJ enforcement, injunctive relief |

---

## B.3 Fair Credit Reporting Act (FCRA)

### Overview

**Table B.6: FCRA Overview**

| Aspect | Detail |
|--------|--------|
| **Citation** | 15 U.S.C. § 1681 |
| **Enforced by** | CFPB, FTC |
| **Applies to** | Credit bureaus, users of credit reports |
| **Purpose** | Ensure accuracy of credit information |

### Key Requirements for Model Users

**Table B.7: FCRA Key Requirements**

| Requirement | What It Means | How We Address It |
|-------------|---------------|-------------------|
| **Permissible purpose** | Can only pull credit for valid reason | Document business purpose |
| **Adverse action notice** | Specific notice when taking adverse action | Automated notice generation |
| **Accuracy obligation** | Must use accurate information | Data quality pipeline |
| **Dispute process** | Must investigate disputes | Escalation procedures |

### Adverse Action Notice Requirements

When denying credit based on a credit report, must provide:

**Table B.8: Adverse Action Notice Requirements**

| Element | Description |
|---------|-------------|
| **Notice of action** | Clear statement of denial |
| **Credit bureau info** | Name, address, phone of bureau used |
| **Right to free report** | Consumer can get free copy within 60 days |
| **Right to dispute** | Consumer can dispute accuracy |
| **Reasons for action** | Specific factors (up to 4) |

### Sample Adverse Action Reasons

- Insufficient credit history
- High debt-to-income ratio
- Recent delinquencies on record
- Too many recent credit inquiries
- High credit utilization

---

## B.4 SR 11-7: Model Risk Management

### Overview

**Table B.9: SR 11-7 Overview**

| Aspect | Detail |
|--------|--------|
| **Citation** | Federal Reserve SR 11-7 |
| **Issued by** | Federal Reserve, OCC |
| **Applies to** | Banks, bank holding companies |
| **Purpose** | Manage risks from model use |

### Model Risk Management Framework

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SR 11-7 FRAMEWORK                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  MODEL DEVELOPMENT                                                  │
│  • Sound theory and methodology                                     │
│  • Robust data and assumptions                                      │
│  • Appropriate documentation                                        │
│                                                                     │
│  MODEL VALIDATION                                                   │
│  • Independent review                                               │
│  • Outcomes analysis                                                │
│  • Ongoing monitoring                                               │
│                                                                     │
│  MODEL GOVERNANCE                                                   │
│  • Clear ownership                                                  │
│  • Change control                                                   │
│  • Model inventory                                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Documentation Requirements

**Table B.10: SR 11-7 Documentation Requirements**

| Document | Contents |
|----------|----------|
| **Model documentation** | Purpose, methodology, assumptions, limitations |
| **Validation report** | Test results, findings, recommendations |
| **Ongoing monitoring** | Performance metrics, drift detection |
| **Change log** | All modifications with rationale |

### Key Principles

**Table B.11: SR 11-7 Key Principles**

| Principle | Application to Credit AI |
|-----------|--------------------------|
| **Effective challenge** | Independent validation of fairness testing |
| **Comprehensive inventory** | All credit models catalogued |
| **Risk-based approach** | Higher scrutiny for high-impact models |
| **Board oversight** | Leadership accountability for model risk |

---

## B.5 EU AI Act

### Overview

**Table B.12: EU AI Act Overview**

| Aspect | Detail |
|--------|--------|
| **Citation** | Regulation (EU) 2024/1689 |
| **Effective** | Phased: 2024-2027 |
| **Applies to** | AI systems in/affecting EU |
| **Purpose** | Regulate AI by risk level |

### Risk Classification

**Table B.13: EU AI Act Risk Classification**

| Risk Level | Examples | Requirements |
|------------|----------|--------------|
| **Unacceptable** | Social scoring, manipulation | Banned |
| **High** | Credit scoring, employment | Full compliance |
| **Limited** | Chatbots, emotion recognition | Transparency only |
| **Minimal** | Spam filters, games | None specific |

### Credit Scoring = High Risk

Credit and insurance AI is explicitly classified as HIGH RISK, requiring:

**Table B.14: EU AI Act High-Risk Requirements**

| Requirement | Description | How We Address It |
|-------------|-------------|-------------------|
| **Risk management system** | Identify and mitigate risks | Fairness monitoring framework |
| **Data governance** | Representative, error-free data | Data quality pipeline (Ch 2) |
| **Technical documentation** | System description, capabilities | Model documentation |
| **Record keeping** | Automatic logging | Monitoring snapshots |
| **Transparency** | Clear info to users | Adverse action notices |
| **Human oversight** | Ability to intervene | Threshold adjustment, manual review |
| **Accuracy & robustness** | Consistent performance | Validation/test evaluation |

### Conformity Assessment

Before deployment, high-risk AI systems must undergo conformity assessment culminating in **CE (Conformité Européenne) Marking**—the EU's certification mark indicating the system meets all regulatory requirements.

```
Development → Internal Assessment → Documentation → CE Marking → Deployment
                                          ↓
                              (Some categories require third-party audit)
```

### Timeline

**Table B.15: EU AI Act Timeline**

| Date | Milestone |
|------|-----------|
| **Aug 2024** | Entry into force |
| **Feb 2025** | Prohibited AI banned |
| **Aug 2025** | High-risk requirements apply |
| **Aug 2026** | Full enforcement |

### Penalties

**Table B.16: EU AI Act Penalties**

| Violation | Maximum Fine |
|-----------|--------------|
| Prohibited AI | €35M or 7% global revenue |
| High-risk non-compliance | €15M or 3% global revenue |
| Incorrect information | €7.5M or 1% global revenue |

---

## B.6 CFPB Guidance on AI

### Key Positions

**Table B.17: CFPB Key Positions on AI**

| Topic | CFPB Position |
|-------|---------------|
| **Black box models** | Complexity doesn't excuse lack of explanation |
| **Adverse action** | Must provide specific, actionable reasons |
| **Proxy discrimination** | Facially neutral features can violate ECOA |
| **Alternative data** | Subject to same fair lending requirements |

### Recent Enforcement Trends

**Table B.18: CFPB Enforcement Trends**

| Trend | Implication |
|-------|-------------|
| Increased AI scrutiny | More exams focused on algorithmic lending |
| Proxy variable focus | Testing for race/gender correlation |
| Explanation quality | Generic reasons insufficient |
| Continuous monitoring | One-time validation not enough |

---

## B.7 State-Level Regulations

### Active State Laws

**Table B.19: Active State AI Laws**

| State | Law | Key Provisions |
|-------|-----|----------------|
| **Colorado** | SB21-169 | Insurance AI must be tested for discrimination |
| **California** | Various | Algorithmic accountability proposals |
| **Illinois** | BIPA extensions | Biometric data in AI |
| **New York** | Local Law 144 | Bias audits for employment AI |

### Emerging State Activity

**Table B.20: Emerging State AI Activity**

| State | Status | Focus |
|-------|--------|-------|
| Connecticut | Enacted | AI in employment |
| Texas | Proposed | Algorithmic transparency |
| Washington | Proposed | Facial recognition limits |

---

## B.8 Compliance Checklist

### Pre-Deployment

**Table B.21: Pre-Deployment Compliance Checklist**

| Requirement | Regulation | ✓ |
|-------------|------------|---|
| Protected attributes excluded from features | ECOA | ☐ |
| DIR ≥ 0.80 for all groups | ECOA | ☐ |
| Adverse action notice system ready | ECOA, FCRA | ☐ |
| Model documentation complete | SR 11-7 | ☐ |
| Independent validation performed | SR 11-7 | ☐ |
| Risk management system in place | EU AI Act | ☐ |
| Human oversight capability | EU AI Act | ☐ |
| Data quality documented | EU AI Act | ☐ |

### Post-Deployment

**Table B.22: Post-Deployment Compliance Checklist**

| Requirement | Frequency | ✓ |
|-------------|-----------|---|
| Fairness metrics monitoring | Daily/Weekly | ☐ |
| Performance monitoring | Daily | ☐ |
| Adverse action notice audit | Monthly | ☐ |
| Full model review | Quarterly | ☐ |
| Regulatory update review | Quarterly | ☐ |
| Comprehensive audit | Annual | ☐ |

---

## B.9 Key Contacts

**Table B.23: Key Regulatory Contacts**

| Agency | Role | Website |
|--------|------|---------|
| **CFPB** | Consumer financial protection | consumerfinance.gov |
| **DOJ Civil Rights** | Fair lending enforcement | justice.gov/crt |
| **OCC** | Bank supervision | occ.treas.gov |
| **Federal Reserve** | Bank holding company supervision | federalreserve.gov |
| **FTC** | Non-bank enforcement | ftc.gov |
| **EU AI Office** | EU AI Act implementation | digital-strategy.ec.europa.eu |

---

*End of Appendix B*
