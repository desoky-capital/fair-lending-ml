# Section 3: Teaching & Deployment Notes

This section provides instructors and team leaders with materials to help learners master data quality concepts through hands-on exercises. Whether you're teaching a university course or training a data engineering team, these materials provide structure and assessment criteria.

---

## 3.1 Learning Objectives & Assessment Rubric

### Core Learning Objectives

By the end of this chapter, learners should be able to:

**LO1: Diagnostic Skills**
- Systematically identify data quality issues across multiple dimensions (completeness, validity, consistency)
- Distinguish between critical issues (must fix) and cosmetic issues (nice to fix)
- Prioritize data quality work based on downstream impact

**LO2: Technical Implementation**
- Build reproducible data cleaning pipelines using pandas and Python
- Implement schema validation and type coercion
- Handle missing data with appropriate strategies (drop, impute, flag)
- Perform deduplication and standardization
- Enforce referential integrity across related tables

**LO3: Documentation & Governance**
- Create data dictionaries that capture schema and business meaning
- Implement lineage tracking for all transformations
- Document data quality decisions with clear rationale
- Generate audit-ready reports

**LO4: Business Context**
- Connect data quality practices to regulatory requirements (BCBS 239, SR 11-7)
- Articulate the business impact of data quality failures
- Evaluate trade-offs between data quality, completeness, and timeliness
- Design data quality rules based on business requirements

**LO5: Critical Thinking**
- Question assumptions embedded in data cleaning decisions
- Recognize when automation is appropriate vs. when human judgment is needed
- Identify gaps or limitations in cleaned data
- Propose validation strategies to catch future issues

### Assessment Rubric

Use this rubric to evaluate learners' data cleaning pipelines and documentation:

#### Dimension 1: Data Quality Improvement (25 points)

| Score | Criteria |
|-------|----------|
| **23-25 (Excellent)** | Pipeline addresses all major data quality issues. Appropriate strategies chosen for each issue type. Final data satisfies all business rules. Minimal data loss while maintaining quality. |
| **20-22 (Good)** | Pipeline addresses most major issues. Strategies are generally appropriate. Final data satisfies core business rules. Reasonable balance between quality and completeness. |
| **17-19 (Adequate)** | Pipeline addresses some major issues. Some strategies may be questionable. Final data has minor business rule violations. Some unnecessary data loss or quality compromises. |
| **0-16 (Needs Work)** | Major issues remain unaddressed. Inappropriate strategies (e.g., dropping all nulls). Business rules frequently violated. Excessive data loss or retained bad data. |

**What to look for:**
- Did they fix the missing account_ids and duplicates?
- Are date formats standardized?
- Is referential integrity enforced?
- Are business rules (e.g., ledger >= available balance) satisfied?

#### Dimension 2: Code Quality & Reproducibility (20 points)

| Score | Criteria |
|-------|----------|
| **18-20 (Excellent)** | Code is well-structured with clear functions. Comprehensive comments explain logic. No hardcoded values. Runs without errors. Version controlled. Dependencies documented. |
| **15-17 (Good)** | Code is reasonably organized. Key sections commented. Few hardcoded values. Runs with minor fixes. Basic documentation present. |
| **12-14 (Adequate)** | Code is functional but poorly organized. Minimal comments. Some hardcoded values. Requires manual intervention to run. Limited documentation. |
| **0-11 (Needs Work)** | Code is disorganized or doesn't run. No comments. Heavily hardcoded. Cannot reproduce results. No documentation. |

**What to look for:**
- Can you run their pipeline without modifying code?
- Are magic numbers explained (e.g., why outlier threshold is 50000)?
- Is the code modular (functions, not just one long script)?
- Are column names and variable names self-documenting?

#### Dimension 3: Documentation (20 points)

| Score | Criteria |
|-------|----------|
| **18-20 (Excellent)** | Comprehensive data dictionary. Complete lineage log. Every decision has documented rationale. Assumptions clearly stated. Quality metrics provided. |
| **15-17 (Good)** | Good data dictionary. Lineage log covers key transformations. Most decisions documented. Core assumptions stated. Basic quality metrics. |
| **12-14 (Adequate)** | Basic data dictionary. Partial lineage log. Some decisions documented. Few assumptions stated. Minimal quality metrics. |
| **0-11 (Needs Work)** | Missing or incomplete data dictionary. No lineage tracking. Decisions not explained. Assumptions not stated. No quality metrics. |

**What to look for:**
- Can someone else understand what they did and why?
- Is there an audit trail showing what changed?
- Are trade-offs and limitations acknowledged?
- Would this satisfy a regulatory audit?

#### Dimension 4: Business Understanding (20 points)

| Score | Criteria |
|-------|----------|
| **18-20 (Excellent)** | Data quality rules clearly tied to business requirements. Impact of issues on downstream use cases articulated. Appropriate trade-offs made with justification. Regulatory context understood. |
| **15-17 (Good)** | Most rules have business justification. Awareness of downstream impact. Trade-offs generally appropriate. Basic regulatory awareness. |
| **12-14 (Adequate)** | Some business context present. Limited discussion of impact. Trade-offs not always justified. Minimal regulatory awareness. |
| **0-11 (Needs Work)** | No business context. Decisions seem arbitrary. No consideration of trade-offs. No regulatory awareness. |

**What to look for:**
- Do they explain *why* an issue matters (not just *what* it is)?
- Can they articulate who would be harmed by bad data?
- Do they reference regulatory requirements appropriately?
- Are decisions grounded in business needs vs. technical convenience?

#### Dimension 5: Validation & Testing (15 points)

| Score | Criteria |
|-------|----------|
| **14-15 (Excellent)** | Comprehensive validation checks. Automated tests for business rules. Comparison to reference data. Edge cases tested. Quality metrics calculated and evaluated. |
| **11-13 (Good)** | Key validation checks present. Some automated tests. Basic comparison to reference. Quality metrics calculated. |
| **8-10 (Adequate)** | Minimal validation. Few or no automated tests. Limited comparison. Some quality metrics. |
| **0-7 (Needs Work)** | No validation. No automated tests. No comparison to reference. No quality metrics. |

**What to look for:**
- Did they validate their cleaned data meets requirements?
- Are there automated tests (e.g., "assert no duplicates in account_id")?
- Did they compare before/after quality metrics?
- Did they test edge cases (e.g., what if all values in a column are NULL)?

### Total: 100 points

**Grading scale:**
- 90-100: Excellent (A)
- 80-89: Good (B)
- 70-79: Adequate (C)
- Below 70: Needs significant improvement

---

## 3.2 Group Exercise: Documenting Data Decisions

**Time required**: 60 minutes (10 min setup, 30 min group work, 20 min debrief)

**Learning objectives**: LO3 (Documentation), LO4 (Business Context), LO5 (Critical Thinking)

### Exercise Overview

Teams explore a mystery dataset with data quality issues, make cleaning decisions, and must justify those decisions as if presenting to stakeholders.

**The scenario**: Your team has just received a dataset from a legacy system that's being decommissioned. The data will be migrated to a new system, but first it must be cleaned. Different teams will make different decisions—you must explain yours.

### Setup (10 minutes)

**Instructor preparation:**

1. Create 4-5 "mystery datasets" (variations of the Atlas Bank data with different issues emphasized)
2. Divide class into teams of 3-4 people
3. Each team gets a different dataset
4. Provide the documentation template (below)

**Materials for each team:**
- Mystery dataset CSV file
- Business rules document (what's acceptable, what's not)
- Decision documentation template
- Access to Python/pandas for exploration

### Group Work (30 minutes)

**Phase 1: Exploration (10 min)**

Teams load their data and systematically identify issues:
```python
import pandas as pd

df = pd.read_csv('mystery_dataset_A.csv')

# Explore structure
print(df.info())
print(df.describe())

# Look for issues
print("\nMissing values:")
print(df.isnull().sum())

print("\nUnique value counts:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique")

print("\nFirst 20 rows:")
print(df.head(20))
```

**Phase 2: Decision-making (15 min)**

For each issue found, teams must decide and document:
1. What is the issue? (Be specific: "45 records have NULL in account_id column")
2. Why is it a problem? (Business impact: "Can't identify these accounts for reporting")
3. What are the options? (Drop records, impute values, flag for review, etc.)
4. What do you recommend? (Choose one option)
5. What's your rationale? (Why this option over alternatives?)

Use the template:

```markdown
## Data Quality Decision Log

### Issue #1
**What**: [Specific description with count/percentage]
**Why it matters**: [Business/regulatory impact]
**Options considered**:
  1. Option A - [description] - Pros: X, Cons: Y
  2. Option B - [description] - Pros: X, Cons: Y
**Recommendation**: [Chosen option]
**Rationale**: [Why this option? Trade-offs accepted?]
**Implementation**: [How would you code this?]

### Issue #2
[Repeat structure]
```

**Phase 3: Preparation (5 min)**

Teams prepare to present ONE decision (the most interesting/controversial one) to the class.

### Debrief (20 minutes)

**Round 1: Presentations (10 min)**

Each team presents their chosen decision (2 min each):
- What was the issue?
- What did you decide?
- Why?

Instructor notes: Don't critique during presentations—just let them present.

**Round 2: Discussion (10 min)**

Instructor facilitates discussion:

"**Observation 1**: Three teams dropped records with missing account_ids, but Team C kept them and flagged for review. Let's discuss..."
- What are the trade-offs?
- In what contexts would each approach be correct?
- How does the downstream use case matter?

"**Observation 2**: Teams handled date format inconsistencies differently..."
- Team A: Parsed all formats, kept everything
- Team B: Kept only standardized format, dropped rest
- Which is more "correct"? (Trick question: depends on risk tolerance and data volume)

"**Observation 3**: No team considered [X]. Why not?"
- Often teams miss temporal validation, referential integrity, or privacy implications
- Use this to highlight blind spots

**Key teaching moments:**
- There's rarely ONE right answer—context matters
- Documentation of rationale is as important as the decision itself
- Business stakeholders need different explanations than technical ones
- Trade-offs should be explicit, not hidden

### Variations for Different Audiences

**🎓 For Academic Settings:**
- Require written reports (3-5 pages) instead of just presentations
- Grade on quality of reasoning, not "correctness" of decision
- Have teams critique another team's decisions

**🏢 For Practitioner Training:**
- Use real (anonymized) data from your organization
- Invite business stakeholders to the debrief to provide real-world context
- Have teams present to "executive committee" (role-play)
- Turn findings into actual data quality backlog items

### Common Issues Teams Will Face

**Issue**: "There's not enough time to document everything properly!"
**Teaching point**: In real projects, this pressure is constant. Documentation is the first thing to get cut. But it's also what kills you in audits. Practice doing it under time pressure.

**Issue**: "We can't agree on what to do with these records."
**Teaching point**: Good! Disagreement means you're thinking critically. Document both viewpoints. In real projects, you'd escalate to a business owner.

**Issue**: "We don't know the business context, so how can we decide?"
**Teaching point**: Exactly. This is why data cleaning can't be purely technical. You need business SMEs involved. When you don't have context, document your assumptions and flag for review.

---

## 3.3 Assignment: Critique the "Bad Pipeline"

**Time required**: 3-5 hours (outside of class)

**Learning objectives**: All LOs, especially LO5 (Critical Thinking)

**Due date**: Typically 1-2 weeks after covering the material

### Assignment Overview

Students receive a working but deeply flawed data cleaning pipeline. Their job: identify what's wrong, explain why it matters, and fix one critical issue.

This assignment forces students to think critically about data quality practices—it's easy to follow a good example, harder to recognize what makes it good by seeing what's bad.

### Provided Materials

**File 1: `bad_pipeline.ipynb`** (Jupyter notebook with problematic code)

The notebook:
- ✅ Loads data and produces output
- ✅ Removes some data quality issues
- ❌ Makes questionable assumptions silently
- ❌ Has no documentation or lineage tracking
- ❌ Uses inconsistent strategies across similar issues
- ❌ Would fail a regulatory audit

**File 2: `bad_pipeline_data.zip`** (Sample data to test the pipeline)
- Raw data with known issues
- Reference clean data (for comparison, not provided to students initially)

**File 3: `memo_template.md`** (Structure for their critique)

### Example "Bad" Patterns Intentionally Planted

To give you a sense of what students will find, here are example problems in the bad pipeline:

**Problem 1: Silent assumptions**
```python
# Bad: No explanation, no logging
df = df[df['balance'] > 0]
```
**What's wrong**: Drops all negative balances without explanation. Is this a data quality rule or a business filter? Why is negative balance invalid? This could be dropping valid credit card or loan accounts.

**Problem 2: Inconsistent handling**
```python
# Bad: Different strategies for similar issues
df['date1'] = pd.to_datetime(df['date1'], errors='coerce')  # Silently converts to NaT
df['date2'] = pd.to_datetime(df['date2'])  # Raises error
```
**What's wrong**: Same type of field, different error handling. No logging of how many dates failed to parse. Inconsistent = not reproducible.

**Problem 3: No lineage**
```python
# Bad: No record of what changed
df = df.drop_duplicates()
print("Removed duplicates")
```
**What's wrong**: How many duplicates? Which records were kept? Why kept first vs. last? No audit trail.

**Problem 4: Dangerous imputations**
```python
# Bad: Imputing financial data
df['transaction_amount'].fillna(df['transaction_amount'].mean(), inplace=True)
```
**What's wrong**: Never impute financial amounts! Missing amounts should be investigated or dropped, not filled with averages.

**Problem 5: Hardcoded business rules**
```python
# Bad: Magic number, no explanation
df = df[df['amount'] < 10000]
```
**What's wrong**: Why 10000? Is this a business rule (max transaction limit) or a data quality check (outlier removal)? No documentation, no configuration.

**Problem 6: No validation**
```python
# Bad: No tests
df_clean = clean_data(df)
df_clean.to_csv('clean_data.csv')
```
**What's wrong**: No validation that the cleaned data actually satisfies business rules. How do you know it worked?

**Problem 7: Manual steps mentioned**
```python
# Bad: Can't reproduce
# Note: Before running this, manually remove any rows 
# where customer_id looks suspicious
```
**What's wrong**: Manual steps destroy reproducibility. What does "suspicious" mean? Who decides? Can't audit this.

**Problem 8: Irreversible operations**
```python
# Bad: Overwrites source data
df.to_csv('accounts.csv')  # Overwrites the input file!
```
**What's wrong**: Destroys the original data. Can't go back, can't compare before/after. Recipe for disaster.

### Assignment Structure

**Part 1: Written Critique (60% of grade)**

Write a 3-5 page memo identifying issues in the bad pipeline:

**Section 1: Critical Issues** (that could lead to compliance failures or wrong decisions)
For each issue:
- Describe what the code does
- Explain why it's problematic (reference chapter concepts)
- Assess the risk/impact (high/medium/low)
- Propose a specific fix

**Section 2: Moderate Issues** (that reduce trustworthiness but don't immediately cause harm)

**Section 3: Minor Issues** (technical debt or code quality)

**Section 4: Overall Assessment**
- Is this pipeline production-ready? Why or why not?
- What would you ask the author before approving this?
- What's the biggest risk if this pipeline ran in production?

Use this structure:
```markdown
## Critical Issue #1: [Title]

**Location**: Cell 5, lines 12-15
**What the code does**: [Describe]
**Why it's problematic**: [Explain with reference to data quality principles]
**Risk/Impact**: HIGH - [Why this could cause serious problems]
**Proposed fix**: [Specific code changes or process changes]
**Rationale**: [Why this fix addresses the root cause]
```

**Part 2: Implement One Fix (40% of grade)**

Choose ONE critical issue and implement a fix:

1. **Create a corrected version** of the problematic code section
2. **Demonstrate it works** with before/after comparison
3. **Document your approach** (why you chose this solution)
4. **Add validation** (tests to ensure it stays fixed)

Submit:
- `fixed_pipeline.ipynb` (your corrected version)
- `fix_explanation.md` (documentation of your changes)

Example fix structure:
```python
# BEFORE (from bad pipeline):
df = df[df['balance'] > 0]  # Silent assumption, no logging

# AFTER (your fix):
def remove_negative_balances(df, logger):
    """
    Remove records with negative balances for deposit accounts.
    
    Business rule: Checking and savings accounts cannot have 
    negative balances (per account terms). Credit accounts CAN
    have negative balances (customer owes bank).
    
    Args:
        df: DataFrame with account data
        logger: DataQualityLogger instance
    
    Returns:
        DataFrame with invalid negative balances removed
    """
    initial_count = len(df)
    
    # Only apply to deposit accounts
    deposit_mask = df['account_type'].isin(['checking', 'savings'])
    invalid_mask = deposit_mask & (df['balance'] < 0)
    
    if invalid_mask.any():
        invalid_count = invalid_mask.sum()
        print(f"Found {invalid_count} deposit accounts with negative balances")
        df_clean = df[~invalid_mask].copy()
        
        logger.log_issue(
            table='accounts',
            column='balance',
            issue_type='business_rule_violation',
            count=invalid_count,
            action='drop_record',
            reason='Deposit accounts cannot have negative balances per account terms'
        )
    else:
        df_clean = df.copy()
    
    final_count = len(df_clean)
    print(f"Removed {initial_count - final_count} records")
    
    return df_clean

# Usage:
logger = DataQualityLogger()
df_clean = remove_negative_balances(df, logger)

# Validation test:
assert (df_clean[df_clean['account_type'].isin(['checking', 'savings'])]['balance'] >= 0).all(), \
    "Deposit accounts should not have negative balances"
```

### Grading Criteria

**Critique (60 points):**
- Identified major issues (25 pts) - Did they catch the serious problems?
- Quality of analysis (20 pts) - Do they understand *why* issues matter?
- Specificity (15 pts) - Are recommendations concrete and actionable?

**Implementation (40 points):**
- Fix works correctly (15 pts) - Does the code run and fix the issue?
- Documentation (15 pts) - Is the fix well-documented with rationale?
- Validation (10 pts) - Are there tests to prevent regression?

### Common Student Mistakes (and How to Address)

**Mistake 1**: "This pipeline is bad because it doesn't use fancy techniques like machine learning imputation."
**Correction**: Simplicity is a virtue in data quality. Advanced ≠ better. Focus on transparency and reproducibility.

**Mistake 2**: Listing dozens of tiny issues (spacing, variable naming) but missing critical ones.
**Correction**: Prioritization matters. What would actually cause a production incident or audit failure?

**Mistake 3**: Fixing everything in Part 2 instead of choosing one issue.
**Correction**: The point is to go deep on ONE fix, showing you can implement best practices, not to rewrite the whole pipeline.

**Mistake 4**: Proposing fixes that aren't actually implementable.
**Correction**: "Add more data quality checks" isn't specific enough. Show the code.

### Variations

**🎓 For advanced students:**
- Provide a more subtle bad pipeline (issues are less obvious)
- Require them to propose a testing strategy for the fixed pipeline
- Have them refactor the entire pipeline, not just fix one issue

**🏢 For practitioners:**
- Use an actual pipeline from your organization (with permission/anonymization)
- Frame as a "code review" exercise
- Have senior engineers review their critiques
- Implement fixes in your actual codebase (if appropriate)

---

## 3.4 Additional Teaching Resources

### Mini-Exercises (5-10 minutes each)

**Exercise 1: Spot the Issue**
Show code snippets, ask students to identify the problem:
```python
# Snippet A
df['date'] = df['date'].apply(lambda x: x.replace('/', '-'))

# Snippet B  
if df['amount'].isnull().any():
    df = df.dropna(subset=['amount'])

# Snippet C
df = df[df['account_type'] != 'unknown']
```

**Discussion**: What's wrong? How would you fix it? What's the impact?

**Exercise 2: Choose Your Strategy**
Present a scenario with missing data:
- "10% of transaction amounts are NULL. What do you do?"
- Have students vote (drop, impute, flag) and justify

**Exercise 3: Write the Documentation**
Give students cleaned data and ask them to reverse-engineer the data dictionary:
- What does each field mean?
- What are valid values?
- What business rules apply?

### Real-World Case Studies

**Case 1: The JPMorgan Whale Trade**
- Trading loss partly attributed to data quality issues in risk models
- Excel error in Value-at-Risk calculation
- Lessons: Testing with realistic data, human review, automation

**Case 2: Knight Capital**
- Covered in Section 1, but can discuss in more detail
- Focus on test data vs. production data mismatches

**Case 3: Robinhood Outages**
- Multiple outages during high-volume trading days
- Data pipeline couldn't handle load
- Lessons: Scalability testing, data quality under stress

### Guest Speakers (if possible)

Invite practitioners to speak about:
- "A time when data quality failed us" (war stories)
- "How we structure data governance in our organization"
- "What regulators actually look for in audits"

### Office Hours Topics

Common questions students ask:

**Q**: "How do I know if my data quality is 'good enough'?"
**A**: Depends on use case. For regulatory reporting, very high bar. For exploratory analysis, lower bar. Always document limitations.

**Q**: "Should I always drop records with missing primary keys?"
**A**: Generally yes—if you can't identify the record, it's useless. But log it and investigate why they're missing.

**Q**: "Is it okay to impute missing values?"
**A**: Depends. Imputing non-critical fields (e.g., middle name) is fine. Imputing financial amounts or dates is dangerous. Document assumptions.

**Q**: "How much data loss is acceptable?"
**A**: No fixed rule, but >10% should trigger investigation. Why so much? Are you being too strict?

---

## 3.5 Instructor Notes for In-Class Delivery

### Suggested Pacing (for 3-hour class session)

**Hour 1: Lecture + Live Demo**
- 20 min: Section 1 (Problem Framing) key points
- 40 min: Live coding - work through Section 2 part 2.1 and 2.2 (load data, first cleaning layer)
  - Show code, run it, explain decisions
  - Invite questions throughout

**Hour 2: Guided Practice**
- 30 min: Students replicate what you just did on their own machines
  - Walk around, help troubleshoot
- 30 min: Continue live coding - Section 2.3 and 2.4 (documentation, lineage)

**Hour 3: Group Exercise**
- 60 min: Group exercise (Section 3.2) with teams

**Homework**: Critique the bad pipeline (Section 3.3)

### Tips for Live Coding

✅ **DO:**
- Type slowly enough for students to follow
- Explain what you're about to do before you type
- Show mistakes and how to debug them
- Use print statements liberally to show what's happening
- Save versions as you go (pipeline_v1.py, pipeline_v2.py)

❌ **DON'T:**
- Just show finished code
- Skip explaining "boring" parts (imports, setup)
- Go too fast (students will get lost)
- Assume everyone knows pandas basics

### Common Sticking Points

**Issue**: Students' code doesn't match yours
**Solution**: Provide notebook checkpoints. "By the end of section 2.2, your data should look like this..."

**Issue**: Students spend 20 minutes debugging installation issues
**Solution**: Provide Docker container or Google Colab link pre-configured

**Issue**: Diverse skill levels (some students finish fast, others struggle)
**Solution**: Provide "extension challenges" for advanced students

### Creating a Supportive Learning Environment

**Normalize mistakes**: Data quality work IS messy. Even experts make mistakes. The key is catching them through validation.

**Encourage questions**: "I don't understand why we're doing this" is a great question. If one person is confused, others are too.

**Connect to their experiences**: Ask students to share data quality issues they've encountered. Make it real.

---

## 3.6 Adapting for Different Audiences

### For Undergraduate Students

**Adjustments:**
- Spend more time on Python/pandas basics
- Simplify business context (less regulatory detail)
- More scaffolding in assignments (templates, starter code)
- Focus on learning the patterns, less on business judgment

**Additional support:**
- Python refresher tutorial
- Pandas cheat sheet
- Worked examples with detailed comments

### For Graduate Students (MS in FinTech, etc.)

**Adjustments:**
- Assume programming competence
- Emphasize business/regulatory context heavily
- More open-ended assignments
- Connect to other courses (risk management, ethics, etc.)

**Higher expectations:**
- Students should be able to design pipelines from scratch
- Should connect data quality to downstream model performance
- Should understand regulatory implications deeply

### For Practitioners (Corporate Training)

**Adjustments:**
- Use real data from your organization (anonymized)
- Focus on immediate applicability
- Less emphasis on grading, more on learning
- Invite business stakeholders to discussions

**Unique elements:**
- Map to your existing data governance framework
- Turn exercises into backlog items
- Follow up with coaching as they implement

**Success metrics:**
- Do they actually improve data quality in production?
- Do they start documenting decisions?
- Do audits go more smoothly?

---

## 3.7 Assessment Alternatives

Beyond the main assignment, consider:

### Option 1: Portfolio Project

Instead of small assignments, one big project:
- Choose a real dataset (e.g., from Kaggle, government open data)
- Clean it end-to-end
- Document everything
- Present to class

**Pros**: More realistic, allows creativity
**Cons**: Higher effort, harder to grade consistently

### Option 2: Peer Review

Students complete assignment, then review a peer's work:
- Use the same rubric you'd use for grading
- Write feedback
- Discuss in pairs

**Pros**: Learn by evaluating others, see different approaches
**Cons**: Requires good rubric, some students uncomfortable critiquing

### Option 3: Kaggle-Style Competition

Provide messy data, define quality metrics:
- Who can achieve highest quality score?
- Must document their approach
- Leaderboard for motivation

**Pros**: Gamification, competitive students love it
**Cons**: Can incentivize gaming the metrics, less emphasis on documentation

### Option 4: Case Study Analysis

Provide real-world data quality failure case:
- What went wrong?
- What could have prevented it?
- Design a pipeline that would catch this issue

**Pros**: Connects to real consequences, critical thinking
**Cons**: Less hands-on coding

---

## 3.8 Resources for Continued Learning

### Recommended Readings

**Books:**
- *Data Quality: The Accuracy Dimension* by Jack Olson
- *Principles of Data Management* by DAMA International
- *The Data Warehouse Toolkit* by Ralph Kimball (Chapter on data quality)

**Papers/Articles:**
- BCBS 239 guidance document (Bank for International Settlements)
- SR 11-7 guidance (Federal Reserve)
- "The Data Quality Framework" (various financial institutions publish these)

**Online Resources:**
- Great Expectations documentation (data validation library)
- Pandas documentation (data cleaning functions)
- SQL Murder Mystery (fun SQL practice)

### Tools to Explore

**Data Quality:**
- Great Expectations (Python) - automated data validation
- dbt (data build tool) - analytics engineering, built-in testing
- Apache Griffin - data quality platform

**Data Lineage:**
- OpenLineage - open standard for lineage
- Marquez - metadata service
- SQLLineage - parse SQL for lineage

**Documentation:**
- DBT Docs - auto-generate documentation
- Data Hub - data discovery platform
- Amundsen - metadata and discovery platform

---

## Final Note for Instructors

This chapter is foundational. If students master these concepts:
- Chapter 3 (credit modeling) will be easier - they'll understand where model data comes from
- Chapter 4 (fraud) will be easier - they'll know how to prepare transaction data
- Chapter 5 (fairness) will be easier - they'll understand biases introduced by data quality choices

But if they skip ahead without mastering data foundations, they'll struggle. Push them to really understand *why* we do things this way, not just *how*.

The best compliment you can get: "I used what I learned in this chapter at my internship, and my manager was impressed with how I documented my data cleaning." That means they get it.

Good luck, and feel free to adapt these materials to your context!
