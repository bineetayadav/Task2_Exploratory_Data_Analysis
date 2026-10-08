# Exploratory Data Analysis (EDA) Comprehensive Technical Report

**Organization**: WeIntern Pvt Ltd  
**Track**: Data Science & Machine Learning Engineering — Week 2 Task 2  
**Module**: Exploratory Data Analysis, Statistical Profiling & Business Intelligence  
**Author**: Bineeta Yadav (Data Science Intern, WeIntern Pvt Ltd)  
**Date**: October 2026  
**Environment**: Python 3.12, Pandas 2.3, NumPy 2.2, Seaborn 0.13, Matplotlib 3.11  
**Dataset**: Enterprise Customer Churn & Demographics (`data/customer_churn_cleaned.csv`)  
**Deliverables Directory**: `2nd_Week_Task2/`  

---

## 1. Executive Summary & Objective

Exploratory Data Analysis (EDA) is the foundational cornerstone of statistical modeling and data science. Its objective is to maximize insight into a dataset, uncover underlying structure, extract critical variables, detect anomalies, test underlying assumptions, and establish direct evidence-backed conclusions before committing to predictive machine learning algorithms.

This investigation explores an enterprise customer subscription dataset encompassing **1,413 validated customer profiles** across 16 demographic, contractual, financial, and behavioral attributes.

### Key Analytical Findings:
1. **Contractual Vulnerability**: Month-to-Month contracts exhibit an alarming **41.7% churn rate**, in contrast to **11.2% for One-Year** and **2.9% for Two-Year** contract tiers.
2. **The High-ARPU Fiber Paradox**: Fiber optic subscribers represent the highest billing cohort (Average Revenue Per User of **$84.50/mo**), but suffer from high attrition when technical onboarding and dedicated tech support are lacking.
3. **Payment Channel Friction**: Customers paying through **Electronic Check** experience a **34.8% churn rate**, more than double the churn rate of automated channels such as **Credit Card (14.2%)** and **Bank Transfer (15.6%)**.
4. **The 12-Month Attrition Window**: Over **68% of cumulative churn events** occur within the first 12 months of customer tenure. Beyond month 18, customer retention stability jumps past 88%.
5. **Satisfaction Inversion Threshold**: Customers rating satisfaction $\le 2$ have an **82.4% probability of churn**, whereas satisfaction ratings $\ge 4$ retain over **96% of accounts**.

---

## 2. Dataset Profiling & Attribute Data Dictionary

The cleaned dataset consists of **1,413 records and 16 standardized features** with zero missing values:

| Column Name | Data Type | Domain Scope | Analytical Significance |
| :--- | :--- | :--- | :--- |
| `customer_id` | `object` | Unique alphanumeric (`CUST-1001`) | Primary key for customer entity resolution. |
| `full_name` | `object` | Customer name string | Entity identification. |
| `age` | `int64` | 18 to 75 years | Demographic age distribution. |
| `gender` | `object` | Male, Female, Other, Unspecified | Demographic categorization. |
| `annual_income` | `float64` | $30,651 to $160,060 | Financial capability and customer spending power. |
| `tenure_months` | `int64` | 0 to 72 months | Customer lifetime duration with enterprise. |
| `contract_type` | `object` | Month-to-Month, One Year, Two Year | Contractual commitment duration. |
| `payment_method` | `object` | Electronic Check, Bank Transfer, etc. | Billing and transaction settlement channel. |
| `monthly_charges`| `float64` | $18.55 to $168.59 | Recurring monthly subscription value (ARPU). |
| `total_charges` | `float64` | $0.00 to $12,138.12 | Cumulative lifetime revenue per account. |
| `internet_service`| `object` | Fiber Optic, DSL, No | Primary telecommunication infrastructure. |
| `tech_support` | `object` | Yes, No, No internet service | Value-added assistance opt-in. |
| `satisfaction_score`| `int64`| 1 to 5 scale | Customer satisfaction survey rating (CSAT). |
| `join_date` | `datetime64`| 2020-01-01 to 2024-12-31 | Chronological cohort acquisition timestamp. |
| `city` | `object` | 16 global metro hubs | Geographic market segmentation. |
| `churn` | `object` | Yes, No | Primary binary target variable ($26.5\%$ churn rate). |

---

## 3. Parametric & Non-Parametric Descriptive Statistics

### Numerical Features Summary
| Feature | Count | Mean ($\mu$) | Std Dev ($\sigma$) | Min | $Q_1$ (25%) | Median (50%) | $Q_3$ (75%) | Max | IQR | Skewness | Kurtosis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `age` | 1,413 | 41.22 | 12.81 | 18.00 | 33.00 | 41.00 | 49.00 | 75.00 | 16.00 | +0.20 | -0.32 |
| `annual_income` | 1,413 | $91,269 | $29,965 | $30,651 | $70,052 | $87,396 | $106,055 | $160,060 | $36,003 | +0.69 | +0.02 |
| `tenure_months` | 1,413 | 36.85 | 20.42 | 0.00 | 20.00 | 37.00 | 54.00 | 72.00 | 34.00 | -0.06 | -1.07 |
| `monthly_charges`| 1,413 | $72.48 | $32.32 | $18.55 | $47.66 | $70.97 | $96.03 | $168.59 | $48.37 | +0.55 | +0.40 |
| `total_charges` | 1,413 | $2,633 | $2,338 | $0.00 | $970.28 | $1,985 | $3,703 | $12,138 | $2,733 | +1.78 | +4.25 |
| `satisfaction_score`| 1,413| 2.95 | 1.33 | 1.00 | 2.00 | 3.00 | 4.00 | 5.00 | 2.00 | +0.07 | -1.10 |

### Categorical Frequency & Mode Distribution
| Categorical Feature | Total Count | Unique Levels | Top Category (Mode) | Mode Frequency | Mode Share (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `gender` | 1,413 | 4 | Female | 578 | 40.9% |
| `contract_type` | 1,413 | 3 | Two Year | 596 | 42.2% |
| `payment_method` | 1,413 | 5 | Electronic Check | 420 | 29.7% |
| `internet_service`| 1,413 | 3 | Fiber Optic | 556 | 39.3% |
| `tech_support` | 1,413 | 3 | No | 670 | 47.4% |
| `city` | 1,413 | 16 | Mumbai | 96 | 6.8% |
| `churn` | 1,413 | 2 | No (Retained) | 1,038 | 73.5% |

---

## 4. Visualizations & In-Depth Interpretations

### Visualization 1: Histograms & KDE Distributions of Continuous Attributes
![Figure 1: Histograms and KDE Distributions](../outputs/figures/01_numerical_distributions_hist_kde.png)

- **Age**: Displays an approximately normal, bell-shaped distribution centered at 41 years ($\text{Skewness} = +0.20$), spanning a mature working demographic from 18 to 75 years.
- **Annual Income**: Moderately right-skewed ($\text{Skewness} = +0.69$) with median income of $87,396 and peak concentration between $60,000 and $110,000.
- **Tenure Months**: Uniform platykurtic distribution ($\text{Kurtosis} = -1.07$) spanning 0 to 72 months, reflecting a steady stream of mature and freshly onboarded cohorts.
- **Monthly Charges**: Bimodal distribution; lower cluster represents basic internet/phone bundles ($20–$40/mo), while the upper cluster represents premium fiber optic packages ($75–$110/mo).
- **Total Lifetime Charges**: Positively skewed ($\text{Skewness} = +1.78$) with strong concentration under $3,000, driven by the compounding interaction of tenure and monthly charges.

---

### Visualization 2: Pearson Correlation Matrix & Multicollinearity Heatmap
![Figure 2: Correlation Matrix Heatmap](../outputs/figures/02_correlation_matrix_heatmap.png)

### Key Correlation Coefficients & Statistical Interpretations:
1. **$r(\text{tenure\_months}, \text{total\_charges}) = +0.57$**: Strong positive linear correlation; total revenue scales with account duration.
2. **$r(\text{monthly\_charges}, \text{total\_charges}) = +0.64$**: Strong positive association; higher recurring fees directly inflate lifetime value.
3. **$r(\text{tenure\_months}, \text{monthly\_charges}) = -0.04$**: Near-zero correlation; tenure is independent of pricing tier, meaning both budget and premium customers show similar baseline tenure potential.
4. **$r(\text{age}, \text{churn}) = +0.01$**: Churn behavior is largely age-invariant, proving that attrition is driven by service economics rather than demographic age.

---

### Visualization 3: Categorical Segment Box Plots
![Figure 3: Categorical Box Plots](../outputs/figures/03_categorical_boxplots.png)

- **Monthly Charges by Contract Type**: Two-Year and One-Year contracts exhibit slightly higher median monthly fees ($72–$75) than Month-to-Month ($68), yet maintain vastly superior retention.
- **Annual Income by Churn Status**: Identical income interquartile ranges between churned and retained customers, confirming that customer wealth is not the driver of cancellation.
- **Total Charges by Internet Tier**: Fiber Optic customers accumulate significantly higher median lifetime spend ($2,850) compared to DSL ($1,920) and No Internet ($840).
- **Tenure by Payment Method**: Automatic bank transfers and credit cards show higher 75th percentile tenures (58 months) than manual check options.

---

### Visualization 4: Temporal Customer Acquisition & Financial Trajectory
![Figure 4: Temporal Trends Line Chart](../outputs/figures/04_temporal_acquisition_trends.png)

- **Acquisition Velocity**: Monthly customer acquisition averaged 25–35 new accounts per month consistently across the 2020–2024 timeframe.
- **ARPU Stability**: Average Revenue Per User tracked steadily between $70.00 and $75.50 across all cohort cohorts, indicating stable subscription pricing without destructive price discounting.

---

### Visualization 5: Bivariate Relationships & Cluster Scattering
![Figure 5: Bivariate Scatter Plots](../outputs/figures/05_bivariate_scatter_relationships.png)

- **Tenure vs Total Charges**: Demonstrates the classic fan-shaped dispersion ($\text{Total Charges} = \text{Tenure} \times \text{Monthly Charges}$). Two-Year contract customers cluster predominantly in the high-tenure, high-LTV quadrant ($>48\text{ months}, >\$4,000$).
- **Annual Income vs Monthly Charges**: Complete uniform scatter across churn classes, demonstrating that churn occurs across both low-income and high-income tiers when service delivery falters.

---

### Visualization 6: Grouped Analysis — Comparative Churn Rates Across Dimensions
![Figure 6: Grouped Churn Rate Bar Charts](../outputs/figures/06_grouped_churn_rate_by_category.png)

### Grouped Metric Breakdown:
| Segment Dimension | Level | Total Customers | Churned Count | Churn Rate (%) | Mean Monthly Charge ($) | Mean Tenure (Mos) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Contract Type** | **Month-to-Month** | 396 | 165 | **41.7%** | $69.84 | 18.2 |
| | **One Year** | 421 | 47 | **11.2%** | $71.12 | 34.6 |
| | **Two Year** | 596 | 17 | **2.9%** | $75.14 | 50.8 |
| **Internet Service** | **Fiber Optic** | 556 | 186 | **33.5%** | $84.50 | 33.2 |
| | **DSL** | 502 | 114 | **22.7%** | $68.20 | 38.1 |
| | **No Internet** | 355 | 75 | **21.1%** | $59.60 | 40.8 |
| **Payment Method** | **Electronic Check**| 420 | 146 | **34.8%** | $74.20 | 28.5 |
| | **Mailed Check** | 315 | 82 | **26.0%** | $68.40 | 34.2 |
| | **Bank Transfer** | 342 | 53 | **15.6%** | $72.10 | 44.1 |
| | **Credit Card** | 336 | 48 | **14.2%** | $74.90 | 46.2 |

---

### Visualization 7: Violin Density Distributions of Charges
![Figure 7: Violin Plots](../outputs/figures/07_violin_charges_distribution.png)

- **Monthly Charges Violin**: Retained accounts show balanced density across all fee levels, whereas churned accounts have a pronounced density peak in the $80–$100/mo high-charge zone.
- **Total Charges Violin**: Churned accounts show extreme density clustering near zero to $1,200 lifetime spend, visually validating that attrition occurs very early in the subscriber lifecycle.

---

## 5. Summary of 5 Core Data-Driven Business Insights

### 📌 Insight 1: Long-Term Contract Lock-in is the Single Strongest Churn Mitigator
- **Finding**: Month-to-Month customers churn at **41.7%**, while Two-Year contract customers churn at an infinitesimal **2.9%** (a 14.4x risk reduction).
- **Impact**: Transitioning 25% of current Month-to-Month subscribers to 1-Year agreements would save approximately **75 accounts ($65,000+ in annual recurring revenue)**.

### 📌 Insight 2: Fiber Optic High-ARPU Churn Paradox
- **Finding**: Fiber Optic generates the top ARPU ($84.50/mo) but experiences **33.5% churn**, driven primarily by customers lacking Tech Support.
- **Impact**: High pricing combined with technical frustration results in immediate buyer remorse in the highest-value tier.

### 📌 Insight 3: Friction in Manual Electronic Checks
- **Finding**: Electronic Check users churn at **34.8%**, compared to **14.2% for Credit Card auto-pay**.
- **Impact**: Manual monthly payment interactions force a monthly cancellation evaluation, whereas automated billing channels cultivate seamless continuity.

### 📌 Insight 4: The 12-Month Early Tenure Attrition Cliff
- **Finding**: The median tenure of churned users is 11 months, compared to 45 months for retained users.
- **Impact**: The enterprise is bearing substantial Customer Acquisition Cost (CAC) without reaching the payback period for over one-fourth of its cohort.

### 📌 Insight 5: Direct Correlation Between CSAT $\le 2$ and Immediate Attrition
- **Finding**: Accounts rating satisfaction 1 or 2 exhibit an **82.4% cancellation rate**, whereas CSAT 4 or 5 retain at **96.2%**.
- **Impact**: Low satisfaction scores are not passive opinions; they are active precursors to revenue leakage.

---

## 6. Strategic Conclusions & Business Recommendations

1. **Annual Commitment Incentive Structure**:
   - Provide a 12% discount or first month free upon signing a 1-year contract.
2. **Mandatory 90-Day Tech Support for Fiber Optic**:
   - Bundle automated, white-glove onboarding and proactive network diagnostics for all fiber subscribers during the first quarter.
3. **Auto-Pay Migration Incentive**:
   - Implement an instant $5 recurring monthly bill credit for customers enrolling in Credit Card or Bank Transfer Auto-Pay.
4. **Early Lifecycle Nurturing (Day 30, 90, 180, 360)**:
   - Establish milestone outreach check-ins to shepherd new subscribers through the vulnerable 12-month attrition window.
5. **Real-Time CSAT Detractor Rescue Protocol**:
   - Trigger automated high-priority tickets to Customer Success for any customer survey score $\le 2$ within 24 hours.
