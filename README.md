# Task 2: Exploratory Data Analysis (EDA)

**Organization**: WeIntern Pvt Ltd  
**Track**: Data Science & Machine Learning Engineering  
**Module**: Exploratory Data Analysis, Statistical Profiling & Business Intelligence  
**Author**: Bineeta Yadav (Data Science Intern, WeIntern Pvt Ltd)  
**Date**: October 2026  
**Language & Tools**: Python 3.12, Pandas 2.3, NumPy 2.2, Seaborn 0.13, Matplotlib 3.11, Jupyter, ReportLab  

---

## 📌 Project Overview & Objective

This repository contains the complete, production-grade submission for **Task 2: Exploratory Data Analysis (EDA)**.

The primary objective is to perform a rigorous statistical and visual Exploratory Data Analysis on the cleaned customer churn and demographic dataset (`data/customer_churn_cleaned.csv`), uncovering underlying behavioral patterns, cohort trajectories, multicollinearities, and actionable business insights.

### Key Analytical Highlights:
- **1,413 Validated Customer Profiles** evaluated across 16 continuous, categorical, and temporal attributes.
- **Parametric & Non-Parametric Statistics**: Computed mean, median, standard deviation, quartiles, IQR, skewness, and kurtosis across all numerical features, alongside categorical frequency breakdowns.
- **7 Publication-Quality Visualization Types**:
  1. **Histograms & KDE Curves**: Univariate density and distribution modeling.
  2. **Pearson Correlation Heatmap**: Linear dependence and multicollinearity screening.
  3. **Categorical Box Plots**: Segment dispersion, quartile spread, and anomaly inspection.
  4. **Time-Series Line Charts**: Chronological cohort sign-up velocity and ARPU trajectories.
  5. **Bivariate Scatter Plots**: Multi-variable interactions with regression trendlines.
  6. **Grouped Bar Charts**: Segmented churn rate comparisons across key dimensions.
  7. **Violin Plots**: Bimodal charge densities stratified by customer retention status.
- **5 Actionable Strategic Insights**: Evidence-backed conclusions linking statistical metrics directly to commercial business decisions.
- **Strategic Action Plan**: Specific, measurable initiatives to mitigate subscriber churn and protect annual recurring revenue.

---

## 📊 Summary of Descriptive Statistics

### Numerical Attributes Summary Table
| Feature | Count | Mean ($\mu$) | Std Dev ($\sigma$) | Min | $Q_1$ (25%) | Median (50%) | $Q_3$ (75%) | Max | IQR | Skewness | Kurtosis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `age` | 1,413 | 41.22 | 12.81 | 18.00 | 33.00 | 41.00 | 49.00 | 75.00 | 16.00 | +0.20 | -0.32 |
| `annual_income` | 1,413 | $91,269 | $29,965 | $30,651 | $70,052 | $87,396 | $106,055 | $160,060 | $36,003 | +0.69 | +0.02 |
| `tenure_months` | 1,413 | 36.85 | 20.42 | 0.00 | 20.00 | 37.00 | 54.00 | 72.00 | 34.00 | -0.06 | -1.07 |
| `monthly_charges`| 1,413 | $72.48 | $32.32 | $18.55 | $47.66 | $70.97 | $96.03 | $168.59 | $48.37 | +0.55 | +0.40 |
| `total_charges` | 1,413 | $2,633 | $2,338 | $0.00 | $970.28 | $1,985 | $3,703 | $12,138 | $2,733 | +1.78 | +4.25 |
| `satisfaction_score`| 1,413| 2.95 | 1.33 | 1.00 | 2.00 | 3.00 | 4.00 | 5.00 | 2.00 | +0.07 | -1.10 |

### Categorical Attributes Mode Breakdown
| Feature | Total Count | Unique Levels | Top Category (Mode) | Mode Frequency | Mode Share (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `gender` | 1,413 | 4 | Female | 578 | 40.9% |
| `contract_type` | 1,413 | 3 | Two Year | 596 | 42.2% |
| `payment_method` | 1,413 | 5 | Electronic Check | 420 | 29.7% |
| `internet_service`| 1,413 | 3 | Fiber Optic | 556 | 39.3% |
| `tech_support` | 1,413 | 3 | No | 670 | 47.4% |
| `city` | 1,413 | 16 | Mumbai | 96 | 6.8% |
| `churn` | 1,413 | 2 | No (Retained) | 1,038 | 73.5% |

---

## 📈 Visualizations Suite & Key Findings

| Figure | Output Path | Chart Type | Core Analytical Insight |
| :--- | :--- | :--- | :--- |
| **Figure 1** | `outputs/figures/01_numerical_distributions_hist_kde.png` | **Histogram + KDE** | Symmetrical bell curve for age; bimodal pricing clusters for monthly charges ($20–$40 budget vs $75–$110 premium). |
| **Figure 2** | `outputs/figures/02_correlation_matrix_heatmap.png` | **Correlation Heatmap** | Strong linear collinearity between total spend and monthly charges ($r = +0.64$) and tenure ($r = +0.57$); age is churn-invariant ($r = +0.01$). |
| **Figure 3** | `outputs/figures/03_categorical_boxplots.png` | **Categorical Box Plots** | Stable income distributions across churn outcomes; two-year contracts show higher median monthly spend ($75/mo) with minimal churn. |
| **Figure 4** | `outputs/figures/04_temporal_acquisition_trends.png` | **Time-Series Line Chart** | Stable monthly customer acquisition (25–35 signups/mo) and consistent ARPU ($70–$75.50) across 2020–2024 cohorts. |
| **Figure 5** | `outputs/figures/05_bivariate_scatter_relationships.png` | **Scatter Plot + Regression** | Fan-shaped total charges expansion across tenure; long-term contracts dominate the high-LTV quadrant ($>\$4,000$). |
| **Figure 6** | `outputs/figures/06_grouped_churn_rate_by_category.png` | **Grouped Bar Charts** | 41.7% churn in Month-to-Month contracts vs 2.9% in Two-Year; Electronic Check payers churn at 34.8% vs 14.2% for Credit Card. |
| **Figure 7** | `outputs/figures/07_violin_charges_distribution.png` | **Violin Plots** | Churned accounts concentrate heavily in the early lifecycle ($<\$1,200$ total charges) and high monthly fee band ($80–$100/mo). |

---

## 💡 5 Core Data-Driven Business Insights

### 1. Contract Commitment is the Primary Retention Shield
- **Quantitative Finding**: Month-to-Month customers churn at **41.7%**, whereas Two-Year contract holders churn at **2.9%** (a **14.4x risk reduction**).
- **Business Impact**: Month-to-Month contracts act as an open door for competitor migration. Locking customers into annual contracts reduces portfolio attrition risk by over **73%**.

### 2. High-ARPU Fiber Optic Vulnerability
- **Quantitative Finding**: Fiber Optic subscribers generate the top Average Revenue Per User (**$84.50/mo**), but suffer from **33.5% churn** when Technical Support is absent.
- **Business Impact**: Customers paying premium prices expect zero friction; when technical onboarding falters, buyer remorse leads to immediate cancellation.

### 3. Payment Channel Friction in Electronic Checks
- **Quantitative Finding**: Customers paying through **Electronic Check** experience a **34.8% churn rate**, compared to **14.2% for automated Credit Cards** and **15.6% for Bank Transfers**.
- **Business Impact**: Manual payment processes trigger a monthly re-evaluation of subscription value and induce involuntary payment failure lapses.

### 4. The 12-Month Early Tenure Attrition Cliff
- **Quantitative Finding**: Over **68% of all churn events** occur within the first 12 months (median tenure of churned users is 11 months vs 45 months for retained users).
- **Business Impact**: The enterprise incurs Customer Acquisition Cost (CAC) without achieving payback maturity for nearly one in four customers.

### 5. Direct CSAT-to-Revenue Sensitivity
- **Quantitative Finding**: Customers rating satisfaction $\le 2$ have an **82.4% probability of churning**, while those rating $\ge 4$ retain at **96.2%**.
- **Business Impact**: CSAT scores are high-fidelity leading indicators; rapid escalation can prevent predictable revenue leakage.

---

## 🎯 Strategic Business Recommendations

1. **Annual Contract Incentive Funnel**:
   - Offer an attractive 10%–15% discount or first month free upon signing a 1-year contract to convert Month-to-Month users.
2. **Dedicated 90-Day Tech Support for Fiber Optic**:
   - Bundle complimentary 24/7 technical assistance with every new Fiber Optic activation to eliminate onboarding churn.
3. **Auto-Pay Migration Bill Credits**:
   - Provide a recurring $5 monthly bill discount for users switching from Electronic Check to automated Credit Card or Bank Transfer Auto-Pay.
4. **Early Lifecycle Milestone Outreach**:
   - Deploy automated customer appreciation rewards and touchpoints at months 3, 6, and 12 to guide accounts past the critical 1-year attrition cliff.
5. **Real-Time CSAT Detractor Rescue Protocol**:
   - Automatically trigger urgent tickets to Customer Success for any customer rating $\le 2$ within 24 hours of survey submission.

---

## 📁 Repository Structure

```
2nd_Week_Task2/
├── data/
│   └── customer_churn_cleaned.csv         <- Validated clean dataset (1,413 rows)
├── notebooks/
│   └── exploratory_data_analysis.ipynb    <- Executed Jupyter Notebook with all outputs & plots
├── outputs/
│   ├── figures/
│   │   ├── 01_numerical_distributions_hist_kde.png  <- Figure 1: Histograms & KDE
│   │   ├── 02_correlation_matrix_heatmap.png        <- Figure 2: Pearson Correlation Heatmap
│   │   ├── 03_categorical_boxplots.png              <- Figure 3: Categorical Box Plots
│   │   ├── 04_temporal_acquisition_trends.png       <- Figure 4: Time-Series Trends
│   │   ├── 05_bivariate_scatter_relationships.png   <- Figure 5: Multi-Variable Scatter Plots
│   │   ├── 06_grouped_churn_rate_by_category.png    <- Figure 6: Grouped Bar Charts
│   │   └── 07_violin_charges_distribution.png       <- Figure 7: Violin Density Plots
│   └── tables/
│       ├── descriptive_statistics_numeric.csv       <- Numeric summary metrics
│       ├── descriptive_statistics_categorical.csv   <- Categorical frequency metrics
│       ├── correlation_matrix.csv                   <- Pearson correlation matrix
│       └── grouped_churn_metrics.csv                <- Consolidated grouped churn metrics
├── reports/
│   ├── eda_summary_report.md              <- Comprehensive 10-section technical report
│   └── eda_report.pdf                     <- Executive PDF report with styled tables & embedded charts
├── src/
│   ├── run_eda_analysis.py                <- Standalone statistical & visualization script
│   ├── build_notebook.py                  <- Notebook builder & execution engine
│   ├── generate_pdf_report.py             <- PDF compiler (ReportLab)
│   └── run_pipeline.py                    <- Master runner executing all steps end-to-end
├── .gitignore                             <- Python and Jupyter ignore rules
├── requirements.txt                       <- Python dependencies
└── README.md                              <- Comprehensive documentation
```

---

## ⚡ Execution Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Complete End-to-End Pipeline
Execute the master runner to perform statistical analysis, generate all 7 figures, execute the Jupyter notebook, and compile the executive PDF report in one command:
```bash
python src/run_pipeline.py
```

### 3. Launch Jupyter Notebook
```bash
jupyter notebook notebooks/exploratory_data_analysis.ipynb
```

---

## 🎯 Deliverables Checklist & Evaluation Alignment

- [x] **Jupyter Notebook (`.ipynb`)**: Available at `notebooks/exploratory_data_analysis.ipynb` with all 10 steps, markdown documentation, code cells, and pre-rendered outputs.
- [x] **At least 5 Distinct Visualizations**: 7 distinct visualization types generated and saved in `outputs/figures/`.
- [x] **Correlation Matrix & Heatmap**: Computed and analyzed in both tabular format and annotated heatmap.
- [x] **At least 3 Actionable Insights**: 5 distinct, data-driven insights detailed with business implications.
- [x] **Distributions Analyzed**: Histograms with overlay KDE curves and summary metrics.
- [x] **Grouped Analysis**: Grouped breakdowns across contract types, services, payment channels, and time cohorts.
- [x] **Written Insights Section within Notebook**: Formatted in Markdown directly inside the notebook.
- [x] **Executive PDF Export**: High-resolution multi-page PDF compiled at `reports/eda_report.pdf`.
