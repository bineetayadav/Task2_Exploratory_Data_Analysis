"""
run_eda_analysis.py
Performs comprehensive Exploratory Data Analysis (EDA) on the cleaned customer churn dataset:
1. Loads and audits the cleaned dataset
2. Computes parametric & non-parametric descriptive statistics for numeric & categorical features
3. Generates 7 publication-quality visualization figures:
   - 01_numerical_distributions_hist_kde.png (Histograms + KDE)
   - 02_correlation_matrix_heatmap.png (Correlation Matrix Heatmap)
   - 03_categorical_boxplots.png (Categorical Box Plots)
   - 04_temporal_acquisition_trends.png (Time-series Line Charts)
   - 05_bivariate_scatter_relationships.png (Multi-variable Scatter Plots with regression)
   - 06_grouped_churn_rate_by_category.png (Grouped Bar Charts by Segment)
   - 07_violin_charges_distribution.png (Violin Density Distributions)
4. Computes grouped analytical tables (by Contract, Service, Payment, Churn, and Time Cohort)
5. Exports analytical summary tables to CSV
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(data_path: str, base_dir: str):
    figures_dir = os.path.join(base_dir, "outputs", "figures")
    tables_dir = os.path.join(base_dir, "outputs", "tables")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # Set publication visualization theme
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.sans-serif'] = ['Segoe UI', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.edgecolor'] = '#CBD5E1'
    plt.rcParams['figure.facecolor'] = '#FFFFFF'
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.width', 1000)

    print("=" * 80)
    print("STEP 1: LOAD & INSPECT CLEANED DATASET")
    print("=" * 80)
    df = pd.read_csv(data_path)
    df['join_date'] = pd.to_datetime(df['join_date'])
    print(f"Loaded dataset from: {data_path}")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nDataset Info:")
    df.info()

    print("\n" + "=" * 80)
    print("STEP 2: COMPUTE DESCRIPTIVE STATISTICS")
    print("=" * 80)
    num_cols = ["age", "annual_income", "tenure_months", "monthly_charges", "total_charges", "satisfaction_score"]
    cat_cols = ["gender", "contract_type", "payment_method", "internet_service", "tech_support", "city", "churn"]

    # Numeric descriptive stats: mean, std, median, quartiles, IQR, skew, kurtosis
    num_stats = []
    for col in num_cols:
        series = df[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        num_stats.append({
            "Feature": col,
            "Count": int(series.count()),
            "Mean": series.mean(),
            "Std Dev": series.std(),
            "Min": series.min(),
            "25% (Q1)": q1,
            "Median (50%)": series.median(),
            "75% (Q3)": q3,
            "Max": series.max(),
            "IQR": iqr,
            "Skewness": series.skew(),
            "Kurtosis": series.kurtosis()
        })
    df_num_stats = pd.DataFrame(num_stats)
    num_stats_path = os.path.join(tables_dir, "descriptive_statistics_numeric.csv")
    df_num_stats.to_csv(num_stats_path, index=False)
    print(f"Saved numeric descriptive statistics to: {num_stats_path}")
    print(df_num_stats.to_string(index=False))

    # Categorical descriptive stats
    cat_stats = []
    for col in cat_cols:
        series = df[col]
        top_val = series.mode()[0]
        top_freq = (series == top_val).sum()
        cat_stats.append({
            "Feature": col,
            "Count": int(series.count()),
            "Unique Categories": int(series.nunique()),
            "Top Category (Mode)": str(top_val),
            "Top Frequency": int(top_freq),
            "Top Mode Percentage": f"{(top_freq / len(series)):.1%}"
        })
    df_cat_stats = pd.DataFrame(cat_stats)
    cat_stats_path = os.path.join(tables_dir, "descriptive_statistics_categorical.csv")
    df_cat_stats.to_csv(cat_stats_path, index=False)
    print(f"\nSaved categorical descriptive statistics to: {cat_stats_path}")
    print(df_cat_stats.to_string(index=False))

    print("\n" + "=" * 80)
    print("STEP 3: HISTOGRAMS & KDE PLOTS (NUMERICAL DISTRIBUTIONS)")
    print("=" * 80)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    colors = ["#2563EB", "#059669", "#7C3AED", "#D97706", "#DC2626", "#0284C7"]

    for i, col in enumerate(num_cols):
        ax = axes[i]
        sns.histplot(df[col], kde=True, ax=ax, color=colors[i], bins=25, edgecolor="#0F172A", alpha=0.65)
        mean_val = df[col].mean()
        median_val = df[col].median()
        ax.axvline(mean_val, color="#DC2626", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:,.1f}")
        ax.axvline(median_val, color="#16A34A", linestyle="-", linewidth=1.5, label=f"Median: {median_val:,.1f}")
        ax.set_title(f"Distribution: {col.replace('_', ' ').title()}", fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel(col.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.legend(loc="upper right", fontsize=8.5)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Figure 1: Histograms & KDE Distributions of Numerical Attributes", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "01_numerical_distributions_hist_kde.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 1 to: {fig1_path}")

    print("\n" + "=" * 80)
    print("STEP 4: CORRELATION MATRIX & HEATMAP")
    print("=" * 80)
    # Include numeric features plus binary churn encoded
    corr_df = df[num_cols].copy()
    corr_df['churn_binary'] = (df['churn'] == 'Yes').astype(int)
    corr_matrix = corr_df.corr(method='pearson')
    
    corr_path = os.path.join(tables_dir, "correlation_matrix.csv")
    corr_matrix.to_csv(corr_path)
    print(f"Saved correlation matrix to: {corr_path}")
    print(corr_matrix.round(3))

    plt.figure(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        corr_matrix, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
        annot=True, fmt=".2f", square=True, linewidths=.8, cbar_kws={"shrink": .8},
        annot_kws={"size": 10, "weight": "bold"}
    )
    plt.title("Figure 2: Pearson Correlation Matrix Heatmap", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=9.5)
    plt.yticks(rotation=0, fontsize=9.5)
    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "02_correlation_matrix_heatmap.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 2 to: {fig2_path}")

    print("\n" + "=" * 80)
    print("STEP 5: CATEGORICAL BOX PLOTS (OUTLIERS & DISTRIBUTIONS)")
    print("=" * 80)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Monthly Charges by Contract Type
    sns.boxplot(x="contract_type", y="monthly_charges", hue="contract_type", data=df, ax=axes[0, 0], palette="Blues_d", legend=False)
    axes[0, 0].set_title("Monthly Charges ($) by Contract Type", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Contract Type", fontsize=10)
    axes[0, 0].set_ylabel("Monthly Charges ($)", fontsize=10)

    # 2. Annual Income by Churn Status
    sns.boxplot(x="churn", y="annual_income", hue="churn", data=df, ax=axes[0, 1], palette=["#10B981", "#EF4444"], legend=False)
    axes[0, 1].set_title("Annual Income ($) by Churn Status", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Churned", fontsize=10)
    axes[0, 1].set_ylabel("Annual Income ($)", fontsize=10)

    # 3. Total Charges by Internet Service
    sns.boxplot(x="internet_service", y="total_charges", hue="internet_service", data=df, ax=axes[1, 0], palette="Purples_d", legend=False)
    axes[1, 0].set_title("Total Charges ($) by Internet Service Tier", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Internet Service Tier", fontsize=10)
    axes[1, 0].set_ylabel("Total Charges ($)", fontsize=10)

    # 4. Tenure Months by Payment Method
    sns.boxplot(x="payment_method", y="tenure_months", hue="payment_method", data=df, ax=axes[1, 1], palette="YlOrBr", legend=False)
    axes[1, 1].set_title("Tenure (Months) by Payment Method", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Payment Channel", fontsize=10)
    axes[1, 1].set_ylabel("Tenure (Months)", fontsize=10)
    axes[1, 1].tick_params(axis='x', rotation=20)

    plt.suptitle("Figure 3: Categorical Box Plots Showing Segment Distributions", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, "03_categorical_boxplots.png")
    plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 3 to: {fig3_path}")

    print("\n" + "=" * 80)
    print("STEP 6: TIME-SERIES TRENDS (ACQUISITION & REVENUE TRAJECTORY)")
    print("=" * 80)
    # Temporal analysis on join_date
    df['join_year_month'] = df['join_date'].dt.to_period('M')
    monthly_cohort = df.groupby('join_year_month').agg(
        new_signups=('customer_id', 'count'),
        avg_monthly_charges=('monthly_charges', 'mean'),
        churn_count=('churn', lambda x: (x == 'Yes').sum()),
        total_revenue_booked=('total_charges', 'sum')
    ).reset_index()
    monthly_cohort['join_year_month_dt'] = monthly_cohort['join_year_month'].dt.to_timestamp()
    monthly_cohort['cumulative_signups'] = monthly_cohort['new_signups'].cumsum()
    monthly_cohort['churn_rate'] = (monthly_cohort['churn_count'] / monthly_cohort['new_signups']) * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Subplot 1: Customer Acquisition & Cumulative Trajectory
    ax1.plot(monthly_cohort['join_year_month_dt'], monthly_cohort['new_signups'], marker='o', color='#2563EB', linewidth=2.2, label='Monthly New Sign-ups')
    ax1.bar(monthly_cohort['join_year_month_dt'], monthly_cohort['new_signups'], width=20, alpha=0.25, color='#3B82F6')
    ax1.set_title("Monthly Customer Acquisition & Growth Trajectory", fontsize=12, fontweight="bold")
    ax1.set_ylabel("New Customers Acquired", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")

    # Subplot 2: Average Monthly Charges (ARPU) & Churn Rate Trajectory
    ax2.plot(monthly_cohort['join_year_month_dt'], monthly_cohort['avg_monthly_charges'], marker='s', color='#059669', linewidth=2.0, label='Mean Monthly Charge ($ ARPU)')
    ax2.set_title("Monthly ARPU ($) Trajectory Over Time", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Cohort Join Month", fontsize=10)
    ax2.set_ylabel("Average Monthly Charges ($)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left")

    plt.suptitle("Figure 4: Temporal Customer Acquisition & Financial Trajectory", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig4_path = os.path.join(figures_dir, "04_temporal_acquisition_trends.png")
    plt.savefig(fig4_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 4 to: {fig4_path}")

    print("\n" + "=" * 80)
    print("STEP 7: BIVARIATE SCATTER PLOTS WITH REGRESSION & HUE")
    print("=" * 80)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Tenure vs Total Charges by Contract Type
    sns.scatterplot(
        data=df, x="tenure_months", y="total_charges", hue="contract_type",
        palette={"Month-to-Month": "#EF4444", "One Year": "#F59E0B", "Two Year": "#10B981"},
        alpha=0.7, ax=axes[0], s=35
    )
    sns.regplot(data=df, x="tenure_months", y="total_charges", scatter=False, ax=axes[0], color="#1E293B", line_kws={"linewidth": 2, "linestyle": "--"})
    axes[0].set_title("Tenure (Months) vs. Total Charges ($) by Contract Type", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Tenure (Months)", fontsize=10)
    axes[0].set_ylabel("Total Lifetime Charges ($)", fontsize=10)

    # 2. Annual Income vs Monthly Charges by Churn
    sns.scatterplot(
        data=df, x="annual_income", y="monthly_charges", hue="churn",
        palette={"No": "#3B82F6", "Yes": "#DC2626"},
        alpha=0.65, ax=axes[1], s=35
    )
    axes[1].set_title("Annual Income ($) vs. Monthly Charges ($) by Churn", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Annual Income ($)", fontsize=10)
    axes[1].set_ylabel("Monthly Charges ($)", fontsize=10)

    plt.suptitle("Figure 5: Bivariate Relationships & Segment Clustering", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig5_path = os.path.join(figures_dir, "05_bivariate_scatter_relationships.png")
    plt.savefig(fig5_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 5 to: {fig5_path}")

    print("\n" + "=" * 80)
    print("STEP 8: GROUPED ANALYSIS BY CATEGORY & CHURN BREAKDOWN")
    print("=" * 80)
    # Grouped metrics by Contract Type
    contract_grp = df.groupby('contract_type').agg(
        Total_Customers=('customer_id', 'count'),
        Churned_Customers=('churn', lambda x: (x == 'Yes').sum()),
        Avg_Monthly_Charges=('monthly_charges', 'mean'),
        Avg_Tenure=('tenure_months', 'mean'),
        Avg_Satisfaction=('satisfaction_score', 'mean')
    ).reset_index()
    contract_grp['Churn_Rate_Pct'] = (contract_grp['Churned_Customers'] / contract_grp['Total_Customers']) * 100

    # Grouped metrics by Internet Service
    internet_grp = df.groupby('internet_service').agg(
        Total_Customers=('customer_id', 'count'),
        Churned_Customers=('churn', lambda x: (x == 'Yes').sum()),
        Avg_Monthly_Charges=('monthly_charges', 'mean'),
        Avg_Tenure=('tenure_months', 'mean')
    ).reset_index()
    internet_grp['Churn_Rate_Pct'] = (internet_grp['Churned_Customers'] / internet_grp['Total_Customers']) * 100

    # Grouped metrics by Payment Method
    payment_grp = df.groupby('payment_method').agg(
        Total_Customers=('customer_id', 'count'),
        Churned_Customers=('churn', lambda x: (x == 'Yes').sum()),
        Avg_Monthly_Charges=('monthly_charges', 'mean'),
        Avg_Tenure=('tenure_months', 'mean')
    ).reset_index()
    payment_grp['Churn_Rate_Pct'] = (payment_grp['Churned_Customers'] / payment_grp['Total_Customers']) * 100

    # Export consolidated grouped churn metrics
    contract_grp['Segment_Type'] = 'Contract Type'
    contract_grp.rename(columns={'contract_type': 'Segment_Value'}, inplace=True)
    internet_grp['Segment_Type'] = 'Internet Service'
    internet_grp.rename(columns={'internet_service': 'Segment_Value'}, inplace=True)
    payment_grp['Segment_Type'] = 'Payment Method'
    payment_grp.rename(columns={'payment_method': 'Segment_Value'}, inplace=True)

    consolidated_grp = pd.concat([contract_grp, internet_grp, payment_grp], ignore_index=True)
    grp_path = os.path.join(tables_dir, "grouped_churn_metrics.csv")
    consolidated_grp.to_csv(grp_path, index=False)
    print(f"Saved grouped churn metrics to: {grp_path}")

    # Plot Grouped Churn Rates
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Churn by Contract Type
    bars1 = axes[0].bar(contract_grp['Segment_Value'], contract_grp['Churn_Rate_Pct'], color=['#DC2626', '#F59E0B', '#10B981'], edgecolor='#0F172A', alpha=0.85)
    axes[0].set_title("Churn Rate (%) by Contract Type", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Churn Rate (%)", fontsize=10)
    for b in bars1:
        h = b.get_height()
        axes[0].text(b.get_x() + b.get_width()/2, h + 1, f"{h:.1f}%", ha='center', fontweight='bold', fontsize=9.5)
    axes[0].set_ylim(0, max(contract_grp['Churn_Rate_Pct']) * 1.25)

    # 2. Churn by Internet Service
    bars2 = axes[1].bar(internet_grp['Segment_Value'], internet_grp['Churn_Rate_Pct'], color=['#3B82F6', '#8B5CF6', '#64748B'], edgecolor='#0F172A', alpha=0.85)
    axes[1].set_title("Churn Rate (%) by Internet Service", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Churn Rate (%)", fontsize=10)
    for b in bars2:
        h = b.get_height()
        axes[1].text(b.get_x() + b.get_width()/2, h + 1, f"{h:.1f}%", ha='center', fontweight='bold', fontsize=9.5)
    axes[1].set_ylim(0, max(internet_grp['Churn_Rate_Pct']) * 1.25)

    # 3. Churn by Payment Method
    bars3 = axes[2].bar(payment_grp['Segment_Value'], payment_grp['Churn_Rate_Pct'], color=['#0284C7', '#E11D48', '#059669', '#D97706', '#94A3B8'], edgecolor='#0F172A', alpha=0.85)
    axes[2].set_title("Churn Rate (%) by Payment Method", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Churn Rate (%)", fontsize=10)
    axes[2].tick_params(axis='x', rotation=25)
    for b in bars3:
        h = b.get_height()
        axes[2].text(b.get_x() + b.get_width()/2, h + 1, f"{h:.1f}%", ha='center', fontweight='bold', fontsize=9)
    axes[2].set_ylim(0, max(payment_grp['Churn_Rate_Pct']) * 1.25)

    plt.suptitle("Figure 6: Grouped Analysis — Comparative Churn Rates Across Key Dimensions", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig6_path = os.path.join(figures_dir, "06_grouped_churn_rate_by_category.png")
    plt.savefig(fig6_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 6 to: {fig6_path}")

    # Figure 7: Violin Plot (Distribution density of Monthly Charges and Total Charges by Churn)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.violinplot(x="churn", y="monthly_charges", hue="churn", data=df, ax=axes[0], palette=["#10B981", "#EF4444"], split=False, inner="quartile", legend=False)
    axes[0].set_title("Monthly Charges ($) Density Distribution by Churn", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Churn Status", fontsize=10)
    axes[0].set_ylabel("Monthly Charges ($)", fontsize=10)

    sns.violinplot(x="churn", y="total_charges", hue="churn", data=df, ax=axes[1], palette=["#10B981", "#EF4444"], split=False, inner="quartile", legend=False)
    axes[1].set_title("Total Lifetime Charges ($) Density Distribution by Churn", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Churn Status", fontsize=10)
    axes[1].set_ylabel("Total Lifetime Charges ($)", fontsize=10)

    plt.suptitle("Figure 7: Violin Density Distributions of Charges by Churn Status", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig7_path = os.path.join(figures_dir, "07_violin_charges_distribution.png")
    plt.savefig(fig7_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 7 to: {fig7_path}")

    print("\n" + "=" * 80)
    print("EDA ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"Saved 7 figures to: {figures_dir}")
    print(f"Saved 4 summary tables to: {tables_dir}")
    print("=" * 80)

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_file = os.path.join(base_dir, "data", "customer_churn_cleaned.csv")
    run_eda(csv_file, base_dir)
