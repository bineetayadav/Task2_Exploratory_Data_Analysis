"""
generate_pdf_report.py
Compiles a publication-grade, multi-page executive PDF report:
reports/eda_report.pdf
Utilizing ReportLab with custom typography, structured data tables, and embedded high-res figures.
"""

import os
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 755, "Task 2: Exploratory Data Analysis (EDA) — Executive Report")
            self.drawRightString(558, 755, "WeIntern Pvt Ltd")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 748, 558, 748)
            
        # Footer
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 45, 558, 45)
        self.drawString(54, 32, "Confidential — Data Science & Machine Learning Engineering Track")
        self.drawRightString(558, 32, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()

def build_pdf_report(base_dir: str):
    pdf_path = os.path.join(base_dir, "reports", "eda_report.pdf")
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=10
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=10,
        spaceAfter=5
    )

    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=5
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#1E293B")
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#0F172A")
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("Exploratory Data Analysis (EDA) Technical Report", title_style))
    story.append(Paragraph("<b>Organization</b>: WeIntern Pvt Ltd &nbsp;|&nbsp; <b>Author</b>: Bineeta Yadav (Data Science Intern) &nbsp;|&nbsp; <b>Date</b>: October 2026", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceAfter=8))

    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary & Core Objectives", h1_style))
    exec_text = (
        "This report provides an in-depth Exploratory Data Analysis (EDA) of an enterprise telecommunications and subscription "
        "dataset spanning <b>1,413 validated customer accounts</b>. The investigation integrates univariate statistical profiling, "
        "correlation diagnostics, multi-dimensional categorical boxplots, chronological acquisition modeling, and grouped churn rate "
        "evaluations to unearth behavioral risk factors and provide strategic business recommendations."
    )
    story.append(Paragraph(exec_text, body_style))

    # KPI Summary Cards
    kpi_data = [
        [
            Paragraph("<b>Analyzed Customers</b><br/><font size=11 color='#1E3A8A'><b>1,413</b></font>", body_style),
            Paragraph("<b>Overall Churn Rate</b><br/><font size=11 color='#DC2626'><b>26.54% (375 Churned)</b></font>", body_style),
            Paragraph("<b>Average Monthly ARPU</b><br/><font size=11 color='#059669'><b>$72.48 / mo</b></font>", body_style),
            Paragraph("<b>Total Lifetime Value</b><br/><font size=11 color='#2563EB'><b>$3.72 Million</b></font>", body_style)
        ]
    ]
    kpi_table = Table(kpi_data, colWidths=[126, 126, 126, 126])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#E2E8F0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
        ('RIGHTPADDING', (0,0), (-1,-1), 7),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 8))

    # 2. Descriptive Statistics Table
    story.append(Paragraph("2. Descriptive Statistics Summary (Continuous Attributes)", h1_style))
    num_csv = os.path.join(base_dir, "outputs", "tables", "descriptive_statistics_numeric.csv")
    if os.path.exists(num_csv):
        ndf = pd.read_csv(num_csv)
        num_rows = [[
            Paragraph("<b>Feature</b>", table_header_style),
            Paragraph("<b>Mean</b>", table_header_style),
            Paragraph("<b>Std Dev</b>", table_header_style),
            Paragraph("<b>Min</b>", table_header_style),
            Paragraph("<b>Median</b>", table_header_style),
            Paragraph("<b>Max</b>", table_header_style),
            Paragraph("<b>Skewness</b>", table_header_style)
        ]]
        for _, row in ndf.iterrows():
            num_rows.append([
                Paragraph(f"<b>{row['Feature']}</b>", table_cell_bold),
                Paragraph(f"{row['Mean']:,.2f}", table_cell_style),
                Paragraph(f"{row['Std Dev']:,.2f}", table_cell_style),
                Paragraph(f"{row['Min']:,.2f}", table_cell_style),
                Paragraph(f"{row['Median (50%)']:,.2f}", table_cell_style),
                Paragraph(f"{row['Max']:,.2f}", table_cell_style),
                Paragraph(f"{row['Skewness']:+.2f}", table_cell_style)
            ])
        t_num = Table(num_rows, colWidths=[90, 70, 70, 60, 70, 75, 65])
        t_num.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E3A8A")),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ]))
        story.append(t_num)
        story.append(Spacer(1, 6))

    # 3. Figure 1: Distributions
    story.append(Paragraph("3. Figure 1: Univariate Histograms & Kernel Density Estimation (KDE)", h1_style))
    fig1_path = os.path.join(base_dir, "outputs", "figures", "01_numerical_distributions_hist_kde.png")
    if os.path.exists(fig1_path):
        story.append(Image(fig1_path, width=490, height=195))
        story.append(Spacer(1, 6))

    story.append(PageBreak())

    # 4. Correlation Analysis & Figure 2
    story.append(Paragraph("4. Figure 2: Pearson Correlation Matrix Heatmap", h1_style))
    story.append(Paragraph(
        "Strongest positive associations occur between total lifetime charges and monthly charges ($r = +0.64$), "
        "as well as tenure ($r = +0.57$). Demographic variables like age exhibit near-zero correlation with churn ($r = +0.01$), "
        "confirming that customer attrition is structurally driven by contract terms and service satisfaction rather than age cohorts.", body_style
    ))
    fig2_path = os.path.join(base_dir, "outputs", "figures", "02_correlation_matrix_heatmap.png")
    if os.path.exists(fig2_path):
        story.append(Image(fig2_path, width=370, height=230))
        story.append(Spacer(1, 6))

    # 5. Categorical Boxplots & Figure 3
    story.append(Paragraph("5. Figure 3: Categorical Segment Box Plots", h1_style))
    fig3_path = os.path.join(base_dir, "outputs", "figures", "03_categorical_boxplots.png")
    if os.path.exists(fig3_path):
        story.append(Image(fig3_path, width=490, height=210))
        story.append(Spacer(1, 6))

    story.append(PageBreak())

    # 6. Time-Series Trends & Figure 4
    story.append(Paragraph("6. Figure 4: Temporal Customer Acquisition & ARPU Trajectory", h1_style))
    fig4_path = os.path.join(base_dir, "outputs", "figures", "04_temporal_acquisition_trends.png")
    if os.path.exists(fig4_path):
        story.append(Image(fig4_path, width=490, height=195))
        story.append(Spacer(1, 6))

    # 7. Scatter Plots & Figure 5
    story.append(Paragraph("7. Figure 5: Multi-Variable Scatter Plots & Segment Clustering", h1_style))
    fig5_path = os.path.join(base_dir, "outputs", "figures", "05_bivariate_scatter_relationships.png")
    if os.path.exists(fig5_path):
        story.append(Image(fig5_path, width=490, height=160))
        story.append(Spacer(1, 6))

    story.append(PageBreak())

    # 8. Grouped Analysis & Figure 6
    story.append(Paragraph("8. Figure 6: Comparative Grouped Churn Rates Across Key Dimensions", h1_style))
    fig6_path = os.path.join(base_dir, "outputs", "figures", "06_grouped_churn_rate_by_category.png")
    if os.path.exists(fig6_path):
        story.append(Image(fig6_path, width=490, height=160))
        story.append(Spacer(1, 6))

    # Grouped Table
    grp_csv = os.path.join(base_dir, "outputs", "tables", "grouped_churn_metrics.csv")
    if os.path.exists(grp_csv):
        gdf = pd.read_csv(grp_csv)
        grp_rows = [[
            Paragraph("<b>Segment Dimension</b>", table_header_style),
            Paragraph("<b>Category Value</b>", table_header_style),
            Paragraph("<b>Total Accounts</b>", table_header_style),
            Paragraph("<b>Churn Count</b>", table_header_style),
            Paragraph("<b>Churn Rate (%)</b>", table_header_style),
            Paragraph("<b>Mean Monthly ($)</b>", table_header_style),
            Paragraph("<b>Mean Tenure (Mos)</b>", table_header_style)
        ]]
        for _, row in gdf.iterrows():
            grp_rows.append([
                Paragraph(str(row['Segment_Type']), table_cell_style),
                Paragraph(f"<b>{row['Segment_Value']}</b>", table_cell_bold),
                Paragraph(str(row['Total_Customers']), table_cell_style),
                Paragraph(str(row['Churned_Customers']), table_cell_style),
                Paragraph(f"{row['Churn_Rate_Pct']:.1f}%", table_cell_bold),
                Paragraph(f"${row['Avg_Monthly_Charges']:.2f}", table_cell_style),
                Paragraph(f"{row['Avg_Tenure']:.1f}", table_cell_style)
            ])
        t_grp = Table(grp_rows, colWidths=[80, 85, 65, 55, 65, 75, 75])
        t_grp.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0F172A")),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
            ('TOPPADDING', (0,0), (-1,-1), 2.5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ]))
        story.append(t_grp)
        story.append(Spacer(1, 6))

    # 9. Figure 7: Violin Distributions
    story.append(Paragraph("9. Figure 7: Violin Density Distributions of Charges by Churn Status", h1_style))
    fig7_path = os.path.join(base_dir, "outputs", "figures", "07_violin_charges_distribution.png")
    if os.path.exists(fig7_path):
        story.append(Image(fig7_path, width=490, height=140))
        story.append(Spacer(1, 6))

    story.append(PageBreak())

    # 10. Insights & Recommendations
    story.append(Paragraph("10. Strategic Business Insights & Action Plan", h1_style))
    insights_text = (
        "<b>1. Contract Commitment is the Primary Retention Driver</b>: Month-to-Month customers churn at <b>41.7%</b>, "
        "compared to <b>2.9%</b> for Two-Year contracts (a 14.4x risk reduction).<br/>"
        "<b>2. High-ARPU Fiber Optic Vulnerability</b>: Fiber optic customers generate the top monthly spend ($84.50/mo) but "
        "churn at 33.5% when tech support is absent, revealing onboarding and troubleshooting friction.<br/>"
        "<b>3. Payment Friction in Electronic Check</b>: Manual check payers churn at <b>34.8%</b> vs. <b>14.2%</b> for auto-pay credit cards.<br/>"
        "<b>4. The 12-Month Attrition Cliff</b>: Median tenure of churned customers is only 11 months, meaning attrition occurs before CAC is recouped.<br/>"
        "<b>5. Direct Return on CSAT</b>: Satisfaction ratings $\\le 2$ lead to 82.4% churn, representing an actionable early-warning signal."
    )
    story.append(Paragraph(insights_text, body_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Executive Recommendations:", h1_style))
    recs_text = (
        "• <b>Annual Plan Incentive</b>: Offer a 10% discount or 1 month free to convert Month-to-Month users to Annual commitments.<br/>"
        "• <b>Complimentary Fiber Tech Support</b>: Bundle 90 days of white-glove tech support with every new Fiber Optic subscription.<br/>"
        "• <b>Auto-Pay Bill Credits</b>: Provide a recurring $5 monthly bill discount for users enrolling in automated Credit Card or Bank Transfer Auto-Pay.<br/>"
        "• <b>Milestone Retention Nurturing</b>: Deploy automated engagement sequences at months 3, 6, and 12.<br/>"
        "• <b>CSAT Detractor Escalation</b>: Automatically route survey ratings $\\le 2$ to dedicated Customer Success reps within 24 hours."
    )
    story.append(Paragraph(recs_text, body_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated executive PDF report: {pdf_path}")

if __name__ == "__main__":
    base_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    build_pdf_report(base_directory)
