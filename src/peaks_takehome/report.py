"""PDF report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
from polars import col
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def build_pdf_report(
    output_path: Path,
    *,
    diagnostics: dict[str, Any],
    numbers: dict[str, Any],
    tables: dict[str, pl.DataFrame],
    model_summary: dict[str, Any],
    figures: dict[str, Path],
) -> None:
    """Write the self-contained PDF report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    story: list[Any] = []

    story.extend(_title(styles))
    story.extend(_executive_summary(styles, numbers, tables, model_summary))
    story.append(PageBreak())
    story.extend(_context_and_data(styles, diagnostics, numbers))
    story.extend(_attribution_section(styles, numbers, tables, figures))
    story.extend(_modelling_section(styles, model_summary, figures))
    story.extend(_skan_section(styles, tables))
    story.extend(_recommendations_section(styles, tables))
    story.extend(_limitations_section(styles))

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=1.4 * cm,
        leftMargin=1.4 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
        title="Peaks Take-Home Analysis",
        author="Codex",
    )
    doc.build(story)


def _title(styles: dict[str, ParagraphStyle]) -> list[Any]:
    return [
        Paragraph("Peaks Marketing Performance and Early-LTV Analysis", styles["Title"]),
        Paragraph("Senior Data Scientist Take-Home Assignment", styles["Subtitle"]),
        Spacer(1, 0.4 * cm),
    ]


def _executive_summary(
    styles: dict[str, ParagraphStyle],
    numbers: dict[str, Any],
    tables: dict[str, pl.DataFrame],
    model_summary: dict[str, Any],
) -> list[Any]:
    best_value = tables["eligible_channel_value"].sort("avg_ltv_365_per_registration", descending=True).head(3)
    best_value_text = ", ".join(best_value.select("source").to_series().to_list())
    skan_ratio = numbers["skan_to_mmp_ratio"]
    top_decile_share = (
        model_summary["decile_lift"].filter(col("predicted_decile") == 1).select("actual_ltv_share").item()
    )

    return [
        Paragraph("Executive Summary", styles["Heading1"]),
        Paragraph(
            (
                f"Peaks spent ${numbers['total_spend']:,.0f} across the period and generated "
                f"{numbers['installs']:,.0f} installs, {numbers['registrations']:,.0f} registrations, and "
                f"{numbers['total_transacting_users']:,.0f} transacting users. "
                f"{numbers['transacting_users']:,.0f} users show funding activity within the first 14 days. "
                "CMO read: last-click is fine for reconciliation, but budget choices should move to cohort value, "
                "iOS privacy correction, and incrementality tests."
            ),
            styles["Body"],
        ),
        Paragraph(
            (
                "The attribution gap is not causal proof; it is a reliability check on whether the source labels can "
                "support budget decisions. iOS looks under-attributed because last-click labels only "
                f"{numbers['ios_paid_share']:.1%} of iOS installs as paid, versus "
                f"{numbers['android_paid_share']:.1%} on Android, while Organic is "
                f"{numbers['ios_organic_share']:.1%} of iOS installs versus "
                f"{numbers['android_organic_share']:.1%} on Android. SKAN reports "
                f"{numbers['skan_total_installs']:,.0f} paid iOS installs, about {skan_ratio:.1f}x the MMP-paid iOS "
                "installs. Use SKAN as the paid iOS volume anchor, IPW LTV as the base value estimate, and the "
                "conversion-value mapping only as sensitivity until the real SKAN schema is available."
            ),
            styles["Body"],
        ),
        Paragraph(
            (
                f"On value, the strongest 12-month fee-revenue cohorts are {best_value_text}. "
                "Protect Referral if marginal quality holds and protect Non-Brand Search under payback caps. Cap Brand "
                "Search credit because it likely harvests existing demand. Keep Meta on LTV bid caps, tighten App "
                "Campaigns unless campaign-level splits improve, and reduce TikTok pending a funded-AUM holdout."
            ),
            styles["Body"],
        ),
        Paragraph(
            (
                f"The LightGBM early-LTV model uses only first-14-day data and its top predicted decile captures "
                f"{top_decile_share:.1%} of validation first-year fee revenue. That gives enough separation for bid "
                "caps and cohort steering, but the model should not be treated as proof of incremental lift."
            ),
            styles["Body"],
        ),
        Paragraph(
            (
                "Immediate moves: launch a weekly cohort value scorecard, replace iOS MMP-only reporting with "
                "SKAN-volume plus IPW-value reporting, and put spend guardrails on Brand Search, App Campaigns, and "
                "TikTok before the next budget increase."
            ),
            styles["Callout"],
        ),
    ]


def _context_and_data(
    styles: dict[str, ParagraphStyle], diagnostics: dict[str, Any], numbers: dict[str, Any]
) -> list[Any]:
    rows = [
        ["Table", "Rows / coverage"],
        *[[name.replace("_", " ").title(), f"{count:,.0f}"] for name, count in diagnostics["row_counts"].items()],
        ["Installed users", f"{numbers['installs']:,.0f}"],
        ["Registered users", f"{numbers['registrations']:,.0f}"],
        ["Transactions in profiles", f"{diagnostics['join_coverage']['transactions_in_profiles']:.1%}"],
    ]
    return [
        Paragraph("Data And Method", styles["Heading1"]),
        Paragraph(
            (
                "The analysis starts from the raw CSVs and parses dates, timestamps, booleans, spend, event counts, "
                "transaction amounts, and SKAN postbacks with explicit Polars schemas. Registered and transacting "
                "users all join back to app events, so the main caveat is attribution quality rather than missing "
                "customer records."
            ),
            styles["Body"],
        ),
        _table(rows, widths=[7 * cm, 7 * cm]),
        Spacer(1, 0.3 * cm),
        Paragraph(
            (
                "I define LTV as first-year fee revenue from AUM: running balance times the current 0.5% annual "
                "management fee, prorated over observed balance intervals. Users without a full 365-day horizon are "
                "excluded from supervised training."
            ),
            styles["Body"],
        ),
    ]


def _attribution_section(
    styles: dict[str, ParagraphStyle],
    numbers: dict[str, Any],
    tables: dict[str, pl.DataFrame],
    figures: dict[str, Path],
) -> list[Any]:
    channel_rows = _format_table(
        tables["channel_performance"]
        .sort("cac", nulls_last=True)
        .select(
            "channel",
            "spend",
            "installs",
            "registrations",
            "cpi",
            "cac",
            "install_to_registration_rate",
            "early_funded_rate",
        ),
        ["Channel", "Spend", "Installs", "Regs", "CPI", "CAC", "Install -> Reg", "Early Funded"],
        money_cols={"Spend", "CPI", "CAC"},
        pct_cols={"Install -> Reg", "Early Funded"},
        max_rows=8,
    )
    anomaly_rows = _format_table(
        tables["attribution_anomalies"].select("signal", "evidence", "readout"),
        ["Signal", "Evidence", "Readout"],
        max_rows=6,
    )
    return [
        Paragraph("Part 1: Attribution And Channel Performance", styles["Heading1"]),
        Paragraph(
            (
                "The last-click view rewards channels close to the decision. Google Brand Search and Referral have "
                "the lowest CACs, while TikTok has the highest CPI and CAC. TikTok also has the weakest early funded "
                "rate, while Referral and Non-Brand Search convert registered users into funded users much more "
                "often. That ranking is incomplete because it mixes intent capture, referrals, prospecting, and brand "
                "activity into one last-click scorecard."
            ),
            styles["Body"],
        ),
        _image(figures["cost_metrics"], width=16 * cm),
        _table(channel_rows, widths=[3.6 * cm, 2.0 * cm, 1.8 * cm, 1.6 * cm, 1.4 * cm, 1.5 * cm, 1.8 * cm, 1.9 * cm]),
        Paragraph(
            (
                "This is not causal proof. It is a check on whether the attribution labels are reliable enough for "
                "budget decisions. iOS looks under-attributed because its paid share is much lower than "
                f"Android ({numbers['ios_paid_share']:.1%} versus {numbers['android_paid_share']:.1%}), Organic is "
                f"much higher ({numbers['ios_organic_share']:.1%} versus {numbers['android_organic_share']:.1%}), "
                "and SKAN reports substantially more paid iOS installs than the MMP source field."
            ),
            styles["Body"],
        ),
        _image(figures["attribution_mix"], width=14.5 * cm),
        Paragraph(
            (
                f"Paid spend and later Organic installs are strongly correlated, peaking at a "
                f"{numbers['best_organic_lag_days']:.0f}-day lag "
                f"({numbers['best_organic_lag_correlation']:.2f}). The shape matters: it is not a smooth decay curve. "
                "It drops after the first few days, rises again around a weekly lag, and then repeats. That points to "
                "shared campaign pacing or day-of-week structure, not a clean estimate of causal lift. Practical read: "
                "Organic is not behaving like an independent unpaid baseline."
            ),
            styles["Body"],
        ),
        _image(figures["organic_lag"], width=13.5 * cm),
        Paragraph("Attribution Anomaly Signals", styles["Heading2"]),
        Paragraph(
            (
                "The anomaly checks do not prove incrementality. They show that last-click labels vary across "
                "platform, tracking state, SKAN vs MMP, and time. That leaves last-click fine for bookkeeping, but "
                "risky for budget allocation."
            ),
            styles["Body"],
        ),
        _table(anomaly_rows, widths=[3.4 * cm, 8.0 * cm, 5.0 * cm]),
    ]


def _modelling_section(
    styles: dict[str, ParagraphStyle],
    model_summary: dict[str, Any],
    figures: dict[str, Path],
) -> list[Any]:
    metrics = model_summary["metrics"]
    feature_rows = _format_table(
        model_summary["feature_importance"].head(10),
        ["Feature", "MAE Increase"],
        money_cols={"MAE Increase"},
        max_rows=10,
    )
    quality_rows = _format_table(
        model_summary["channel_quality"].select(
            "source",
            "installs",
            "registrations",
            "predicted_ltv_365_per_install",
            "predicted_ltv_365_per_registration",
        ),
        ["Channel", "Installs", "Regs", "Pred LTV / Install", "Pred LTV / Reg"],
        money_cols={"Pred LTV / Install", "Pred LTV / Reg"},
        max_rows=8,
    )
    return [
        Paragraph("Part 2: Early-LTV Prediction", styles["Heading1"]),
        Paragraph(
            (
                "The native LightGBM model uses only fields available within 14 days after install: install source, "
                "campaign, platform, country, registration/profile fields if the user registered, and first-14-day "
                "transaction behavior. Training uses older installs with a full 365-day observation window, then "
                "validates on later eligible cohorts."
            ),
            styles["Body"],
        ),
        Paragraph(
            (
                f"The two-stage LightGBM model improves over a transparent segment baseline. Validation MAE is "
                f"${metrics['gbdt_validation_mae']:.2f} versus ${metrics['baseline_validation_mae']:.2f}; "
                f"log RMSE is {metrics['gbdt_validation_log_rmse']:.2f} versus "
                f"{metrics['baseline_validation_log_rmse']:.2f}. The classifier AUC for positive LTV is "
                f"{metrics['gbdt_validation_auc']:.2f}."
            ),
            styles["Body"],
        ),
        Paragraph(
            (
                "The feature ranking fits the business model. Early funding behavior carries most of the "
                "signal: net deposit, gross deposit, transaction count, deposit count, balance, and timing to first "
                "transaction. That makes sense for Peaks because revenue comes from AUM fees; early balance formation "
                "is the best early clue about future fee revenue. One caveat: gender appears predictive in this "
                "dataset, but I would treat it as diagnostic, not automatically operational. A real investment-product "
                "bidding system should review demographic features with compliance and fairness constraints before use."
            ),
            styles["Body"],
        ),
        Paragraph(
            (
                "The concentration chart splits ranked users into the top 1%, the next 9%, and the remaining 90%. "
                "Nearly all realized first-year fee revenue in validation sits in the highest-ranked users."
            ),
            styles["Body"],
        ),
        _image(figures["decile_lift"], width=14.5 * cm),
        KeepTogether([Paragraph("Most predictive feature groups", styles["Heading2"]), _table(feature_rows)]),
        Paragraph(
            (
                "Re-scoring channels by predicted value changes the CPI story. Google Non-Brand Search has the "
                "highest predicted LTV per registration, while Referral has the strongest predicted LTV per install "
                "and the fastest observed payback on complete cohorts. Google Brand Search still looks cheap by CPI "
                "and CAC, but that is probably demand capture rather than demand creation. TikTok is the weakest read "
                "here: high acquisition cost and low predicted customer value."
            ),
            styles["Body"],
        ),
        KeepTogether([Paragraph("Predicted channel quality", styles["Heading2"]), _table(quality_rows)]),
        _image(figures["channel_value"], width=14.5 * cm),
    ]


def _skan_section(styles: dict[str, ParagraphStyle], tables: dict[str, pl.DataFrame]) -> list[Any]:
    skan_value = tables["skan_value_method_comparison"]
    ipw_min = skan_value.select(col("ipw_ltv_365_per_skan_install").min()).item()
    ipw_max = skan_value.select(col("ipw_ltv_365_per_skan_install").max()).item()
    cv_ratio_min = skan_value.select(col("cv_to_ipw_ltv_ratio").min()).item()
    cv_ratio_max = skan_value.select(col("cv_to_ipw_ltv_ratio").max()).item()
    skan_rows = _format_table(
        tables["skan_mmp_comparison"].select(
            "network",
            "skan_installs",
            "mmp_ios_paid_installs",
            "unattributed_ios_paid_gap",
            "skan_to_mmp_ratio",
        ),
        ["Network", "SKAN Installs", "MMP iOS Paid", "Gap", "SKAN / MMP"],
        max_rows=8,
    )
    skan_value_rows = _format_table(
        skan_value.select(
            "network",
            "skan_installs",
            "ipw_ltv_365_per_skan_install",
            "cv_ltv_365_per_skan_install",
            "cv_to_ipw_ltv_ratio",
        ),
        ["Network", "SKAN Installs", "IPW LTV / SKAN Install", "CV LTV / SKAN Install", "CV / IPW"],
        money_cols={"IPW LTV / SKAN Install", "CV LTV / SKAN Install"},
        max_rows=8,
    )
    return [
        Paragraph("iOS And SKAN Estimation", styles["Heading1"]),
        Paragraph(
            (
                "SKAN sees far more paid iOS volume than the MMP last-click source field. I treat that as a missing-data "
                "and selection-bias problem, not as proof of incremental lift. SKAN gives aggregate paid iOS volume, "
                "while user-level MMP data only sees a biased subset of paid iOS users. A cross-fitted LightGBM "
                "propensity model estimates each complete-cohort iOS user's chance of being observable as MMP-paid, "
                "then inverse-probability weights the observed paid iOS users before applying value to SKAN volume."
            ),
            styles["Body"],
        ),
        Paragraph(
            (
                "I use the IPW estimate as the primary planning number because it corrects the observable selection "
                "problem directly. The conversion-value rank calibration stays in the table as a sensitivity check "
                "because the true SKAN conversion schema is not available in the assignment data."
            ),
            styles["Body"],
        ),
        Paragraph(
            (
                f"The comparison changes the read: IPW puts these networks at roughly ${ipw_min:,.2f} to "
                f"${ipw_max:,.2f} per SKAN install, while the rank-based conversion-value method is "
                f"{cv_ratio_min:.1f}x to {cv_ratio_max:.1f} higher. I would treat the CV method as upside sensitivity, "
                "not the base case."
            ),
            styles["Body"],
        ),
        _table(skan_rows),
        Spacer(1, 0.2 * cm),
        _table(skan_value_rows, widths=[3.8 * cm, 2.1 * cm, 3.1 * cm, 3.0 * cm, 1.6 * cm]),
    ]


def _recommendations_section(styles: dict[str, ParagraphStyle], tables: dict[str, pl.DataFrame]) -> list[Any]:
    budget_rows = _format_table(
        tables["budget_reallocation_plan"].select(
            "channel",
            "budget_action",
            "valuation_read",
            "predicted_ltv_365_per_install",
            "payback_years_at_y1_rate",
            "ipw_ltv_365_per_skan_install",
        ),
        ["Channel", "Action", "Value read", "Pred LTV / Install", "Payback Years", "IPW iOS LTV"],
        money_cols={"Pred LTV / Install", "IPW iOS LTV"},
        max_rows=8,
    )
    role_rows = _format_table(
        tables["campaign_roles"].select("channel", "role", "measurement_note"),
        ["Channel", "Role", "Measurement note"],
        max_rows=8,
    )
    measurement_rows = _format_table(
        tables["measurement_framework"].select(
            "measurement_layer",
            "primary_metrics",
            "cadence",
            "decision_use",
        ),
        ["Layer", "Primary metrics", "Cadence", "Decision use"],
        max_rows=8,
    )
    quick_win_rows = _format_table(
        tables["quick_wins"].select(
            "quick_win",
            "owner",
            "first_action",
            "success_metric",
        ),
        ["Quick win", "Owner", "First action", "Success metric"],
        max_rows=3,
    )
    return [
        Paragraph("Part 3: Recommendations", styles["Heading1"]),
        Paragraph(
            (
                "Budget allocation should move from CPI to expected fee revenue and payback, with separate rules for "
                "channel roles. Recommended move: protect Referral and Non-Brand Search, cap Brand Search credit, "
                "hold Meta under LTV guardrails, tighten App Campaigns, and reduce TikTok until it proves incremental "
                "funded AUM."
            ),
            styles["Body"],
        ),
        _table(budget_rows, widths=[2.6 * cm, 2.6 * cm, 3.6 * cm, 2.3 * cm, 2.0 * cm, 2.2 * cm]),
        Spacer(1, 0.2 * cm),
        Paragraph(
            (
                "I classify channel roles from campaign names, media mechanics, and cohort value. Low CPM with "
                "awareness naming points to reach, high-CTR Search points to intent capture, "
                "Referral spend behaves like an incentive cost, and Organic is a mixed attribution bucket. That is why "
                "the KPI changes by channel: not every channel should be judged on CPI or last-click CAC alone."
            ),
            styles["Body"],
        ),
        _table(role_rows, widths=[3.2 * cm, 5.0 * cm, 7.3 * cm]),
        Paragraph("Forward measurement framework", styles["Heading2"]),
        Paragraph(
            (
                "Operating rule: keep last-click for reconciliation, but do not let it choose the "
                "budget. Budget decisions should come from cohort value, iOS privacy correction, and incrementality "
                "tests. This avoids treating Organic as free, Brand Search as pure growth, or SKAN as user-level truth."
            ),
            styles["Body"],
        ),
        _table(measurement_rows, widths=[3.0 * cm, 5.8 * cm, 2.2 * cm, 5.0 * cm]),
        Paragraph("Quick wins", styles["Heading2"]),
        Paragraph(
            (
                "These are the first moves I would make before a broader attribution rebuild. They use the current "
                "data products and directly address the weak spots found in the analysis."
            ),
            styles["Body"],
        ),
        _table(quick_win_rows, widths=[3.6 * cm, 3.0 * cm, 6.0 * cm, 4.4 * cm]),
        Paragraph(
            (
                "The team does not need perfect attribution to make better budget calls. Start by "
                "changing the reporting grain, correcting iOS volume and value, and putting guardrails on channels where "
                "last-click is most likely to mislead."
            ),
            styles["Body"],
        ),
        Paragraph(
            (
                "Brand Search should have a smaller credit window or be reported as demand capture unless a holdout "
                "shows lift. Referral should stay outside paid-media auction reporting because its economics are "
                "incentive-led. For iOS, SKAN should anchor volume, IPW should be the base value estimate, and the "
                "conversion-value mapping should stay as sensitivity until the real schema is available."
            ),
            styles["Body"],
        ),
    ]


def _limitations_section(styles: dict[str, ParagraphStyle]) -> list[Any]:
    return [
        Paragraph("Limitations And Next Steps", styles["Heading1"]),
        Paragraph(
            (
                "The LightGBM model is fit for ranking cohorts and setting bid guardrails, but it predicts value from "
                "historical behavior; it does not prove incremental lift. Training also excludes newer installs without a "
                "full 365-day observation window, so calibration should be monitored as channel mix and campaign strategy "
                "change. Gender appears predictive in this dataset, but I would keep it diagnostic unless compliance and "
                "fairness review approves operational use."
            ),
            styles["Body"],
        ),
        Paragraph(
            (
                "LTV is based on observed AUM balances and the current 0.5% fee, so fee changes, balance mix changes, or "
                "product changes would need a refresh. The SKAN IPW correction assumes observed features explain MMP-paid "
                "observability; it does not recover unobserved network assignment or causal lift. The conversion-value "
                "mapping is a sensitivity check and should be replaced with the actual SKAN schema if the marketing team "
                "can provide it."
            ),
            styles["Body"],
        ),
        Paragraph(
            (
                "The budget recommendations are guardrails, not a guarantee that marginal spend will perform the same way. "
                "Referral and Non-Brand Search may saturate, Brand Search may keep harvesting demand created elsewhere, and "
                "Organic remains a mixed bucket of owned demand and privacy-masked paid demand. Next: pair the new "
                "reporting system with holdouts and cohort monitoring before making large permanent reallocations."
            ),
            styles["Body"],
        ),
    ]


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "CustomBody",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        spaceAfter=7,
    )
    return {
        "Title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            spaceAfter=5,
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=11,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#555555"),
            spaceAfter=12,
        ),
        "Heading1": ParagraphStyle(
            "Heading1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=17,
            spaceBefore=8,
            spaceAfter=6,
        ),
        "Heading2": ParagraphStyle(
            "Heading2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            spaceBefore=6,
            spaceAfter=4,
        ),
        "Body": body,
        "Callout": ParagraphStyle(
            "Callout",
            parent=body,
            leftIndent=8,
            rightIndent=8,
            borderColor=colors.HexColor("#577590"),
            borderWidth=0.75,
            borderPadding=7,
            backColor=colors.HexColor("#f4f7f9"),
            spaceBefore=4,
            spaceAfter=8,
        ),
    }


def _image(path: Path, *, width: float) -> Image:
    image = Image(str(path))
    ratio = image.imageHeight / image.imageWidth
    image.drawWidth = width
    image.drawHeight = width * ratio
    return image


def _table(rows: list[list[Any]], widths: list[float] | None = None) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#273043")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.6),
                ("LEADING", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f7f7")]),
            ]
        )
    )
    return table


def _format_table(
    frame: pl.DataFrame,
    headers: list[str],
    *,
    money_cols: set[str] | None = None,
    pct_cols: set[str] | None = None,
    max_rows: int = 10,
) -> list[list[str]]:
    money_cols = money_cols or set()
    pct_cols = pct_cols or set()
    return [
        headers,
        *[
            [
                _format_value(value, money=header in money_cols, pct=header in pct_cols)
                for header, value in zip(headers, raw_row, strict=True)
            ]
            for raw_row in frame.head(max_rows).iter_rows()
        ],
    ]


def _format_value(value: Any, *, money: bool = False, pct: bool = False) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if pct:
            return f"{value:.1%}"
        if money:
            return f"${value:,.2f}"
        if abs(value) >= 1_000:
            return f"{value:,.0f}"
        return f"{value:.2f}"
    if isinstance(value, int):
        if money:
            return f"${value:,.0f}"
        return f"{value:,}"
    return str(value)
