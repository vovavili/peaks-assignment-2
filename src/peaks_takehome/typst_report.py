"""Typst report generation and compilation."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import polars as pl
from polars import col


def build_typst_report(
    output_path: Path,
    *,
    diagnostics: dict[str, Any],
    numbers: dict[str, Any],
    tables: dict[str, pl.DataFrame],
    model_summary: dict[str, Any],
    figures: dict[str, Path],
) -> Path | None:
    """Write Typst source and compile it when the Typst binary is available."""
    typ_path = output_path.with_suffix(".typ")
    typ_path.write_text(
        _document(
            diagnostics=diagnostics,
            numbers=numbers,
            tables=tables,
            model_summary=model_summary,
            figures=figures,
        ),
        encoding="utf-8",
    )

    typst = _find_typst()
    if typst is None:
        return None

    subprocess.run(
        [typst, "compile", str(typ_path), str(output_path)],
        cwd=typ_path.parent,
        check=True,
    )
    return output_path


def _document(
    *,
    diagnostics: dict[str, Any],
    numbers: dict[str, Any],
    tables: dict[str, pl.DataFrame],
    model_summary: dict[str, Any],
    figures: dict[str, Path],
) -> str:
    best_value = tables["eligible_channel_value"].sort("avg_ltv_365_per_registration", descending=True).head(3)
    best_value_text = ", ".join(best_value.select("source").to_series().to_list())
    top_decile_share = (
        model_summary["decile_lift"].filter(col("predicted_decile") == 1).select("actual_ltv_share").item()
    )

    channel_perf = (
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
        )
    )
    feature_importance = model_summary["feature_importance"].head(10)
    channel_quality = model_summary["channel_quality"].select(
        "source",
        "installs",
        "registrations",
        "predicted_ltv_365_per_install",
        "predicted_ltv_365_per_registration",
    )
    skan_comparison = tables["skan_mmp_comparison"].select(
        "network",
        "skan_installs",
        "mmp_ios_paid_installs",
        "unattributed_ios_paid_gap",
        "skan_to_mmp_ratio",
    )
    skan_value = tables["skan_value_method_comparison"].select(
        "network",
        "skan_installs",
        "ipw_ltv_365_per_skan_install",
        "cv_ltv_365_per_skan_install",
        "cv_to_ipw_ltv_ratio",
    )
    ipw_min = skan_value.select(col("ipw_ltv_365_per_skan_install").min()).item()
    ipw_max = skan_value.select(col("ipw_ltv_365_per_skan_install").max()).item()
    cv_ratio_min = skan_value.select(col("cv_to_ipw_ltv_ratio").min()).item()
    cv_ratio_max = skan_value.select(col("cv_to_ipw_ltv_ratio").max()).item()
    budget_reallocation = tables["budget_reallocation_plan"].select(
        "channel",
        "budget_action",
        "valuation_read",
        "predicted_ltv_365_per_install",
        "payback_years_at_y1_rate",
        "ipw_ltv_365_per_skan_install",
    )
    roles = tables["campaign_roles"].select("channel", "role", "measurement_note")
    measurement_framework = tables["measurement_framework"].select(
        "measurement_layer",
        "primary_metrics",
        "cadence",
        "decision_use",
    )
    quick_wins = tables["quick_wins"].select(
        "quick_win",
        "owner",
        "first_action",
        "success_metric",
    )
    attribution_anomalies = tables["attribution_anomalies"].select("signal", "evidence", "readout")

    diagnostics_rows = [
        ["Table", "Rows / coverage"],
        *[[name.replace("_", " ").title(), _int(count)] for name, count in diagnostics["row_counts"].items()],
        ["Installed users", _int(numbers["installs"])],
        ["Registered users", _int(numbers["registrations"])],
        ["Transactions in profiles", _pct(diagnostics["join_coverage"]["transactions_in_profiles"])],
    ]

    return "\n\n".join(
        [
            _preamble(),
            _title(),
            _metrics_grid(numbers),
            _heading("Executive Summary", level=1),
            _para(
                f"Peaks spent {_money(numbers['total_spend'])} across the period and generated "
                f"{_int(numbers['installs'])} installs, {_int(numbers['registrations'])} registrations, and "
                f"{_int(numbers['total_transacting_users'])} transacting users. "
                f"{_int(numbers['transacting_users'])} users show funding activity within the first 14 days. "
                "CMO read: last-click is fine for reconciliation, but budget choices should move to cohort value, "
                "iOS privacy correction, and incrementality tests."
            ),
            _para(
                "The attribution gap is not causal proof; it is a reliability check on whether the source labels can "
                "support budget decisions. iOS looks under-attributed because last-click labels only "
                f"{_pct(numbers['ios_paid_share'])} of iOS installs as paid, versus "
                f"{_pct(numbers['android_paid_share'])} on Android, while Organic is "
                f"{_pct(numbers['ios_organic_share'])} of iOS installs versus "
                f"{_pct(numbers['android_organic_share'])} on Android. SKAN reports "
                f"{_int(numbers['skan_total_installs'])} paid iOS installs, about "
                f"{numbers['skan_to_mmp_ratio']:.1f}x the MMP-paid iOS installs. "
                "Use SKAN as the paid iOS volume anchor, IPW LTV as the base value estimate, and the conversion-value "
                "mapping only as sensitivity until the real SKAN schema is available."
            ),
            _para(
                f"On value, the strongest 12-month fee-revenue cohorts are {best_value_text}. Google Brand Search "
                "and Referral look efficient in last-click CAC, but Brand Search is likely harvesting demand created "
                "elsewhere. Protect Referral if marginal quality holds and protect Non-Brand Search under payback caps. "
                "Keep Meta on LTV bid caps, tighten App Campaigns unless campaign-level splits improve, and reduce "
                "TikTok pending a funded-AUM holdout."
            ),
            _para(
                "The LightGBM early-LTV model uses only first-14-day data and its top predicted decile captures "
                f"{_pct(top_decile_share)} of validation first-year fee revenue. That gives enough separation for bid "
                "caps and cohort steering, but the model should not be treated as proof of incremental lift."
            ),
            _callout(
                "Immediate moves: launch a weekly cohort value scorecard, replace iOS MMP-only reporting with "
                "SKAN-volume plus IPW-value reporting, and put spend guardrails on Brand Search, App Campaigns, and "
                "TikTok before the next budget increase."
            ),
            "#pagebreak()",
            _heading("Data And Method", level=1),
            _para(
                "The analysis starts from the raw CSVs and parses dates, timestamps, booleans, spend, event counts, "
                "transaction amounts, and SKAN postbacks with explicit Polars schemas. Registered and transacting users "
                "all join back to app events, so the main caveat is attribution quality rather than missing customer records."
            ),
            _table(diagnostics_rows),
            _para(
                "I define LTV as first-year fee revenue from AUM: running balance times the current 0.5% annual "
                "management fee, prorated over observed balance intervals. Users without a full 365-day horizon are "
                "excluded from supervised training."
            ),
            _heading("Part 1: Attribution And Channel Performance", level=1),
            _para(
                "The last-click view rewards channels close to the decision. Google Brand Search and Referral have "
                "the lowest CACs, while TikTok has the highest CPI and CAC. TikTok also has the weakest early funded "
                "rate, while Referral and Non-Brand Search convert registered users into funded users much more "
                "often. That ranking is incomplete because it mixes intent capture, referrals, prospecting, and brand "
                "activity into one last-click scorecard."
            ),
            _figure(
                figures["cost_metrics"],
                "Last-click CAC rewards intent capture and referral mechanics; it is not enough for budget allocation.",
            ),
            _table(
                _frame_rows(
                    channel_perf,
                    ["Channel", "Spend", "Installs", "Regs", "CPI", "CAC", "Install -> Reg", "Early Funded"],
                    money_cols={"Spend", "CPI", "CAC"},
                    pct_cols={"Install -> Reg", "Early Funded"},
                )
            ),
            _para(
                "This is not causal proof. It is a check on whether the attribution labels are reliable enough for "
                "budget decisions. iOS looks under-attributed because its paid share is much lower than "
                f"Android ({_pct(numbers['ios_paid_share'])} versus {_pct(numbers['android_paid_share'])}), Organic is "
                f"much higher ({_pct(numbers['ios_organic_share'])} versus {_pct(numbers['android_organic_share'])}), "
                "and SKAN reports substantially more paid iOS installs than the MMP source field."
            ),
            _figure(
                figures["attribution_mix"],
                "iOS has a much higher Organic share and a much lower paid-attributed share than Android.",
            ),
            _para(
                "Paid spend and later Organic installs are strongly correlated, peaking at a "
                f"{numbers['best_organic_lag_days']:.0f}-day lag "
                f"({numbers['best_organic_lag_correlation']:.2f}). The shape matters: it is not a smooth decay curve. "
                "It drops after the first few days, rises again around a weekly lag, and then repeats. That points to "
                "shared campaign pacing or day-of-week structure, not a clean estimate of causal lift. Practical read: "
                "Organic is not behaving like an independent unpaid baseline."
            ),
            _figure(
                figures["organic_lag"],
                "Organic installs move with paid spend on a weekly rhythm, not as a clean causal decay curve.",
            ),
            _heading("Attribution Anomaly Signals", level=2),
            _para(
                "The anomaly checks do not prove incrementality. They show that last-click labels vary across "
                "platform, tracking state, SKAN vs MMP, and time. That leaves last-click fine for bookkeeping, but "
                "risky for budget allocation."
            ),
            _table(_frame_rows(attribution_anomalies, ["Signal", "Evidence", "Readout"], max_rows=6)),
            _heading("Part 2: Early-LTV Prediction", level=1),
            _para(
                "The native LightGBM model uses only fields available within 14 days after install: install source, "
                "campaign, platform, country, registration/profile fields if the user registered, and first-14-day "
                "transaction behavior. Training uses older installs with a full 365-day observation window, then validates "
                "on later eligible cohorts."
            ),
            _para(
                "The two-stage LightGBM model improves over a transparent segment baseline. Validation MAE is "
                f"{_money(model_summary['metrics']['gbdt_validation_mae'])} versus "
                f"{_money(model_summary['metrics']['baseline_validation_mae'])}; log RMSE is "
                f"{model_summary['metrics']['gbdt_validation_log_rmse']:.2f} versus "
                f"{model_summary['metrics']['baseline_validation_log_rmse']:.2f}. The classifier AUC for positive LTV is "
                f"{model_summary['metrics']['gbdt_validation_auc']:.2f}."
            ),
            _para(
                "The feature ranking fits the business model. Early funding behavior carries most of the signal: "
                "net deposit, gross deposit, transaction count, deposit count, balance, and timing to first transaction. "
                "That makes sense for Peaks because revenue comes from AUM fees; early balance formation is the best "
                "early clue about future fee revenue. One caveat: gender appears predictive in this dataset, but I would "
                "treat it as diagnostic, not automatically operational. A real investment-product bidding system should "
                "review demographic features with compliance and fairness constraints before use."
            ),
            _figure(
                figures["decile_lift"],
                "After users are ranked by predicted LTV, revenue is concentrated in the highest-ranked users.",
            ),
            _heading("Most Predictive Feature Groups", level=2),
            _table(_frame_rows(feature_importance, ["Feature", "MAE Increase"], money_cols={"MAE Increase"})),
            _para(
                "Re-scoring channels by predicted value changes the CPI story. Google Non-Brand Search has the highest "
                "predicted LTV per registration, while Referral has the strongest predicted LTV per install and the "
                "fastest observed payback on complete cohorts. Google Brand Search still looks cheap by CPI and CAC, "
                "but that is probably demand capture rather than demand creation. TikTok is the weakest read here: "
                "high acquisition cost and low predicted customer value."
            ),
            _heading("Predicted Channel Quality", level=2),
            _table(
                _frame_rows(
                    channel_quality,
                    ["Channel", "Installs", "Regs", "Pred LTV / Install", "Pred LTV / Reg"],
                    money_cols={"Pred LTV / Install", "Pred LTV / Reg"},
                )
            ),
            _figure(
                figures["channel_value"],
                "Predicted customer value and acquisition cost point to a different story than CPI alone.",
            ),
            _heading("iOS And SKAN Estimation", level=1),
            _para(
                "SKAN sees far more paid iOS volume than the MMP last-click source field. I treat that as a missing-data "
                "and selection-bias problem, not as proof of incremental lift. SKAN gives aggregate paid iOS volume, "
                "while user-level MMP data only sees a biased subset of paid iOS users. A cross-fitted LightGBM "
                "propensity model estimates each complete-cohort iOS user's chance of being observable as MMP-paid, "
                "then inverse-probability weights the observed paid iOS users before applying value to SKAN volume."
            ),
            _para(
                "I use the IPW estimate as the primary planning number because it corrects the observable selection "
                "problem directly. The conversion-value rank calibration stays in the table as a sensitivity check "
                "because the true SKAN conversion schema is not available in the assignment data."
            ),
            _para(
                f"The comparison changes the read: IPW puts these networks at roughly {_money(ipw_min)} to "
                f"{_money(ipw_max)} per SKAN install, while the rank-based conversion-value method is "
                f"{cv_ratio_min:.1f}x to {cv_ratio_max:.1f}x higher. I would treat the CV method as upside sensitivity, "
                "not the base case."
            ),
            _table(
                _frame_rows(
                    skan_comparison,
                    ["Network", "SKAN Installs", "MMP iOS Paid", "Gap", "SKAN / MMP"],
                )
            ),
            _table(
                _frame_rows(
                    skan_value,
                    [
                        "Network",
                        "SKAN Installs",
                        "IPW LTV / SKAN Install",
                        "CV LTV / SKAN Install",
                        "CV / IPW",
                    ],
                    money_cols={"IPW LTV / SKAN Install", "CV LTV / SKAN Install"},
                )
            ),
            _heading("Part 3: Recommendations", level=1),
            _para(
                "Budget allocation should move from CPI to expected fee revenue and payback, with separate rules for channel "
                "roles. Recommended move: protect Referral and Non-Brand Search, cap Brand Search credit, hold Meta "
                "under LTV guardrails, tighten App Campaigns, and reduce TikTok until it proves incremental funded AUM."
            ),
            _table(
                _frame_rows(
                    budget_reallocation,
                    ["Channel", "Action", "Value read", "Pred LTV / Install", "Payback Years", "IPW iOS LTV"],
                    money_cols={"Pred LTV / Install", "IPW iOS LTV"},
                    max_rows=8,
                )
            ),
            _para(
                "I classify channel roles from campaign names, media mechanics, and cohort value. Low CPM with awareness "
                "naming points to reach, high-CTR Search points to intent capture, Referral spend "
                "behaves like an incentive cost, and Organic is a mixed attribution bucket. That is why the KPI changes by "
                "channel: not every channel should be judged on CPI or last-click CAC alone."
            ),
            _table(_frame_rows(roles, ["Channel", "Role", "Measurement note"])),
            _heading("Forward Measurement Framework", level=2),
            _para(
                "Operating rule: keep last-click for reconciliation, but do not let it choose the "
                "budget. Budget decisions should come from cohort value, iOS privacy correction, and incrementality "
                "tests. This avoids treating Organic as free, Brand Search as pure growth, or SKAN as user-level truth."
            ),
            _table(
                _frame_rows(
                    measurement_framework,
                    ["Layer", "Primary metrics", "Cadence", "Decision use"],
                    max_rows=8,
                )
            ),
            _heading("Quick Wins", level=2),
            _para(
                "These are the first moves I would make before a broader attribution rebuild. They use the current "
                "data products and directly address the weak spots found in the analysis."
            ),
            _table(
                _frame_rows(
                    quick_wins,
                    ["Quick win", "Owner", "First action", "Success metric"],
                    max_rows=3,
                )
            ),
            _para(
                "The team does not need perfect attribution to make better budget calls. Start by "
                "changing the reporting grain, correcting iOS volume and value, and putting guardrails on channels where "
                "last-click is most likely to mislead."
            ),
            _para(
                "Brand Search should have a smaller credit window or be reported as demand capture unless a holdout shows "
                "lift. Referral should stay outside paid-media auction reporting because its economics are incentive-led. "
                "For iOS, SKAN should anchor volume, IPW should be the base value estimate, and the conversion-value mapping "
                "should stay as sensitivity until the real schema is available."
            ),
            _heading("Limitations And Next Steps", level=1),
            _para(
                "The LightGBM model is fit for ranking cohorts and setting bid guardrails, but it predicts value from "
                "historical behavior; it does not prove incremental lift. Training also excludes newer installs without a "
                "full 365-day observation window, so calibration should be monitored as channel mix and campaign strategy "
                "change. Gender appears predictive in this dataset, but I would keep it diagnostic unless compliance and "
                "fairness review approves operational use."
            ),
            _para(
                "LTV is based on observed AUM balances and the current 0.5% fee, so fee changes, balance mix changes, or "
                "product changes would need a refresh. The SKAN IPW correction assumes observed features explain MMP-paid "
                "observability; it does not recover unobserved network assignment or causal lift. The conversion-value "
                "mapping is a sensitivity check and should be replaced with the actual SKAN schema if the marketing team "
                "can provide it."
            ),
            _para(
                "The budget recommendations are guardrails, not a guarantee that marginal spend will perform the same way. "
                "Referral and Non-Brand Search may saturate, Brand Search may keep harvesting demand created elsewhere, and "
                "Organic remains a mixed bucket of owned demand and privacy-masked paid demand. Next: pair the new "
                "reporting system with holdouts and cohort monitoring before making large permanent reallocations."
            ),
        ]
    )


def _preamble() -> str:
    return """#let navy = rgb("#17212b")
#let ink = rgb("#24211d")
#let muted = rgb("#6f6a60")
#let gold = rgb("#b98b2d")
#let paper = rgb("#fbfaf6")
#let wash = rgb("#f1eee7")
#let hairline = rgb("#d9d1c2")

#set page(paper: "a4", margin: (x: 1.55cm, y: 1.35cm))
#set text(font: ("New Computer Modern", "Libertinus Serif"), size: 9.35pt, lang: "en", fill: ink)
#set par(justify: true, leading: 0.62em, spacing: 0.72em)
#set heading(numbering: none)
#show heading.where(level: 1): it => [
  #v(0.75em)
  #align(center)[#text(size: 14.2pt, weight: "bold", fill: navy)[#it.body]]
  #v(0.05em)
  #align(center)[#line(length: 62%, stroke: 0.55pt + gold)]
  #v(0.18em)
]
#show heading.where(level: 2): it => [
  #v(0.45em)
  #align(center)[#text(size: 10.6pt, weight: "bold", fill: navy)[#it.body]]
  #v(0.05em)
]
#show table: set text(size: 7.25pt)
#show table.cell.where(y: 0): set text(weight: "bold", fill: white)
#show table.cell.where(y: 0): set table.cell(fill: navy)

#let metric(label, value, note) = block(
  width: 100%,
  fill: paper,
  stroke: 0.45pt + hairline,
  radius: 3pt,
  inset: (x: 7pt, y: 6pt),
)[
  #align(center)[
    #text(size: 16.5pt, weight: "bold", fill: navy)[#value]
    #linebreak()
    #text(size: 6.6pt, fill: muted, tracking: 0.35pt)[#label]
    #linebreak()
    #text(size: 7.1pt, fill: muted)[#note]
  ]
]
"""


def _title() -> str:
    return """#block(
  width: 100%,
  fill: wash,
  stroke: 0.55pt + hairline,
  radius: 4pt,
  inset: (x: 14pt, y: 12pt),
)[
  #align(center)[
    #text(size: 6.8pt, fill: muted, tracking: 1.1pt)[SENIOR DATA SCIENTIST TAKE-HOME]
    #linebreak()
    #v(0.18em)
    #text(size: 24pt, weight: "bold", fill: navy)[Peaks Marketing Performance]
    #linebreak()
    #text(size: 16pt, fill: navy)[Attribution, early LTV, and next-quarter budget choices]
    #v(0.35em)
    #line(length: 34%, stroke: 1pt + gold)
  ]
]"""


def _metrics_grid(numbers: dict[str, Any]) -> str:
    return f"""#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 7pt,
  metric("TOTAL SPEND", "{_money_short(numbers["total_spend"])}", "730 days"),
  metric("INSTALLS", "{_int(numbers["installs"])}", "all platforms"),
  metric("IOS PAID SHARE", "{_pct(numbers["ios_paid_share"])}", "MMP last-click"),
  metric("SKAN / MMP IOS", "{numbers["skan_to_mmp_ratio"]:.1f}x", "paid install gap"),
)"""


def _heading(text: str, *, level: int) -> str:
    return f"{'=' * level} {text}"


def _para(text: str) -> str:
    return _escape(text)


def _callout(text: str) -> str:
    return f'#block(fill: rgb("#f4f7f9"), stroke: 0.7pt + rgb("#577590"), inset: 8pt, radius: 2pt)[{_escape(text)}]'


def _figure(path: Path, caption: str) -> str:
    return f'#figure(\n  image("{_relative_output_path(path)}", width: 86%),\n  caption: [{_escape(caption)}],\n)'


def _table(rows: list[list[str]]) -> str:
    columns = ", ".join(["auto"] * len(rows[0]))
    cells = ",\n  ".join(
        _header_cell(value) if row_index == 0 else _content(value)
        for row_index, row in enumerate(rows)
        for value in row
    )
    return f"""#align(center)[#table(
  columns: ({columns}),
  inset: (x: 4.2pt, y: 3.2pt),
  stroke: 0.3pt + hairline,
  {cells},
)]"""


def _frame_rows(
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
            return _pct(value)
        if money:
            return _money(value)
        if abs(value) >= 1_000:
            return _int(value)
        return f"{value:.2f}"
    if isinstance(value, int):
        if money:
            return _money(value)
        return _int(value)
    return str(value)


def _money(value: float | int) -> str:
    return f"${value:,.2f}"


def _money_short(value: float | int) -> str:
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.1f}K"
    return _money(value)


def _int(value: float | int) -> str:
    return f"{value:,.0f}"


def _pct(value: float) -> str:
    return f"{value:.1%}"


def _content(text: str) -> str:
    return f"[{_escape(text)}]"


def _header_cell(text: str) -> str:
    return f'table.cell(fill: navy)[#text(fill: white, weight: "bold")[{_escape(text)}]]'


def _escape(text: str) -> str:
    replacements = {
        "\\": "\\\\",
        "[": "\\[",
        "]": "\\]",
        "#": "\\#",
        "$": "\\$",
        "*": "\\*",
        "_": "\\_",
        "`": "\\`",
    }
    return "".join(replacements.get(char, char) for char in str(text))


def _relative_output_path(path: Path) -> str:
    try:
        return path.relative_to(path.parents[1]).as_posix()
    except ValueError:
        return path.as_posix()


def _find_typst() -> str | None:
    if found := shutil.which("typst"):
        return found

    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        return None

    packages_dir = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
    candidates = sorted(packages_dir.glob("Typst.Typst_*/**/typst.exe"))
    if candidates:
        return str(candidates[-1])
    return None
