"""Summary tables and diagnostics for the report."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import lightgbm as lgb
import numpy as np
import polars as pl
from polars import col

from peaks_takehome.features import PAID_CHANNELS

if TYPE_CHECKING:
    from peaks_takehome.data import RawInputs

CHANNEL_TOTAL_COLUMNS = ("spend", "impressions", "clicks", "installs", "registrations")
SKAN_IPW_CATEGORICAL_FEATURES = ("country", "gender")
SKAN_IPW_NUMERIC_FEATURES = (
    "age",
    "tracking_enabled_num",
    "registered_num",
    "days_to_registration",
    "txn_count_14d",
    "deposit_count_14d",
    "withdrawal_count_14d",
    "gross_deposit_14d",
    "gross_withdrawal_14d",
    "net_deposit_14d",
    "balance_14d",
    "max_balance_14d",
    "days_to_first_transaction",
    "funded_14d_num",
    "install_month",
)
SKAN_IPW_FEATURES = (*SKAN_IPW_CATEGORICAL_FEATURES, *SKAN_IPW_NUMERIC_FEATURES)
SKAN_IPW_CATEGORICAL_INDICES = tuple(range(len(SKAN_IPW_CATEGORICAL_FEATURES)))
SKAN_IPW_FOLDS = 5
SKAN_IPW_PROBABILITY_FLOOR = 0.02
SKAN_IPW_PROBABILITY_CEILING = 0.95
REGISTRATION_FEATURE_CUTOFF_DAYS = 14
CategoryMaps = dict[str, dict[str, int]]


def build_report_tables(
    inputs: RawInputs,
    user_panel: pl.DataFrame,
    channel_daily_metrics: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Create the compact tables used by the PDF and CSV exports."""
    channel_performance = _channel_performance(user_panel, channel_daily_metrics)
    eligible_channel_value = _eligible_channel_value(inputs, user_panel)
    attribution_mix = _attribution_mix(user_panel)
    tracking_reliability = _tracking_reliability(user_panel)
    organic_lag_correlation = _organic_lag_correlation(inputs)
    skan_mmp_comparison = _skan_mmp_comparison(inputs)
    skan_cv_value_estimate = _skan_cv_value_estimate(inputs, user_panel)
    skan_ipw_value_estimate = _skan_ipw_value_estimate(inputs, user_panel)
    skan_value_method_comparison = _skan_value_method_comparison(skan_cv_value_estimate, skan_ipw_value_estimate)
    attribution_anomalies = _attribution_anomalies(
        attribution_mix,
        tracking_reliability,
        organic_lag_correlation,
        skan_mmp_comparison,
        channel_performance,
    )
    campaign_roles = _campaign_roles(inputs, channel_performance, eligible_channel_value)

    return {
        "channel_performance": channel_performance,
        "eligible_channel_value": eligible_channel_value,
        "attribution_mix": attribution_mix,
        "tracking_reliability": tracking_reliability,
        "organic_lag_correlation": organic_lag_correlation,
        "skan_mmp_comparison": skan_mmp_comparison,
        "skan_cv_value_estimate": skan_cv_value_estimate,
        "skan_ipw_value_estimate": skan_ipw_value_estimate,
        "skan_value_method_comparison": skan_value_method_comparison,
        "attribution_anomalies": attribution_anomalies,
        "campaign_roles": campaign_roles,
    }


def build_budget_reallocation_plan(
    tables: dict[str, pl.DataFrame],
    model_summary: dict[str, Any],
) -> pl.DataFrame:
    """Create the CMO-facing Part 3 budget action table.

    This table deliberately mixes model output, observed payback, and attribution caveats. The
    assignment asks what to do with budget, not just which channel has the lowest CPI, so each row
    has a decision label plus the evidence needed to defend it.
    """
    decision_labels = pl.DataFrame(
        [
            {
                "channel": "Referral",
                "budget_action": "Increase if scalable",
                "valuation_read": "Undervalued by paid-media scorecards",
                "rationale": "Best payback and strongest predicted value per install; keep fraud and saturation checks.",
                "guardrail": "Scale only while marginal funded quality holds.",
            },
            {
                "channel": "Google Non-Brand Search",
                "budget_action": "Protect / test more",
                "valuation_read": "Undervalued by CPI-only views",
                "rationale": "Highest predicted value per registration; CAC is high, so expand with payback caps.",
                "guardrail": "Bid to predicted LTV, not position or volume.",
            },
            {
                "channel": "Google Brand Search",
                "budget_action": "Cap and de-credit",
                "valuation_read": "Overvalued by last-click",
                "rationale": "Cheap CAC and good payback, but likely harvests demand created elsewhere.",
                "guardrail": "Report as demand capture until a holdout proves incrementality.",
            },
            {
                "channel": "Meta",
                "budget_action": "Hold / optimize",
                "valuation_read": "Possibly undervalued on iOS, weak on raw payback",
                "rationale": "Broad prospecting has mediocre payback, but iOS under-attribution means MMP may be too harsh.",
                "guardrail": "Keep only audiences that clear LTV bid caps.",
            },
            {
                "channel": "Google App Campaigns",
                "budget_action": "Tighten or reduce",
                "valuation_read": "Overvalued if treated as scaled performance",
                "rationale": "Long payback and the lowest IPW iOS value estimate among paid SKAN networks.",
                "guardrail": "Split by platform and campaign before adding spend.",
            },
            {
                "channel": "TikTok",
                "budget_action": "Reduce pending test",
                "valuation_read": "Overvalued by cheap-reach logic",
                "rationale": "Highest CPI/CAC, weakest predicted customer value, and very long observed payback.",
                "guardrail": "Only scale after geo or budget-holdout lift.",
            },
            {
                "channel": "Organic",
                "budget_action": "Measure, do not treat as free",
                "valuation_read": "Mixed bucket, partly privacy-masked demand",
                "rationale": "Organic has real funded users and moves with paid spend; do not use it as a clean baseline.",
                "guardrail": "Split owned baseline from paid spillover.",
            },
        ]
    )

    channel_quality = model_summary["channel_quality"].rename({"source": "channel"})
    skan_value = tables["skan_value_method_comparison"].rename({"network": "channel"})

    metrics = (
        tables["channel_performance"]
        .select("channel", "spend", "cpi", "cac", "early_funded_rate")
        .join(
            tables["eligible_channel_value"].select(
                col("source").alias("channel"),
                "first_year_roas",
                "payback_years_at_y1_rate",
            ),
            on="channel",
            how="left",
        )
        .join(
            channel_quality.select(
                "channel",
                "predicted_ltv_365_per_install",
                "predicted_ltv_365_per_registration",
            ),
            on="channel",
            how="left",
        )
        .join(
            skan_value.select("channel", "ipw_ltv_365_per_skan_install"),
            on="channel",
            how="left",
        )
    )

    return (
        decision_labels.join(metrics, on="channel", how="left")
        .select(
            "channel",
            "budget_action",
            "valuation_read",
            "spend",
            "cpi",
            "cac",
            "predicted_ltv_365_per_install",
            "predicted_ltv_365_per_registration",
            "first_year_roas",
            "payback_years_at_y1_rate",
            "ipw_ltv_365_per_skan_install",
            "early_funded_rate",
            "rationale",
            "guardrail",
        )
        .sort("channel")
    )


def build_measurement_framework() -> pl.DataFrame:
    """Create the CMO-facing Part 3 measurement framework table.

    The framework separates bookkeeping, value measurement, privacy correction, and
    incrementality. That separation matters because the analysis showed that one KPI cannot
    handle paid search, referrals, Organic, broad prospecting, and iOS privacy loss at once.
    """
    return pl.DataFrame(
        [
            {
                "measurement_layer": "Operational reconciliation",
                "question_answered": "Did spend and attributed funnel counts reconcile?",
                "primary_metrics": "Spend, impressions, clicks, installs, registrations, CPI, CAC",
                "data_sources": "marketing_spend + app_events",
                "cadence": "Daily / weekly",
                "decision_use": "Finance, pacing, QA; not budget allocation by itself.",
            },
            {
                "measurement_layer": "Cohort economics",
                "question_answered": "Did acquired users create funded AUM and fee revenue?",
                "primary_metrics": "Funded rate, predicted 365-day LTV, observed fee revenue, payback, ROAS",
                "data_sources": "app_events + user_profiles + user_transactions + model_predictions",
                "cadence": "Weekly cohorts",
                "decision_use": "Bid caps, budget guardrails, and channel quality ranking.",
            },
            {
                "measurement_layer": "iOS privacy correction",
                "question_answered": "How much paid iOS volume and value is hidden by MMP limits?",
                "primary_metrics": "SKAN installs, SKAN / MMP gap, IPW LTV per SKAN install, CV sensitivity",
                "data_sources": "skan_attribution_ios + app_events + user_panel",
                "cadence": "Weekly / monthly",
                "decision_use": "Use SKAN as iOS volume anchor and IPW as the value base case.",
            },
            {
                "measurement_layer": "Incrementality testing",
                "question_answered": "Which spend creates incremental funded users and AUM?",
                "primary_metrics": "Lift in registrations, funded users, net deposits, AUM, fee revenue, payback",
                "data_sources": "Geo holdouts, budget holdouts, campaign experiments, cohort LTV",
                "cadence": "Monthly / quarterly",
                "decision_use": "Validate Brand Search credit, TikTok scale, and major budget shifts.",
            },
            {
                "measurement_layer": "Channel-role scorecards",
                "question_answered": "Is each channel judged by the job it is meant to do?",
                "primary_metrics": "Search payback, referral marginal quality, prospecting LTV caps, brand lift",
                "data_sources": "campaign_roles + channel_performance + cohort value tables",
                "cadence": "Weekly review, monthly reset",
                "decision_use": "Stop comparing brand, referral, search, and prospecting on CPI alone.",
            },
            {
                "measurement_layer": "Organic baseline",
                "question_answered": "How much Organic is owned demand versus paid spillover?",
                "primary_metrics": "Organic trend, paid/Organic lag correlation, platform mix, tracking mix",
                "data_sources": "marketing_spend + app_events + attribution anomalies",
                "cadence": "Monthly",
                "decision_use": "Prevent privacy-masked paid demand from being treated as free growth.",
            },
        ]
    )


def build_quick_wins() -> pl.DataFrame:
    """Create the 2-3 immediate actions for the CMO summary.

    These are deliberately operational. Each action can start without waiting for a new vendor,
    a full attribution rebuild, or a longer modelling cycle.
    """
    return pl.DataFrame(
        [
            {
                "quick_win": "Launch a weekly cohort value scorecard",
                "owner": "Growth Analytics + Finance",
                "first_action": "Publish channel x platform x campaign cohorts with CPI, CAC, funded rate, predicted LTV, observed LTV, and payback.",
                "why_now": "CPI alone misranks Brand Search, Referral, TikTok, and App Campaigns.",
                "success_metric": "Weekly budget review uses predicted LTV and payback before spend moves.",
                "timeline": "1 week",
            },
            {
                "quick_win": "Replace iOS MMP-only reporting",
                "owner": "Marketing Analytics + Mobile Measurement",
                "first_action": "Use SKAN as paid iOS volume anchor, IPW LTV as the base value estimate, and CV-rank only as sensitivity.",
                "why_now": "SKAN shows about 6.7x the MMP-paid iOS installs, so raw MMP is too low for iOS budget decisions.",
                "success_metric": "Every iOS channel readout includes SKAN / MMP gap, IPW LTV, and CV sensitivity.",
                "timeline": "1-2 weeks",
            },
            {
                "quick_win": "Put spend guardrails on weak or ambiguous channels",
                "owner": "Growth Lead",
                "first_action": "Cap Brand Search credit, tighten App Campaigns, and pause TikTok scale-ups until a holdout proves funded-AUM lift.",
                "why_now": "Brand Search is likely demand capture, App Campaigns have the weakest IPW iOS value, and TikTok has the longest payback.",
                "success_metric": "No increase clears unless it passes LTV bid caps or an incrementality test.",
                "timeline": "2 weeks",
            },
        ]
    )


def summary_numbers(
    inputs: RawInputs,
    user_panel: pl.DataFrame,
    tables: dict[str, pl.DataFrame],
    model_summary: dict[str, Any],
) -> dict[str, Any]:
    """Collect headline values for the narrative report."""
    skan_gap = tables["skan_mmp_comparison"].with_columns(
        (col("skan_installs") / col("mmp_ios_paid_installs")).alias("skan_to_mmp_ratio")
    )
    top_lag = tables["organic_lag_correlation"].sort("correlation", descending=True).head(1)
    platform_mix = tables["attribution_mix"]
    ios_paid_share = platform_mix.filter((col("platform") == "ios") & (col("source_bucket") == "paid_attributed"))
    android_paid_share = platform_mix.filter(
        (col("platform") == "android") & (col("source_bucket") == "paid_attributed")
    )
    organic_rows = platform_mix.filter(col("source_bucket") == "organic")

    return {
        "total_spend": inputs.marketing_spend.select(col("spend").sum()).item(),
        "installs": user_panel.height,
        "registrations": user_panel.filter(col("registered")).height,
        "total_transacting_users": inputs.user_transactions.select(col("user_id").n_unique()).item(),
        "transacting_users": user_panel.filter(col("txn_count_14d") > 0).height,
        "ios_paid_share": ios_paid_share.select("share").item(),
        "android_paid_share": android_paid_share.select("share").item(),
        "ios_organic_share": organic_rows.filter(col("platform") == "ios").select("share").item(),
        "android_organic_share": organic_rows.filter(col("platform") == "android").select("share").item(),
        "skan_total_installs": skan_gap.select(col("skan_installs").sum()).item(),
        "mmp_ios_paid_installs": skan_gap.select(col("mmp_ios_paid_installs").sum()).item(),
        "skan_to_mmp_ratio": skan_gap.select(col("skan_installs").sum() / col("mmp_ios_paid_installs").sum()).item(),
        "best_organic_lag_days": top_lag.select("lag_days").item(),
        "best_organic_lag_correlation": top_lag.select("correlation").item(),
        "model_validation_mae": model_summary["metrics"]["gbdt_validation_mae"],
        "model_validation_log_rmse": model_summary["metrics"]["gbdt_validation_log_rmse"],
    }


def _channel_performance(user_panel: pl.DataFrame, channel_daily_metrics: pl.DataFrame) -> pl.DataFrame:
    """Collapse daily channel metrics into the Part 1 channel scorecard.

    `build_channel_daily_metrics` gives one row per date/channel. For the report, we also need
    a channel-level view that can be sorted and compared: total spend, total reach/click volume,
    installs, registrations, and the standard acquisition metrics.

    A second aggregation comes from `user_panel`, because funded behavior is a user-level outcome. Joining these
    two views lets the scorecard show both acquisition efficiency and early customer quality in the same table.
    """
    # First collapse the daily table to one row per channel. These totals are the denominator
    # base for CPI, CAC, CPM, CTR, and CPC.
    totals = channel_daily_metrics.group_by("channel").agg([col(column).sum() for column in CHANNEL_TOTAL_COLUMNS])

    # The user panel is one row per install, so it is the right place to count distinct users
    # and early funded users. We group by `source` because it is the install-side channel label.
    funded = user_panel.group_by("source").agg(
        col("user_id").n_unique().alias("panel_installs"),
        col("registered").sum().alias("panel_registrations"),
        (col("txn_count_14d") > 0).sum().alias("funded_14d_users"),
    )

    return (
        # Match media-channel totals to user-panel outcomes. Paid channels, Organic, and Referral
        # all use the same channel/source labels after the earlier normalization step.
        totals.join(funded, left_on="channel", right_on="source", how="left")
        # These are descriptive last-click metrics. They are useful for comparing the observed
        # funnel, but they should not be read as incrementality or true customer value.
        .with_columns(
            _safe_cost_divide(col("spend"), col("installs")).alias("cpi"),
            _safe_cost_divide(col("spend"), col("registrations")).alias("cac"),
            _safe_cost_divide(col("spend") * 1000, col("impressions")).alias("cpm"),
            _safe_divide(col("clicks"), col("impressions")).alias("ctr"),
            _safe_cost_divide(col("spend"), col("clicks")).alias("cpc"),
            _safe_divide(col("registrations"), col("installs")).alias("install_to_registration_rate"),
            _safe_divide(col("funded_14d_users"), col("registrations")).alias("early_funded_rate"),
        )
        # Spend-descending order makes the table read like a budget review: largest media levers first.
        .sort("spend", descending=True)
    )


def _eligible_channel_value(inputs: RawInputs, user_panel: pl.DataFrame) -> pl.DataFrame:
    """Compare channel cost against observed 365-day value on uncensored cohorts.

    This table is the backward-looking payback check for Part 2.3. The model can predict value
    for all installs, but payback needs both known first-year LTV and spend from the matching
    acquisition window. That is why the table is restricted to users with a full 365-day outcome.
    """
    # Limit the value comparison to installs old enough to have a complete 365-day LTV target.
    # Without this filter, newer high-volume channels would look artificially weak because their
    # balances have not had a full year to generate fee revenue.
    eligible = user_panel.filter(col("eligible_365d"))
    cutoff = eligible.select(col("install_date").max()).item()

    # Spend is cut at the same install-date boundary as the eligible cohort. This keeps CPI, CAC,
    # ROAS, and payback from mixing complete-LTV users with media spend from later censored cohorts.
    spend = (
        inputs.marketing_spend.filter(col("date") <= cutoff)
        .group_by("channel")
        .agg(col("spend").sum().alias("eligible_period_spend"))
    )

    return (
        eligible.group_by("source")
        .agg(
            pl.len().alias("eligible_installs"),
            col("registered").sum().alias("eligible_registrations"),
            (col("ltv_365_fee_usd") > 0).sum().alias("positive_ltv_users"),
            col("ltv_365_fee_usd").sum().alias("total_ltv_365_fee_usd"),
            col("ltv_365_fee_usd").mean().alias("avg_ltv_365_per_install"),
            col("ltv_365_fee_usd").filter(col("registered")).mean().alias("avg_ltv_365_per_registration"),
            col("ltv_365_fee_usd").quantile(0.9).alias("p90_ltv_365_per_install"),
        )
        .join(spend, left_on="source", right_on="channel", how="left")
        .with_columns(col("eligible_period_spend").fill_null(0.0))
        .with_columns(
            # Cost metrics let us compare against the Part 1 CPI/CAC view.
            _safe_cost_divide(col("eligible_period_spend"), col("eligible_installs")).alias("cohort_cpi"),
            _safe_cost_divide(col("eligible_period_spend"), col("eligible_registrations")).alias("cohort_cac"),
            # Value metrics answer whether those cheaper customers actually generated fee revenue.
            _safe_divide(col("positive_ltv_users"), col("eligible_registrations")).alias("positive_ltv_rate"),
            _safe_cost_divide(col("total_ltv_365_fee_usd"), col("eligible_period_spend")).alias("first_year_roas"),
            # This is a simple payback proxy: spend divided by first-year fee revenue. For example,
            # 0.5 means the channel recouped spend in about half a year at the observed year-one run
            # rate; 6.0 means the first-year fee revenue would need roughly six years to repay spend.
            _safe_cost_divide(col("eligible_period_spend"), col("total_ltv_365_fee_usd")).alias(
                "payback_years_at_y1_rate"
            ),
        )
        # Sort by observed customer value so the top rows show the highest-quality cohorts before
        # adding the modelled forward-looking view.
        .sort("avg_ltv_365_per_registration", descending=True)
    )


def _attribution_mix(user_panel: pl.DataFrame) -> pl.DataFrame:
    """Compare Organic, paid-attributed, and Referral mix by platform.

    This is the first attribution-gap check. If iOS privacy loss is material, the MMP source
    mix should look different on iOS than Android: more installs pushed into Organic and fewer
    installs explicitly labelled as paid.
    """
    return (
        # `user_panel` is one row per install, so this count is an install mix rather than an
        # event-count mix. Registrations are included to show whether the same pattern survives
        # after the first conversion step.
        user_panel.group_by("platform", "source_bucket")
        .agg(pl.len().alias("installs"), col("registered").sum().alias("registrations"))
        .with_columns(
            # Shares are normalized within each platform so Android and iOS can be compared
            # even if their absolute install volumes differ.
            (col("installs") / col("installs").sum().over("platform")).alias("share"),
            (col("registrations") / col("registrations").sum().over("platform")).alias("registration_share"),
        )
        .sort("platform", "source_bucket")
    )


def _tracking_reliability(user_panel: pl.DataFrame) -> pl.DataFrame:
    """Show whether attribution labels change when registered users opted into tracking.

    This isolates the privacy mechanism more directly than platform alone. If non-tracked iOS
    users are disproportionately labelled Organic, that supports the interpretation that some
    paid iOS demand is being hidden by consent and attribution limits.
    """
    # `tracking_enabled` only exists in the profile table, so this check is limited to users who
    # registered. That makes it narrower than install mix, but more diagnostic.
    registered = user_panel.filter(col("registered"))
    return (
        registered.group_by("platform", "tracking_enabled", "source_bucket")
        .agg(pl.len().alias("registrations"))
        .with_columns(
            # Normalize inside each platform x tracking state to compare source composition
            # between tracked and non-tracked users.
            (col("registrations") / col("registrations").sum().over("platform", "tracking_enabled")).alias("share")
        )
        .sort("platform", "tracking_enabled", "source_bucket")
    )


def _organic_lag_correlation(inputs: RawInputs) -> pl.DataFrame:
    """Test whether Organic installs move with recent paid spend.

    This is a lightweight anomaly test, not a causal model. If Organic were a stable unpaid
    baseline, daily Organic installs should not strongly track paid media intensity. A positive
    lag correlation means paid spend on day t is compared with Organic installs on day t + lag,
    which is the expected direction if paid campaigns create same-day or delayed uncredited
    installs.
    """
    # Collapse all paid media into a daily intensity signal. Referral is excluded because it is
    # incentive spend rather than auction media with impressions and clicks.
    paid_daily = (
        inputs.marketing_spend.filter(col("channel").is_in(PAID_CHANNELS))
        .group_by("date")
        .agg(col("spend").sum().alias("paid_spend"))
    )

    # Count installs that the MMP source field calls Organic. If privacy masking or view-through
    # effects are present, this bucket can contain demand influenced by paid media.
    organic_daily = (
        inputs.app_events.filter((col("event_name") == "install") & (col("source") == "Organic"))
        .with_columns(col("event_timestamp").dt.date().alias("date"))
        .group_by("date")
        .agg(pl.len().alias("organic_installs"))
    )

    # Keep every calendar day seen in either table. Filling nulls with zero avoids losing days
    # where spend or Organic installs are absent.
    daily = paid_daily.join(organic_daily, on="date", how="full", coalesce=True).fill_null(0).sort("date")

    # For each lag, correlate paid spend with Organic installs shifted earlier in the table:
    # lag 0 compares same-day values; lag 7 compares paid spend today with Organic installs
    # seven days later. The output is intentionally small so the report can show the full
    # sensitivity curve rather than only the single best lag.
    return pl.DataFrame(
        [
            {
                "lag_days": lag_days,
                "correlation": daily.select(
                    pl.corr(col("paid_spend"), col("organic_installs").shift(-lag_days))
                ).item(),
            }
            for lag_days in range(15)
        ]
    )


def _attribution_anomalies(
    attribution_mix: pl.DataFrame,
    tracking_reliability: pl.DataFrame,
    organic_lag_correlation: pl.DataFrame,
    skan_mmp_comparison: pl.DataFrame,
    channel_performance: pl.DataFrame,
) -> pl.DataFrame:
    """Summarize cross-source signals that make last-click attribution suspect.

    The assignment asks for unusual patterns, time periods, or correlations across all available
    data. This table keeps the result presentation-sized: each row names a signal, the data
    source combination behind it, the numerical evidence, and the practical readout.
    """
    top_lag = organic_lag_correlation.sort("correlation", descending=True).row(0, named=True)
    ios_paid_share = _mix_share(attribution_mix, "ios", "paid_attributed")
    android_paid_share = _mix_share(attribution_mix, "android", "paid_attributed")
    ios_organic_share = _mix_share(attribution_mix, "ios", "organic")
    android_organic_share = _mix_share(attribution_mix, "android", "organic")
    skan_total = skan_mmp_comparison.select(col("skan_installs").sum()).item()
    mmp_total = skan_mmp_comparison.select(col("mmp_ios_paid_installs").sum()).item()
    organic = channel_performance.filter(col("channel") == "Organic")
    referral = channel_performance.filter(col("channel") == "Referral")

    non_tracking_organic = tracking_reliability.filter(
        (~col("tracking_enabled")) & (col("source_bucket") == "organic")
    ).select(
        col("share").filter(col("platform") == "ios").first().fill_null(0.0).alias("ios"),
        col("share").filter(col("platform") == "android").first().fill_null(0.0).alias("android"),
    )

    return pl.DataFrame(
        [
            {
                "signal": "Paid/Organic daily co-movement",
                "evidence": (
                    f"marketing_spend + app_events: best lag r={top_lag['correlation']:.2f} "
                    f"at {top_lag['lag_days']:.0f} days"
                ),
                "readout": "Organic is not a clean unpaid baseline.",
            },
            {
                "signal": "iOS source mix discontinuity",
                "evidence": (
                    f"app_events: paid {ios_paid_share:.1%} iOS vs {android_paid_share:.1%} Android; "
                    f"Organic {ios_organic_share:.1%} iOS vs {android_organic_share:.1%} Android"
                ),
                "readout": "Platform-level last-click labels are not comparable.",
            },
            {
                "signal": "SKAN exceeds MMP paid iOS",
                "evidence": f"skan + app_events: {_fmt_int(skan_total)} SKAN paid iOS installs vs {_fmt_int(mmp_total)}"
                " MMP",
                "readout": "Use SKAN to calibrate iOS paid volume.",
            },
            {
                "signal": "Non-tracking collapses to Organic",
                "evidence": (
                    "user_profiles + app_events: non-tracking Organic share is "
                    f"{non_tracking_organic.select('ios').item():.1%} on iOS and "
                    f"{non_tracking_organic.select('android').item():.1%} on Android"
                ),
                "readout": "Consent state affects attribution availability.",
            },
            {
                "signal": "Organic has real funded users",
                "evidence": (
                    "app_events + transactions: "
                    f"{_fmt_int(organic.select('registrations').item())} Organic regs; "
                    f"{organic.select('early_funded_rate').item():.1%} early funded"
                ),
                "readout": "Zero assigned spend does not mean zero acquisition influence.",
            },
            {
                "signal": "Referral is not auction media",
                "evidence": (
                    f"marketing_spend: {_fmt_money(referral.select('spend').item())} Referral spend with "
                    f"{_fmt_int(referral.select('impressions').item())} impressions and "
                    f"{_fmt_int(referral.select('clicks').item())} clicks"
                ),
                "readout": "Separate incentive/referral economics from paid-media CPI.",
            },
        ]
    )


def _skan_mmp_comparison(inputs: RawInputs) -> pl.DataFrame:
    """Compare Apple's aggregate SKAN iOS paid installs with user-level MMP paid labels.

    SKAN is not user-level and arrives through a different measurement path, but it is designed
    to preserve aggregate iOS ad attribution under privacy constraints. When SKAN counts more
    paid iOS installs than the MMP source field, the gap is evidence that user-level MMP labels
    are undercounting paid iOS acquisition.
    """
    # SKAN arrives as network-level postbacks, so the clean comparison grain is network/channel.
    skan = inputs.skan_attribution_ios.group_by("network").agg(col("install_count").sum().alias("skan_installs"))

    # The MMP side is the subset of app installs that are both iOS and explicitly labelled as a
    # paid channel. Privacy-masked paid users often appear elsewhere, usually Organic.
    mmp = (
        inputs.app_events.filter(
            (col("event_name") == "install") & (col("platform") == "ios") & (col("source").is_in(PAID_CHANNELS))
        )
        .group_by("source")
        .agg(pl.len().alias("mmp_ios_paid_installs"))
    )

    return (
        skan.join(mmp, left_on="network", right_on="source", how="left")
        .with_columns(
            # Positive gap: SKAN sees paid iOS installs that the user-level MMP source field does
            # not expose. Ratio > 1 means SKAN volume exceeds MMP-paid iOS volume.
            (col("skan_installs") - col("mmp_ios_paid_installs")).alias("unattributed_ios_paid_gap"),
            _safe_divide(col("skan_installs"), col("mmp_ios_paid_installs")).alias("skan_to_mmp_ratio"),
        )
        .sort("unattributed_ios_paid_gap", descending=True)
    )


def _skan_cv_value_estimate(inputs: RawInputs, user_panel: pl.DataFrame) -> pl.DataFrame:
    """Estimate iOS paid value when SKAN only gives aggregate conversion values.

    SKAN does not expose user-level installs or balances, so we cannot join a SKAN postback to
    one customer's 365-day LTV. The compromise here is a calibration approach:

    1. Use tracked paid iOS users as the value reference population.
    2. Treat SKAN conversion values as ordered early-value buckets.
    3. Map each SKAN bucket to a slice of the tracked paid iOS LTV distribution.
    4. Apply the bucket-level value estimate back to SKAN install counts by network.

    This gives a network-level planning estimate. It should be replaced by the real SKAN
    conversion schema if the marketing team can provide it.
    """
    # Calibration population: users who are iOS, paid-attributed, tracking-enabled, and old
    # enough to have complete 365-day fee revenue. This is not all iOS paid traffic; it is the
    # subset where both attribution and value are observable.
    tracked_paid_ios = user_panel.filter(
        (col("platform") == "ios")
        & (col("tracking_enabled"))
        & (col("source").is_in(PAID_CHANNELS))
        & (col("eligible_365d"))
    )
    if tracked_paid_ios.height == 0:
        return pl.DataFrame()

    # SKAN conversion values are aggregate counts, not users. We collapse them into a distribution
    # of bucket weights: for example, if conversion value 3 accounts for 10% of SKAN installs, it
    # gets mapped to a 10% slice of the tracked paid iOS value distribution.
    cv_distribution = (
        inputs.skan_attribution_ios.group_by("skan_conversion_value")
        .agg(col("install_count").sum().alias("skan_installs"))
        .sort("skan_conversion_value", nulls_last=True)
    )
    known_cv = cv_distribution.filter(col("skan_conversion_value").is_not_null())
    weights = known_cv.select(col("skan_installs") / col("skan_installs").sum()).to_series().to_list()
    values = known_cv.select("skan_conversion_value").to_series().to_list()

    # Sorting observed LTV lets us map low conversion values to lower-value users and higher
    # conversion values to higher-value users without pretending we know the real SKAN schema.
    ltv = tracked_paid_ios.select("ltv_365_fee_usd").to_series().to_numpy()
    sorted_ltv = np.sort(ltv)

    rows: list[dict[str, float | int | None]] = []
    lower = 0.0
    for conversion_value, weight in zip(values, weights, strict=True):
        # Each conversion value receives the next quantile band of tracked iOS paid LTV, with the
        # band width determined by that conversion value's share of SKAN installs.
        upper = min(lower + float(weight), 1.0)
        start = int(np.floor(lower * len(sorted_ltv)))
        stop = max(start + 1, int(np.floor(upper * len(sorted_ltv))))
        rows.append(
            {
                "skan_conversion_value": conversion_value,
                "estimated_ltv_365_fee_usd": float(sorted_ltv[start:stop].mean()),
                "tracked_quantile_low": lower,
                "tracked_quantile_high": upper,
            }
        )
        lower = upper

    # Missing conversion values are treated conservatively. They are still paid iOS installs, but
    # without a conversion signal we assign them a low tracked-value percentile.
    null_share = cv_distribution.filter(col("skan_conversion_value").is_null()).select("skan_installs")
    if null_share.height:
        rows.append(
            {
                "skan_conversion_value": None,
                "estimated_ltv_365_fee_usd": float(np.quantile(sorted_ltv, 0.05)),
                "tracked_quantile_low": 0.0,
                "tracked_quantile_high": 0.05,
            }
        )
    mapping = pl.DataFrame(rows)

    # Apply the estimated value per conversion bucket back to each SKAN postback row, then roll up
    # to network. The output is deliberately network-level because SKAN is aggregate by design.
    return (
        inputs.skan_attribution_ios.join(mapping, on="skan_conversion_value", how="left")
        .with_columns((col("install_count") * col("estimated_ltv_365_fee_usd")).alias("estimated_total_ltv_365"))
        .group_by("network")
        .agg(
            col("install_count").sum().alias("skan_installs"),
            col("estimated_total_ltv_365").sum().alias("estimated_total_ltv_365"),
        )
        .with_columns(
            _safe_divide(col("estimated_total_ltv_365"), col("skan_installs")).alias(
                "estimated_ltv_365_per_skan_install"
            )
        )
        .sort("estimated_ltv_365_per_skan_install", descending=True)
    )


def _skan_ipw_value_estimate(inputs: RawInputs, user_panel: pl.DataFrame) -> pl.DataFrame:
    """Estimate SKAN value with an inverse-probability weighting sensitivity model.

    The CV-rank method above assumes SKAN conversion values are ordered value buckets. This
    method attacks the missing-data problem from the other side: among complete iOS cohorts,
    which users are observable as MMP-paid at all?

    The output is still not causal lift. It reweights observed paid iOS users to look less like
    the subset that survives MMP attribution, then applies the reweighted value estimate to
    SKAN's aggregate paid iOS volume by network.
    """
    ipw_frame = _skan_ipw_frame(user_panel)
    observed_paid = ipw_frame.get_column("observed_paid_ios").to_numpy().astype(bool)
    if ipw_frame.height == 0 or observed_paid.sum() == 0:
        return pl.DataFrame()

    category_maps = _fit_skan_ipw_category_maps(ipw_frame)
    features = _skan_ipw_feature_matrix(ipw_frame.select(*SKAN_IPW_FEATURES), category_maps)
    probabilities = _crossfit_skan_observability_probability(features, observed_paid)
    observed_paid_rate = float(observed_paid.mean())

    # Stabilized weights avoid changing the overall sample size too much: rows that are unlikely
    # to be observed as paid receive more weight, while very observable rows receive less.
    weighted_paid_ios = (
        ipw_frame.with_columns(pl.Series("observability_probability", probabilities))
        .filter(col("observed_paid_ios") == 1)
        .with_columns(
            col("observability_probability")
            .clip(SKAN_IPW_PROBABILITY_FLOOR, SKAN_IPW_PROBABILITY_CEILING)
            .alias("clipped_observability_probability")
        )
        .with_columns(
            (pl.lit(observed_paid_rate) / col("clipped_observability_probability")).alias("stabilized_weight")
        )
        .with_columns((col("ltv_365_fee_usd") * col("stabilized_weight")).alias("weighted_ltv_365_fee_usd"))
    )

    network_value = (
        weighted_paid_ios.group_by("source")
        .agg(
            pl.len().alias("mmp_paid_ios_installs"),
            col("ltv_365_fee_usd").mean().alias("unweighted_mmp_ltv_365_per_install"),
            col("observability_probability").mean().alias("mean_observability_probability"),
            col("stabilized_weight").mean().alias("mean_stabilized_weight"),
            col("stabilized_weight").max().alias("max_stabilized_weight"),
            col("weighted_ltv_365_fee_usd").sum().alias("_weighted_ltv_sum"),
            col("stabilized_weight").sum().alias("_weight_sum"),
            ((col("stabilized_weight").sum() ** 2) / (col("stabilized_weight") ** 2).sum()).alias(
                "effective_sample_size"
            ),
        )
        .with_columns(_safe_divide(col("_weighted_ltv_sum"), col("_weight_sum")).alias("ipw_ltv_365_per_skan_install"))
        .drop("_weighted_ltv_sum", "_weight_sum")
    )

    skan_volume = inputs.skan_attribution_ios.group_by("network").agg(col("install_count").sum().alias("skan_installs"))

    return (
        skan_volume.join(network_value, left_on="network", right_on="source", how="left")
        .with_columns(
            (col("skan_installs") * col("ipw_ltv_365_per_skan_install")).alias("estimated_total_ltv_365_ipw"),
            pl.lit(observed_paid_rate).alias("observed_paid_ios_rate"),
            pl.lit(ipw_frame.height).alias("selection_model_rows"),
            pl.lit(int(observed_paid.sum())).alias("selection_model_observed_paid_ios"),
            pl.lit(SKAN_IPW_FOLDS).alias("selection_model_folds"),
            pl.lit(SKAN_IPW_PROBABILITY_FLOOR).alias("propensity_floor"),
            pl.lit(SKAN_IPW_PROBABILITY_CEILING).alias("propensity_ceiling"),
        )
        .sort("ipw_ltv_365_per_skan_install", descending=True)
    )


def _skan_value_method_comparison(
    skan_cv_value_estimate: pl.DataFrame,
    skan_ipw_value_estimate: pl.DataFrame,
) -> pl.DataFrame:
    """Put the old CV-rank estimate beside the IPW estimate.

    IPW is the primary planning estimate because it explicitly addresses observed iOS selection
    bias. The CV-rank estimate remains useful as a sensitivity check because the true SKAN
    conversion-value schema is not available in the assignment data.
    """
    if skan_cv_value_estimate.height == 0 or skan_ipw_value_estimate.height == 0:
        return pl.DataFrame()

    return (
        skan_ipw_value_estimate.select(
            "network",
            "skan_installs",
            "mmp_paid_ios_installs",
            "ipw_ltv_365_per_skan_install",
            "estimated_total_ltv_365_ipw",
            "mean_observability_probability",
            "mean_stabilized_weight",
            "max_stabilized_weight",
            "effective_sample_size",
        )
        .join(
            skan_cv_value_estimate.select(
                "network",
                col("estimated_ltv_365_per_skan_install").alias("cv_ltv_365_per_skan_install"),
                col("estimated_total_ltv_365").alias("estimated_total_ltv_365_cv"),
            ),
            on="network",
            how="left",
        )
        .with_columns(
            _safe_divide(col("cv_ltv_365_per_skan_install"), col("ipw_ltv_365_per_skan_install")).alias(
                "cv_to_ipw_ltv_ratio"
            )
        )
        .sort("ipw_ltv_365_per_skan_install", descending=True)
    )


def _skan_ipw_frame(user_panel: pl.DataFrame) -> pl.DataFrame:
    """Build the complete-cohort iOS frame used to model MMP-paid observability."""
    registration_available_14d = col("days_to_registration").is_not_null() & (
        col("days_to_registration") <= REGISTRATION_FEATURE_CUTOFF_DAYS
    )
    return (
        user_panel.filter((col("platform") == "ios") & col("eligible_365d") & col("feature_complete_14d"))
        .with_columns(
            registration_available_14d.alias("_registration_available_14d"),
            col("source").is_in(PAID_CHANNELS).cast(pl.Int8).alias("observed_paid_ios"),
            col("funded_14d").cast(pl.Int8).alias("funded_14d_num"),
        )
        .with_columns(
            pl.when(col("_registration_available_14d")).then(1).otherwise(0).cast(pl.Int8).alias("registered_num"),
            pl.when(col("_registration_available_14d"))
            .then(col("tracking_enabled").fill_null(False).cast(pl.Int8))
            .otherwise(0)
            .cast(pl.Int8)
            .alias("tracking_enabled_num"),
            pl.when(col("_registration_available_14d")).then(col("age")).otherwise(None).alias("age"),
            pl.when(col("_registration_available_14d"))
            .then(col("gender"))
            .otherwise(pl.lit("unknown"))
            .alias("gender"),
            pl.when(col("_registration_available_14d"))
            .then(col("days_to_registration"))
            .otherwise(None)
            .alias("days_to_registration"),
        )
        .select(
            "user_id",
            "source",
            "ltv_365_fee_usd",
            "observed_paid_ios",
            *SKAN_IPW_FEATURES,
        )
        .with_columns([col(feature).cast(pl.String).fill_null("missing") for feature in SKAN_IPW_CATEGORICAL_FEATURES])
    )


def _fit_skan_ipw_category_maps(frame: pl.DataFrame) -> CategoryMaps:
    return {feature: _skan_ipw_category_map(frame, feature) for feature in SKAN_IPW_CATEGORICAL_FEATURES}


def _skan_ipw_category_map(frame: pl.DataFrame, feature: str) -> dict[str, int]:
    values = frame.select(col(feature).cast(pl.String).fill_null("missing").unique().sort()).get_column(feature)
    ordered_values = ["missing", *(value for value in values.to_list() if value != "missing")]
    return {value: code for code, value in enumerate(ordered_values)}


def _skan_ipw_feature_matrix(features: pl.DataFrame, category_maps: CategoryMaps) -> np.ndarray:
    encoded = features.select(
        *[
            col(feature)
            .cast(pl.String)
            .fill_null("missing")
            .replace_strict(category_maps[feature], default=0, return_dtype=pl.Int32)
            .alias(feature)
            for feature in SKAN_IPW_CATEGORICAL_FEATURES
        ],
        *[col(feature).cast(pl.Float64) for feature in SKAN_IPW_NUMERIC_FEATURES],
    )
    return encoded.to_numpy().astype(np.float32, copy=False)


def _crossfit_skan_observability_probability(
    features: np.ndarray,
    observed_paid: np.ndarray,
    *,
    random_state: int = 42,
) -> np.ndarray:
    """Predict MMP-paid observability out-of-fold for every eligible iOS row."""
    row_count = observed_paid.size
    probabilities = np.empty(row_count, dtype=np.float64)
    rng = np.random.default_rng(random_state)
    shuffled_indices = rng.permutation(row_count)
    fold_ids = np.empty(row_count, dtype=np.int16)
    fold_ids[shuffled_indices] = np.arange(row_count) % SKAN_IPW_FOLDS

    for fold in range(SKAN_IPW_FOLDS):
        holdout_mask = fold_ids == fold
        train_mask = ~holdout_mask
        train_observed = observed_paid[train_mask]
        if train_observed.sum() == 0 or train_observed.sum() == train_observed.size:
            probabilities[holdout_mask] = train_observed.mean()
            continue

        model = _train_skan_observability_model(
            features[train_mask],
            train_observed,
            random_state=random_state + fold,
        )
        probabilities[holdout_mask] = np.asarray(model.predict(features[holdout_mask]), dtype=np.float64)

    return probabilities


def _train_skan_observability_model(
    features: np.ndarray,
    observed_paid: np.ndarray,
    *,
    random_state: int,
) -> lgb.Booster:
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 100,
        "lambda_l2": 1.0,
        "seed": random_state,
        "feature_pre_filter": False,
        "force_col_wise": True,
        "verbosity": -1,
    }
    dataset = lgb.Dataset(
        features,
        label=observed_paid.astype(np.float32),
        feature_name=list(SKAN_IPW_FEATURES),
        categorical_feature=list(SKAN_IPW_CATEGORICAL_INDICES),
        free_raw_data=True,
    )
    return lgb.train(
        params,
        dataset,
        num_boost_round=180,
        callbacks=[lgb.log_evaluation(period=0)],
    )


def _campaign_roles(
    inputs: RawInputs,
    channel_performance: pl.DataFrame,
    eligible_channel_value: pl.DataFrame,
) -> pl.DataFrame:
    """Classify each channel's funnel role and the right measurement lens.

    This is deliberately not a statistical model. The assignment asks for a marketing read
    based on campaign names, CPMs, CTRs, and value. The output keeps those ingredients together:
    media metrics explain how the channel behaves, campaign names explain the intended buying
    motion, and cohort LTV/ROAS keeps the recommendation from being based on cheap traffic alone.
    """
    # Campaign names are qualitative evidence. "Brand Awareness", "Search", "App Campaign",
    # and "Member Get Member" tell us what the channel was probably bought to do.
    campaigns = inputs.marketing_spend.group_by("channel").agg(
        col("campaign_name").unique().str.join("; ").alias("campaign_names")
    )

    # These labels are business rules, not inferred classes. Each note says how the channel
    # should be valued if last-click does not capture its actual role in the funnel.
    role_labels = pl.DataFrame(
        [
            {
                "channel": "TikTok",
                "role": "Upper-funnel reach / brand demand creation",
                "measurement_note": "Low CPM and campaign naming point to awareness; last-click will miss assisted "
                "demand.",
            },
            {
                "channel": "Meta",
                "role": "Scaled paid prospecting",
                "measurement_note": "Broad reach with better CTR than TikTok; should be judged on cohort value and "
                "incrementality.",
            },
            {
                "channel": "Google App Campaigns",
                "role": "Algorithmic app acquisition",
                "measurement_note": "Performance channel, but iOS privacy loss means MMP counts are incomplete.",
            },
            {
                "channel": "Google Non-Brand Search",
                "role": "High-intent category demand capture",
                "measurement_note": "Higher CPC/CPM, but strong LTV can justify spend if incrementality holds.",
            },
            {
                "channel": "Google Brand Search",
                "role": "Brand demand harvesting",
                "measurement_note": "Very efficient last-click economics, but likely captures demand created "
                "elsewhere.",
            },
            {
                "channel": "Organic",
                "role": "Unattributed, owned, and privacy-masked demand",
                "measurement_note": "Do not assume it is unpaid; split owned baseline from paid spillover using tests.",
            },
            {
                "channel": "Referral",
                "role": "Member-get-member acquisition",
                "measurement_note": "High conversion and quality; treat separately from paid media because spend is an "
                "incentive cost.",
            },
        ]
    )

    # Join the qualitative role labels back to the numeric scorecard. CPM/CTR/CPC describe
    # reach and intent, CPI/CAC describe last-click efficiency, and LTV/ROAS describe quality.
    return (
        channel_performance.select("channel", "cpm", "ctr", "cpc", "cpi", "cac")
        .join(
            eligible_channel_value.select("source", "avg_ltv_365_per_registration", "first_year_roas"),
            left_on="channel",
            right_on="source",
            how="left",
        )
        .join(campaigns, on="channel", how="left")
        .join(role_labels, on="channel", how="left")
        .sort("channel")
    )


def _safe_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return pl.when(denominator > 0).then(numerator / denominator).otherwise(None)


def _safe_cost_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return pl.when((denominator > 0) & (numerator > 0)).then(numerator / denominator).otherwise(None)


def _mix_share(attribution_mix: pl.DataFrame, platform: str, source_bucket: str) -> float:
    return (
        attribution_mix.filter((col("platform") == platform) & (col("source_bucket") == source_bucket))
        .select("share")
        .item()
    )


def _fmt_int(value: float | int) -> str:
    return f"{value:,.0f}"


def _fmt_money(value: float | int) -> str:
    return f"${value:,.0f}"
