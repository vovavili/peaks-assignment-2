"""Build reusable Polars data products for the analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars import col

from peaks_takehome.ltv import compute_first14_transaction_features, compute_horizon_ltv, data_end_timestamp

if TYPE_CHECKING:
    import datetime as dt

    from peaks_takehome.data import RawInputs


PAID_CHANNELS = (
    "TikTok",
    "Meta",
    "Google App Campaigns",
    "Google Non-Brand Search",
    "Google Brand Search",
)

SPEND_METRICS = ("spend", "impressions", "clicks")
CHANNEL_DAILY_FILL_VALUES = (("spend", 0.0), *((v, 0) for v in ("impressions", "clicks", "installs", "registrations")))


def build_installs(events: pl.DataFrame) -> pl.DataFrame:
    """Return one row per installed user."""
    return (
        events.filter(col("event_name") == "install")
        .sort("user_id", "event_timestamp")
        .unique(subset=["user_id"], keep="first", maintain_order=True)
        .select(
            "user_id",
            col("event_timestamp").alias("install_ts"),
            col("event_timestamp").dt.date().alias("install_date"),
            col("platform").alias("platform"),
            col("source").alias("source"),
            col("campaign_id").alias("campaign_id"),
            col("country").alias("country"),
        )
    )


def build_user_panel(inputs: RawInputs) -> pl.DataFrame:
    """Build the one-row-per-install panel used by reporting and modelling."""
    installs = build_installs(inputs.app_events)
    registrations = (
        inputs.app_events.filter(col("event_name") == "registration")
        .sort("user_id", "event_timestamp")
        .unique(subset=["user_id"], keep="first", maintain_order=True)
        .select("user_id", col("event_timestamp").alias("registration_ts"))
    )
    profiles = inputs.user_profiles.rename(
        {
            "registration_date": "profile_registration_date",
            "platform": "profile_platform",
        }
    )
    data_end_ts = data_end_timestamp(inputs.user_transactions)

    panel = (
        installs.join(registrations, on="user_id", how="left")
        .join(profiles, on="user_id", how="left")
        .with_columns(
            col("campaign_id").fill_null("none"),
            col("age").cast(pl.Float64),
            col("gender").fill_null("unknown"),
            col("tracking_enabled").fill_null(False),
            col("profile_platform").fill_null(col("platform")),
            col("registration_ts").is_not_null().alias("registered"),
            ((col("registration_ts").cast(pl.Int64) - col("install_ts").cast(pl.Int64)) / 86_400_000_000).alias(
                "days_to_registration"
            ),
        )
    )

    # The model features use only the first 14 days after install, while the targets use later
    # balance history. Keeping those calls separate makes the feature/target cutoff explicit.
    first14 = compute_first14_transaction_features(installs, inputs.user_transactions, data_end_ts=data_end_ts)

    # LTV is fee revenue: running AUM balance * 0.5% annual fee, integrated over each horizon.
    # The 365-day target is the main modelling label; 180 days is kept as a shorter-horizon
    # sensitivity check for newer cohorts.
    ltv_180 = compute_horizon_ltv(
        installs,
        inputs.user_transactions,
        horizon_days=180,
        data_end_ts=data_end_ts,
    )
    ltv_365 = compute_horizon_ltv(
        installs,
        inputs.user_transactions,
        horizon_days=365,
        data_end_ts=data_end_ts,
    )

    return (
        panel.join(first14.drop("install_ts"), on="user_id", how="left")
        .join(ltv_180, on="user_id", how="left")
        .join(ltv_365, on="user_id", how="left")
        .with_columns(
            # Normalize raw source labels into three attribution buckets used in Part 1.2.
            # The point is not to decide whether traffic is truly paid or organic; it is to
            # compare how the MMP labels users across platform and privacy states.
            pl.when(col("source").is_in(PAID_CHANNELS))
            .then(pl.lit("paid_attributed"))
            .when(col("source") == "Referral")
            .then(pl.lit("referral"))
            .otherwise(pl.lit("organic"))
            .alias("source_bucket"),
            col("install_date").dt.year().alias("install_year"),
            col("install_date").dt.month().alias("install_month"),
        )
    )


def build_channel_daily_metrics(inputs: RawInputs) -> pl.DataFrame:
    """Build the Part 1 channel-performance table at a date x channel grain.

    Marketing spend is already daily, but campaign-level rows can still create more than one
    record per date and channel. App events are user-level, so installs and registrations are
    rolled up to the same daily channel grain using the event `source` as the channel label.

    The output keeps all date/channel pairs from either side of the join. That matters because
    Organic and Referral can have installs with no media spend, while paid channels can have
    spend days with no measured installs or registrations.
    """
    # Spend is the media ledger: dollars, impressions, and clicks by date/channel.
    # Summing here protects the downstream metrics if the raw file has several campaign rows
    # for the same channel on the same day.
    spend_daily = inputs.marketing_spend.group_by("date", "channel").agg(
        [col(metric).sum() for metric in SPEND_METRICS]
    )

    # App events are the product-side funnel counts. They are aggregated separately so installs
    # and registrations can be joined to spend without assuming every channel has paid media.
    install_daily = _event_daily_counts(inputs.app_events, "install", "installs")
    registration_daily = _event_daily_counts(inputs.app_events, "registration", "registrations")

    return (
        # Full joins preserve Organic/Referral rows and paid spend rows with no observed funnel event.
        spend_daily.join(install_daily, on=("date", "channel"), how="full", coalesce=True)
        .join(registration_daily, on=("date", "channel"), how="full", coalesce=True)
        # After the full joins, nulls in these metric columns mean "no activity observed" for
        # that date/channel. Filling them gives later ratios a clean numeric base.
        .with_columns([col(name).fill_null(value) for name, value in CHANNEL_DAILY_FILL_VALUES])
        # Derived metrics used in Part 1:
        # CPI = cost per install, CAC = cost per registration, CPM = cost per 1,000 impressions,
        # CTR = click-through rate, CPC = cost per click, and install -> registration is the
        # first product-funnel conversion rate.
        .with_columns(
            _safe_cost_divide(col("spend"), col("installs")).alias("cpi"),
            _safe_cost_divide(col("spend"), col("registrations")).alias("cac"),
            _safe_cost_divide(col("spend") * 1000, col("impressions")).alias("cpm"),
            _safe_divide(col("clicks"), col("impressions")).alias("ctr"),
            _safe_cost_divide(col("spend"), col("clicks")).alias("cpc"),
            _safe_divide(col("registrations"), col("installs")).alias("install_to_registration_rate"),
        )
        .sort("date", "channel")
    )


def channel_value_cutoff(user_panel: pl.DataFrame) -> dt.datetime:
    """Return the latest install timestamp with a full 365-day target."""
    return user_panel.filter(col("eligible_365d")).select(col("install_ts").max()).item()


def _event_daily_counts(events: pl.DataFrame, event_name: str, output_column: str) -> pl.DataFrame:
    """Roll one app event type onto the same daily channel grain as media spend."""
    return (
        events.filter(col("event_name") == event_name)
        .with_columns(col("event_timestamp").dt.date().alias("date"), col("source").alias("channel"))
        .group_by("date", "channel")
        .agg(col("user_id").n_unique().alias(output_column))
    )


def _safe_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return pl.when(denominator > 0).then(numerator / denominator).otherwise(None)


def _safe_cost_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return pl.when((denominator > 0) & (numerator > 0)).then(numerator / denominator).otherwise(None)
