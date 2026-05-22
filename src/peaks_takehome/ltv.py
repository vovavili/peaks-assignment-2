"""LTV and early-behavior feature engineering."""

from __future__ import annotations

import datetime as dt

import polars as pl
from polars import col

FEE_RATE = 0.005
MICROSECONDS_PER_DAY = 86_400_000_000
TRANSACTION_AMOUNT_COLUMNS = (
    ("deposit", "deposit_amount"),
    ("withdrawal", "withdrawal_amount"),
)
TRANSACTION_AMOUNT_FEATURES = (
    ("deposit_amount", "gross_deposit_14d"),
    ("withdrawal_amount", "gross_withdrawal_14d"),
)


def data_end_timestamp(transactions: pl.DataFrame) -> dt.datetime:
    """Use the last transaction date plus one day as the balance observation end."""
    max_ts = transactions.select(col("transaction_timestamp").max()).item()
    return max_ts.replace(hour=0, minute=0, second=0, microsecond=0) + dt.timedelta(days=1)


def compute_horizon_ltv(
    installs: pl.DataFrame,
    transactions: pl.DataFrame,
    *,
    horizon_days: int,
    data_end_ts: dt.datetime,
    fee_rate: float = FEE_RATE,
) -> pl.DataFrame:
    """Compute fee revenue over a fixed post-install horizon.

    The function treats each transaction's `running_balance_usd` as the AUM that applies until the
    next transaction or the horizon end. Users with a complete horizon and no transactions receive
    zero revenue. Users without a complete horizon receive a null target so they cannot slip into
    supervised training by accident.
    """
    target_col = f"ltv_{horizon_days}_fee_usd"
    eligible_col = f"eligible_{horizon_days}d"
    horizon_end_col = f"horizon_end_{horizon_days}d"
    data_end_literal = pl.lit(data_end_ts, dtype=pl.Datetime)

    install_horizons = installs.select("user_id", "install_ts").with_columns(
        (col("install_ts") + pl.duration(days=horizon_days)).alias(horizon_end_col),
        ((col("install_ts") + pl.duration(days=horizon_days)) <= data_end_literal).alias(eligible_col),
    )

    # Revenue is earned on balances, not on deposit volume. Each transaction row gives the
    # running balance after that transaction; we carry that balance forward until the user's
    # next transaction or the end of the requested post-install horizon.
    interval_revenue = (
        transactions.join(install_horizons, on="user_id", how="inner")
        # Ignore any transaction before install and any transaction after the target window.
        .filter(
            (col("transaction_timestamp") >= col("install_ts")) & (col("transaction_timestamp") < col(horizon_end_col))
        )
        .sort("user_id", "transaction_timestamp")
        # The next transaction closes the current balance interval. If there is no later
        # transaction in the horizon, the balance remains active until the horizon end.
        .with_columns(col("transaction_timestamp").shift(-1).over("user_id").alias("next_transaction_ts"))
        .with_columns(
            pl.min_horizontal(
                col("next_transaction_ts").fill_null(col(horizon_end_col)),
                col(horizon_end_col),
            ).alias("interval_end_ts")
        )
        # Convert the interval length to days and apply the annual fee pro rata:
        # running_balance_usd * annual_fee_rate * active_days / 365.
        .with_columns(
            pl.max_horizontal(
                col("interval_end_ts").cast(pl.Int64) - col("transaction_timestamp").cast(pl.Int64),
                pl.lit(0),
            ).alias("active_microseconds")
        )
        .with_columns(
            (col("running_balance_usd") * fee_rate * (col("active_microseconds") / MICROSECONDS_PER_DAY) / 365.0).alias(
                "fee_revenue"
            )
        )
        .group_by("user_id")
        .agg(col("fee_revenue").sum().alias(target_col))
    )

    return (
        install_horizons.select("user_id", eligible_col)
        .join(interval_revenue, on="user_id", how="left")
        # A complete-horizon user with no transaction has true zero fee revenue. A user without
        # enough elapsed time is censored, so the target stays null and is excluded from training.
        .with_columns(col(target_col).fill_null(0.0))
        .with_columns(pl.when(col(eligible_col)).then(col(target_col)).otherwise(None).alias(target_col))
    )


def compute_first14_transaction_features(
    installs: pl.DataFrame,
    transactions: pl.DataFrame,
    *,
    data_end_ts: dt.datetime,
) -> pl.DataFrame:
    """Build transaction features available during the first 14 days after install."""
    feature_complete_col = "feature_complete_14d"
    install_windows = installs.select("user_id", "install_ts").with_columns(
        (col("install_ts") + pl.duration(days=14)).alias("feature_window_end_ts"),
        ((col("install_ts") + pl.duration(days=14)) <= pl.lit(data_end_ts, dtype=pl.Datetime)).alias(
            feature_complete_col
        ),
    )

    features = (
        transactions.join(install_windows, on="user_id", how="inner")
        .filter(
            (col("transaction_timestamp") >= col("install_ts"))
            & (col("transaction_timestamp") < col("feature_window_end_ts"))
        )
        .sort("user_id", "transaction_timestamp")
        .with_columns(
            [
                pl.when(col("transaction_type") == transaction_type)
                .then(col("amount_usd"))
                .otherwise(0.0)
                .alias(amount_column)
                for transaction_type, amount_column in TRANSACTION_AMOUNT_COLUMNS
            ]
        )
        .group_by("user_id")
        .agg(
            pl.len().alias("txn_count_14d"),
            (col("transaction_type") == "deposit").sum().alias("deposit_count_14d"),
            (col("transaction_type") == "withdrawal").sum().alias("withdrawal_count_14d"),
            *[col(amount_column).sum().alias(feature) for amount_column, feature in TRANSACTION_AMOUNT_FEATURES],
            (col("deposit_amount").sum() - col("withdrawal_amount").sum()).alias("net_deposit_14d"),
            col("running_balance_usd").last().alias("balance_14d"),
            col("running_balance_usd").max().alias("max_balance_14d"),
            col("transaction_timestamp").first().alias("first_transaction_ts"),
        )
    )

    zero_columns = (
        "txn_count_14d",
        "deposit_count_14d",
        "withdrawal_count_14d",
        "gross_deposit_14d",
        "gross_withdrawal_14d",
        "net_deposit_14d",
        "balance_14d",
        "max_balance_14d",
    )

    return (
        install_windows.join(features, on="user_id", how="left")
        .with_columns([col(column).fill_null(0) for column in zero_columns])
        .with_columns(
            (
                (col("first_transaction_ts").cast(pl.Int64) - col("install_ts").cast(pl.Int64)) / MICROSECONDS_PER_DAY
            ).alias("days_to_first_transaction"),
            (col("txn_count_14d") > 0).alias("funded_14d"),
        )
        .drop("feature_window_end_ts")
    )
