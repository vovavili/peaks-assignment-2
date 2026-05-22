"""Data loading and validation for the Peaks take-home datasets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import polars as pl
from polars import col

if TYPE_CHECKING:
    from polars._typing import SchemaDict


DATA_DIR = "data"
CSV_FILES = {
    "marketing_spend": "marketing_spend.csv",
    "app_events": "app_events.csv",
    "user_profiles": "user_profiles.csv",
    "user_transactions": "user_transactions.csv",
    "skan_attribution_ios": "skan_attribution_ios.csv",
}

MARKETING_SPEND_SCHEMA: SchemaDict = {
    "date": pl.Date,
    "channel": pl.String,
    "campaign_id": pl.String,
    "campaign_name": pl.String,
    "spend": pl.Float64,
    "impressions": pl.Int64,
    "clicks": pl.Int64,
}

APP_EVENTS_SCHEMA: SchemaDict = {
    "user_id": pl.String,
    "event_timestamp": pl.Datetime,
    "event_name": pl.String,
    "platform": pl.String,
    "source": pl.String,
    "campaign_id": pl.String,
    "country": pl.String,
}

USER_PROFILES_SCHEMA: SchemaDict = {
    "user_id": pl.String,
    "age": pl.Int64,
    "gender": pl.String,
    "registration_date": pl.Date,
    "platform": pl.String,
    "tracking_enabled": pl.Boolean,
}

USER_TRANSACTIONS_SCHEMA: SchemaDict = {
    "user_id": pl.String,
    "transaction_timestamp": pl.Datetime,
    "transaction_type": pl.String,
    "amount_usd": pl.Float64,
    "running_balance_usd": pl.Float64,
    "platform": pl.String,
}

SKAN_ATTRIBUTION_IOS_SCHEMA: SchemaDict = {
    "postback_date": pl.Date,
    "network": pl.String,
    "campaign_id": pl.String,
    "skan_conversion_value": pl.Float64,
    "install_count": pl.Int64,
}

CsvScanner = Callable[[Path], pl.LazyFrame]


@dataclass(slots=True, frozen=True)
class RawInputs:
    """The five source tables after strict parsing."""

    marketing_spend: pl.DataFrame
    app_events: pl.DataFrame
    user_profiles: pl.DataFrame
    user_transactions: pl.DataFrame
    skan_attribution_ios: pl.DataFrame


def _csv_path(root: Path, key: str) -> Path:
    path = root / DATA_DIR / CSV_FILES[key]
    if not path.exists():
        msg = f"Missing required input file: {path}"
        raise FileNotFoundError(msg)
    return path


def _scan_csv(root: Path, key: str, schema: SchemaDict) -> pl.LazyFrame:
    return pl.scan_csv(_csv_path(root, key), schema=schema, null_values=[""])


def _csv_scanner_factory(key: str, schema: SchemaDict, doc: str) -> CsvScanner:
    scanner: CsvScanner = partial(_scan_csv, key=key, schema=schema)
    scanner.__doc__ = doc
    return scanner


scan_marketing_spend = _csv_scanner_factory(
    "marketing_spend",
    MARKETING_SPEND_SCHEMA,
    "Scan daily media spend using the assignment's declared schema.",
)
scan_app_events = _csv_scanner_factory(
    "app_events",
    APP_EVENTS_SCHEMA,
    "Scan user-level events using the assignment's declared schema.",
)
scan_user_profiles = _csv_scanner_factory(
    "user_profiles",
    USER_PROFILES_SCHEMA,
    "Scan registered-user profiles using the assignment's declared schema.",
)
scan_user_transactions = _csv_scanner_factory(
    "user_transactions",
    USER_TRANSACTIONS_SCHEMA,
    "Scan deposit and withdrawal history using the assignment's declared schema.",
)
scan_skan_attribution_ios = _csv_scanner_factory(
    "skan_attribution_ios",
    SKAN_ATTRIBUTION_IOS_SCHEMA,
    "Scan aggregated iOS SKAN postbacks using the assignment's declared schema.",
)


def load_inputs(root: Path) -> RawInputs:
    """Collect the source lazy scans into memory once."""
    return RawInputs(
        marketing_spend=_collect(scan_marketing_spend(root)),
        app_events=_collect(scan_app_events(root)),
        user_profiles=_collect(scan_user_profiles(root)),
        user_transactions=_collect(scan_user_transactions(root)),
        skan_attribution_ios=_collect(scan_skan_attribution_ios(root)),
    )


def _collect(frame: pl.LazyFrame) -> pl.DataFrame:
    return cast("pl.DataFrame", frame.collect())


def validate_inputs(inputs: RawInputs) -> dict[str, Any]:
    """Return compact diagnostics and raise on hard data contract failures."""
    diagnostics: dict[str, Any] = {
        "row_counts": {
            "marketing_spend": inputs.marketing_spend.height,
            "app_events": inputs.app_events.height,
            "user_profiles": inputs.user_profiles.height,
            "user_transactions": inputs.user_transactions.height,
            "skan_attribution_ios": inputs.skan_attribution_ios.height,
        },
        "date_ranges": {
            "marketing_spend": _date_range(inputs.marketing_spend, "date"),
            "app_events": _date_range(inputs.app_events, "event_timestamp"),
            "user_profiles": _date_range(inputs.user_profiles, "registration_date"),
            "user_transactions": _date_range(inputs.user_transactions, "transaction_timestamp"),
            "skan_attribution_ios": _date_range(inputs.skan_attribution_ios, "postback_date"),
        },
        "duplicate_event_rows": inputs.app_events.height - inputs.app_events.unique().height,
        "campaign_ids": inputs.marketing_spend.select(col("campaign_id").n_unique()).item(),
    }

    _require_no_negative(inputs.marketing_spend, "spend", "marketing_spend.spend")
    _require_no_negative(inputs.marketing_spend, "impressions", "marketing_spend.impressions")
    _require_no_negative(inputs.marketing_spend, "clicks", "marketing_spend.clicks")
    _require_no_negative(inputs.user_transactions, "running_balance_usd", "user_transactions.running_balance_usd")

    installs = inputs.app_events.filter(col("event_name") == "install").select("user_id")
    profiles = inputs.user_profiles.select("user_id")
    transactions = inputs.user_transactions.select("user_id").unique()
    diagnostics["join_coverage"] = {
        "profiles_in_events": profiles.join(inputs.app_events.select("user_id").unique(), on="user_id").height
        / profiles.height,
        "transactions_in_profiles": transactions.join(profiles, on="user_id").height / transactions.height,
        "installs": installs.height,
        "unique_installed_users": installs.select(col("user_id").n_unique()).item(),
    }

    if diagnostics["join_coverage"]["profiles_in_events"] < 0.999:
        msg = "Some registered users cannot be joined to app events."
        raise ValueError(msg)
    if diagnostics["join_coverage"]["transactions_in_profiles"] < 0.999:
        msg = "Some transacting users cannot be joined to user profiles."
        raise ValueError(msg)
    if diagnostics["join_coverage"]["installs"] != diagnostics["join_coverage"]["unique_installed_users"]:
        msg = "Expected exactly one install event per installed user."
        raise ValueError(msg)

    return diagnostics


def _date_range(frame: pl.DataFrame, column: str) -> tuple[str, str]:
    result = frame.select(col(column).min().alias("min_date"), col(column).max().alias("max_date")).row(0)
    return (str(result[0]), str(result[1]))


def _require_no_negative(frame: pl.DataFrame, column: str, label: str) -> None:
    if frame.filter(col(column) < 0).height:
        msg = f"Negative values found in {label}."
        raise ValueError(msg)
