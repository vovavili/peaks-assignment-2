"""Tests for LTV and first-14-day feature engineering."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from functools import partial

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from peaks_takehome.ltv import FEE_RATE, compute_first14_transaction_features, compute_horizon_ltv

datetime_tz = partial(dt.datetime, tzinfo=dt.UTC)

HORIZON_DAYS = 30
INSTALL_TS = datetime_tz(2024, 1, 1)
DATA_END_TS = datetime_tz(2024, 3, 1)

INSTALL_SCHEMA = {
    "user_id": pl.Utf8,
    "install_ts": pl.Datetime(time_zone="UTC"),
}
TRANSACTION_SCHEMA = {
    "user_id": pl.Utf8,
    "transaction_timestamp": pl.Datetime(time_zone="UTC"),
    "transaction_type": pl.Utf8,
    "amount_usd": pl.Float64,
    "running_balance_usd": pl.Float64,
    "platform": pl.Utf8,
}
FIRST14_FEATURE_COLUMNS = (
    "feature_complete_14d",
    "txn_count_14d",
    "deposit_count_14d",
    "withdrawal_count_14d",
    "gross_deposit_14d",
    "gross_withdrawal_14d",
    "net_deposit_14d",
    "balance_14d",
    "max_balance_14d",
    "first_transaction_ts",
    "days_to_first_transaction",
    "funded_14d",
)

TransactionSpec = tuple[int, str, float, float]


def _installs(install_ts: dt.datetime = INSTALL_TS) -> pl.DataFrame:
    """Build the one-user install table used by the LTV helpers."""
    return pl.DataFrame({"user_id": ["u1"], "install_ts": [install_ts]}, schema=INSTALL_SCHEMA)


def _transactions(
    specs: Sequence[TransactionSpec],
    *,
    install_ts: dt.datetime = INSTALL_TS,
    user_id: str = "u1",
) -> pl.DataFrame:
    """Build a transaction table from day offsets, transaction types, amounts, and balances."""
    return pl.DataFrame(
        {
            "user_id": [user_id] * len(specs),
            "transaction_timestamp": [install_ts + dt.timedelta(days=offset) for offset, _, _, _ in specs],
            "transaction_type": [transaction_type for _, transaction_type, _, _ in specs],
            "amount_usd": [amount for _, _, amount, _ in specs],
            "running_balance_usd": [running_balance for _, _, _, running_balance in specs],
            "platform": ["ios"] * len(specs),
        },
        schema=TRANSACTION_SCHEMA,
    )


def _transaction_specs(*, min_day: int, max_day: int) -> st.SearchStrategy[tuple[TransactionSpec, ...]]:
    """Generate valid transaction rows inside an inclusive day-offset range."""
    if min_day > max_day:
        return st.just(())

    return st.lists(
        st.integers(min_value=min_day, max_value=max_day),
        unique=True,
        max_size=max_day - min_day + 1,
    ).flatmap(lambda offsets: _transaction_specs_for_offsets(sorted(offsets)))


def _transaction_specs_for_offsets(offsets: Sequence[int]) -> st.SearchStrategy[tuple[TransactionSpec, ...]]:
    """Generate transaction attributes after the timestamp offsets are known."""
    return st.tuples(
        st.lists(st.sampled_from(("deposit", "withdrawal")), min_size=len(offsets), max_size=len(offsets)),
        st.lists(st.integers(min_value=0, max_value=100_000), min_size=len(offsets), max_size=len(offsets)),
        st.lists(st.integers(min_value=0, max_value=100_000), min_size=len(offsets), max_size=len(offsets)),
    ).map(lambda values: _combine_transaction_specs(offsets, values))


def _combine_transaction_specs(
    offsets: Sequence[int],
    values: tuple[list[str], list[int], list[int]],
) -> tuple[TransactionSpec, ...]:
    """Combine generated timestamp offsets with generated transaction attributes."""
    transaction_types, amounts, balances = values
    return tuple(
        (offset, transaction_type, float(amount), float(balance))
        for offset, transaction_type, amount, balance in zip(offsets, transaction_types, amounts, balances, strict=True)
    )


def _incomplete_horizon_case() -> st.SearchStrategy[tuple[dt.datetime, tuple[TransactionSpec, ...]]]:
    """Generate users whose observation window is shorter than the target horizon."""
    return st.integers(min_value=0, max_value=HORIZON_DAYS - 1).flatmap(_incomplete_horizon_case_for_age)


def _incomplete_horizon_case_for_age(
    observed_age_days: int,
) -> st.SearchStrategy[tuple[dt.datetime, tuple[TransactionSpec, ...]]]:
    """Generate an incomplete-horizon case for one observed user age."""
    install_ts = DATA_END_TS - dt.timedelta(days=observed_age_days)
    specs = _transaction_specs(min_day=0, max_day=observed_age_days - 1) if observed_age_days else st.just(())
    return specs.map(lambda generated_specs: (install_ts, generated_specs))


def _reference_horizon_ltv(specs: Sequence[TransactionSpec], horizon_days: int) -> float:
    """Plain-Python reference for running-balance interval revenue."""
    return sum(
        running_balance * FEE_RATE * ((specs[index + 1][0] if index + 1 < len(specs) else horizon_days) - offset) / 365
        for index, (offset, _, _, running_balance) in enumerate(specs)
    )


def test_horizon_ltv_integrates_running_balance_intervals() -> None:
    """Revenue should accrue from each transaction balance until the next transaction or horizon."""
    installs = _installs()
    transactions = _transactions(
        (
            (0, "deposit", 100.0, 100.0),
            (10, "withdrawal", 50.0, 50.0),
        )
    )

    result = compute_horizon_ltv(
        installs,
        transactions,
        horizon_days=20,
        data_end_ts=datetime_tz(2024, 2, 1),
    )

    expected = (100.0 * FEE_RATE * 10 / 365) + (50.0 * FEE_RATE * 10 / 365)
    assert result.select("ltv_20_fee_usd").item() == pytest.approx(expected)
    assert result.select("eligible_20d").item()


def test_horizon_ltv_zero_for_complete_horizon_without_transactions() -> None:
    """A complete no-transaction horizon is a true zero-value label."""
    result = compute_horizon_ltv(
        _installs(),
        _transactions(((1, "deposit", 100.0, 100.0),), user_id="other"),
        horizon_days=20,
        data_end_ts=datetime_tz(2024, 2, 1),
    )

    assert result.select("ltv_20_fee_usd").item() == 0.0
    assert result.select("eligible_20d").item()


def test_incomplete_horizon_target_is_null() -> None:
    """Incomplete horizons should be excluded from supervised target training."""
    result = compute_horizon_ltv(
        _installs(datetime_tz(2024, 1, 20)),
        _transactions(((1, "deposit", 100.0, 100.0),), install_ts=datetime_tz(2024, 1, 20)),
        horizon_days=20,
        data_end_ts=datetime_tz(2024, 2, 1),
    )

    assert result.select("ltv_20_fee_usd").item() is None
    assert not result.select("eligible_20d").item()


def test_first14_features_ignore_late_transactions() -> None:
    """First-14-day features must not include post-window transactions."""
    result = compute_first14_transaction_features(
        _installs(),
        _transactions(
            (
                (1, "deposit", 100.0, 100.0),
                (19, "deposit", 500.0, 600.0),
            )
        ),
        data_end_ts=datetime_tz(2024, 2, 1),
    )

    assert result.select("gross_deposit_14d").item() == 100.0
    assert result.select("balance_14d").item() == 100.0
    assert result.select("txn_count_14d").item() == 1


@settings(max_examples=100, deadline=None)
@given(specs=_transaction_specs(min_day=0, max_day=HORIZON_DAYS - 1))
def test_horizon_ltv_matches_reference_interval_integration(specs: tuple[TransactionSpec, ...]) -> None:
    """Generated running-balance schedules should match the reference interval calculation."""
    result = compute_horizon_ltv(
        _installs(),
        _transactions(specs),
        horizon_days=HORIZON_DAYS,
        data_end_ts=DATA_END_TS,
    )

    assert result.select(f"ltv_{HORIZON_DAYS}_fee_usd").item() == pytest.approx(
        _reference_horizon_ltv(specs, HORIZON_DAYS)
    )
    assert result.select(f"eligible_{HORIZON_DAYS}d").item()


@settings(max_examples=50, deadline=None)
@given(case=_incomplete_horizon_case())
def test_incomplete_horizon_is_always_censored(case: tuple[dt.datetime, tuple[TransactionSpec, ...]]) -> None:
    """A user with less than a full target horizon should never receive a numeric target."""
    install_ts, specs = case

    result = compute_horizon_ltv(
        _installs(install_ts),
        _transactions(specs, install_ts=install_ts),
        horizon_days=HORIZON_DAYS,
        data_end_ts=DATA_END_TS,
    )

    assert result.select(f"ltv_{HORIZON_DAYS}_fee_usd").item() is None
    assert not result.select(f"eligible_{HORIZON_DAYS}d").item()


@settings(max_examples=100, deadline=None)
@given(
    early_specs=_transaction_specs(min_day=0, max_day=13),
    late_specs=_transaction_specs(min_day=14, max_day=45),
)
def test_first14_features_are_invariant_to_late_transactions(
    early_specs: tuple[TransactionSpec, ...],
    late_specs: tuple[TransactionSpec, ...],
) -> None:
    """Adding transactions after day 14 should not change first-14-day features."""
    base = compute_first14_transaction_features(
        _installs(),
        _transactions(early_specs),
        data_end_ts=DATA_END_TS,
    )
    with_late_transactions = compute_first14_transaction_features(
        _installs(),
        _transactions((*early_specs, *late_specs)),
        data_end_ts=DATA_END_TS,
    )

    assert base.select(FIRST14_FEATURE_COLUMNS).equals(with_late_transactions.select(FIRST14_FEATURE_COLUMNS))
