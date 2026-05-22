"""Early-LTV modelling for the Peaks take-home."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
from polars import col

# All model inputs must be known by day 14 after install. These columns come from the install
# event, registration/profile data available by then, and first-14-day transaction behavior.
CATEGORICAL_FEATURES = (
    "platform",
    "source",
    "campaign_id",
    "country",
    "gender",
    "source_bucket",
)

NUMERIC_FEATURES = (
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
    "install_year",
    "install_month",
)

MODEL_FEATURES = (*CATEGORICAL_FEATURES, *NUMERIC_FEATURES)
CATEGORICAL_FEATURE_INDICES = tuple(range(len(CATEGORICAL_FEATURES)))
REGISTRATION_FEATURE_CUTOFF_DAYS = 14

# LightGBM consumes integer-coded categorical features. Code 0 is reserved for missing or
# previously unseen categories, and the maps are fitted on the training cohort only.
CategoryMaps = dict[str, dict[str, int]]

# Registration/profile fields are only valid model inputs if registration happened by day 14.
REGISTRATION_AVAILABLE_14D = col("days_to_registration").is_not_null() & (
    col("days_to_registration") <= REGISTRATION_FEATURE_CUTOFF_DAYS
)

# This frame intentionally includes both features and labels. The final feature matrices are
# selected from MODEL_FEATURES only; the LTV columns are targets and evaluation outputs.
MODEL_FRAME_COLUMNS = (
    "user_id",
    "install_ts",
    "install_date",
    "platform",
    "source",
    "campaign_id",
    "country",
    "gender",
    "source_bucket",
    "registered",
    "tracking_enabled_num",
    "registered_num",
    "age",
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
    "install_year",
    "install_month",
    "eligible_365d",
    "feature_complete_14d",
    "ltv_365_fee_usd",
    "ltv_180_fee_usd",
)
PREDICTION_OUTPUT_COLUMNS = (
    "user_id",
    "install_ts",
    "install_date",
    "platform",
    "source",
    "campaign_id",
    "registered",
    "eligible_365d",
    "ltv_365_fee_usd",
    "ltv_180_fee_usd",
    "predicted_ltv_365_fee_usd",
    "positive_ltv_probability",
)
BASELINE_DETAILED_KEYS = ("platform", "source_bucket", "funded_14d_num", "deposit_band")
BASELINE_SOURCE_KEYS = ("source_bucket", "funded_14d_num", "deposit_band")
BASELINE_BEHAVIOR_KEYS = ("funded_14d_num", "deposit_band")


@dataclass(slots=True, frozen=True)
class ModelArtifacts:
    """Fitted models and their report-ready outputs."""

    classifier: lgb.Booster
    regressor: lgb.Booster
    category_maps: CategoryMaps
    predictions: pl.DataFrame
    validation_predictions: pl.DataFrame
    decile_lift: pl.DataFrame
    feature_importance: pl.DataFrame
    channel_quality: pl.DataFrame
    metrics: dict[str, float | str | int]


def train_ltv_models(user_panel: pl.DataFrame, random_state: int = 42) -> ModelArtifacts:
    """Train a native LightGBM two-stage early-LTV model.

    The model predicts 365-day fee revenue using only information available within 14 days
    after install. Users without a complete 365-day target or complete 14-day feature window
    are excluded from supervised training and validation.
    """
    modelling_frame = _prepare_model_frame(user_panel)

    # Sort by install time and validate on later cohorts. A random split would make the model
    # look better than it should for budget decisions because future acquisition patterns could
    # leak into training.
    eligible = modelling_frame.filter(col("eligible_365d") & col("feature_complete_14d")).sort(
        ("install_ts", "user_id")
    )
    split_index = int(eligible.height * 0.75)
    split_ts = eligible.item(split_index - 1, "install_ts")
    train = eligible.slice(0, split_index)
    valid = eligible.slice(split_index)

    category_maps = _fit_category_maps(train)
    x_train = _feature_matrix(train.select(*MODEL_FEATURES), category_maps)
    y_train = train.get_column("ltv_365_fee_usd").to_numpy()
    x_valid = _feature_matrix(valid.select(*MODEL_FEATURES), category_maps)
    y_valid = valid.get_column("ltv_365_fee_usd").to_numpy()
    positive_train = y_train > 0

    # LTV is sparse: many installs never generate fee revenue, while positive-value users have
    # a skewed continuous target. A two-stage LightGBM model handles that directly: first estimate
    # whether LTV is positive, then estimate the amount conditional on being positive.
    classifier = _train_lightgbm_classifier(x_train, positive_train, random_state=random_state)
    regressor = _train_lightgbm_regressor(
        _feature_matrix(train.filter(col("ltv_365_fee_usd") > 0).select(*MODEL_FEATURES), category_maps),
        np.log1p(y_train[positive_train]),
        random_state=random_state,
    )

    lightgbm_valid_pred = _predict_two_stage(classifier, regressor, x_valid)
    baseline_valid_pred = _baseline_predictions(train, valid)
    valid_positive = y_valid > 0

    all_features = modelling_frame.select(*MODEL_FEATURES)
    all_matrix = _feature_matrix(all_features, category_maps)

    # Predictions are written for every install, not just training-eligible installs. Newer users
    # still need expected value for bidding, even though their actual 365-day label is censored.
    predictions_pl = modelling_frame.with_columns(
        pl.Series("predicted_ltv_365_fee_usd", _predict_two_stage(classifier, regressor, all_matrix)),
        pl.Series("positive_ltv_probability", _predict_positive_probability(classifier, all_matrix)),
    ).select(*PREDICTION_OUTPUT_COLUMNS)

    # Validation output keeps actual and predicted LTV side by side for calibration, lift charts,
    # and comparison with the interpretable segment baseline.
    validation_predictions = valid.select(
        "user_id",
        "install_ts",
        "platform",
        "source",
        "campaign_id",
        "ltv_365_fee_usd",
    ).with_columns(
        pl.Series("predicted_ltv_365_fee_usd", lightgbm_valid_pred),
        pl.Series("baseline_predicted_ltv_365_fee_usd", baseline_valid_pred),
    )

    metrics = {
        "model_family": "LightGBM native two-stage GBDT",
        "split_timestamp": str(split_ts),
        "train_rows": train.height,
        "validation_rows": valid.height,
        "positive_ltv_rate_validation": float(valid_positive.mean()),
        "gbdt_validation_auc": _binary_auc(valid_positive, _predict_positive_probability(classifier, x_valid)),
        "gbdt_validation_mae": _mean_absolute_error(y_valid, lightgbm_valid_pred),
        "baseline_validation_mae": _mean_absolute_error(y_valid, baseline_valid_pred),
        "gbdt_validation_log_rmse": _log_rmse(y_valid, lightgbm_valid_pred),
        "baseline_validation_log_rmse": _log_rmse(y_valid, baseline_valid_pred),
    }

    return ModelArtifacts(
        classifier=classifier,
        regressor=regressor,
        category_maps=category_maps,
        predictions=predictions_pl,
        validation_predictions=validation_predictions,
        decile_lift=_decile_lift(validation_predictions),
        feature_importance=_permutation_importance(
            classifier,
            regressor,
            category_maps,
            valid.select(*MODEL_FEATURES),
            y_valid,
            random_state=random_state,
        ),
        channel_quality=_channel_quality(predictions_pl),
        metrics=metrics,
    )


def model_summary_dict(artifacts: ModelArtifacts) -> dict[str, Any]:
    """Return JSON-serializable model outputs for report generation."""
    return {
        "metrics": artifacts.metrics,
        "decile_lift": artifacts.decile_lift,
        "feature_importance": artifacts.feature_importance,
        "channel_quality": artifacts.channel_quality,
    }


def _prepare_model_frame(user_panel: pl.DataFrame) -> pl.DataFrame:
    """Select model columns, enforce the day-14 cutoff, and normalize missing categories."""
    return (
        user_panel.with_columns(
            REGISTRATION_AVAILABLE_14D.alias("_registration_available_14d"),
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
        .select(*MODEL_FRAME_COLUMNS)
        .with_columns([col(column).cast(pl.String).fill_null("missing") for column in CATEGORICAL_FEATURES])
    )


def _fit_category_maps(train: pl.DataFrame) -> CategoryMaps:
    """Fit training-cohort category maps for LightGBM categorical features."""
    return {feature: _category_map(train, feature) for feature in CATEGORICAL_FEATURES}


def _category_map(train: pl.DataFrame, feature: str) -> dict[str, int]:
    values = (
        train.select(col(feature).cast(pl.String).fill_null("missing").unique().sort()).get_column(feature).to_list()
    )
    ordered_values = ["missing", *(value for value in values if value != "missing")]
    return {value: code for code, value in enumerate(ordered_values)}


def _feature_matrix(features: pl.DataFrame, category_maps: CategoryMaps) -> np.ndarray:
    """Encode Polars features into the numeric matrix expected by native LightGBM."""
    encoded = features.select(
        *[
            col(feature)
            .cast(pl.String)
            .fill_null("missing")
            .replace_strict(category_maps[feature], default=0, return_dtype=pl.Int32)
            .alias(feature)
            for feature in CATEGORICAL_FEATURES
        ],
        *[col(feature).cast(pl.Float64) for feature in NUMERIC_FEATURES],
    )
    return encoded.to_numpy().astype(np.float32, copy=False)


def _lightgbm_dataset(features: np.ndarray, label: np.ndarray) -> lgb.Dataset:
    return lgb.Dataset(
        features,
        label=label,
        feature_name=list(MODEL_FEATURES),
        categorical_feature=list(CATEGORICAL_FEATURE_INDICES),
        free_raw_data=True,
    )


def _train_lightgbm_classifier(x_train: np.ndarray, positive_train: np.ndarray, *, random_state: int) -> lgb.Booster:
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 30,
        "lambda_l2": 0.05,
        "seed": random_state,
        "feature_pre_filter": False,
        "force_col_wise": True,
        "verbosity": -1,
    }
    return lgb.train(
        params,
        _lightgbm_dataset(x_train, positive_train.astype(np.float32)),
        num_boost_round=260,
        callbacks=[lgb.log_evaluation(period=0)],
    )


def _train_lightgbm_regressor(
    x_positive_train: np.ndarray, y_positive_log_ltv: np.ndarray, *, random_state: int
) -> lgb.Booster:
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 30,
        "lambda_l2": 0.05,
        "seed": random_state,
        "feature_pre_filter": False,
        "force_col_wise": True,
        "verbosity": -1,
    }
    return lgb.train(
        params,
        _lightgbm_dataset(x_positive_train, y_positive_log_ltv.astype(np.float32)),
        num_boost_round=300,
        callbacks=[lgb.log_evaluation(period=0)],
    )


def _predict_two_stage(classifier: lgb.Booster, regressor: lgb.Booster, features: np.ndarray) -> np.ndarray:
    """Combine positive-LTV probability with conditional positive-value prediction."""
    positive_probability = _predict_positive_probability(classifier, features)
    positive_log_value = np.asarray(regressor.predict(features), dtype=np.float64)
    positive_value = np.expm1(positive_log_value)
    return positive_probability * np.clip(positive_value, 0, None)


def _predict_positive_probability(classifier: lgb.Booster, features: np.ndarray) -> np.ndarray:
    return np.asarray(classifier.predict(features), dtype=np.float64)


def _baseline_predictions(train: pl.DataFrame, frame: pl.DataFrame) -> np.ndarray:
    """Predict LTV from transparent first-14-day segments fitted on the training cohort."""
    global_mean = train.select(col("ltv_365_fee_usd").mean()).item() or 0.0
    train_segments = _with_deposit_band(train)
    frame_segments = _with_deposit_band(frame)

    detailed = _segment_mean(train_segments, BASELINE_DETAILED_KEYS, "_baseline_detailed_ltv")
    source_fallback = _segment_mean(train_segments, BASELINE_SOURCE_KEYS, "_baseline_source_ltv")
    behavior_fallback = _segment_mean(train_segments, BASELINE_BEHAVIOR_KEYS, "_baseline_behavior_ltv")

    return (
        frame_segments.join(detailed, on=list(BASELINE_DETAILED_KEYS), how="left")
        .join(source_fallback, on=list(BASELINE_SOURCE_KEYS), how="left")
        .join(behavior_fallback, on=list(BASELINE_BEHAVIOR_KEYS), how="left")
        .with_columns(
            pl.coalesce(
                "_baseline_detailed_ltv",
                "_baseline_source_ltv",
                "_baseline_behavior_ltv",
                pl.lit(float(global_mean)),
            ).alias("baseline_predicted_ltv_365_fee_usd")
        )
        .get_column("baseline_predicted_ltv_365_fee_usd")
        .to_numpy()
    )


def _with_deposit_band(frame: pl.DataFrame) -> pl.DataFrame:
    net_deposit = col("net_deposit_14d").fill_null(0.0)
    return frame.with_columns(
        pl.when(net_deposit <= 0)
        .then(pl.lit("none_or_negative"))
        .when(net_deposit <= 100)
        .then(pl.lit("0_to_100"))
        .when(net_deposit <= 1_000)
        .then(pl.lit("100_to_1000"))
        .when(net_deposit <= 10_000)
        .then(pl.lit("1000_to_10000"))
        .otherwise(pl.lit("10000_plus"))
        .alias("deposit_band")
    )


def _segment_mean(train_segments: pl.DataFrame, keys: tuple[str, ...], value_name: str) -> pl.DataFrame:
    return train_segments.group_by(*keys).agg(col("ltv_365_fee_usd").mean().alias(value_name))


def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _log_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(np.clip(y_pred, 0, None))) ** 2)))


def _binary_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Compute ROC AUC with average ranks without adding a metrics dependency."""
    y_bool = np.asarray(y_true, dtype=bool)
    n_positive = int(y_bool.sum())
    n_negative = int(y_bool.size - n_positive)
    if n_positive == 0 or n_negative == 0:
        return float("nan")

    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(score.size, dtype=np.float64)
    start = 0
    while start < score.size:
        stop = start + 1
        while stop < score.size and sorted_score[stop] == sorted_score[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2
        start = stop

    positive_rank_sum = ranks[y_bool].sum()
    return float((positive_rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))


def _decile_lift(validation_predictions: pl.DataFrame) -> pl.DataFrame:
    """Show how much actual validation LTV is captured by the highest predicted users."""
    # Decile 1 is the top 10% by predicted value. If the model is useful for bidding, actual
    # revenue should be concentrated in the early deciles rather than spread evenly.
    ranked = validation_predictions.sort(
        ["predicted_ltv_365_fee_usd", "user_id"],
        descending=[True, False],
    ).with_row_index("rank_index")
    total_actual = ranked.select(col("ltv_365_fee_usd").sum()).item()
    return (
        ranked.with_columns(
            ((col("rank_index") * 10 / ranked.height).floor() + 1).clip(1, 10).cast(pl.Int64).alias("predicted_decile")
        )
        .group_by("predicted_decile")
        .agg(
            pl.len().alias("users"),
            col("predicted_ltv_365_fee_usd").mean().alias("avg_predicted_ltv"),
            col("ltv_365_fee_usd").mean().alias("avg_actual_ltv"),
            col("ltv_365_fee_usd").sum().alias("total_actual_ltv"),
        )
        .sort("predicted_decile")
        .with_columns(
            pl.when(pl.lit(total_actual) > 0)
            .then(col("total_actual_ltv") / pl.lit(total_actual))
            .otherwise(0.0)
            .alias("actual_ltv_share")
        )
        .with_columns(col("actual_ltv_share").cum_sum().alias("cumulative_actual_ltv_share"))
    )


def _permutation_importance(
    classifier: lgb.Booster,
    regressor: lgb.Booster,
    category_maps: CategoryMaps,
    x_valid: pl.DataFrame,
    y_valid: np.ndarray,
    *,
    random_state: int,
) -> pl.DataFrame:
    """Estimate feature importance as validation MAE increase after shuffling one feature."""
    rng = np.random.default_rng(random_state)
    sample_size = min(8_000, x_valid.height)
    sample_idx = np.sort(rng.choice(x_valid.height, size=sample_size, replace=False))
    sample_x = (
        x_valid.with_row_index("sample_index")
        .filter(col("sample_index").is_in(sample_idx.tolist()))
        .sort("sample_index")
        .drop("sample_index")
    )
    sample_y = y_valid[sample_idx]
    baseline_pred = _predict_two_stage(classifier, regressor, _feature_matrix(sample_x, category_maps))
    baseline_mae = _mean_absolute_error(sample_y, baseline_pred)

    def feature_mae_increase(feature: str) -> float:
        # Shuffling breaks the association between one feature and the target while leaving the
        # rest of the validation rows intact. Larger MAE increase means the model relied more on
        # that feature for out-of-sample predictions.
        shuffled = sample_x.with_columns(
            pl.Series(feature, rng.permutation(sample_x.get_column(feature).to_numpy()).tolist())
        )
        pred = _predict_two_stage(classifier, regressor, _feature_matrix(shuffled, category_maps))
        return _mean_absolute_error(sample_y, pred) - baseline_mae

    return pl.DataFrame(
        [
            {
                "feature": feature,
                "mae_increase": feature_mae_increase(feature),
            }
            for feature in MODEL_FEATURES
        ]
    ).sort("mae_increase", descending=True)


def _channel_quality(predictions: pl.DataFrame) -> pl.DataFrame:
    """Re-score acquisition channels using modelled expected LTV.

    The raw channel-performance table answers "what did this channel cost?" This table answers
    "what value do users from this channel look likely to create?" It uses predictions for every
    install, including newer cohorts whose true 365-day LTV is still censored, so it is the better
    table for forward-looking bidding and budget allocation.
    """
    return (
        predictions.group_by("source")
        .agg(
            # Keep volume next to value. A channel can look strong per user but still be small,
            # or create most total value simply because it has much more traffic.
            pl.len().alias("installs"),
            col("registered").sum().alias("registrations"),
            col("predicted_ltv_365_fee_usd").sum().alias("predicted_total_ltv_365"),
            # Per-install value is the fairest metric for bid ceilings because non-registering
            # users are real acquisition outcomes and should keep their predicted near-zero value.
            col("predicted_ltv_365_fee_usd").mean().alias("predicted_ltv_365_per_install"),
            # Per-registration value is easier for business interpretation: once a user registers,
            # how valuable does this channel's customer base look?
            col("predicted_ltv_365_fee_usd")
            .filter(col("registered"))
            .mean()
            .alias("predicted_ltv_365_per_registration"),
            # The observed actual is limited to users with a complete 365-day horizon. It is a
            # useful sanity check, but not the main planning metric for newer cohorts.
            col("ltv_365_fee_usd").filter(col("eligible_365d")).mean().alias("actual_ltv_365_per_eligible_install"),
        )
        # Sorting by registered-customer value makes the table answer the assignment question
        # directly: which channels deliver the highest-value users?
        .sort("predicted_ltv_365_per_registration", descending=True)
    )
