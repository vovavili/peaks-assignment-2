"""Chart generation for the PDF report."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import altair as alt
import polars as pl
from polars import col

SOURCE_BUCKET_ORDER = ("organic", "paid_attributed", "referral")
SOURCE_BUCKET_COLORS = ("#90be6d", "#577590", "#f9c74f")
LTV_CONCENTRATION_COLORS = ("#277da1", "#43aa8b", "#f9844a")


def build_figures(
    output_dir: Path,
    channel_performance: pl.DataFrame,
    attribution_mix: pl.DataFrame,
    organic_lag_correlation: pl.DataFrame,
    validation_predictions: pl.DataFrame,
    channel_quality: pl.DataFrame,
    channel_value: pl.DataFrame,
) -> dict[str, Path]:
    """Create all report figures and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = {
        "cost_metrics": output_dir / "cost_metrics.png",
        "attribution_mix": output_dir / "attribution_mix.png",
        "organic_lag": output_dir / "organic_lag.png",
        "decile_lift": output_dir / "decile_lift.png",
        "channel_value": output_dir / "channel_value.png",
    }
    _plot_cost_metrics(channel_performance, figure_paths["cost_metrics"])
    _plot_attribution_mix(attribution_mix, figure_paths["attribution_mix"])
    _plot_organic_lag(organic_lag_correlation, figure_paths["organic_lag"])
    _plot_ltv_concentration(validation_predictions, figure_paths["decile_lift"])
    _plot_channel_value(channel_quality, channel_value, figure_paths["channel_value"])
    return figure_paths


def _plot_cost_metrics(channel_performance: pl.DataFrame, path: Path) -> None:
    frame = channel_performance.drop_nulls("cac").sort("cac")
    channel_order = frame["channel"].to_list()
    bar_records = [row | {"metric": "CAC", "value": row["cac"]} for row in _records(frame)]
    point_records = [row | {"metric": "CPI", "value": row["cpi"]} for row in _records(frame)]
    metric_color = alt.Color(
        "metric:N",
        title=None,
        scale=alt.Scale(domain=["CAC", "CPI"], range=["#577590", "#f3722c"]),
        legend=alt.Legend(orient="bottom", direction="horizontal"),
    )
    base_encodings = {
        "y": alt.Y("channel:N", sort=channel_order, title=None),
        "x": alt.X("value:Q", title="USD", axis=alt.Axis(format="$,.0f")),
        "color": metric_color,
    }
    bars = (
        alt.Chart(_inline_records(bar_records))
        .mark_bar(opacity=0.86)
        .encode(
            **base_encodings,
            tooltip=[
                alt.Tooltip("channel:N", title="Channel"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("value:Q", title="Value", format="$,.2f"),
            ],
        )
    )
    points = (
        alt.Chart(_inline_records(point_records))
        .mark_point(filled=True, size=90)
        .encode(
            **base_encodings,
            tooltip=[
                alt.Tooltip("channel:N", title="Channel"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("value:Q", title="Value", format="$,.2f"),
            ],
        )
    )
    _save_altair_chart(
        (bars + points)
        .properties(width=660, height=300, title="Last-click CPI and CAC by channel")
        .resolve_scale(x="shared")
        .configure_legend(orient="bottom", title=None),
        path,
    )


def _plot_attribution_mix(attribution_mix: pl.DataFrame, path: Path) -> None:
    records = [
        row | {"bucket_order": SOURCE_BUCKET_ORDER.index(row["source_bucket"])}
        for row in attribution_mix.to_dicts()
        if row["source_bucket"] in SOURCE_BUCKET_ORDER
    ]
    chart = (
        alt.Chart(_inline_records(records))
        .mark_bar(size=82)
        .encode(
            x=alt.X("platform:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("share:Q", title="Share of installs", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "source_bucket:N",
                title=None,
                scale=alt.Scale(domain=list(SOURCE_BUCKET_ORDER), range=list(SOURCE_BUCKET_COLORS)),
                legend=alt.Legend(orient="bottom", direction="horizontal"),
            ),
            order=alt.Order("bucket_order:Q"),
            tooltip=[
                alt.Tooltip("platform:N", title="Platform"),
                alt.Tooltip("source_bucket:N", title="Attribution bucket"),
                alt.Tooltip("share:Q", title="Install share", format=".1%"),
            ],
        )
        .properties(width=560, height=300, title="Install attribution mix by platform")
    )
    _save_altair_chart(chart, path)


def _plot_organic_lag(organic_lag_correlation: pl.DataFrame, path: Path) -> None:
    frame = organic_lag_correlation.sort("lag_days")
    chart = (
        alt.Chart(_inline_data(frame))
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=60), color="#43aa8b", strokeWidth=2.5)
        .encode(
            x=alt.X(
                "lag_days:Q",
                title="Organic installs shifted after paid spend (days)",
                axis=alt.Axis(tickMinStep=1),
            ),
            y=alt.Y("correlation:Q", title="Pearson r: paid spend vs Organic installs", scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                alt.Tooltip("lag_days:Q", title="Lag days", format=".0f"),
                alt.Tooltip("correlation:Q", title="Pearson r", format=".2f"),
            ],
        )
        .properties(width=580, height=285, title="Paid spend and later Organic installs")
    )
    _save_altair_chart(chart, path)


def _plot_ltv_concentration(validation_predictions: pl.DataFrame, path: Path) -> None:
    frame = validation_predictions.sort("predicted_ltv_365_fee_usd", descending=True)
    total_ltv = float(frame["ltv_365_fee_usd"].sum() or 0.0)
    top_1_count = max(1, math.ceil(frame.height * 0.01))
    top_10_count = max(top_1_count, math.ceil(frame.height * 0.10))

    group_values = {
        "Top 1% predicted users": float(frame["ltv_365_fee_usd"].head(top_1_count).sum() or 0.0),
        "Next 9% predicted users": float(
            frame["ltv_365_fee_usd"].slice(top_1_count, top_10_count - top_1_count).sum() or 0.0
        ),
        "Bottom 90% predicted users": float(frame["ltv_365_fee_usd"].slice(top_10_count).sum() or 0.0),
    }
    group_shares = {label: value / total_ltv if total_ltv > 0 else 0.0 for label, value in group_values.items()}
    arc_order = {
        "Top 1% predicted users": 0,
        "Bottom 90% predicted users": 1,
        "Next 9% predicted users": 2,
    }
    legend_order = {
        "Top 1% predicted users": 0,
        "Next 9% predicted users": 1,
        "Bottom 90% predicted users": 2,
    }
    records = [
        {
            "segment": label,
            "share": group_shares[label],
            "share_label": f"{group_shares[label]:.1%} of validation LTV",
            "arc_order": arc_order[label],
            "legend_order": legend_order[label],
            "color": color,
        }
        for label, color in zip(group_shares, LTV_CONCENTRATION_COLORS, strict=True)
    ]
    domain = [record["segment"] for record in records]
    range_ = [record["color"] for record in records]
    color = alt.Color("segment:N", scale=alt.Scale(domain=domain, range=range_), legend=None)

    donut = (
        alt.Chart(_inline_records(records))
        .mark_arc(innerRadius=78, outerRadius=138, padAngle=0, strokeWidth=0)
        .encode(
            theta=alt.Theta("share:Q", stack=True),
            color=color,
            order=alt.Order("arc_order:Q"),
            tooltip=[
                alt.Tooltip("segment:N", title="Rank bucket"),
                alt.Tooltip("share:Q", title="Share of validation LTV", format=".1%"),
            ],
        )
        .properties(width=310, height=290)
    )
    center_pct = (
        alt.Chart(_inline_records([{"text": f"{group_shares['Top 1% predicted users']:.1%}"}]))
        .mark_text(fontSize=26, fontWeight="bold", color="black")
        .encode(x=alt.value(155), y=alt.value(135), text="text:N")
    )
    center_label = (
        alt.Chart(_inline_records([{"text": "Top 1%"}]))
        .mark_text(fontSize=20, fontWeight="bold", color="black")
        .encode(x=alt.value(155), y=alt.value(165), text="text:N")
    )

    legend_base = alt.Chart(_inline_records(records)).encode(y=alt.Y("legend_order:O", sort=[0, 1, 2], axis=None))
    legend_swatch = legend_base.mark_rect(width=48, height=8).encode(
        x=alt.value(0),
        color=color,
    )
    legend_title = legend_base.mark_text(
        align="left", baseline="bottom", dx=62, dy=-5, fontSize=13, fontWeight="bold"
    ).encode(
        x=alt.value(0),
        text="segment:N",
        color=color,
    )
    legend_value = legend_base.mark_text(align="left", baseline="top", dx=62, dy=8, fontSize=12, color="black").encode(
        x=alt.value(0),
        text="share_label:N",
    )
    legend = (legend_swatch + legend_title + legend_value).properties(width=360, height=230)

    _save_altair_chart(
        alt.hconcat(
            donut + center_pct + center_label,
            legend,
            spacing=16,
            title="Validation LTV concentration by predicted user rank",
        ),
        path,
        view_stroke=None,
    )


def _plot_channel_value(channel_quality: pl.DataFrame, channel_value: pl.DataFrame, path: Path) -> None:
    """Plot modelled customer value against observed acquisition cost.

    This is the visual version of the 2.3 answer: CPI/CAC alone says "what did the channel cost?",
    while predicted LTV says "what did the acquired customers look worth?" Channels in the upper
    left are attractive because they combine lower acquisition cost with higher expected value.
    """
    # `channel_quality` is the modelled forward-looking table. `channel_value` contributes the
    # complete-cohort CAC, so the chart compares expected value with a cost metric that is not
    # affected by censored 365-day outcomes.
    frame = channel_quality.join(channel_value.select("source", "cohort_cac"), on="source", how="left").drop_nulls(
        "cohort_cac"
    )
    cohort_cac_min = cast("float", frame["cohort_cac"].min())
    cohort_cac_max = cast("float", frame["cohort_cac"].max())
    ltv_per_registration_min = cast("float", frame["predicted_ltv_365_per_registration"].min())
    ltv_per_registration_max = cast("float", frame["predicted_ltv_365_per_registration"].max())
    x_min = max(0, cohort_cac_min - 20)
    x_max = cohort_cac_max + 55
    y_min = max(0, ltv_per_registration_min - 5)
    y_max = ltv_per_registration_max + 8
    records = _records(frame)
    points = (
        alt.Chart(_inline_records(records))
        .mark_circle(size=115, color="#f9844a", opacity=0.9)
        .encode(
            x=alt.X(
                "cohort_cac:Q",
                title="Cohort CAC (USD)",
                scale=alt.Scale(domain=[x_min, x_max]),
                axis=alt.Axis(format="$,.0f"),
            ),
            y=alt.Y(
                "predicted_ltv_365_per_registration:Q",
                title="Predicted 12-month fee revenue per registration (USD)",
                scale=alt.Scale(domain=[y_min, y_max]),
                axis=alt.Axis(format="$,.0f"),
            ),
            tooltip=[
                alt.Tooltip("source:N", title="Channel"),
                alt.Tooltip("cohort_cac:Q", title="Cohort CAC", format="$,.2f"),
                alt.Tooltip(
                    "predicted_ltv_365_per_registration:Q",
                    title="Predicted LTV / registration",
                    format="$,.2f",
                ),
            ],
        )
    )
    labels = (
        alt.Chart(_inline_records(records))
        .mark_text(align="left", baseline="middle", dx=8, fontSize=11)
        .encode(
            x=alt.X("cohort_cac:Q", scale=alt.Scale(domain=[x_min, x_max])),
            y=alt.Y("predicted_ltv_365_per_registration:Q", scale=alt.Scale(domain=[y_min, y_max])),
            text="source:N",
        )
    )
    _save_altair_chart(
        (points + labels).properties(width=610, height=330, title="Customer value versus acquisition cost"),
        path,
    )


def build_weekly_paid_organic(inputs_daily: pl.DataFrame) -> pl.DataFrame:
    """Prepare a weekly series for ad hoc review outside the PDF."""
    return (
        inputs_daily.with_columns(col("date").dt.truncate("1w").alias("week"))
        .group_by("week")
        .agg(col("spend").sum(), col("installs").sum())
        .sort("week")
    )


def _records(frame: pl.DataFrame) -> list[dict[str, Any]]:
    return frame.to_dicts()


def _inline_data(frame: pl.DataFrame) -> dict[str, list[dict[str, Any]]]:
    return _inline_records(_records(frame))


def _inline_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {"values": records}


def _save_altair_chart(chart: alt.TopLevelMixin, path: Path, *, view_stroke: str | None = "#d9d1c2") -> None:
    styled = (
        chart.configure_axis(
            labelFontSize=11,
            titleFontSize=12,
            gridColor="#e7e2d8",
            gridOpacity=0.7,
            domainColor="#8d8678",
            tickColor="#8d8678",
        )
        .configure_title(anchor="middle", fontSize=15, fontWeight="normal")
        .configure_legend(labelFontSize=11, titleFontSize=11)
    )
    if view_stroke is not None:
        styled = styled.configure_view(stroke=view_stroke)
    else:
        styled = styled.configure_view(stroke=None)
    styled.save(path, scale_factor=2.0)
