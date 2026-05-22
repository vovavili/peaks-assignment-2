# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair==6.1.0",
#     "marimo>=0.13",
#     "polars==1.41.0",
#     "vegafusion==2.0.3",
#     "vl-convert-python==1.9.0.post1",
# ]
# ///

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="wide")


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl
    import vegafusion as vf

    vf.set_local_tz("Europe/Amsterdam")
    _ = alt.data_transformers.enable("vegafusion")
    return Path, alt, mo, pl


@app.cell(hide_code=True)
def _(center, mo):
    intro = mo.Html("""
<section style="max-width: 860px; text-align: center;">
  <h1>Peaks executive review</h1>
  <p>
    This marimo notebook is an optional audit surface for the PDF. It reads the generated
    CSV outputs and lets you inspect the channel, attribution, model, and recommendation
    evidence without rerunning the full analysis.
  </p>
  <p>Run <code>uv run peaks-takehome</code> first so the <code>outputs/tables</code> files are current.</p>
</section>
""")
    center(intro)


@app.cell
def _(Path):
    project_root = Path(__file__).resolve().parents[1]
    tables_dir = project_root / "outputs" / "tables"
    return project_root, tables_dir


@app.cell
def _(pl, tables_dir):
    channel_performance = pl.read_csv(tables_dir / "channel_performance.csv")
    model_channel_quality = pl.read_csv(tables_dir / "model_channel_quality.csv")
    budget_reallocation = pl.read_csv(tables_dir / "budget_reallocation_plan.csv")
    measurement_framework = pl.read_csv(tables_dir / "measurement_framework.csv")
    quick_wins = pl.read_csv(tables_dir / "quick_wins.csv")
    attribution_mix = pl.read_csv(tables_dir / "attribution_mix.csv")
    organic_lag = pl.read_csv(tables_dir / "organic_lag_correlation.csv")
    feature_importance = pl.read_csv(tables_dir / "model_feature_importance.csv")
    skan_value = pl.read_csv(tables_dir / "skan_value_method_comparison.csv")
    decile_lift = pl.read_csv(tables_dir / "model_decile_lift.csv")
    return (
        attribution_mix,
        budget_reallocation,
        channel_performance,
        decile_lift,
        feature_importance,
        measurement_framework,
        model_channel_quality,
        organic_lag,
        quick_wins,
        skan_value,
    )


@app.cell
def _(mo):
    def center(item):
        return mo.hstack([item], justify="center", align="center")

    def title(text: str):
        return mo.Html(f'<h2 style="text-align: center; margin: 1.2rem 0 0.4rem;">{text}</h2>')

    def section(*items):
        return center(mo.vstack(items, align="center", gap=0.8))

    return center, section, title


@app.cell
def _():
    def money(value: float | int | None) -> str:
        if value is None:
            return "n/a"
        return f"${value:,.2f}"

    def whole(value: float | int | None) -> str:
        if value is None:
            return "n/a"
        return f"{value:,.0f}"

    def pct(value: float | int | None) -> str:
        if value is None:
            return "n/a"
        return f"{value:.1%}"

    return money, pct, whole


@app.cell
def _(center, channel_performance, decile_lift, money, mo, pct, skan_value, whole):
    total_spend = channel_performance["spend"].sum()
    installs = channel_performance["installs"].sum()
    registrations = channel_performance["registrations"].sum()
    skan_installs = skan_value["skan_installs"].sum()
    mmp_ios_installs = skan_value["mmp_paid_ios_installs"].sum()
    top_decile_share = decile_lift.filter(decile_lift["predicted_decile"] == 1)["actual_ltv_share"].item()

    headline_checks = mo.Html(f"""
<section style="text-align: center;">
  <h2 style="margin: 1.2rem 0 0.75rem;">Headline checks</h2>
  <table style="margin: 0 auto; border-collapse: collapse; min-width: 360px;">
    <thead>
      <tr>
        <th style="padding: 0.45rem 0.7rem; border-bottom: 1px solid #9ca3af; text-align: right;">Metric</th>
        <th style="padding: 0.45rem 0.7rem; border-bottom: 1px solid #9ca3af; text-align: right;">Value</th>
      </tr>
    </thead>
    <tbody>
      <tr><td style="padding: 0.45rem 0.7rem; text-align: right;">Total spend</td><td style="padding: 0.45rem 0.7rem; text-align: right;">{money(total_spend)}</td></tr>
      <tr><td style="padding: 0.45rem 0.7rem; text-align: right;">Installs</td><td style="padding: 0.45rem 0.7rem; text-align: right;">{whole(installs)}</td></tr>
      <tr><td style="padding: 0.45rem 0.7rem; text-align: right;">Registrations</td><td style="padding: 0.45rem 0.7rem; text-align: right;">{whole(registrations)}</td></tr>
      <tr><td style="padding: 0.45rem 0.7rem; text-align: right;">SKAN paid iOS installs</td><td style="padding: 0.45rem 0.7rem; text-align: right;">{whole(skan_installs)}</td></tr>
      <tr><td style="padding: 0.45rem 0.7rem; text-align: right;">MMP-paid iOS installs</td><td style="padding: 0.45rem 0.7rem; text-align: right;">{whole(mmp_ios_installs)}</td></tr>
      <tr><td style="padding: 0.45rem 0.7rem; text-align: right;">SKAN / MMP paid iOS gap</td><td style="padding: 0.45rem 0.7rem; text-align: right;">{skan_installs / mmp_ios_installs:.1f}x</td></tr>
      <tr><td style="padding: 0.45rem 0.7rem; text-align: right;">Top predicted decile LTV capture</td><td style="padding: 0.45rem 0.7rem; text-align: right;">{pct(top_decile_share)}</td></tr>
    </tbody>
  </table>
</section>
""")
    center(headline_checks)


@app.cell
def _(center, channel_performance, mo):
    channel_options = sorted(channel_performance["channel"].to_list())
    channel_selector = mo.ui.multiselect(
        options=channel_options,
        value=channel_options,
        label="Channels",
    )
    center(channel_selector)
    return channel_options, channel_selector


@app.cell
def _(
    budget_reallocation,
    channel_options,
    channel_performance,
    channel_selector,
    model_channel_quality,
    pl,
):
    selected_channels = channel_selector.value or channel_options
    selected_performance = channel_performance.filter(pl.col("channel").is_in(selected_channels))
    selected_budget = budget_reallocation.filter(pl.col("channel").is_in(selected_channels))
    selected_quality = model_channel_quality.filter(pl.col("source").is_in(selected_channels))
    return selected_budget, selected_channels, selected_performance, selected_quality


@app.cell
def _(alt, center, pl, selected_performance):
    cost_long = (
        selected_performance.select("channel", "cpi", "cac")
        .unpivot(index="channel", on=("cpi", "cac"), variable_name="metric", value_name="usd")
        .drop_nulls("usd")
        .with_columns(pl.col("metric").replace({"cpi": "CPI", "cac": "CAC"}))
    )
    cost_chart = (
        alt.Chart(cost_long)
        .mark_bar()
        .encode(
            x=alt.X("usd:Q", title="USD", axis=alt.Axis(format="$,.0f")),
            y=alt.Y("channel:N", sort="-x", title=None),
            color=alt.Color("metric:N", title=None, scale=alt.Scale(range=["#f3722c", "#577590"])),
            tooltip=[
                alt.Tooltip("channel:N", title="Channel"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("usd:Q", title="USD", format="$,.2f"),
            ],
        )
        .properties(width=720, height=280, title="Last-click cost view")
    )
    center(cost_chart)


@app.cell
def _(alt, center, selected_budget):
    value_chart = (
        alt.Chart(selected_budget.drop_nulls("payback_years_at_y1_rate"))
        .mark_circle(size=120, color="#f9844a", opacity=0.9)
        .encode(
            x=alt.X(
                "payback_years_at_y1_rate:Q",
                title="Observed payback years at year-one run rate",
                scale=alt.Scale(domain=[0, 15.5]),
            ),
            y=alt.Y(
                "predicted_ltv_365_per_install:Q",
                title="Predicted 365-day LTV per install",
                axis=alt.Axis(format="$,.0f"),
            ),
            tooltip=[
                alt.Tooltip("channel:N", title="Channel"),
                alt.Tooltip("budget_action:N", title="Action"),
                alt.Tooltip("valuation_read:N", title="Value read"),
                alt.Tooltip("predicted_ltv_365_per_install:Q", title="Pred LTV / install", format="$,.2f"),
                alt.Tooltip("payback_years_at_y1_rate:Q", title="Payback years", format=".2f"),
            ],
        )
        .properties(width=760, height=320, title="Budget guardrails: value versus payback")
    )
    labels = (
        alt.Chart(selected_budget.drop_nulls("payback_years_at_y1_rate"))
        .mark_text(align="left", dx=8, fontSize=11)
        .encode(
            x="payback_years_at_y1_rate:Q",
            y="predicted_ltv_365_per_install:Q",
            text="channel:N",
        )
    )
    center(value_chart + labels)


@app.cell
def _(section, selected_budget, title):
    section(
        title("Budget actions"),
        selected_budget.select(
            "channel",
            "budget_action",
            "valuation_read",
            "rationale",
            "guardrail",
        ),
    )


@app.cell
def _(alt, attribution_mix, center):
    attribution_chart = (
        alt.Chart(attribution_mix)
        .mark_bar()
        .encode(
            x=alt.X("platform:N", title=None),
            y=alt.Y("share:Q", title="Install share", axis=alt.Axis(format="%")),
            color=alt.Color(
                "source_bucket:N",
                title=None,
                scale=alt.Scale(
                    domain=["organic", "paid_attributed", "referral"],
                    range=["#90be6d", "#577590", "#f9c74f"],
                ),
            ),
            tooltip=[
                alt.Tooltip("platform:N", title="Platform"),
                alt.Tooltip("source_bucket:N", title="Bucket"),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
            ],
        )
        .properties(width=420, height=280, title="Attribution mix by platform")
    )
    center(attribution_chart)


@app.cell
def _(alt, center, organic_lag):
    organic_lag_chart = (
        alt.Chart(organic_lag)
        .mark_line(point=True, color="#43aa8b")
        .encode(
            x=alt.X("lag_days:Q", title="Organic installs shifted after paid spend (days)"),
            y=alt.Y("correlation:Q", title="Pearson r", scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                alt.Tooltip("lag_days:Q", title="Lag days", format=".0f"),
                alt.Tooltip("correlation:Q", title="Pearson r", format=".2f"),
            ],
        )
        .properties(width=620, height=280, title="Paid spend and later Organic installs")
    )
    center(organic_lag_chart)


@app.cell
def _(alt, center, pl, skan_value):
    skan_long = (
        skan_value.select(
            "network",
            "ipw_ltv_365_per_skan_install",
            "cv_ltv_365_per_skan_install",
        )
        .unpivot(
            index="network",
            on=("ipw_ltv_365_per_skan_install", "cv_ltv_365_per_skan_install"),
            variable_name="method",
            value_name="ltv_per_install",
        )
        .with_columns(
            pl.col("method").replace(
                {
                    "ipw_ltv_365_per_skan_install": "IPW base",
                    "cv_ltv_365_per_skan_install": "CV sensitivity",
                }
            )
        )
    )
    skan_chart = (
        alt.Chart(skan_long)
        .mark_bar()
        .encode(
            x=alt.X("ltv_per_install:Q", title="LTV per SKAN install", axis=alt.Axis(format="$,.0f")),
            y=alt.Y("network:N", sort="-x", title=None),
            color=alt.Color("method:N", title=None, scale=alt.Scale(range=["#277da1", "#f9844a"])),
            tooltip=[
                alt.Tooltip("network:N", title="Network"),
                alt.Tooltip("method:N", title="Method"),
                alt.Tooltip("ltv_per_install:Q", title="LTV / SKAN install", format="$,.2f"),
            ],
        )
        .properties(width=720, height=300, title="iOS SKAN value: IPW base versus CV sensitivity")
    )
    center(skan_chart)


@app.cell
def _(alt, center, feature_importance):
    top_features = feature_importance.head(10)
    feature_chart = (
        alt.Chart(top_features)
        .mark_bar(color="#577590")
        .encode(
            x=alt.X("mae_increase:Q", title="Validation MAE increase", axis=alt.Axis(format="$,.0f")),
            y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("mae_increase:Q", title="MAE increase", format="$,.2f"),
            ],
        )
        .properties(width=660, height=300, title="Most predictive early-LTV features")
    )
    center(feature_chart)


@app.cell
def _(quick_wins, section, title):
    section(title("Quick wins"), quick_wins)


@app.cell
def _(measurement_framework, section, title):
    section(title("Measurement framework"), measurement_framework)


if __name__ == "__main__":
    app.run()
