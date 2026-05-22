"""End-to-end pipeline for the Peaks take-home assignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from peaks_takehome.data import load_inputs, validate_inputs
from peaks_takehome.features import build_channel_daily_metrics, build_user_panel
from peaks_takehome.modeling import model_summary_dict, train_ltv_models
from peaks_takehome.report import build_pdf_report
from peaks_takehome.summaries import (
    build_budget_reallocation_plan,
    build_measurement_framework,
    build_quick_wins,
    build_report_tables,
    summary_numbers,
)
from peaks_takehome.typst_report import build_typst_report
from peaks_takehome.visuals import build_figures

if TYPE_CHECKING:
    import polars as pl


def run(root: Path | None = None) -> dict[str, Path]:
    """Run the full analysis and write tables, figures, model outputs, and PDF."""
    root = root or Path.cwd()
    outputs = root / "outputs"
    tables_dir = outputs / "tables"
    figures_dir = outputs / "figures"
    for directory in [outputs, tables_dir, figures_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs(root)
    diagnostics = validate_inputs(inputs)
    user_panel = build_user_panel(inputs)
    channel_daily_metrics = build_channel_daily_metrics(inputs)
    tables = build_report_tables(inputs, user_panel, channel_daily_metrics)
    model_artifacts = train_ltv_models(user_panel)
    model_summary = model_summary_dict(model_artifacts)
    tables["budget_reallocation_plan"] = build_budget_reallocation_plan(tables, model_summary)
    tables["measurement_framework"] = build_measurement_framework()
    tables["quick_wins"] = build_quick_wins()
    numbers = summary_numbers(inputs, user_panel, tables, model_summary)

    _write_frame(channel_daily_metrics, tables_dir / "channel_daily_metrics.csv")
    _write_frame(user_panel, tables_dir / "user_panel.csv")
    _write_frame(model_artifacts.predictions, tables_dir / "model_predictions.csv")
    _write_frame(model_artifacts.validation_predictions, tables_dir / "validation_predictions.csv")
    _write_frame(model_artifacts.decile_lift, tables_dir / "model_decile_lift.csv")
    _write_frame(model_artifacts.feature_importance, tables_dir / "model_feature_importance.csv")
    _write_frame(model_artifacts.channel_quality, tables_dir / "model_channel_quality.csv")
    for name, frame in tables.items():
        _write_frame(frame, tables_dir / f"{name}.csv")

    figures = build_figures(
        figures_dir,
        tables["channel_performance"],
        tables["attribution_mix"],
        tables["organic_lag_correlation"],
        model_artifacts.validation_predictions,
        model_artifacts.channel_quality,
        tables["eligible_channel_value"],
    )

    diagnostics_path = outputs / "diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(_json_ready(diagnostics | {"numbers": numbers, "model_metrics": model_artifacts.metrics}), indent=2)
    )

    report_path = outputs / "Peaks_Take_Home_Report.pdf"
    build_pdf_report(
        report_path,
        diagnostics=diagnostics,
        numbers=numbers,
        tables=tables,
        model_summary=model_summary,
        figures=figures,
    )

    typst_report_path = outputs / "Peaks_Take_Home_Report_Typst.pdf"
    compiled_typst_report = build_typst_report(
        typst_report_path,
        diagnostics=diagnostics,
        numbers=numbers,
        tables=tables,
        model_summary=model_summary,
        figures=figures,
    )

    output_paths = {
        "report": report_path,
        "diagnostics": diagnostics_path,
        "tables": tables_dir,
        "figures": figures_dir,
    }
    if compiled_typst_report is not None:
        output_paths["typst_report"] = compiled_typst_report
    output_paths["typst_source"] = typst_report_path.with_suffix(".typ")
    return output_paths


def main() -> None:
    """CLI entry point."""
    outputs = run()
    for label, path in outputs.items():
        print(f"{label}: {path}")


def _write_frame(frame: pl.DataFrame, path: Path) -> None:
    frame.write_csv(path)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    main()
