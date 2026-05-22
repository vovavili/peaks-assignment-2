# Peaks Take-Home Analysis

This repo contains a report-first solution for the Peaks senior data scientist assignment. The analysis keeps the heavy data work in Polars, uses native LightGBM for the early-LTV model, and writes a self-contained PDF report for non-technical review.

## Run

```powershell
uv sync --extra dev
uv run peaks-takehome
uv run pytest
ruff format .
ruff check . --fix-only --exit-zero --quiet
```

The pipeline reads the five source CSV files from `data/` and writes tables, figures, model outputs, and the PDF under
`outputs/`.

## Main Outputs

- `deliverables/Peaks_Take_Home_Report_Typst.pdf` (tracked submission copy)
- `outputs/Peaks_Take_Home_Report.pdf`
- `outputs/Peaks_Take_Home_Report_Typst.pdf`
- `outputs/tables/channel_daily_metrics.csv`
- `outputs/tables/user_panel.csv`
- `outputs/tables/model_predictions.csv`
- `outputs/figures/*.png`

## Optional marimo Review Notebook

The PDF is the standalone deliverable. The optional marimo notebook is a thin review surface over the generated CSV
outputs, useful for checking channel decisions, attribution gaps, SKAN/IPW estimates, and quick wins interactively.

```powershell
uv run --extra dev marimo edit notebooks/peaks_executive_review.py
uv run --extra dev marimo run notebooks/peaks_executive_review.py
```
