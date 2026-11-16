"""
ResultExporter — serialises ModelResult to model_results.json and writes
the table of model predictions to Excel.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from research_agents.agents.ml.models import ModelResult


class ResultExporter:

    def model_results_path(self, result: ModelResult, output_dir: Path) -> Path:
        """Write model_results.json and return its path."""
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "model_results.json"
        path.write_text(
            json.dumps(result.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("[ResultExporter] model_results.json written ({} coefficients)", len(result.coefficients))
        return path

    def predictions_path(self, predictions: pd.DataFrame, output_dir: Path) -> Path:
        """Write model_predictions.xlsx (input table + model values + errors) and return its path."""
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "model_predictions.xlsx"
        predictions.to_excel(path, index=False)
        logger.info("[ResultExporter] model_predictions.xlsx written ({} rows)", len(predictions))
        return path
