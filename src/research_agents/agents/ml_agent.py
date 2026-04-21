"""
MLAgent — fits a GLM model on the assembled dataset via outboxml
and exports model_results.json + diagnostic figures.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from research_agents.agents.ml.figure_plotter import FigurePlotter
from research_agents.agents.ml.model_runner import ModelRunner
from research_agents.agents.ml.result_exporter import ResultExporter
from research_agents.base_agent import BaseAgent
from research_agents.pydantic_models import RunContext


class MLAgent(BaseAgent):
    name = "ml"

    def __init__(
        self,
        ctx: RunContext,
        model_runner: ModelRunner,
        figure_plotter: FigurePlotter,
        result_exporter: ResultExporter,
    ) -> None:
        super().__init__(ctx)
        self._model_runner   = model_runner
        self._figure_plotter = figure_plotter
        self._result_exporter = result_exporter

    def run(self) -> None:
        output_dir = Path(self.ctx.output_dir)

        df = self._dataset()

        configs_dir  = output_dir / "ml_configs"
        figures_dir  = output_dir / "figures"

        model_result, glm_result = self._model_runner.fit(df, configs_dir)

        self._figure_plotter.plot(
            glm_result,
            figures_dir,
            fmt=self.ctx.config.get("report", {}).get("figures", {}).get("format", "pdf"),
        )

        results_path = self._result_exporter.model_results_path(model_result, output_dir)

        self.ctx.set_artifact("model_results", str(results_path))
        self.ctx.set_artifact("figures_dir", str(figures_dir))

    # ------------------------------------------------------------------

    def _dataset(self) -> pd.DataFrame:
        dataset_path = self.ctx.artifact_path("dataset")
        if not dataset_path or not dataset_path.exists():
            raise FileNotFoundError(
                "dataset artifact not found in RunContext — DataAgent must run first"
            )
        logger.info("[MLAgent] loading dataset from {}", dataset_path.name)
        suffix = dataset_path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(dataset_path)
        return pd.read_json(dataset_path, orient="records")
