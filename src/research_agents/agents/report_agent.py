"""
ReportAgent — generates a Markdown technical review of the trained model.

Reads model_results.json + dataset summary + figure list, asks the injected LLM
client to produce a Markdown review, and writes it as report.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from research_agents.base_agent import BaseAgent
from research_agents.pydantic_models import RunContext


class ReportAgent(BaseAgent):
    name = "report"

    def __init__(
        self,
        ctx: RunContext,
        client: Any,
        system_prompt: str,
        user_template: str,
    ) -> None:
        super().__init__(ctx)
        self._client = client
        self._system_prompt = system_prompt
        self._user_template = user_template

    def run(self) -> None:
        output_dir = Path(self.ctx.output_dir)

        results_path = self.ctx.artifact_path("model_results")
        if not results_path or not results_path.exists():
            raise FileNotFoundError(
                "model_results artifact not found — MLAgent must run first"
            )

        dataset_path = self.ctx.artifact_path("dataset")
        figures_dir  = self.ctx.artifact_path("figures_dir")

        model_results_json = results_path.read_text(encoding="utf-8")
        dataset_summary    = self._dataset_summary(dataset_path)
        figure_list        = self._figure_list(figures_dir)

        content = self._user_template.format(
            goal=self.ctx.goal,
            dataset_summary=dataset_summary,
            model_results_json=model_results_json,
            figure_list=figure_list,
        )

        logger.info("[ReportAgent] requesting review from LLM")
        response = self._client.generate_content(
            contents=[
                f"{self._system_prompt}\n\n{content}"
            ],
            generation_config={
                "max_output_tokens": 8192,
                "temperature": 0.7,
            },
        )
        markdown = response.text.strip()

        report_path = output_dir / "report.md"
        report_path.write_text(markdown, encoding="utf-8")
        logger.info("[ReportAgent] wrote {} ({} chars)", report_path.name, len(markdown))

        self.ctx.set_artifact("article", str(report_path))

    # ------------------------------------------------------------------

    def _dataset_summary(self, dataset_path: Path | None) -> str:
        if not dataset_path or not dataset_path.exists():
            return "(dataset not available)"
        df = pd.read_csv(dataset_path) if dataset_path.suffix == ".csv" else pd.read_json(dataset_path)
        numeric_cols = df.select_dtypes("number").columns.tolist()
        summary = {
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "columns": df.columns.tolist(),
            "numeric_describe": json.loads(
                df[numeric_cols].describe().round(2).to_json()
            ) if numeric_cols else {},
        }
        return json.dumps(summary, ensure_ascii=False, indent=2)

    def _figure_list(self, figures_dir: Path | None) -> str:
        if not figures_dir or not figures_dir.exists():
            return "(no figures)"
        figures = sorted(p.name for p in figures_dir.iterdir() if p.is_file())
        return "\n".join(f"- {name}" for name in figures) or "(no figures)"
