"""
DataAgent — collects raw training data from three sources and assembles a dataset.

Sources:
  1. Paper extraction  — LLM copies values from paper abstracts
  2. Standards calc    — LLM applies RMRS/GOST/ISO formulas to parameter grids
  3. User CSV          — optional file provided in config

All dependencies are injected via constructor.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from research_agents.agents.data.assembler import DatasetAssembler, EmptyDatasetError
from research_agents.agents.data.engineering_calculator import EngineeringCalculator
from research_agents.agents.data.paper_extractor import PaperExtractor
from research_agents.agents.data.standards_calculator import StandardsCalculator
from research_agents.agents.research.models import LiteratureReport
from research_agents.base_agent import BaseAgent
from research_agents.config import DataConfig
from research_agents.pydantic_models import RunContext


class DataAgent(BaseAgent):
    name = "data"

    def __init__(
        self,
        ctx: RunContext,
        paper_extractor: PaperExtractor,
        standards_calculator: StandardsCalculator,
        engineering_calculator: EngineeringCalculator,
        assembler: DatasetAssembler,
    ) -> None:
        super().__init__(ctx)
        self._paper_extractor        = paper_extractor
        self._standards_calculator   = standards_calculator
        self._engineering_calculator = engineering_calculator
        self._assembler              = assembler

    def run(self) -> None:
        cfg        = DataConfig(**self.ctx.config["data"])
        output_dir = Path(self.ctx.output_dir)

        # 1. Paper extraction
        paper_rows = self._paper_rows(cfg)

        # 2. Standards calculations (formula in config)
        calc_rows = self._calculation_rows(cfg)

        # 3. Engineering calculations (LLM knows the standard)
        engineering_rows = self._engineering_rows(cfg)

        # 4. User-provided data
        user_rows = self._user_rows(cfg)

        # 5. Assemble & export
        try:
            dataset_path, metadata_path = self._assembler.assembled_dataset(
                paper_rows=paper_rows,
                calculation_rows=calc_rows,
                engineering_rows=engineering_rows,
                user_rows=user_rows,
                cfg=cfg,
                output_dir=output_dir,
            )
        except EmptyDatasetError as exc:
            raise RuntimeError(str(exc)) from exc

        # 5. Update context
        self.ctx.set_artifact("dataset", str(dataset_path))
        self.ctx.set_artifact("dataset_metadata", str(metadata_path))

    # -------------------------------------------------------------------------

    def _paper_rows(self, cfg: DataConfig) -> list[dict]:
        if not cfg.extraction_rules:
            logger.info("[DataAgent] no extraction_rules configured — skipping paper extraction")
            return []

        papers_path = self.ctx.artifact_path("papers_data")
        if not papers_path or not papers_path.exists():
            logger.warning("[DataAgent] papers_data artifact not found — skipping paper extraction")
            return []

        report = LiteratureReport.model_validate_json(papers_path.read_text(encoding="utf-8"))
        relevant = report.relevant()
        logger.info("[DataAgent] extracting from {} relevant papers", len(relevant))

        rows: list[dict] = []
        for analysis in relevant:
            rows.extend(
                self._paper_extractor.extracted_rows(analysis, cfg.extraction_rules)
            )
        return rows

    def _calculation_rows(self, cfg: DataConfig) -> list[dict]:
        if not cfg.calculations:
            logger.info("[DataAgent] no calculations configured — skipping standards calc")
            return []

        rows: list[dict] = []
        for rule in cfg.calculations:
            rows.extend(self._standards_calculator.calculated_rows(rule))
        return rows

    def _engineering_rows(self, cfg: DataConfig) -> list[dict]:
        if not cfg.engineering_calculations:
            logger.info("[DataAgent] no engineering_calculations configured — skipping")
            return []

        rows: list[dict] = []
        for rule in cfg.engineering_calculations:
            rows.extend(self._engineering_calculator.calculated_rows(rule))
        return rows

    def _user_rows(self, cfg: DataConfig) -> list[dict]:
        if not cfg.user_data:
            return []

        path = Path(cfg.user_data)
        if not path.exists():
            logger.warning("[DataAgent] user_data file not found: {} — skipping", path)
            return []

        try:
            df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_json(path)
            df["source"] = "user"
            df["source_type"] = "user"
            logger.info("[DataAgent] loaded {} user rows from {}", len(df), path.name)
            return df.to_dict(orient="records")
        except Exception as exc:
            logger.warning("[DataAgent] failed to load user_data: {}", exc)
            return []
