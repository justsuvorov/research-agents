"""
Stage 2 runner: load existing dataset from output_crane, fit GLM, generate review.

Assumes run_crane_loads.py has already produced output_crane/dataset.csv +
output_crane/run_context.json.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from research_agents.agents.data.gemini_adapter import GeminiAdapter
from research_agents.agents.ml.figure_plotter import FigurePlotter
from research_agents.agents.ml.model_runner import ModelRunner
from research_agents.agents.ml.result_exporter import ResultExporter
from research_agents.agents.ml_agent import MLAgent
from research_agents.agents.report_agent import ReportAgent
from research_agents.config import MLConfig
from research_agents.pydantic_models import RunContext

load_dotenv()

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    colorize=True,
)

TARGET = "Sk_kN"
DROP_COLS = ["Px1_kN", "Py1_kN", "Pz1_kN", "theta_d_deg", "_steps", "source", "source_type"]


def _llm_client():
    model_name = os.getenv("AI_MODEL_NAME", "")
    if model_name.startswith("gemini"):
        logger.info("LLM client: Gemini ({})", model_name)
        return GeminiAdapter(api_key=os.environ["GEMINI_API_KEY"], model=model_name)
    logger.info("LLM client: Anthropic")
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _ml_dataset(raw_path: Path, ml_path: Path) -> list[str]:
    """One-hot encode load_case and drop unused columns. Return feature list."""
    df = pd.read_csv(raw_path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    if "load_case" in df.columns:
        dummies = pd.get_dummies(df["load_case"], prefix="lc", drop_first=True).astype(int)
        df = pd.concat([df.drop(columns=["load_case"]), dummies], axis=1)

    df.to_csv(ml_path, index=False)
    features = [c for c in df.columns if c != TARGET]
    logger.info("[runner] preprocessed dataset: {} rows × {} cols", len(df), df.shape[1])
    logger.info("[runner] features: {}", features)
    return features


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./output_crane")
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    if not (output_dir / "run_context.json").exists():
        logger.error("No run_context.json in {} — run run_crane_loads.py first", output_dir)
        return 1

    ctx = RunContext.run_context(str(output_dir))

    raw_dataset_path = ctx.artifact_path("dataset")
    if not raw_dataset_path or not raw_dataset_path.exists():
        logger.error("dataset artifact missing")
        return 1

    ml_dataset_path = output_dir / "dataset_ml.csv"
    features = _ml_dataset(raw_dataset_path, ml_dataset_path)

    ml_cfg = MLConfig(
        model="glm",
        target_variable=TARGET,
        features=features,
        hyperparameters={"objective": "gamma"},
    )
    ctx.config["ml"] = ml_cfg.model_dump()
    ctx.set_artifact("dataset", str(ml_dataset_path))

    # Force re-run of ml + report agents on this invocation
    ctx.agent_status.ml = "pending"
    ctx.agent_status.report = "pending"
    ctx.errors.pop("ml", None)
    ctx.errors.pop("report", None)
    ctx.save()

    llm = _llm_client()

    ml_agent = MLAgent(
        ctx=ctx,
        model_runner=ModelRunner(cfg=ml_cfg),
        figure_plotter=FigurePlotter(),
        result_exporter=ResultExporter(),
    )
    ml_agent.execute()

    if ctx.errors.get("ml"):
        logger.error("ML agent failed — skipping report")
        return 1

    report_agent = ReportAgent(
        ctx=ctx,
        client=llm,
        system_prompt=(Path("prompts") / "report" / "system.txt").read_text(encoding="utf-8"),
        user_template=(Path("prompts") / "report" / "review.txt").read_text(encoding="utf-8"),
    )
    report_agent.execute()

    if ctx.errors.get("report"):
        logger.error("Report agent failed")
        return 1

    logger.info("model_results: {}", ctx.artifact_path("model_results"))
    logger.info("figures_dir:   {}", ctx.artifact_path("figures_dir"))
    logger.info("report:        {}", ctx.artifact_path("article"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
