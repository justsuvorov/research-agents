"""
Stage 2 runner: fits the GLM regression model of the slewing torque on the table
produced by run_crane_loads.py and exports coefficients, metrics, predictions and figures.

Model settings (target, features, encodings, wrapper) are taken from the `ml` section
of the config file.  The LLM review (ReportAgent) is optional: --report.

Usage:
    python run_ml_report.py [--config crane_loads_config.yaml] [--output ./output_crane] [--report]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from research_agents.agents.ml.figure_plotter import FigurePlotter
from research_agents.agents.ml.model_runner import ModelRunner
from research_agents.agents.ml.result_exporter import ResultExporter
from research_agents.agents.ml_agent import MLAgent
from research_agents.config import agent_config, ConfigError
from research_agents.pydantic_models import RunContext

load_dotenv()

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    colorize=True,
)

DEFAULT_CONFIG = Path(__file__).parent / "crane_loads_config.yaml"


def _llm_client():
    import anthropic
    from research_agents.agents.data.gemini_adapter import GeminiAdapter

    model_name = os.getenv("AI_MODEL_NAME", "")
    if model_name.startswith("gemini"):
        logger.info("LLM client: Gemini ({})", model_name)
        return GeminiAdapter(api_key=os.environ["GEMINI_API_KEY"], model=model_name)
    logger.info("LLM client: Anthropic")
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default="./output_crane")
    parser.add_argument("--report", action="store_true", help="generate LLM review of the results")
    args = parser.parse_args()

    try:
        cfg = agent_config(args.config)
    except ConfigError as exc:
        logger.error("Config error: {}", exc)
        return 1

    output_dir = Path(args.output).resolve()
    if not (output_dir / "run_context.json").exists():
        logger.error("No run_context.json in {} — run run_crane_loads.py first", output_dir)
        return 1

    ctx = RunContext.run_context(str(output_dir))

    dataset_path = ctx.artifact_path("dataset")
    if not dataset_path or not dataset_path.exists():
        logger.error("dataset artifact missing")
        return 1

    ctx.config["ml"] = cfg.ml.model_dump()
    ctx.config["report"] = cfg.report.model_dump()

    # Force re-run of ml (+ report) agents on this invocation
    ctx.agent_status.ml = "pending"
    ctx.agent_status.report = "pending"
    ctx.errors.pop("ml", None)
    ctx.errors.pop("report", None)
    ctx.save()

    ml_agent = MLAgent(
        ctx=ctx,
        model_runner=ModelRunner(cfg=cfg.ml),
        figure_plotter=FigurePlotter(),
        result_exporter=ResultExporter(),
    )
    ml_agent.execute()

    if ctx.errors.get("ml"):
        logger.error("ML agent failed: {}", ctx.errors["ml"])
        return 1

    logger.info("model_results: {}", ctx.artifact_path("model_results"))
    logger.info("predictions:   {}", ctx.artifact_path("predictions"))
    logger.info("figures_dir:   {}", ctx.artifact_path("figures_dir"))

    if not args.report:
        return 0

    from research_agents.agents.report_agent import ReportAgent

    report_agent = ReportAgent(
        ctx=ctx,
        client=_llm_client(),
        system_prompt=(Path("prompts") / "report" / "system.txt").read_text(encoding="utf-8"),
        user_template=(Path("prompts") / "report" / "review.txt").read_text(encoding="utf-8"),
    )
    report_agent.execute()

    if ctx.errors.get("report"):
        logger.error("Report agent failed")
        return 1

    logger.info("report:        {}", ctx.artifact_path("article"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
