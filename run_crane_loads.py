"""
Standalone runner: generates 100 ship crane load calculation cases via DataAgent.

Usage:
    python run_crane_loads.py [--output ./output_crane]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from loguru import logger

from research_agents.agents.data.assembler import DatasetAssembler
from research_agents.agents.data.engineering_calculator import EngineeringCalculator
from research_agents.agents.data.gemini_adapter import GeminiAdapter
from research_agents.agents.data.paper_extractor import PaperExtractor
from research_agents.agents.data.standards_calculator import StandardsCalculator
from research_agents.agents.data_agent import DataAgent
from research_agents.config import agent_config, ConfigError
from research_agents.prompt_loader import PromptLoader
from research_agents.pydantic_models import RunContext

load_dotenv()

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    colorize=True,
)

CONFIG_PATH = Path(__file__).parent / "crane_loads_config.yaml"
GOAL = "Generate a comprehensive dataset of ship crane load cases for structural analysis."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./output_crane")
    args = parser.parse_args()

    try:
        cfg = agent_config(CONFIG_PATH)
    except ConfigError as exc:
        logger.error("Config error: {}", exc)
        return 1

    prompts = PromptLoader(os.getenv("PROMPTS_DIR", "./prompts"))

    model_name = os.getenv("AI_MODEL_NAME", "")
    if model_name.startswith("gemini"):
        api_key = os.environ["GEMINI_API_KEY"]
        llm = GeminiAdapter(api_key=api_key, model=model_name)
        logger.info("LLM client: Gemini ({})", model_name)
    else:
        llm = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        logger.info("LLM client: Anthropic (claude-sonnet-4-6)")

    output_dir = str(Path(args.output).resolve())
    ctx = RunContext.run_context_or_new(
        goal=GOAL,
        config=cfg.model_dump(),
        output_dir=output_dir,
    )

    data_agent = DataAgent(
        ctx=ctx,
        paper_extractor=PaperExtractor(
            client=llm,
            system_prompt=prompts.prompt_text("data", "system.txt"),
            user_template=prompts.prompt_text("data", "paper_extractor.txt"),
        ),
        standards_calculator=StandardsCalculator(
            client=llm,
            system_prompt=prompts.prompt_text("data", "system.txt"),
            user_template=prompts.prompt_text("data", "standards_calculator.txt"),
        ),
        engineering_calculator=EngineeringCalculator(
            client=llm,
            system_prompt=prompts.prompt_text("data", "engineering_calculator_system.txt"),
            user_template=prompts.prompt_text("data", "engineering_calculator_user.txt"),
            batch_size=10,
        ),
        assembler=DatasetAssembler(),
    )

    data_agent.execute()

    dataset_path = ctx.artifact_path("dataset")
    if dataset_path and dataset_path.exists():
        import pandas as pd
        df = pd.read_csv(dataset_path)
        logger.info("Dataset saved to: {}", dataset_path)
        logger.info("Rows: {}, Columns: {}", len(df), list(df.columns))
        print(df.head(10).to_string())
        return 0

    logger.error("Dataset artifact not found after agent run")
    return 1


if __name__ == "__main__":
    sys.exit(main())
