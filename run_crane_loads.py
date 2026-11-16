"""
Stage 1 runner: calculates loads on the slewing bearing (ОПУ) and the slewing torque
for ship cranes and writes the initial table for the regression model.

The calculation is deterministic (dissertation, section 2.1.4); no LLM or API key is used.

Usage:
    python run_crane_loads.py [--config crane_loads_config.yaml] [--output ./output_crane]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from research_agents.agents.data.assembler import DatasetAssembler
from research_agents.agents.data.opu_load_calculator import OpuLoadCalculator
from research_agents.agents.data_agent import DataAgent
from research_agents.config import agent_config, ConfigError
from research_agents.pydantic_models import RunContext

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    colorize=True,
)

DEFAULT_CONFIG = Path(__file__).parent / "crane_loads_config.yaml"
GOAL = "Calculate loads on the slewing bearing of ship cranes and the slewing torque."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default="./output_crane")
    args = parser.parse_args()

    try:
        cfg = agent_config(args.config)
    except ConfigError as exc:
        logger.error("Config error: {}", exc)
        return 1

    output_dir = str(Path(args.output).resolve())
    ctx = RunContext.run_context_or_new(
        goal=GOAL,
        config=cfg.model_dump(),
        output_dir=output_dir,
    )

    data_agent = DataAgent(
        ctx=ctx,
        opu_load_calculator=OpuLoadCalculator(),
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
