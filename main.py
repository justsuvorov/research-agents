"""
Research-agents pipeline entry point.

Usage:
    python main.py --goal research_goal.txt [--config agent_config.yaml] [--output ./output]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from loguru import logger

from research_agents.agents.data_agent import DataAgent
from research_agents.agents.ml_agent import MLAgent
from research_agents.agents.report_agent import ReportAgent
from research_agents.agents.research_agent import ResearchAgent
from research_agents.agents.research.paper_analyzer import PaperAnalyzer
from research_agents.agents.research.query_builder import QueryBuilder
from research_agents.agents.research.searchers import (
    ArxivSearcher,
    ElibrarySearcher,
    MdpiSearcher,
    SemanticScholarSearcher,
)
from research_agents.agents.research.synthesizer import Synthesizer
from research_agents.ai_model import GeminiModel
from research_agents.config import agent_config, research_goal, ConfigError
from research_agents.pipeline import ResearchPipeline
from research_agents.prompt_loader import PromptLoader
from research_agents.pydantic_models import RunContext


load_dotenv()

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    colorize=True,
)


def _cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research-agents: goal + config → scientific article"
    )
    parser.add_argument("--goal",   required=True,      help="Path to research_goal.txt")
    parser.add_argument("--config", default=None,       help="Path to agent_config.yaml (optional)")
    parser.add_argument("--output", default="./output", help="Output directory (default: ./output)")
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    # --- Config & goal ---
    try:
        goal = research_goal(args.goal)
        cfg  = agent_config(args.config)
    except ConfigError as exc:
        logger.error("Configuration error: {}", exc)
        return 1

    # --- Shared infrastructure ---
    prompts   = PromptLoader(os.getenv("PROMPTS_DIR", "./prompts"))
    ai_model  = GeminiModel(client=genai.Client(api_key=os.environ["GEMINI_API_KEY"]))

    # --- RunContext ---
    ctx = RunContext.run_context_or_new(
        goal=goal,
        config=cfg.model_dump(),
        output_dir=str(Path(os.getenv("OUTPUT_DIR", args.output)).resolve()),
    )

    # --- Research agent dependencies ---
    query_builder = QueryBuilder(
        model=ai_model,
        system_prompt=prompts.prompt_text("research", "system.txt"),
        user_template=prompts.prompt_text("research", "query_builder.txt"),
    )

    paper_analyzer = PaperAnalyzer(
        model=ai_model,
        system_prompt=prompts.prompt_text("research", "system.txt"),
        user_template=prompts.prompt_text("research", "paper_analyzer.txt"),
    )

    synthesizer = Synthesizer(
        model=ai_model,
        system_prompt=prompts.prompt_text("research", "system.txt"),
        user_template=prompts.prompt_text("research", "synthesizer.txt"),
    )

    searchers = [
        SemanticScholarSearcher(api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY")),
        ArxivSearcher(),
        MdpiSearcher(mailto=os.getenv("MAILTO")),
        ElibrarySearcher(api_token=os.getenv("ELIBRARY_API_TOKEN")),
    ]

    # --- Pipeline ---
    pipeline = ResearchPipeline(
        ctx=ctx,
        research_agent=ResearchAgent(
            ctx=ctx,
            query_builder=query_builder,
            searchers=searchers,
            paper_analyzer=paper_analyzer,
            synthesizer=synthesizer,
        ),
        data_agent=DataAgent(ctx),
        ml_agent=MLAgent(ctx),
        report_agent=ReportAgent(ctx),
    )

    result = pipeline.result()
    return 1 if result.errors else 0


if __name__ == "__main__":
    sys.exit(main(_cli_args()))
