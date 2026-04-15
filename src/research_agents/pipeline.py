"""
ResearchPipeline — orchestrator that runs agents in sequence.

All agents are injected via constructor. The pipeline does not
instantiate or import agents itself — that is the caller's responsibility.
"""

from __future__ import annotations

from loguru import logger

from research_agents.base_agent import BaseAgent
from research_agents.pydantic_models import RunContext


class ResearchPipeline:

    def __init__(
        self,
        ctx: RunContext,
        research_agent: BaseAgent,
        data_agent: BaseAgent,
        ml_agent: BaseAgent,
        report_agent: BaseAgent,
    ) -> None:
        self.ctx = ctx
        self._agents: list[tuple[str, BaseAgent]] = [
            ("research", research_agent),
            ("data",     data_agent),
            ("ml",       ml_agent),
            ("report",   report_agent),
        ]

    def result(self) -> RunContext:
        """Run all agents sequentially. Returns final RunContext."""
        logger.info("Pipeline started  run_id={}", self.ctx.run_id)

        for name, agent in self._agents:
            agent.execute()
            if self.ctx.errors.get(name):
                logger.error("Pipeline stopped — failure in agent: {}", name)
                break

        logger.info("Pipeline finished  run_id={}", self.ctx.run_id)
        return self.ctx
