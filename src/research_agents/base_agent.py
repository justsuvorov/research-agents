"""
BaseAgent — abstract base class for all pipeline agents.

Each agent:
  - receives RunContext
  - does its work in run()
  - updates RunContext artifacts and status
  - is idempotent: skips work if already completed
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from loguru import logger

from research_agents.pydantic_models import AgentStatus, RunContext


class BaseAgent(ABC):
    """Abstract base class for pipeline agents."""

    #: Must match the key used in AgentStatuses and Artifacts
    name: str

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx

    def execute(self) -> None:
        """Entry point. Handles status transitions and error capture."""
        if self.ctx.is_completed(self.name):
            logger.info("[%s] already completed — skipping", self.name)
            return

        logger.info("[%s] starting", self.name)
        self.ctx.set_status(self.name, AgentStatus.RUNNING)

        try:
            self.run()
            self.ctx.set_status(self.name, AgentStatus.COMPLETED)
            logger.info("[%s] completed", self.name)
        except Exception as exc:
            logger.exception("[%s] failed: %s", self.name, exc)
            self.ctx.set_error(self.name, str(exc))
            raise

    @abstractmethod
    def run(self) -> None:
        """Agent-specific logic. Must update ctx.artifacts via ctx.set_artifact()."""
        ...
