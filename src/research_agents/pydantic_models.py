"""
Pydantic models for pipeline state.

RunContext — shared state passed between all agents.
Persisted as run_context.json in the output directory.

Naming convention: method/function name = name of the returned object.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Artifacts(BaseModel):
    literature_review: Optional[str] = None   # path to literature_review.md
    references: Optional[str] = None          # path to references.bib
    papers_data: Optional[str] = None         # path to papers.json (analyzed papers)
    dataset: Optional[str] = None             # path to dataset.csv
    dataset_metadata: Optional[str] = None    # path to dataset_metadata.json
    model_results: Optional[str] = None       # path to model_results.json
    figures_dir: Optional[str] = None         # path to figures/
    article: Optional[str] = None             # path to article.tex
    article_pdf: Optional[str] = None         # path to article.pdf


class AgentStatuses(BaseModel):
    research: AgentStatus = AgentStatus.PENDING
    data: AgentStatus = AgentStatus.PENDING
    ml: AgentStatus = AgentStatus.PENDING
    report: AgentStatus = AgentStatus.PENDING


class AgentCheckpoint(BaseModel):
    agent_name: str
    status: str  # pending, in_progress, completed, failed
    timestamp: str  # ISO format
    checkpoint_dir: str  # path to checkpoints/{agent}/
    step_completed: int = 0  # which step completed (e.g., 1-6 for ResearchAgent)
    total_items: int = 0  # total items to process
    tokens_used: int = 0
    intermediate_artifacts: dict[str, str] = Field(default_factory=dict)  # step_name -> file path
    errors: list[str] = Field(default_factory=list)
    recovery_possible: bool = True


class RunContext(BaseModel):
    goal: str
    config: dict
    output_dir: str
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    artifacts: Artifacts = Field(default_factory=Artifacts)
    agent_status: AgentStatuses = Field(default_factory=AgentStatuses)
    errors: dict[str, str] = Field(default_factory=dict)
    checkpoints: dict[str, AgentCheckpoint] = Field(default_factory=dict)

    model_config = {"use_enum_values": True}

    # --- Persistence ---

    def save(self) -> None:
        path = Path(self.output_dir) / "run_context.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            self.model_dump_json(indent=2),
            encoding="utf-8",
        )

    @classmethod
    def run_context(cls, output_dir: str) -> "RunContext":
        """Return RunContext loaded from output_dir."""
        path = Path(output_dir) / "run_context.json"
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    @classmethod
    def run_context_or_new(cls, goal: str, config: dict, output_dir: str) -> "RunContext":
        """Return existing RunContext from output_dir, or create and save a new one."""
        path = Path(output_dir) / "run_context.json"
        if path.exists():
            return cls.run_context(output_dir)
        ctx = cls(goal=goal, config=config, output_dir=output_dir)
        ctx.save()
        return ctx

    # --- Mutation helpers (save after each change) ---

    def set_status(self, agent: str, status: AgentStatus) -> None:
        setattr(self.agent_status, agent, status)
        self.save()

    def set_artifact(self, key: str, path: str) -> None:
        setattr(self.artifacts, key, str(Path(path).resolve()))
        self.save()

    def set_error(self, agent: str, message: str) -> None:
        self.errors[agent] = message
        self.set_status(agent, AgentStatus.FAILED)

    # --- Query helpers ---

    def is_completed(self, agent: str) -> bool:
        return getattr(self.agent_status, agent) == AgentStatus.COMPLETED

    def artifact_path(self, key: str) -> Optional[Path]:
        """Return artifact path as Path, or None if not set."""
        value = getattr(self.artifacts, key, None)
        return Path(value) if value else None

    # --- Checkpoint management ---

    def checkpoint(
        self,
        agent: str,
        step: int,
        total_items: int = 0,
        tokens_used: int = 0,
        intermediate_artifacts: Optional[dict[str, str]] = None,
    ) -> None:
        """Save checkpoint for agent after completing a step."""
        checkpoint_dir = Path(self.output_dir) / "checkpoints" / agent
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        cp = AgentCheckpoint(
            agent_name=agent,
            status="in_progress",
            timestamp=datetime.now(timezone.utc).isoformat(),
            checkpoint_dir=str(checkpoint_dir.resolve()),
            step_completed=step,
            total_items=total_items,
            tokens_used=tokens_used,
            intermediate_artifacts=intermediate_artifacts or {},
            recovery_possible=True,
        )

        self.checkpoints[agent] = cp
        self.save()

    def checkpoint_completed(self, agent: str) -> None:
        """Mark checkpoint as completed."""
        if agent in self.checkpoints:
            self.checkpoints[agent].status = "completed"
        self.save()

    def checkpoint_failed(self, agent: str, error: str) -> None:
        """Mark checkpoint as failed."""
        if agent in self.checkpoints:
            self.checkpoints[agent].status = "failed"
            self.checkpoints[agent].errors.append(error)
            self.checkpoints[agent].recovery_possible = False
        self.save()

    def last_completed_agent(self) -> Optional[str]:
        """Return name of last completed agent, or None if no agents completed."""
        completed = [
            name
            for name in ["research", "data", "ml", "report"]
            if self.is_completed(name)
        ]
        return completed[-1] if completed else None

    def can_resume(self) -> bool:
        """Return True if pipeline can resume from last checkpoint."""
        if not self.checkpoints:
            return False
        last_agent = self.last_completed_agent()
        if not last_agent:
            return False
        cp = self.checkpoints.get(last_agent)
        return cp is not None and cp.recovery_possible
