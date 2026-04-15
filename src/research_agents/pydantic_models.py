"""
Pydantic models for pipeline state.

RunContext — shared state passed between all agents.
Persisted as run_context.json in the output directory.

Naming convention: method/function name = name of the returned object.
"""

from __future__ import annotations

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
