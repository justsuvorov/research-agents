"""
Unit tests for pydantic_models (RunContext, AgentCheckpoint, etc.)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from research_agents.pydantic_models import (
    AgentCheckpoint,
    AgentStatus,
    AgentStatuses,
    Artifacts,
    RunContext,
)


class TestAgentCheckpoint:
    """Test AgentCheckpoint model."""

    def test_create_checkpoint_with_defaults(self) -> None:
        cp = AgentCheckpoint(
            agent_name="research",
            status="in_progress",
            timestamp=datetime.now(timezone.utc).isoformat(),
            checkpoint_dir="/tmp/checkpoints/research",
        )
        assert cp.agent_name == "research"
        assert cp.status == "in_progress"
        assert cp.step_completed == 0
        assert cp.total_items == 0
        assert cp.tokens_used == 0
        assert cp.intermediate_artifacts == {}
        assert cp.errors == []
        assert cp.recovery_possible is True

    def test_checkpoint_with_artifacts(self) -> None:
        artifacts = {"step1": "/path/to/queries.json", "step2": "/path/to/papers.json"}
        cp = AgentCheckpoint(
            agent_name="research",
            status="in_progress",
            timestamp=datetime.now(timezone.utc).isoformat(),
            checkpoint_dir="/tmp/checkpoints/research",
            step_completed=2,
            total_items=50,
            tokens_used=5000,
            intermediate_artifacts=artifacts,
        )
        assert cp.intermediate_artifacts == artifacts
        assert cp.step_completed == 2
        assert cp.total_items == 50
        assert cp.tokens_used == 5000

    def test_checkpoint_serializes_to_json(self) -> None:
        cp = AgentCheckpoint(
            agent_name="research",
            status="completed",
            timestamp=datetime.now(timezone.utc).isoformat(),
            checkpoint_dir="/tmp/checkpoints/research",
            step_completed=6,
        )
        json_str = cp.model_dump_json()
        data = json.loads(json_str)
        assert data["agent_name"] == "research"
        assert data["status"] == "completed"
        assert data["step_completed"] == 6


class TestRunContext:
    """Test RunContext model and methods."""

    def test_run_context_creation(self, tmp_path: Path) -> None:
        ctx = RunContext(
            goal="Test research goal",
            config={"research": {"sources": ["arxiv"]}},
            output_dir=str(tmp_path),
        )
        assert ctx.goal == "Test research goal"
        assert ctx.output_dir == str(tmp_path)
        assert ctx.run_id is not None
        assert ctx.created_at is not None

    def test_run_context_save_and_load(self, tmp_path: Path) -> None:
        ctx = RunContext(
            goal="Test goal",
            config={"key": "value"},
            output_dir=str(tmp_path),
        )
        ctx.save()

        ctx_loaded = RunContext.run_context(str(tmp_path))
        assert ctx_loaded.goal == ctx.goal
        assert ctx_loaded.run_id == ctx.run_id
        assert ctx_loaded.config == ctx.config

    def test_run_context_or_new_creates_new(self, tmp_path: Path) -> None:
        ctx = RunContext.run_context_or_new(
            goal="Test goal",
            config={"key": "value"},
            output_dir=str(tmp_path),
        )
        assert ctx.goal == "Test goal"
        assert (tmp_path / "run_context.json").exists()

    def test_run_context_or_new_loads_existing(self, tmp_path: Path) -> None:
        ctx1 = RunContext.run_context_or_new(
            goal="First goal",
            config={"key": "value"},
            output_dir=str(tmp_path),
        )
        run_id_1 = ctx1.run_id

        ctx2 = RunContext.run_context_or_new(
            goal="Second goal (ignored)",
            config={"key": "other"},
            output_dir=str(tmp_path),
        )
        assert ctx2.run_id == run_id_1  # Same run
        assert ctx2.goal == "First goal"  # Loaded, not new

    def test_set_status(self, tmp_path: Path) -> None:
        ctx = RunContext(
            goal="Test",
            config={},
            output_dir=str(tmp_path),
        )
        ctx.set_status("research", AgentStatus.RUNNING)
        assert ctx.agent_status.research == AgentStatus.RUNNING

        ctx.set_status("research", AgentStatus.COMPLETED)
        assert ctx.agent_status.research == AgentStatus.COMPLETED
        assert (tmp_path / "run_context.json").exists()

    def test_set_artifact(self, tmp_path: Path) -> None:
        ctx = RunContext(
            goal="Test",
            config={},
            output_dir=str(tmp_path),
        )
        artifact_path = tmp_path / "output.txt"
        artifact_path.write_text("test")

        ctx.set_artifact("literature_review", str(artifact_path))
        assert ctx.artifacts.literature_review == str(artifact_path.resolve())
        assert (tmp_path / "run_context.json").exists()

    def test_is_completed(self, tmp_path: Path) -> None:
        ctx = RunContext(
            goal="Test",
            config={},
            output_dir=str(tmp_path),
        )
        assert not ctx.is_completed("research")

        ctx.set_status("research", AgentStatus.COMPLETED)
        assert ctx.is_completed("research")

    def test_artifact_path_returns_path_or_none(self, tmp_path: Path) -> None:
        ctx = RunContext(
            goal="Test",
            config={},
            output_dir=str(tmp_path),
        )
        artifact_path = tmp_path / "output.txt"
        artifact_path.write_text("test")

        ctx.set_artifact("literature_review", str(artifact_path))
        returned_path = ctx.artifact_path("literature_review")
        assert returned_path == artifact_path.resolve()

        assert ctx.artifact_path("nonexistent") is None

    def test_checkpoint_saves_to_context(self, tmp_path: Path) -> None:
        ctx = RunContext(
            goal="Test",
            config={},
            output_dir=str(tmp_path),
        )
        ctx.checkpoint(
            agent="research",
            step=1,
            total_items=10,
            tokens_used=1000,
            intermediate_artifacts={"queries": "/path/to/queries.json"},
        )

        assert "research" in ctx.checkpoints
        cp = ctx.checkpoints["research"]
        assert cp.agent_name == "research"
        assert cp.step_completed == 1
        assert cp.total_items == 10
        assert cp.tokens_used == 1000
        assert cp.status == "in_progress"
        assert (tmp_path / "run_context.json").exists()

    def test_checkpoint_completed(self, tmp_path: Path) -> None:
        ctx = RunContext(
            goal="Test",
            config={},
            output_dir=str(tmp_path),
        )
        ctx.checkpoint(agent="research", step=6)
        ctx.checkpoint_completed("research")

        assert ctx.checkpoints["research"].status == "completed"

    def test_checkpoint_failed(self, tmp_path: Path) -> None:
        ctx = RunContext(
            goal="Test",
            config={},
            output_dir=str(tmp_path),
        )
        ctx.checkpoint(agent="research", step=3)
        ctx.checkpoint_failed("research", "Network timeout")

        cp = ctx.checkpoints["research"]
        assert cp.status == "failed"
        assert "Network timeout" in cp.errors
        assert cp.recovery_possible is False

    def test_last_completed_agent_returns_none_if_empty(self, tmp_path: Path) -> None:
        ctx = RunContext(
            goal="Test",
            config={},
            output_dir=str(tmp_path),
        )
        assert ctx.last_completed_agent() is None

    def test_last_completed_agent_returns_last_completed(self, tmp_path: Path) -> None:
        ctx = RunContext(
            goal="Test",
            config={},
            output_dir=str(tmp_path),
        )
        ctx.set_status("research", AgentStatus.COMPLETED)
        assert ctx.last_completed_agent() == "research"

        ctx.set_status("data", AgentStatus.COMPLETED)
        assert ctx.last_completed_agent() == "data"

    def test_can_resume_false_if_no_checkpoints(self, tmp_path: Path) -> None:
        ctx = RunContext(
            goal="Test",
            config={},
            output_dir=str(tmp_path),
        )
        assert ctx.can_resume() is False

    def test_can_resume_false_if_no_completed_agent(self, tmp_path: Path) -> None:
        ctx = RunContext(
            goal="Test",
            config={},
            output_dir=str(tmp_path),
        )
        ctx.checkpoint(agent="research", step=3)
        assert ctx.can_resume() is False

    def test_can_resume_true_if_recovery_possible(self, tmp_path: Path) -> None:
        ctx = RunContext(
            goal="Test",
            config={},
            output_dir=str(tmp_path),
        )
        ctx.checkpoint(agent="research", step=6)
        ctx.set_status("research", AgentStatus.COMPLETED)
        assert ctx.can_resume() is True

    def test_can_resume_false_if_recovery_not_possible(self, tmp_path: Path) -> None:
        ctx = RunContext(
            goal="Test",
            config={},
            output_dir=str(tmp_path),
        )
        ctx.checkpoint(agent="research", step=3)
        ctx.checkpoint_failed("research", "Error")
        ctx.set_status("research", AgentStatus.FAILED)
        assert ctx.can_resume() is False

    def test_checkpoint_creates_directory(self, tmp_path: Path) -> None:
        ctx = RunContext(
            goal="Test",
            config={},
            output_dir=str(tmp_path),
        )
        ctx.checkpoint(agent="research", step=1)
        checkpoint_dir = tmp_path / "checkpoints" / "research"
        assert checkpoint_dir.exists()

    def test_set_error(self, tmp_path: Path) -> None:
        ctx = RunContext(
            goal="Test",
            config={},
            output_dir=str(tmp_path),
        )
        ctx.set_error("research", "Test error message")
        assert ctx.errors["research"] == "Test error message"
        assert ctx.agent_status.research == AgentStatus.FAILED
