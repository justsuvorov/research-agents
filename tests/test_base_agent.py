"""
Unit tests for BaseAgent and idempotency logic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from research_agents.base_agent import BaseAgent
from research_agents.pydantic_models import AgentStatus, RunContext


class ConcreteTestAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""

    name = "test_agent"
    run_count = 0

    def run(self) -> None:
        ConcreteTestAgent.run_count += 1


class TestBaseAgent:
    """Test BaseAgent idempotency and status management."""

    def setup_method(self) -> None:
        ConcreteTestAgent.run_count = 0

    def test_execute_runs_agent(self, tmp_path: Path) -> None:
        ctx = RunContext(goal="Test", config={}, output_dir=str(tmp_path))
        agent = ConcreteTestAgent(ctx)

        agent.execute()
        assert ConcreteTestAgent.run_count == 1
        assert ctx.is_completed("test_agent")

    def test_execute_idempotent_skips_on_second_call(self, tmp_path: Path) -> None:
        ctx = RunContext(goal="Test", config={}, output_dir=str(tmp_path))
        agent = ConcreteTestAgent(ctx)

        agent.execute()  # First call
        first_count = ConcreteTestAgent.run_count
        agent.execute()  # Second call

        assert ConcreteTestAgent.run_count == first_count  # Not incremented

    def test_execute_sets_running_status_before_run(self, tmp_path: Path) -> None:
        ctx = RunContext(goal="Test", config={}, output_dir=str(tmp_path))
        agent = ConcreteTestAgent(ctx)

        # Verify initial status
        assert ctx.agent_status.test_agent == AgentStatus.PENDING

        agent.execute()

        # After execute, should be COMPLETED
        assert ctx.agent_status.test_agent == AgentStatus.COMPLETED

    def test_execute_sets_error_on_exception(self, tmp_path: Path) -> None:
        ctx = RunContext(goal="Test", config={}, output_dir=str(tmp_path))

        class FailingAgent(BaseAgent):
            name = "failing_agent"

            def run(self) -> None:
                raise RuntimeError("Test error")

        agent = FailingAgent(ctx)

        with pytest.raises(RuntimeError):
            agent.execute()

        assert ctx.agent_status.failing_agent == AgentStatus.FAILED
        assert "Test error" in ctx.errors["failing_agent"]

    def test_execute_with_existing_run_context(self, tmp_path: Path) -> None:
        ctx = RunContext(goal="Test", config={}, output_dir=str(tmp_path))
        ctx.save()

        # Load existing context
        loaded_ctx = RunContext.run_context_or_new(
            goal="Test", config={}, output_dir=str(tmp_path)
        )

        agent = ConcreteTestAgent(loaded_ctx)
        agent.execute()

        assert loaded_ctx.is_completed("test_agent")

    def test_is_completed_returns_true_after_successful_execute(
        self, tmp_path: Path
    ) -> None:
        ctx = RunContext(goal="Test", config={}, output_dir=str(tmp_path))
        agent = ConcreteTestAgent(ctx)

        agent.execute()
        assert ctx.is_completed("test_agent")

    def test_is_completed_returns_false_before_execute(self, tmp_path: Path) -> None:
        ctx = RunContext(goal="Test", config={}, output_dir=str(tmp_path))
        agent = ConcreteTestAgent(ctx)

        assert not ctx.is_completed("test_agent")
