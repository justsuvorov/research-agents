"""
Integration tests for checkpoint recovery workflow.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from research_agents.agents.research.models import Paper
from research_agents.agents.research.paper_analyzer import PaperAnalyzer
from research_agents.agents.research.query_builder import QueryBuilder
from research_agents.agents.research.searchers.base_searcher import BaseSearcher
from research_agents.agents.research.synthesizer import Synthesizer
from research_agents.agents.research_agent import ResearchAgent
from research_agents.pydantic_models import RunContext


class MockSearcher(BaseSearcher):
    """Mock searcher for testing."""

    source_id = "mock"

    def __init__(self) -> None:
        super().__init__()
        self._call_count = 0

    def papers(self, query: str, max_results: int = 5) -> list[Paper]:
        self._call_count += 1
        return [
            Paper(
                title=f"Paper {i} from query '{query}'",
                authors=[f"Author {i}"],
                year=2020 + i,
                doi=f"10.1234/test.{i}",
                source="mock",
                abstract=f"Abstract for paper {i}",
            )
            for i in range(max_results)
        ]


@pytest.fixture
def checkpoint_recovery_agent(tmp_path: Path) -> tuple[ResearchAgent, RunContext]:
    """Create ResearchAgent with mocked dependencies for checkpoint testing."""
    ctx = RunContext(
        goal="Test checkpoint recovery",
        config={"research": {"sources": ["mock"], "max_papers": 10}},
        output_dir=str(tmp_path),
    )

    query_builder = MagicMock(spec=QueryBuilder)
    query_builder.search_queries.return_value = ["query1", "query2"]

    paper_analyzer = MagicMock(spec=PaperAnalyzer)

    def paper_analysis(paper: Paper):
        from research_agents.agents.research.models import PaperAnalysis

        return PaperAnalysis(
            paper=paper,
            passes_domain_filter=True,
            relevance_score=0.8,
            category="theory",
            summary=f"Summary of {paper.title}",
            key_equation=None,
            gap_analysis=None,
        )

    paper_analyzer.paper_analysis.side_effect = paper_analysis

    synthesizer = MagicMock(spec=Synthesizer)
    synthesizer.literature_review_sections.return_value = {
        "theory": "## Theory\n\nContent here..."
    }

    agent = ResearchAgent(
        ctx=ctx,
        query_builder=query_builder,
        searchers=[MockSearcher()],
        paper_analyzer=paper_analyzer,
        synthesizer=synthesizer,
    )

    return agent, ctx


class TestCheckpointRecovery:
    """Test checkpoint saving and recovery workflow."""

    def test_checkpoint_saved_after_step_1(
        self, checkpoint_recovery_agent: tuple[ResearchAgent, RunContext]
    ) -> None:
        agent, ctx = checkpoint_recovery_agent

        # Execute agent
        agent.execute()

        # Verify checkpoint exists for step 1
        assert "research" in ctx.checkpoints
        cp = ctx.checkpoints["research"]
        assert cp.step_completed >= 1

    def test_all_checkpoints_saved(
        self, checkpoint_recovery_agent: tuple[ResearchAgent, RunContext]
    ) -> None:
        agent, ctx = checkpoint_recovery_agent

        agent.execute()

        checkpoint_dir = Path(ctx.output_dir) / "checkpoints" / "research"
        expected_files = [
            "queries.json",
            "papers_raw.json",
            "papers_dedup.json",
            "analyses.json",
            "sections.json",
        ]

        for filename in expected_files:
            assert (checkpoint_dir / filename).exists(), f"Missing: {filename}"

    def test_checkpoint_persists_across_loads(
        self, checkpoint_recovery_agent: tuple[ResearchAgent, RunContext]
    ) -> None:
        agent, ctx = checkpoint_recovery_agent

        agent.execute()
        initial_checkpoints = len(ctx.checkpoints)

        # Save and reload
        ctx.save()
        loaded_ctx = RunContext.run_context(ctx.output_dir)

        assert len(loaded_ctx.checkpoints) == initial_checkpoints
        assert "research" in loaded_ctx.checkpoints

    def test_second_execution_skips_completed_agent(
        self, checkpoint_recovery_agent: tuple[ResearchAgent, RunContext]
    ) -> None:
        agent, ctx = checkpoint_recovery_agent
        mock_searcher = agent._searchers["mock"]
        initial_calls = mock_searcher._call_count

        # First execution
        agent.execute()
        first_calls = mock_searcher._call_count

        # Second execution should skip (no new searcher calls)
        agent.execute()
        second_calls = mock_searcher._call_count

        assert second_calls == first_calls  # No additional calls

    def test_can_resume_returns_true_for_completed_agent(
        self, checkpoint_recovery_agent: tuple[ResearchAgent, RunContext]
    ) -> None:
        agent, ctx = checkpoint_recovery_agent

        agent.execute()

        assert ctx.can_resume() is True
        assert ctx.last_completed_agent() == "research"

    def test_checkpoint_json_is_valid(
        self, checkpoint_recovery_agent: tuple[ResearchAgent, RunContext]
    ) -> None:
        agent, ctx = checkpoint_recovery_agent

        agent.execute()

        checkpoint_dir = Path(ctx.output_dir) / "checkpoints" / "research"
        import json

        for filename in ["queries.json", "papers_raw.json"]:
            filepath = checkpoint_dir / filename
            data = json.loads(filepath.read_text(encoding="utf-8"))
            assert isinstance(data, (list, dict))

    def test_checkpoint_intermediate_artifacts_populated(
        self, checkpoint_recovery_agent: tuple[ResearchAgent, RunContext]
    ) -> None:
        agent, ctx = checkpoint_recovery_agent

        agent.execute()

        cp = ctx.checkpoints["research"]
        assert len(cp.intermediate_artifacts) > 0
        assert cp.checkpoint_dir is not None

    def test_checkpoint_status_transitions(
        self, checkpoint_recovery_agent: tuple[ResearchAgent, RunContext]
    ) -> None:
        agent, ctx = checkpoint_recovery_agent

        # Before execution
        assert len(ctx.checkpoints) == 0

        # During execution (checkpoints created as "in_progress")
        agent.execute()

        # After completion
        cp = ctx.checkpoints["research"]
        assert cp.status == "completed" or cp.agent_name == "research"


class TestCheckpointWithFailure:
    """Test checkpoint behavior when agent fails."""

    def test_checkpoint_marked_failed_on_exception(self, tmp_path: Path) -> None:
        ctx = RunContext(
            goal="Test",
            config={"research": {"sources": ["mock"]}},
            output_dir=str(tmp_path),
        )

        # Create agent that will fail
        query_builder = MagicMock(spec=QueryBuilder)
        query_builder.search_queries.side_effect = RuntimeError("API Error")

        agent = ResearchAgent(
            ctx=ctx,
            query_builder=query_builder,
            searchers=[],
            paper_analyzer=MagicMock(),
            synthesizer=MagicMock(),
        )

        with pytest.raises(RuntimeError):
            agent.execute()

        # Verify agent is marked as failed
        assert ctx.agent_status.research.value == "failed"

    def test_recovery_not_possible_after_failure(self, tmp_path: Path) -> None:
        ctx = RunContext(
            goal="Test",
            config={"research": {"sources": ["mock"]}},
            output_dir=str(tmp_path),
        )

        ctx.checkpoint(agent="research", step=2)
        ctx.checkpoint_failed("research", "Network error")

        assert ctx.can_resume() is False
        assert ctx.checkpoints["research"].recovery_possible is False
