"""
Integration tests for ResearchAgent with mocked components.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from research_agents.agents.research.models import Paper, PaperAnalysis, LiteratureReport
from research_agents.agents.research.paper_analyzer import PaperAnalyzer
from research_agents.agents.research.query_builder import QueryBuilder
from research_agents.agents.research.searchers.base_searcher import BaseSearcher
from research_agents.agents.research.synthesizer import Synthesizer
from research_agents.agents.research_agent import ResearchAgent
from research_agents.config import ResearchConfig
from research_agents.pydantic_models import RunContext


class MockSearcher(BaseSearcher):
    """Mock searcher that returns fixed papers."""

    source_id = "mock"

    def __init__(self, papers: list[Paper] | None = None) -> None:
        super().__init__()
        self._papers = papers or []

    def papers(self, query: str, max_results: int = 5) -> list[Paper]:
        return self._papers[:max_results]


@pytest.fixture
def mock_papers() -> list[Paper]:
    """Create mock papers for testing."""
    return [
        Paper(
            title="Paper 1: Gear Wear Analysis",
            authors=["Author A", "Author B"],
            year=2020,
            doi="10.1234/test.1",
            source="arxiv",
            abstract="This paper discusses gear wear under dynamic loading conditions.",
        ),
        Paper(
            title="Paper 2: Composite Materials in Mechanics",
            authors=["Author C"],
            year=2021,
            doi="10.1234/test.2",
            source="arxiv",
            abstract="Study of polymeric composite materials in engineering applications.",
        ),
        Paper(
            title="Paper 3: Dynamic Loading Effects",
            authors=["Author D", "Author E", "Author F"],
            year=2022,
            doi="10.1234/test.3",
            source="arxiv",
            abstract="Analysis of dynamic loading on rotating machinery.",
        ),
        Paper(
            title="Paper 4: DNV Standards Review",
            authors=["Author G"],
            year=2019,
            doi="10.1234/test.4",
            source="arxiv",
            abstract="Classification standards for marine equipment and their application.",
        ),
        Paper(
            title="Paper 5: Tribology Fundamentals",
            authors=["Author H"],
            year=2021,
            doi="10.1234/test.5",
            source="arxiv",
            abstract="Fundamental concepts of tribology and wear mechanisms.",
        ),
        Paper(
            title="Paper 6: GLM Models for Engineering",
            authors=["Author I"],
            year=2023,
            doi="10.1234/test.6",
            source="arxiv",
            abstract="Generalized Linear Models applied to mechanical engineering problems.",
        ),
    ]


@pytest.fixture
def mock_query_builder() -> QueryBuilder:
    """Create mock QueryBuilder that returns fixed queries."""
    builder = MagicMock(spec=QueryBuilder)
    builder.search_queries.return_value = [
        "gear wear dynamic loading",
        "composite materials mechanics",
        "marine crane specifications",
        "tribology wear models",
        "GLM engineering prediction",
        "DNV classification standards",
    ]
    return builder


@pytest.fixture
def mock_paper_analyzer() -> PaperAnalyzer:
    """Create mock PaperAnalyzer that returns fixed analyses."""
    analyzer = MagicMock(spec=PaperAnalyzer)

    def paper_analysis(paper: Paper) -> PaperAnalysis:
        # All papers pass domain filter and have varied relevance
        return PaperAnalysis(
            paper=paper,
            passes_domain_filter=True,
            relevance_score=0.8 if paper.year >= 2021 else 0.6,
            category="theory" if "theory" in paper.title.lower() else "experiments",
            summary=f"Summary of {paper.title}",
            key_equation="E = mc²" if "dynamic" in paper.abstract.lower() else None,
            gap_analysis="Needs more recent data" if paper.year < 2022 else None,
        )

    analyzer.paper_analysis.side_effect = paper_analysis
    return analyzer


@pytest.fixture
def mock_synthesizer() -> Synthesizer:
    """Create mock Synthesizer that returns fixed sections."""
    synthesizer = MagicMock(spec=Synthesizer)

    def literature_review_sections(
        report: LiteratureReport,
    ) -> dict[str, str]:
        return {
            "theory": "## Theoretical Foundations\n\nThis section covers theoretical aspects...",
            "experiments": "## Experimental Studies\n\nThis section covers experimental work...",
            "standards": "## Standards and Guidelines\n\nThis section covers applicable standards...",
        }

    synthesizer.literature_review_sections.side_effect = literature_review_sections
    return synthesizer


@pytest.fixture
def research_agent(
    tmp_path: Path,
    mock_query_builder: QueryBuilder,
    mock_paper_analyzer: PaperAnalyzer,
    mock_papers: list[Paper],
    mock_synthesizer: Synthesizer,
) -> tuple[ResearchAgent, RunContext]:
    """Create ResearchAgent with mocked dependencies."""
    ctx = RunContext(
        goal="Test research on gear wear",
        config={"research": {"sources": ["mock"], "max_papers": 20}},
        output_dir=str(tmp_path),
    )

    mock_searchers = [MockSearcher(papers=mock_papers)]

    agent = ResearchAgent(
        ctx=ctx,
        query_builder=mock_query_builder,
        searchers=mock_searchers,
        paper_analyzer=mock_paper_analyzer,
        synthesizer=mock_synthesizer,
    )

    return agent, ctx


class TestResearchAgent:
    """Integration tests for ResearchAgent."""

    def test_agent_executes_successfully(
        self, research_agent: tuple[ResearchAgent, RunContext]
    ) -> None:
        agent, ctx = research_agent
        agent.execute()

        assert ctx.is_completed("research")
        assert ctx.artifact_path("literature_review") is not None
        assert ctx.artifact_path("references") is not None
        assert ctx.artifact_path("papers_data") is not None

    def test_artifacts_files_exist(
        self, research_agent: tuple[ResearchAgent, RunContext]
    ) -> None:
        agent, ctx = research_agent
        agent.execute()

        review_path = ctx.artifact_path("literature_review")
        bib_path = ctx.artifact_path("references")
        papers_path = ctx.artifact_path("papers_data")

        assert review_path is not None and review_path.exists()
        assert bib_path is not None and bib_path.exists()
        assert papers_path is not None and papers_path.exists()

    def test_literature_review_contains_content(
        self, research_agent: tuple[ResearchAgent, RunContext]
    ) -> None:
        agent, ctx = research_agent
        agent.execute()

        review_path = ctx.artifact_path("literature_review")
        content = review_path.read_text(encoding="utf-8")

        assert len(content) > 100
        assert "##" in content  # Markdown headers

    def test_bibtex_has_entries(
        self, research_agent: tuple[ResearchAgent, RunContext]
    ) -> None:
        agent, ctx = research_agent
        agent.execute()

        bib_path = ctx.artifact_path("references")
        content = bib_path.read_text(encoding="utf-8")

        assert "@" in content  # BibTeX entries
        assert content.count("@") >= 5  # At least 5 papers

    def test_papers_json_is_valid(
        self, research_agent: tuple[ResearchAgent, RunContext]
    ) -> None:
        agent, ctx = research_agent
        agent.execute()

        papers_path = ctx.artifact_path("papers_data")
        import json

        data = json.loads(papers_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert "analyses" in data
        assert len(data["analyses"]) >= 5

    def test_checkpoints_created(
        self, research_agent: tuple[ResearchAgent, RunContext]
    ) -> None:
        agent, ctx = research_agent
        agent.execute()

        checkpoint_dir = ctx.output_dir / "checkpoints" / "research"
        assert checkpoint_dir.exists()
        assert (checkpoint_dir / "queries.json").exists()
        assert (checkpoint_dir / "papers_raw.json").exists()
        assert (checkpoint_dir / "papers_dedup.json").exists()
        assert (checkpoint_dir / "analyses.json").exists()
        assert (checkpoint_dir / "sections.json").exists()

    def test_run_context_saved_after_completion(
        self, research_agent: tuple[ResearchAgent, RunContext]
    ) -> None:
        agent, ctx = research_agent
        agent.execute()

        run_context_file = Path(ctx.output_dir) / "run_context.json"
        assert run_context_file.exists()

        loaded_ctx = RunContext.run_context(ctx.output_dir)
        assert loaded_ctx.is_completed("research")

    def test_idempotent_on_second_execute(
        self, research_agent: tuple[ResearchAgent, RunContext]
    ) -> None:
        agent, ctx = research_agent

        agent.execute()
        first_artifacts = {
            "review": str(ctx.artifact_path("literature_review")),
            "bib": str(ctx.artifact_path("references")),
            "papers": str(ctx.artifact_path("papers_data")),
        }

        agent.execute()  # Second call should be skipped
        second_artifacts = {
            "review": str(ctx.artifact_path("literature_review")),
            "bib": str(ctx.artifact_path("references")),
            "papers": str(ctx.artifact_path("papers_data")),
        }

        assert first_artifacts == second_artifacts

    def test_minimum_papers_requirement_enforced(
        self, tmp_path: Path,
        mock_query_builder: QueryBuilder,
        mock_paper_analyzer: PaperAnalyzer,
        mock_synthesizer: Synthesizer,
    ) -> None:
        """Test that agent fails if fewer than 5 relevant papers found."""
        ctx = RunContext(
            goal="Test research",
            config={"research": {"sources": ["mock"], "max_papers": 20}},
            output_dir=str(tmp_path),
        )

        # Create searcher with only 2 papers
        few_papers = [
            Paper(
                title="Paper 1",
                authors=["A"],
                year=2020,
                doi="10.1234/1",
                source="arxiv",
                abstract="Test",
            ),
            Paper(
                title="Paper 2",
                authors=["B"],
                year=2021,
                doi="10.1234/2",
                source="arxiv",
                abstract="Test",
            ),
        ]
        mock_searchers = [MockSearcher(papers=few_papers)]

        agent = ResearchAgent(
            ctx=ctx,
            query_builder=mock_query_builder,
            searchers=mock_searchers,
            paper_analyzer=mock_paper_analyzer,
            synthesizer=mock_synthesizer,
        )

        with pytest.raises(RuntimeError, match="Insufficient relevant sources"):
            agent.execute()
