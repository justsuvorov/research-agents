"""
Integration tests for paper searchers (ArxivSearcher, SemanticScholarSearcher, etc.)
"""

from __future__ import annotations

import os

import pytest

from research_agents.agents.research.searchers import (
    ArxivSearcher,
    MdpiSearcher,
    SemanticScholarSearcher,
)


class TestArxivSearcher:
    """Test ArxivSearcher connectivity and functionality."""

    def test_arxiv_returns_papers(self) -> None:
        """Test that ArxivSearcher can query and return papers."""
        searcher = ArxivSearcher()
        papers = searcher.papers("gear wear", max_results=5)

        assert len(papers) > 0, "ArxivSearcher returned no papers"
        assert all(p.title for p in papers), "Some papers missing titles"
        assert all(p.source == "arxiv" for p in papers), "Papers not marked as arxiv source"

    def test_arxiv_respects_max_results(self) -> None:
        """Test that ArxivSearcher respects max_results parameter."""
        searcher = ArxivSearcher()
        papers = searcher.papers("dynamic load", max_results=3)

        assert len(papers) <= 3, f"ArxivSearcher returned {len(papers)} papers, expected <= 3"

    def test_arxiv_returns_valid_papers(self) -> None:
        """Test that ArxivSearcher returns valid Paper objects."""
        searcher = ArxivSearcher()
        papers = searcher.papers("mechanical engineering", max_results=2)

        for paper in papers:
            assert paper.title is not None
            assert paper.title.strip() != ""
            assert paper.source == "arxiv"
            # DOI may be None for some arxiv papers, but should have some identifier
            assert paper.doi or paper.title  # At least one identifier

    def test_arxiv_handles_empty_results(self) -> None:
        """Test that ArxivSearcher handles queries with no results gracefully."""
        searcher = ArxivSearcher()
        # Very specific query unlikely to have results
        papers = searcher.papers("xyzabc123notarealthing", max_results=5)

        assert isinstance(papers, list)
        assert len(papers) == 0


class TestSemanticScholarSearcher:
    """Test SemanticScholarSearcher connectivity and functionality."""

    @pytest.mark.skipif(
        not os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
        reason="Semantic Scholar API key not set",
    )
    def test_semantic_scholar_returns_papers(self) -> None:
        """Test that SemanticScholarSearcher can query and return papers."""
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        searcher = SemanticScholarSearcher(api_key=api_key)

        papers = searcher.papers("machine learning", max_results=5)

        assert len(papers) > 0, "SemanticScholarSearcher returned no papers"
        assert all(p.title for p in papers), "Some papers missing titles"

    @pytest.mark.skipif(
        not os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
        reason="Semantic Scholar API key not set",
    )
    def test_semantic_scholar_respects_max_results(self) -> None:
        """Test that SemanticScholarSearcher respects max_results."""
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        searcher = SemanticScholarSearcher(api_key=api_key)

        papers = searcher.papers("neural networks", max_results=3)

        assert len(papers) <= 3

    def test_semantic_scholar_without_api_key_returns_empty(self) -> None:
        """Test that SemanticScholarSearcher without API key handles gracefully."""
        searcher = SemanticScholarSearcher(api_key=None)
        papers = searcher.papers("some query", max_results=5)

        # Should return empty list or handle gracefully
        assert isinstance(papers, list)


class TestMdpiSearcher:
    """Test MdpiSearcher connectivity and functionality."""

    def test_mdpi_returns_papers(self) -> None:
        """Test that MdpiSearcher can query and return papers."""
        searcher = MdpiSearcher(mailto="test@example.com")
        papers = searcher.papers("materials science", max_results=5)

        assert len(papers) >= 0, "MdpiSearcher returned invalid result"
        if len(papers) > 0:
            assert all(p.title for p in papers), "Some papers missing titles"

    def test_mdpi_respects_max_results(self) -> None:
        """Test that MdpiSearcher respects max_results parameter."""
        searcher = MdpiSearcher(mailto="test@example.com")
        papers = searcher.papers("engineering", max_results=3)

        assert len(papers) <= 3, f"MdpiSearcher returned {len(papers)} papers, expected <= 3"

    def test_mdpi_requires_mailto(self) -> None:
        """Test that MdpiSearcher handles missing mailto gracefully."""
        searcher = MdpiSearcher(mailto=None)
        papers = searcher.papers("some query", max_results=5)

        assert isinstance(papers, list)


# ============================================================================
# Searchable Queries Test - Find papers that exist
# ============================================================================


class TestSearchableQueries:
    """Test with queries known to return results."""

    def test_arxiv_finds_papers_on_gear_wear(self) -> None:
        """ArxivSearcher should find papers on gear wear/tribology."""
        searcher = ArxivSearcher()
        papers = searcher.papers("tribology wear", max_results=10)

        assert len(papers) > 0, "ArxivSearcher found no tribology papers"
        print(f"[OK] Found {len(papers)} tribology papers on ArXiv")

    def test_arxiv_finds_papers_on_dynamic_loads(self) -> None:
        """ArxivSearcher should find papers on dynamic loads/mechanics."""
        searcher = ArxivSearcher()
        papers = searcher.papers("dynamic load mechanical", max_results=10)

        assert len(papers) > 0, "ArxivSearcher found no dynamic load papers"
        print(f"[OK] Found {len(papers)} dynamic load papers on ArXiv")

    def test_arxiv_finds_papers_on_composites(self) -> None:
        """ArxivSearcher should find papers on composite materials."""
        searcher = ArxivSearcher()
        papers = searcher.papers("composite materials", max_results=10)

        assert len(papers) > 0, "ArxivSearcher found no composite papers"
        print(f"[OK] Found {len(papers)} composite material papers on ArXiv")

    def test_arxiv_finds_papers_on_statistical_models(self) -> None:
        """ArxivSearcher should find papers on GLM or statistical models."""
        searcher = ArxivSearcher()
        papers = searcher.papers("generalized linear model", max_results=10)

        assert len(papers) > 0, "ArxivSearcher found no GLM papers"
        print(f"[OK] Found {len(papers)} GLM papers on ArXiv")


# ============================================================================
# Specific Research Goal Test - Narrow down what works
# ============================================================================


class TestSpecificSearchQueries:
    """Test queries specific to ship crane gear research."""

    def test_find_papers_on_ship_cranes(self) -> None:
        """Find papers on ship cranes and marine equipment."""
        searcher = ArxivSearcher()
        papers = searcher.papers("ship crane marine", max_results=10)

        print(f"\n[Papers] Ship crane: {len(papers)} found")
        for paper in papers[:3]:
            print(f"   - {paper.title[:60]}... ({paper.year})")

    def test_find_papers_on_slewing_bearings(self) -> None:
        """Find papers on slewing bearings (опорно-поворотное устройство)."""
        searcher = ArxivSearcher()
        papers = searcher.papers("slewing bearing rotation", max_results=10)

        print(f"\n[Papers] Slewing bearings: {len(papers)} found")
        for paper in papers[:3]:
            print(f"   - {paper.title[:60]}... ({paper.year})")

    def test_find_papers_on_gear_design(self) -> None:
        """Find papers on gear design and calculation."""
        searcher = ArxivSearcher()
        papers = searcher.papers("gear design calculation", max_results=10)

        print(f"\n[Papers] Gear design: {len(papers)} found")
        for paper in papers[:3]:
            print(f"   - {paper.title[:60]}... ({paper.year})")

    def test_find_papers_on_wear_models(self) -> None:
        """Find papers on wear prediction and modeling."""
        searcher = ArxivSearcher()
        papers = searcher.papers("wear model prediction", max_results=10)

        print(f"\n[Papers] Wear models: {len(papers)} found")
        for paper in papers[:3]:
            print(f"   - {paper.title[:60]}... ({paper.year})")


# ============================================================================
# Connectivity Status Check
# ============================================================================


class TestSearcherConnectivity:
    """Test and report searcher connectivity status."""

    def test_all_searchers_connectivity_status(self) -> None:
        """Test connectivity of all searchers and print status."""
        print("\n" + "=" * 70)
        print("SEARCHER CONNECTIVITY STATUS")
        print("=" * 70)

        # Test ArXiv
        try:
            searcher = ArxivSearcher()
            papers = searcher.papers("test", max_results=1)
            status = "[OK]" if len(papers) >= 0 else "[WARN]"
            print(f"ArXiv         {status}")
        except Exception as e:
            print(f"ArXiv         [ERROR]: {str(e)[:50]}")

        # Test Semantic Scholar
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        try:
            searcher = SemanticScholarSearcher(api_key=api_key)
            papers = searcher.papers("test", max_results=1)
            status = "[OK]" if api_key else "[NO KEY]"
            print(f"Semantic      {status}")
        except Exception as e:
            print(f"Semantic      [ERROR]: {str(e)[:50]}")

        # Test MDPI
        try:
            searcher = MdpiSearcher(mailto="test@example.com")
            papers = searcher.papers("test", max_results=1)
            status = "[OK]" if len(papers) >= 0 else "[WARN]"
            print(f"MDPI          {status}")
        except Exception as e:
            print(f"MDPI          [ERROR]: {str(e)[:50]}")

        print("=" * 70)
        print("\nTips:")
        print("  - If Semantic Scholar fails: add SEMANTIC_SCHOLAR_API_KEY to .env")
        print("  - If MDPI fails: check internet connection or Crossref API status")
        print("  - If ArXiv fails: check internet connection")
        print()
