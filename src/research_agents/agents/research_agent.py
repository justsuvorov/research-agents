"""
ResearchAgent — orchestrates literature search, filtering, synthesis, and export.

All dependencies are injected: LLM client, searchers, analyzers, synthesizer, exporters.
No prompts or credentials live inside this module.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from research_agents.agents.research.exporters.bibtex import bib_file
from research_agents.agents.research.exporters.markdown_writer import markdown_review
from research_agents.agents.research.models import LiteratureReport, Paper
from research_agents.agents.research.paper_analyzer import PaperAnalyzer
from research_agents.agents.research.query_builder import QueryBuilder
from research_agents.agents.research.searchers.base_searcher import BaseSearcher
from research_agents.agents.research.synthesizer import Synthesizer
from research_agents.base_agent import BaseAgent
from research_agents.config import ResearchConfig
from research_agents.pydantic_models import RunContext


class ResearchAgent(BaseAgent):
    name = "research"

    def __init__(
        self,
        ctx: RunContext,
        query_builder: QueryBuilder,
        searchers: list[BaseSearcher],
        paper_analyzer: PaperAnalyzer,
        synthesizer: Synthesizer,
    ) -> None:
        super().__init__(ctx)
        self._query_builder = query_builder
        self._searchers = {s.source_id: s for s in searchers}
        self._paper_analyzer = paper_analyzer
        self._synthesizer = synthesizer

    def run(self) -> None:
        cfg = ResearchConfig(**self.ctx.config["research"])
        output_dir = Path(self.ctx.output_dir)

        # 1. Generate queries
        queries = self._query_builder.search_queries(self.ctx.goal, n_queries=6)

        # 2. Search configured sources
        papers = self._search_all(queries, cfg)
        logger.info("[ResearchAgent] collected {} papers (before dedup)", len(papers))

        # 3. Deduplicate
        papers = self._deduplicated(papers)
        logger.info("[ResearchAgent] {} papers after deduplication", len(papers))

        # 4. Analyze and filter
        report = LiteratureReport(goal=self.ctx.goal)
        for paper in papers[: cfg.max_papers]:
            analysis = self._paper_analyzer.paper_analysis(paper)
            report.analyses.append(analysis)

        relevant = report.relevant()
        logger.info("[ResearchAgent] {} papers passed domain filter", len(relevant))

        if len(relevant) < 5:
            raise RuntimeError(
                f"Insufficient relevant sources: {len(relevant)} found, minimum 5 required."
            )

        # 5. Synthesize sections
        sections = self._synthesizer.literature_review_sections(report)

        # 6. Export
        review_path = output_dir / "literature_review.md"
        bib_path = output_dir / "references.bib"

        markdown_review(report, sections, review_path)
        bib_file([a.paper for a in relevant], bib_path)

        # 7. Update context
        self.ctx.set_artifact("literature_review", str(review_path))
        self.ctx.set_artifact("references", str(bib_path))

    def _search_all(self, queries: list[str], cfg: ResearchConfig) -> list[Paper]:
        papers: list[Paper] = []
        per_source = max(1, cfg.max_papers // len(cfg.sources))
        for source_id in cfg.sources:
            searcher = self._searchers.get(source_id)
            if not searcher:
                logger.warning("[ResearchAgent] searcher not registered: {}", source_id)
                continue
            for query in queries:
                papers.extend(searcher.papers(query, max_results=per_source))
        return papers

    def _deduplicated(self, papers: list[Paper]) -> list[Paper]:
        seen: set[str] = set()
        unique: list[Paper] = []
        for p in papers:
            key = (p.doi or p.title).strip().lower()
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique
