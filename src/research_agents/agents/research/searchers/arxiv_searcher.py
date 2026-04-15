"""
ArxivSearcher — wraps the arxiv Python library.
"""

from __future__ import annotations

import arxiv
from loguru import logger

from research_agents.agents.research.models import Paper
from research_agents.agents.research.searchers.base_searcher import BaseSearcher


class ArxivSearcher(BaseSearcher):
    source_id = "arxiv"

    def __init__(self, timeout: int = 30) -> None:
        self._client = arxiv.Client(num_retries=3, delay_seconds=3)

    def papers(self, query: str, max_results: int) -> list[Paper]:
        logger.debug("[arxiv] query={!r} max={}", query, max_results)
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = []
        try:
            for r in self._client.results(search):
                results.append(Paper(
                    title=r.title,
                    authors=[a.name for a in r.authors],
                    year=r.published.year if r.published else None,
                    doi=r.doi,
                    abstract=r.summary,
                    url=r.entry_id,
                    source=self.source_id,
                ))
        except Exception as exc:
            logger.warning("[arxiv] error: {}", exc)
        return results
