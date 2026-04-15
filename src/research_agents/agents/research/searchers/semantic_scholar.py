"""
SemanticScholarSearcher — wraps the Semantic Scholar public API.
No API key required for basic access; pass api_key for higher rate limits.
"""

from __future__ import annotations

from typing import Optional

import httpx
from loguru import logger

from research_agents.agents.research.models import Paper
from research_agents.agents.research.searchers.base_searcher import BaseSearcher

_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "title,authors,year,externalIds,abstract,url"


class SemanticScholarSearcher(BaseSearcher):
    source_id = "semantic_scholar"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 30) -> None:
        headers = {"x-api-key": api_key} if api_key else {}
        self._client = httpx.Client(
            base_url=_BASE_URL,
            headers=headers,
            timeout=timeout,
        )

    def papers(self, query: str, max_results: int) -> list[Paper]:
        logger.debug("[semantic_scholar] query={!r} max={}", query, max_results)
        try:
            response = self._client.get(
                "/paper/search",
                params={"query": query, "limit": min(max_results, 100), "fields": _FIELDS},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("[semantic_scholar] HTTP error: {}", exc)
            return []

        return [
            self._paper(item)
            for item in response.json().get("data", [])
        ]

    def _paper(self, item: dict) -> Paper:
        return Paper(
            title=item.get("title", ""),
            authors=[a.get("name", "") for a in item.get("authors", [])],
            year=item.get("year"),
            doi=item.get("externalIds", {}).get("DOI"),
            abstract=item.get("abstract") or "",
            url=item.get("url"),
            source=self.source_id,
        )
