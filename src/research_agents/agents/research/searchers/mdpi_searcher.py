"""
MdpiSearcher — searches MDPI journals via CrossRef API (publisher = MDPI AG).
CrossRef covers all MDPI open-access articles and is free to use.
"""

from __future__ import annotations

from typing import Optional

import httpx
from loguru import logger

from research_agents.agents.research.models import Paper
from research_agents.agents.research.searchers.base_searcher import BaseSearcher

_CROSSREF_URL = "https://api.crossref.org/works"
_MDPI_MEMBER_ID = "1968"   # CrossRef member ID for MDPI AG
_MAILTO = "research-agents@example.com"  # Polite pool — set to real email in config


class MdpiSearcher(BaseSearcher):
    source_id = "mdpi"

    def __init__(self, mailto: Optional[str] = None, timeout: int = 30) -> None:
        self._mailto = mailto or _MAILTO
        self._client = httpx.Client(timeout=timeout)

    def papers(self, query: str, max_results: int) -> list[Paper]:
        logger.debug("[mdpi] query={!r} max={}", query, max_results)
        try:
            response = self._client.get(
                _CROSSREF_URL,
                params={
                    "query": query,
                    "member": _MDPI_MEMBER_ID,
                    "rows": min(max_results, 100),
                    "select": "DOI,title,author,published,abstract,URL",
                    "mailto": self._mailto,
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("[mdpi] HTTP error: {}", exc)
            return []

        items = response.json().get("message", {}).get("items", [])
        return [self._paper(item) for item in items]

    def _paper(self, item: dict) -> Paper:
        title = " ".join(item.get("title", [""])) or ""
        authors = [
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in item.get("author", [])
        ]
        pub = item.get("published", {}).get("date-parts", [[None]])[0]
        year = pub[0] if pub else None
        return Paper(
            title=title,
            authors=authors,
            year=year,
            doi=item.get("DOI"),
            abstract=item.get("abstract", ""),
            url=item.get("URL"),
            source=self.source_id,
        )
