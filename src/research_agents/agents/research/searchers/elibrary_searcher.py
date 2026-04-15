"""
ElibrarySearcher — searches eLIBRARY.ru (РИНЦ) via their API.

eLIBRARY.ru does not have a fully open public API.
Access requires registration and an API token from:
https://elibrary.ru/projects/api/api_info.asp

Without a token the searcher logs a warning and returns an empty list,
so the pipeline continues with the other configured sources.
"""

from __future__ import annotations

from typing import Optional

import httpx
from loguru import logger

from research_agents.agents.research.models import Paper
from research_agents.agents.research.searchers.base_searcher import BaseSearcher

_API_URL = "https://elibrary.ru/api/search"


class ElibrarySearcher(BaseSearcher):
    source_id = "elibrary"

    def __init__(self, api_token: Optional[str] = None, timeout: int = 30) -> None:
        if not api_token:
            logger.warning(
                "[elibrary] No API token provided. "
                "Register at https://elibrary.ru/projects/api/api_info.asp "
                "and pass api_token= to ElibrarySearcher. Searches will be skipped."
            )
        self._token = api_token
        self._client = httpx.Client(timeout=timeout)

    def papers(self, query: str, max_results: int) -> list[Paper]:
        if not self._token:
            return []

        logger.debug("[elibrary] query={!r} max={}", query, max_results)
        try:
            response = self._client.get(
                _API_URL,
                params={
                    "token": self._token,
                    "query": query,
                    "pagesize": min(max_results, 100),
                    "format": "json",
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("[elibrary] HTTP error: {}", exc)
            return []

        results = []
        for item in response.json().get("articles", []):
            results.append(Paper(
                title=item.get("title", ""),
                authors=item.get("authors", []),
                year=item.get("year"),
                doi=item.get("doi"),
                abstract=item.get("abstract", ""),
                url=item.get("url"),
                source=self.source_id,
            ))
        return results
