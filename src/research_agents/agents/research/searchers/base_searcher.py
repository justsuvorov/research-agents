"""
BaseSearcher — abstract interface for all academic source adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from research_agents.agents.research.models import Paper


class BaseSearcher(ABC):
    """Abstract searcher. Each subclass wraps one academic database."""

    source_id: str  # must match agent_config sources list

    @abstractmethod
    def papers(self, query: str, max_results: int) -> list[Paper]:
        """Return up to max_results papers matching query."""
        ...
