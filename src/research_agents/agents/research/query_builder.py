"""
QueryBuilder — generates targeted search queries from the research goal via LLM.
All prompt text is injected via constructor — no inline strings here.
"""

from __future__ import annotations

import json

from loguru import logger

from research_agents.ai_model import AIModel

_FALLBACK_QUERIES = [
    "slewing bearing internal gear wear boundary lubrication",
    "Kraghelsky Timofeev wear model adhesive fatigue",
    "dynamic factor crane slewing mechanism gear tribology",
    "composite boom crane weight reduction contact stress",
    "GLM generalized linear model wear engineering",
    "RMRS slewing ring bearing marine crane requirements",
]


class QueryBuilder:

    def __init__(
        self,
        model: AIModel,
        system_prompt: str,
        user_template: str,
    ) -> None:
        self._model = model
        self._system_prompt = system_prompt
        self._user_template = user_template

    def search_queries(self, goal: str, n_queries: int = 6) -> list[str]:
        """Return list of search query strings generated from the research goal."""
        logger.debug("[QueryBuilder] generating {} queries", n_queries)

        raw = self._model.complete(
            system=self._system_prompt,
            user=self._user_template.format(goal=goal, n_queries=n_queries),
            max_tokens=512,
        ).strip()
        try:
            queries = json.loads(raw)
            if not isinstance(queries, list):
                raise ValueError("expected JSON array")
            logger.info("[QueryBuilder] generated {} queries", len(queries))
            return [str(q) for q in queries]
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("[QueryBuilder] failed to parse LLM response: {}", exc)
            return _FALLBACK_QUERIES
