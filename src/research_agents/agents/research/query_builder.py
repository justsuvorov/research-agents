"""
QueryBuilder — generates targeted search queries from the research goal via LLM.
All prompt text is injected via constructor — no inline strings here.
"""

from __future__ import annotations

import json

import google.generativeai as genai
from loguru import logger

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
        client: genai.GenerativeModel,
        system_prompt: str,
        user_template: str,
    ) -> None:
        self._client = client
        self._system_prompt = system_prompt
        self._user_template = user_template

    def search_queries(self, goal: str, n_queries: int = 6) -> list[str]:
        """Return list of search query strings generated from the research goal."""
        logger.debug("[QueryBuilder] generating {} queries", n_queries)

        user_message = self._user_template.format(goal=goal, n_queries=n_queries)

        response = self._client.generate_content(
            contents=[
                f"{self._system_prompt}\n\n{user_message}"
            ],
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=512,
                temperature=0.7,
            ),
        )

        raw = response.text.strip()
        try:
            queries = json.loads(raw)
            if not isinstance(queries, list):
                raise ValueError("expected JSON array")
            logger.info("[QueryBuilder] generated {} queries", len(queries))
            return [str(q) for q in queries]
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("[QueryBuilder] failed to parse LLM response: {}", exc)
            return _FALLBACK_QUERIES
