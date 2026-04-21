"""
PaperExtractor — extracts structured rows from paper abstracts via LLM.
One paper can yield multiple rows (one per experiment/observation).
All prompt text is injected via constructor.
"""

from __future__ import annotations

import json

import anthropic
from loguru import logger

from research_agents.agents.research.models import PaperAnalysis
from research_agents.config import ExtractionRule


class PaperExtractor:

    def __init__(
        self,
        client: anthropic.Anthropic,
        system_prompt: str,
        user_template: str,
    ) -> None:
        self._client = client
        self._system_prompt = system_prompt
        self._user_template = user_template

    def extracted_rows(
        self,
        analysis: PaperAnalysis,
        rules: list[ExtractionRule],
    ) -> list[dict]:
        """Return rows extracted from a single paper. May return 0 or many rows."""
        if not rules:
            return []

        columns_block = "\n".join(
            f"- {r.name} ({r.type}){', ' + r.unit if r.unit else ''}: {r.description}"
            for r in rules
        )
        row_schema = ", ".join(
            f'"{r.name}": <{r.type} or null>' for r in rules
        )

        content = self._user_template.format(
            title=analysis.paper.title,
            authors=", ".join(analysis.paper.authors) or "N/A",
            year=analysis.paper.year or "N/A",
            abstract=analysis.paper.abstract or "No abstract.",
            columns_block=columns_block,
            row_schema=row_schema,
        )

        logger.debug("[PaperExtractor] extracting from: {}", analysis.paper.title[:60])

        response = self._client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=[
                {
                    "type": "text",
                    "text": self._system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": content}],
        )

        return self._parse_rows(response.content[0].text, analysis, rules)

    def _parse_rows(
        self,
        raw: str,
        analysis: PaperAnalysis,
        rules: list[ExtractionRule],
    ) -> list[dict]:
        try:
            rows = json.loads(raw.strip())
            if not isinstance(rows, list):
                raise ValueError("expected JSON array")
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("[PaperExtractor] parse error for '{}': {}", analysis.paper.title[:40], exc)
            return []

        result = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            record: dict = {
                "source": analysis.paper.bibtex_key or analysis.paper.title[:40],
                "source_type": "paper",
            }
            for rule in rules:
                record[rule.name] = row.get(rule.name)
            result.append(record)

        if result:
            logger.info("[PaperExtractor] {} rows from '{}'", len(result), analysis.paper.title[:50])
        return result
