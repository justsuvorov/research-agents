"""
PaperAnalyzer — filters papers by domain constraints and extracts structured analysis.
All prompt text is injected via constructor.
"""

from __future__ import annotations

import json

import anthropic
from loguru import logger

from research_agents.agents.research.models import KnowledgeCategory, Paper, PaperAnalysis


class PaperAnalyzer:

    def __init__(
        self,
        client: anthropic.Anthropic,
        system_prompt: str,
        user_template: str,
    ) -> None:
        self._client = client
        self._system_prompt = system_prompt
        self._user_template = user_template

    def paper_analysis(self, paper: Paper) -> PaperAnalysis:
        """Return PaperAnalysis for a single paper."""
        logger.debug("[PaperAnalyzer] analyzing: {}", paper.title[:60])

        content = self._user_template.format(
            title=paper.title,
            authors=", ".join(paper.authors) or "N/A",
            year=paper.year or "N/A",
            abstract=paper.abstract or "No abstract available.",
        )

        response = self._client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": self._system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": content}],
        )

        return self._parse_analysis(response.content[0].text, paper)

    def _parse_analysis(self, raw: str, paper: Paper) -> PaperAnalysis:
        try:
            data = json.loads(raw.strip())
            return PaperAnalysis(
                paper=paper,
                passes_domain_filter=bool(data.get("passes_domain_filter", False)),
                relevance_score=float(data.get("relevance_score", 0.0)),
                category=KnowledgeCategory(
                    data.get("category", KnowledgeCategory.OTHER)
                ),
                summary=data.get("summary", ""),
                key_equation=data.get("key_equation", ""),
                gap_analysis=data.get("gap_analysis", ""),
            )
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.warning("[PaperAnalyzer] parse error for '{}': {}", paper.title[:40], exc)
            return PaperAnalysis(paper=paper, passes_domain_filter=False)
