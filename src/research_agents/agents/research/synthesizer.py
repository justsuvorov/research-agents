"""
Synthesizer — groups PaperAnalyses by category and writes section texts via LLM.
All prompt text is injected via constructor.
"""

from __future__ import annotations

import google.generativeai as genai
from loguru import logger

from research_agents.agents.research.models import (
    KnowledgeCategory,
    LiteratureReport,
    PaperAnalysis,
)


class Synthesizer:

    def __init__(
        self,
        client: genai.GenerativeModel,
        system_prompt: str,
        user_template: str,
    ) -> None:
        self._client = client
        self._system_prompt = system_prompt
        self._user_template = user_template

    def section_text(self, goal: str, category: KnowledgeCategory, analyses: list[PaperAnalysis]) -> str:
        """Return LLM-generated review section text for one category."""
        if not analyses:
            return ""

        sources_block = "\n\n".join(
            f"[{i+1}] {a.paper.title} ({a.paper.year})\n"
            f"Резюме: {a.summary}\n"
            f"Ключевое уравнение: {a.key_equation or '—'}\n"
            f"Gap: {a.gap_analysis}"
            for i, a in enumerate(analyses)
        )

        user_content = self._user_template.format(
            goal=goal,
            category=category.value,
            sources_block=sources_block,
        )

        response = self._client.generate_content(
            contents=[
                f"{self._system_prompt}\n\n{user_content}"
            ],
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=2048,
                temperature=0.7,
            ),
        )
        return response.text.strip()

    def literature_review_sections(self, report: LiteratureReport) -> dict[KnowledgeCategory, str]:
        """Return dict of category → section text for all categories with relevant papers."""
        sections: dict[KnowledgeCategory, str] = {}
        by_cat = report.by_category()

        for category, analyses in by_cat.items():
            relevant = [a for a in analyses if a.passes_domain_filter]
            if not relevant:
                continue
            logger.info("[Synthesizer] writing section '{}' ({} papers)", category.value, len(relevant))
            sections[category] = self.section_text(report.goal, category, relevant)

        return sections
