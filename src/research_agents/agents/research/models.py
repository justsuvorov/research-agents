"""
Pydantic models for ResearchAgent.
Naming convention: class name = the concept it represents.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class KnowledgeCategory(str, Enum):
    WEAR_THEORY   = "Теория износа"
    CRANE_DYNAMICS = "Динамика кранов"
    PKM_MACHINES  = "Машины и механизмы (ПКМ)"
    OTHER         = "Прочее"


class Paper(BaseModel):
    """A single academic paper retrieved from a search source."""
    title: str
    authors: list[str] = []
    year: Optional[int] = None
    doi: Optional[str] = None
    abstract: str = ""
    source: str = ""                    # arxiv | semantic_scholar | scopus | ...
    url: Optional[str] = None
    keywords: list[str] = []
    bibtex_key: str = ""                # generated on deduplication


class PaperAnalysis(BaseModel):
    """LLM-generated analysis of a single paper."""
    paper: Paper
    summary: str                        # why important for the dissertation
    key_equation: str                   # latex formula to integrate into the model
    gap_analysis: str                   # what is missing → scientific novelty
    category: KnowledgeCategory = KnowledgeCategory.OTHER
    relevance_score: float = Field(ge=0.0, le=1.0, default=0.0)
    passes_domain_filter: bool = False  # passed all domain constraints


class LiteratureReport(BaseModel):
    """Full output of ResearchAgent — ready for export."""
    goal: str
    analyses: list[PaperAnalysis] = []

    def by_category(self) -> dict[KnowledgeCategory, list[PaperAnalysis]]:
        """Return analyses grouped by KnowledgeCategory."""
        groups: dict[KnowledgeCategory, list[PaperAnalysis]] = {c: [] for c in KnowledgeCategory}
        for a in self.analyses:
            groups[a.category].append(a)
        return groups

    def relevant(self) -> list[PaperAnalysis]:
        """Return only analyses that passed domain filter."""
        return [a for a in self.analyses if a.passes_domain_filter]
