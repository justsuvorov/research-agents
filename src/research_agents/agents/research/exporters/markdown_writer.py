"""
MarkdownWriter — renders LiteratureReport to a structured Markdown file.
"""

from __future__ import annotations

from pathlib import Path

from research_agents.agents.research.models import KnowledgeCategory, LiteratureReport


def markdown_review(
    report: LiteratureReport,
    sections: dict[KnowledgeCategory, str],
    path: str | Path,
) -> None:
    """Write full literature review Markdown to path."""
    lines: list[str] = [
        f"# Literature Review",
        f"",
        f"**Research Goal:** {report.goal}",
        f"",
        f"---",
        f"",
    ]

    for category, text in sections.items():
        if not text:
            continue
        lines += [
            f"## {category.value}",
            f"",
            text,
            f"",
        ]

    lines += ["---", "", "## References", ""]
    for analysis in report.relevant():
        p = analysis.paper
        ref_num = p.bibtex_key or "?"
        doi_part = f" DOI: {p.doi}." if p.doi else ""
        authors = "; ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
        lines.append(
            f"[{ref_num}] {authors} ({p.year}). *{p.title}*.{doi_part}"
        )

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
