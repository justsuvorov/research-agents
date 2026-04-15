"""
BibTeX exporter — converts a list of Paper objects to a .bib file.
"""

from __future__ import annotations

import re
from pathlib import Path

from research_agents.agents.research.models import Paper


def bibtex_key(paper: Paper, index: int) -> str:
    """Return a unique BibTeX key for a paper."""
    first_author = (paper.authors[0].split()[-1] if paper.authors else "Unknown").lower()
    first_author = re.sub(r"[^a-z]", "", first_author)
    year = paper.year or "0000"
    return f"{first_author}{year}_{index}"


def bibtex_entry(paper: Paper, key: str) -> str:
    """Return a BibTeX @article entry string."""
    authors_str = " and ".join(paper.authors) if paper.authors else "Unknown"
    lines = [
        f"@article{{{key},",
        f"  author  = {{{authors_str}}},",
        f"  title   = {{{paper.title}}},",
        f"  year    = {{{paper.year or ''}}},",
    ]
    if paper.doi:
        lines.append(f"  doi     = {{{paper.doi}}},")
    if paper.url:
        lines.append(f"  url     = {{{paper.url}}},")
    lines.append("}")
    return "\n".join(lines)


def bib_file(papers: list[Paper], path: str | Path) -> None:
    """Write all papers as BibTeX entries to path."""
    entries = []
    for i, paper in enumerate(papers):
        key = bibtex_key(paper, i + 1)
        paper.bibtex_key = key
        entries.append(bibtex_entry(paper, key))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n\n".join(entries), encoding="utf-8")
