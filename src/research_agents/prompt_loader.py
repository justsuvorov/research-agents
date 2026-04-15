"""
PromptLoader — loads prompt templates from the prompts/ directory.
Naming convention: function name = returned object.
"""

from __future__ import annotations

from pathlib import Path


class PromptLoader:
    def __init__(self, prompts_dir: str | Path) -> None:
        self._root = Path(prompts_dir)

    def prompt_text(self, *path_parts: str) -> str:
        """Return prompt text loaded from prompts_dir / path_parts."""
        full_path = self._root.joinpath(*path_parts)
        if not full_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {full_path}")
        return full_path.read_text(encoding="utf-8").strip()
