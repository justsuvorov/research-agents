"""
StandardsCalculator — computes rows by applying engineering formulas
from RMRS / GOST / ISO standards via LLM over a parameter grid.
All prompt text is injected via constructor.
"""

from __future__ import annotations

import itertools
import json

import anthropic
from loguru import logger

from research_agents.config import CalculationRule


class StandardsCalculator:

    def __init__(
        self,
        client: anthropic.Anthropic,
        system_prompt: str,
        user_template: str,
    ) -> None:
        self._client = client
        self._system_prompt = system_prompt
        self._user_template = user_template

    def calculated_rows(self, rule: CalculationRule) -> list[dict]:
        """Return rows computed by applying rule.formula to the parameter grid."""
        grid = self._parameter_grid(rule)
        if not grid:
            logger.warning("[StandardsCalculator] empty parameter grid for '{}'", rule.name)
            return []

        logger.info(
            "[StandardsCalculator] calculating '{}' ({}) — {} combinations",
            rule.name, rule.standard, len(grid),
        )

        output_columns_block = "\n".join(
            f"- {col.name} ({col.type}){', ' + col.unit if col.unit else ''}"
            for col in rule.output_columns
        )

        # Build result schema example from first grid row + output cols
        example_row = {**grid[0], **{col.name: "<float or null>" for col in rule.output_columns}}
        result_schema_example = json.dumps([example_row], ensure_ascii=False, indent=2)

        content = self._user_template.format(
            name=rule.name,
            standard=rule.standard,
            description=rule.description,
            formula=rule.formula,
            output_columns_block=output_columns_block,
            parameter_grid_json=json.dumps(grid, ensure_ascii=False),
            result_schema_example=result_schema_example,
        )

        response = self._client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": self._system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": content}],
        )

        return self._parse_rows(response.content[0].text, rule)

    def _parameter_grid(self, rule: CalculationRule) -> list[dict]:
        """Return Cartesian product of all parameter ranges as list of dicts."""
        keys = list(rule.parameter_ranges.keys())
        values = list(rule.parameter_ranges.values())
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def _parse_rows(self, raw: str, rule: CalculationRule) -> list[dict]:
        try:
            rows = json.loads(raw.strip())
            if not isinstance(rows, list):
                raise ValueError("expected JSON array")
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("[StandardsCalculator] parse error for '{}': {}", rule.name, exc)
            return []

        result = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            row["source"] = rule.standard
            row["source_type"] = "calculation"
            result.append(row)

        logger.info("[StandardsCalculator] '{}' → {} rows", rule.name, len(result))
        return result
