"""
EngineeringCalculator — applies chained engineering calculations from published
standards (DNV, FEM, ISO, RMRS) to a parameter grid via LLM.

The LLM acts as the formula engine: it knows the standard calculation procedures
and executes the full chain (with intermediate variables) for each parameter
combination.  The user specifies mechanism, standards, and input ranges — not
the formulas themselves.

All dependencies are injected via constructor.
"""

from __future__ import annotations

import itertools
import json

import anthropic
from loguru import logger

from research_agents.config import EngineeringCalculationRule


class EngineeringCalculator:

    def __init__(
        self,
        client: anthropic.Anthropic,
        system_prompt: str,
        user_template: str,
        batch_size: int = 20,
    ) -> None:
        self._client = client
        self._system_prompt = system_prompt
        self._user_template = user_template
        self._batch_size = batch_size

    def calculated_rows(self, rule: EngineeringCalculationRule) -> list[dict]:
        """Return rows computed by applying rule's standard calculation chain."""
        grid = self._parameter_grid(rule)
        if not grid:
            logger.warning(
                "[EngineeringCalculator] empty parameter grid for '{}'", rule.name
            )
            return []

        logger.info(
            "[EngineeringCalculator] '{}' — {} combinations, standards: {}",
            rule.name,
            len(grid),
            ", ".join(rule.standards),
        )

        batches = [
            grid[i : i + self._batch_size]
            for i in range(0, len(grid), self._batch_size)
        ]

        rows: list[dict] = []
        for idx, batch in enumerate(batches, start=1):
            logger.debug(
                "[EngineeringCalculator] '{}' batch {}/{} ({} rows)",
                rule.name, idx, len(batches), len(batch),
            )
            rows.extend(self._call_llm(rule, batch))

        logger.info(
            "[EngineeringCalculator] '{}' → {} rows total", rule.name, len(rows)
        )
        return rows

    # -------------------------------------------------------------------------

    def _parameter_grid(self, rule: EngineeringCalculationRule) -> list[dict]:
        """Return Cartesian product of all input parameter ranges as list of dicts."""
        keys = list(rule.input_parameters.keys())
        values = list(rule.input_parameters.values())
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def _call_llm(
        self, rule: EngineeringCalculationRule, batch: list[dict]
    ) -> list[dict]:
        output_columns_block = "\n".join(
            f"- {col.name} ({col.type}){', ' + col.unit if col.unit else ''}"
            for col in rule.output_columns
        )

        example_row = {
            **batch[0],
            **{col.name: "<float or null>" for col in rule.output_columns},
            "_steps": {"<intermediate_var>": "<value>"},
        }
        result_schema_example = json.dumps([example_row], ensure_ascii=False, indent=2)

        content = self._user_template.format(
            mechanism=rule.mechanism,
            standards=", ".join(rule.standards),
            description=rule.description,
            n_combinations=len(batch),
            parameter_grid_json=json.dumps(batch, ensure_ascii=False),
            output_columns_block=output_columns_block,
            result_schema_example=result_schema_example,
        )

        try:
            response = self._client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=8192,
                system=[
                    {
                        "type": "text",
                        "text": self._system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": content}],
            )
        except anthropic.APIError as exc:
            logger.warning(
                "[EngineeringCalculator] API error for '{}': {}", rule.name, exc
            )
            return []

        return self._parse_rows(response.content[0].text, rule, len(batch))

    def _parse_rows(
        self, raw: str, rule: EngineeringCalculationRule, expected: int
    ) -> list[dict]:
        text = raw.strip()
        # Strip markdown fences if LLM added them despite instructions
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            text = text.rsplit("```", 1)[0]

        try:
            rows = json.loads(text)
            if not isinstance(rows, list):
                raise ValueError("expected JSON array")
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "[EngineeringCalculator] parse error for '{}': {}", rule.name, exc
            )
            return []

        if len(rows) != expected:
            logger.warning(
                "[EngineeringCalculator] '{}' expected {} rows, got {}",
                rule.name, expected, len(rows),
            )

        result: list[dict] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if "_error" in row:
                logger.debug(
                    "[EngineeringCalculator] '{}' invalid combination: {}",
                    rule.name, row["_error"],
                )
            row["source"] = rule.name
            row["source_type"] = "engineering_calculation"
            result.append(row)

        return result
