"""
AgentConfig — loads, validates, and merges agent_config.yaml with defaults.
Uses Pydantic models for validation.

Naming convention: function name = name of the returned object.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Section models
# ---------------------------------------------------------------------------

class ResearchConfig(BaseModel):
    sources: list[Literal["arxiv", "semantic_scholar", "pubmed", "mdpi", "elibrary"]] = [
        "arxiv", "semantic_scholar", "mdpi"
    ]
    max_papers: int = Field(default=30, ge=1)
    citation_format: Literal["APA", "IEEE", "GOST"] = "APA"
    language: str = "en"


class ExtractionRule(BaseModel):
    name: str
    type: Literal["numeric", "categorical", "text", "boolean"]
    description: str
    unit: Optional[str] = None


class OutputColumn(BaseModel):
    name: str
    type: Literal["numeric", "categorical", "boolean"]
    unit: Optional[str] = None


class CalculationRule(BaseModel):
    name: str
    standard: str
    description: str
    formula: str
    output_columns: list[OutputColumn]
    parameter_ranges: dict[str, list[float]]


class DataConfig(BaseModel):
    output_format: Literal["csv", "json"] = "csv"
    extraction_rules: list[ExtractionRule] = []
    calculations: list[CalculationRule] = []
    user_data: Optional[str] = None


class MLConfig(BaseModel):
    library: Optional[str] = None
    model: str = "GLM"
    target_variable: Optional[str] = None
    features: list[str] = []
    hyperparameters: dict[str, Any] = {}


class FiguresConfig(BaseModel):
    format: Literal["pdf", "png"] = "pdf"
    dpi: int = Field(default=300, ge=72, le=600)


class ReportConfig(BaseModel):
    template: Optional[str] = None
    sections: list[
        Literal["abstract", "introduction", "methods", "results", "discussion", "conclusion"]
    ] = ["abstract", "introduction", "methods", "results", "discussion", "conclusion"]
    figures: FiguresConfig = Field(default_factory=FiguresConfig)

    @field_validator("template")
    @classmethod
    def template_must_exist(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not Path(v).exists():
            raise ValueError(f"LaTeX template not found: {v}")
        return v


class AgentConfig(BaseModel):
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ConfigError(Exception):
    pass


# ---------------------------------------------------------------------------
# Public functions  (name = returned object)
# ---------------------------------------------------------------------------

def agent_config(path: Optional[str | Path] = None) -> AgentConfig:
    """Return AgentConfig merged from YAML file and defaults.
    If path is None, returns default AgentConfig."""
    if path is None:
        return AgentConfig()
    raw = _yaml_dict(Path(path))
    try:
        return AgentConfig.model_validate(raw)
    except Exception as exc:
        raise ConfigError(f"Invalid configuration in {path}:\n{exc}") from exc


def research_goal(path: str | Path) -> str:
    """Return research goal text read from file."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ConfigError(f"Goal file is empty: {path}")
    return text


# ---------------------------------------------------------------------------
# Private helpers  (name = returned object)
# ---------------------------------------------------------------------------

def _yaml_dict(path: Path) -> dict:
    """Return raw dict parsed from a YAML file."""
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"Config file must be a YAML mapping: {path}")
    return data
