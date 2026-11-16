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


class GearParameters(BaseModel):
    """Открытая зубчатая передача «шестерня — венец ОПУ» механизма поворота."""
    pinion_teeth: int = Field(gt=0)                 # z1
    ring_teeth: int = Field(gt=0)                   # z2
    module_mm: float = Field(gt=0)                  # m, мм
    pressure_angle_deg: float = 20.0                # α_w, град
    efficiency: float = Field(default=0.95, gt=0, le=1)     # η_зп
    drives: int = Field(default=1, ge=1)                    # n_пр
    load_sharing: float = Field(default=0.95, gt=0, le=1)   # η_н
    duty_factor: float = Field(default=1.0, gt=0)           # γ_m по FEM 1.001


class CraneParameters(BaseModel):
    """Исходные данные одного крана для расчёта нагрузок на ОПУ."""
    name: str
    crane_type: str                                  # M, T, P
    description: str = ""
    boom_mass_t: float = Field(gt=0)                 # масса стрелы, т
    reach_m: float = Field(gt=0)                     # вылет R, м
    boom_cog_m: Optional[float] = None               # плечо ЦМ стрелы, м (по умолчанию R/2)
    rotating_mass_t: float = Field(default=0.0, ge=0)    # масса поворотной части без стрелы, т
    rotating_cog_m: float = 0.0                      # плечо ЦМ поворотной части, м (< 0 — за осью)
    cog_height_m: float = Field(default=0.0, ge=0)   # высота общего ЦМ над плоскостью ОПУ, м
    boom_wind_area_m2: float = Field(default=0.0, ge=0)
    load_wind_area_m2: float = Field(default=0.0, ge=0)
    wind_height_m: float = Field(default=0.0, ge=0)  # высота приложения ветровой силы, м
    slew_speed_rpm: float = Field(gt=0)              # частота вращения крана, об/мин
    slew_start_time_s: float = Field(gt=0)           # время пуска механизма поворота, с
    bearing_track_diameter_m: float = Field(gt=0)    # D_L
    bearing_friction: float = Field(gt=0)            # μ
    gear: GearParameters
    cargo_t: list[float] = Field(min_length=1)       # массы груза для расчётной сетки, т


class LoadCase(BaseModel):
    """Расчётный случай: коэффициент динамичности, ветер, крен."""
    name: str
    description: str = ""
    psi: float = Field(ge=1.0)                       # ψ
    wind_pressure_pa: float = Field(default=0.0, ge=0)
    heel_deg: float = 0.0                            # угол наклона основания крана (крен), град


class OpuLoadRule(BaseModel):
    """Набор расчётов нагрузок на ОПУ: краны × грузы × расчётные случаи × углы поворота."""
    name: str
    cranes: list[CraneParameters] = Field(min_length=1)
    load_cases: list[LoadCase] = Field(min_length=1)
    slew_angles_deg: list[float] = [0.0, 45.0, 90.0]    # угол поворота стрелы от ДП судна


class DataConfig(BaseModel):
    output_format: Literal["csv", "json"] = "csv"
    extraction_rules: list[ExtractionRule] = []
    calculations: list[CalculationRule] = []
    opu_load_calculations: list[OpuLoadRule] = []
    user_data: Optional[str] = None


class MLConfig(BaseModel):
    library: Optional[str] = None
    model: str = "GLM"
    model_name: Optional[str] = None                 # имя модели в outboxml (по умолчанию model)
    wrapper: str = "glm"                             # glm | glm_without_scaler
    target_variable: Optional[str] = None
    features: list[str] = []
    feature_encodings: dict[str, Optional[str]] = {}  # признак → кодирование (WoE_num_to_cat …)
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
