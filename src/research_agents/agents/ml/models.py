from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class ModelResult(BaseModel):
    model: str
    library: str
    target: str
    features: list[str]
    coefficients: dict[str, float]
    metrics: dict[str, Optional[float]]
    diagnostics: dict[str, Any]
