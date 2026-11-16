"""
ModelRunner — trains a GLM model via outboxml and returns extracted results.

Workflow:
  1. Build two JSON config files (models_config + auto_ml_config) in configs_dir.
     The models config follows the torque model concept: GLM (wrapper from MLConfig),
     numerical features, optional per-feature encoding (e.g. WoE_num_to_cat binning).
  2. Create AutoMLManager with a DataFrameExtractor wrapper.
  3. Call update_models() to train.
  4. Extract statsmodels GLM result via get_result()[model_name].model.model.
  5. Return (ModelResult, raw_glm_result) for downstream export and plotting.
  6. predictions() returns the input table with model values and errors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from outboxml.automl_manager import AutoMLManager
from outboxml.automl_utils import build_default_auto_ml_config
from outboxml.extractors import Extractor
from scipy.stats import normaltest

from research_agents.agents.ml.errors import LibraryImportError, ModelFitError
from research_agents.agents.ml.models import ModelResult
from research_agents.config import MLConfig

_RESERVED_MODEL_PARAMS = frozenset({"wrapper", "name"})


class DataFrameExtractor(Extractor):
    """Thin Extractor wrapper that serves a pre-loaded DataFrame to AutoMLManager."""

    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__()
        self._data = data

    def extract_dataset(self) -> pd.DataFrame:
        return self._data


class ModelRunner:

    def __init__(self, cfg: MLConfig) -> None:
        self._cfg = cfg

    def fit(self, df: pd.DataFrame, configs_dir: Path) -> tuple[ModelResult, Any]:
        """Train GLM via outboxml. Returns (ModelResult, statsmodels GLMResults)."""
        cfg = self._cfg

        if not cfg.target_variable:
            raise ModelFitError("ml.target_variable must be set in agent config")

        target = cfg.target_variable
        non_service = [c for c in df.columns if c not in ("source", "source_type")]
        features = cfg.features or [c for c in non_service if c != target]
        model_name = cfg.model_name or cfg.model.lower()

        missing = [c for c in [*features, target] if c not in df.columns]
        if missing:
            raise ModelFitError(f"columns not found in dataset: {missing}")
        df_model = df[[*features, target]].reset_index(drop=True)

        overrides = {k: v for k, v in cfg.hyperparameters.items() if k not in _RESERVED_MODEL_PARAMS}
        if len(overrides) < len(cfg.hyperparameters):
            ignored = set(cfg.hyperparameters) - set(overrides)
            logger.warning("[ModelRunner] ignoring reserved hyperparameter keys: {}", ignored)

        configs_dir.mkdir(parents=True, exist_ok=True)

        logger.info("[ModelRunner] building outboxml config: target={}, features={}", target, len(features))
        all_models_config = self.all_models_config(model_name, target, features, overrides)

        models_config_path  = configs_dir / "models_config.json"
        auto_ml_config_path = configs_dir / "auto_ml_config.json"

        models_config_path.write_text(
            json.dumps(all_models_config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        auto_ml_cfg = build_default_auto_ml_config()
        auto_ml_config_path.write_text(
            json.dumps(auto_ml_cfg.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info("[ModelRunner] training model '{}' via outboxml", model_name)
        try:
            auto_ml = AutoMLManager(
                auto_ml_config=str(auto_ml_config_path),
                models_config=str(models_config_path),
                extractor=DataFrameExtractor(df_model),
                retro=False,
                hp_tune=False,
            )
            auto_ml.update_models(send_mail=False)
        except Exception as exc:
            raise ModelFitError(f"outboxml training failed: {exc}") from exc

        logger.info("[ModelRunner] extracting GLM result for model '{}'", model_name)
        try:
            ds_result = auto_ml.get_result()[model_name]
            glm_result = ds_result.model.model
        except (KeyError, AttributeError) as exc:
            raise ModelFitError(
                f"Cannot access GLM result for model '{model_name}': {exc}"
            ) from exc

        model_result = self._model_result(
            glm_result, model_name, target, features, self.predictions(ds_result, df_model)
        )
        return model_result, ds_result

    def all_models_config(
        self,
        model_name: str,
        target: str,
        features: list[str],
        model_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Return outboxml AllModelsConfig as dict: one GLM model, numerical features,
        optional encoding per feature from MLConfig.feature_encodings."""
        params = dict(model_params)
        objective = params.pop("objective", "RMSE")
        feature_configs: list[dict[str, Any]] = []
        for name in features:
            feature: dict[str, Any] = {"name": name, "default": 0, "replace": {"_TYPE_": "_NUM_"}}
            encoding = self._cfg.feature_encodings.get(name)
            if encoding:
                feature["encoding"] = encoding
            feature_configs.append(feature)

        model_config: dict[str, Any] = {
            "name": model_name,
            "column_target": target,
            "objective": objective,
            "wrapper": self._cfg.wrapper,
            "relative_features": [],
            "treatment_dict": None,
            "features": feature_configs,
        }
        if params:
            model_config["params_glm"] = params

        return {
            "group_name": model_name,
            "project": model_name,
            "version": "1",
            "data_config": {
                "source": "csv",
                "local_name_source": "",
                "separation": {"kind": "none", "random_state": 42},
                "extra_columns": [],
                "data": {"targetslices": []},
            },
            "models_configs": [model_config],
        }

    def predictions(self, ds_result: Any, df: pd.DataFrame) -> pd.DataFrame:
        """Return df (index-aligned with the training table) with model values and errors."""
        target = self._cfg.target_variable
        parts = [p for p in (ds_result.predictions.get("train"), ds_result.predictions.get("test")) if p is not None]
        if not parts:
            raise ModelFitError("outboxml returned no predictions")
        fitted = pd.concat(parts).squeeze()
        if isinstance(fitted, pd.DataFrame):
            fitted = fitted.iloc[:, 0]

        table = df.reset_index(drop=True).copy()
        if fitted.index.isin(table.index).all():
            table[f"{target}_model"] = fitted.reindex(table.index).to_numpy()
        else:
            table[f"{target}_model"] = fitted.to_numpy()[: len(table)]
        table[f"{target}_error"] = table[f"{target}_model"] - table[target]
        table[f"{target}_error_pct"] = 100.0 * table[f"{target}_error"] / table[target].where(table[target] != 0)
        return table

    # ------------------------------------------------------------------

    def _model_result(
        self,
        glm_result: Any,
        model_name: str,
        target: str,
        features: list[str],
        predictions: pd.DataFrame,
    ) -> ModelResult:
        coefficients = {k: float(v) for k, v in glm_result.params.items()}

        try:
            ci_df = glm_result.conf_int()
            confidence_intervals: dict[str, list[float]] = {
                k: [float(ci_df.at[k, 0]), float(ci_df.at[k, 1])]
                for k in ci_df.index
            }
        except Exception as exc:
            logger.warning("[ModelRunner] could not extract confidence intervals: {}", exc)
            confidence_intervals = {}

        try:
            p_values = {k: float(v) for k, v in glm_result.pvalues.items()}
        except Exception as exc:
            logger.warning("[ModelRunner] could not extract p-values: {}", exc)
            p_values = {}

        try:
            _, norm_p = normaltest(glm_result.resid_pearson)
            residuals_normality_p: float | None = float(norm_p)
        except Exception as exc:
            logger.warning("[ModelRunner] could not compute residuals normality test: {}", exc)
            residuals_normality_p = None

        llf    = self._safe_float(getattr(glm_result, "llf", None))
        llnull = self._safe_float(getattr(glm_result, "llnull", None))
        pseudo_r2 = (1.0 - llf / llnull) if llf is not None and llnull else None

        actual = predictions[target].to_numpy(dtype=float)
        model  = predictions[f"{target}_model"].to_numpy(dtype=float)
        error  = model - actual
        ss_tot = float(((actual - actual.mean()) ** 2).sum())

        metrics: dict[str, float | None] = {
            "aic":      self._safe_float(getattr(glm_result, "aic", None)),
            "bic":      self._safe_float(getattr(glm_result, "bic_llf", None)),
            "deviance": self._safe_float(getattr(glm_result, "deviance", None)),
            "r2":       pseudo_r2,
            "r2_predictions": (1.0 - float((error ** 2).sum()) / ss_tot) if ss_tot else None,
            "mae":      float(abs(error).mean()),
            "rmse":     float((error ** 2).mean() ** 0.5),
        }

        return ModelResult(
            model=model_name,
            library="outboxml",
            target=target,
            features=features,
            coefficients=coefficients,
            metrics=metrics,
            diagnostics={
                "p_values": p_values,
                "confidence_intervals": confidence_intervals,
                "residuals_normality_p": residuals_normality_p,
            },
        )

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None
