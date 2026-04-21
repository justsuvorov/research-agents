"""
Integration tests for MLAgent using the California housing dataset
(mirrors the outboxml housing_pricing example).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import fetch_california_housing

from research_agents.agents.ml.errors import ModelFitError
from research_agents.agents.ml.figure_plotter import FigurePlotter
from research_agents.agents.ml.model_runner import ModelRunner
from research_agents.agents.ml.result_exporter import ResultExporter
from research_agents.agents.ml_agent import MLAgent
from research_agents.config import MLConfig
from research_agents.pydantic_models import RunContext

TARGET = "MedHouseVal"
FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup"]
SAMPLE_SIZE = 2000


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def housing_df() -> pd.DataFrame:
    housing = fetch_california_housing(as_frame=True)
    df = pd.concat([housing["data"], housing["target"]], axis=1)
    return df[FEATURES + [TARGET]].dropna().sample(SAMPLE_SIZE, random_state=42)


@pytest.fixture(scope="module")
def ml_config() -> MLConfig:
    return MLConfig(
        model="glm",
        target_variable=TARGET,
        features=FEATURES,
        hyperparameters={"objective": "gamma"},
    )


@pytest.fixture(scope="module")
def fitted_model(
    housing_df: pd.DataFrame,
    ml_config: MLConfig,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple:
    """Train once per test session; all tests share the result."""
    configs_dir = tmp_path_factory.mktemp("ml_configs")
    runner = ModelRunner(cfg=ml_config)
    return runner.fit(housing_df, configs_dir)


@pytest.fixture()
def run_context(
    tmp_path: Path,
    housing_df: pd.DataFrame,
    ml_config: MLConfig,
) -> RunContext:
    dataset_path = tmp_path / "dataset.csv"
    housing_df.to_csv(dataset_path, index=False)
    ctx = RunContext(
        goal="Predict California house prices",
        config={"ml": ml_config.model_dump(), "report": {"figures": {"format": "pdf"}}},
        output_dir=str(tmp_path),
    )
    ctx.set_artifact("dataset", str(dataset_path))
    return ctx


@pytest.fixture()
def ml_agent(run_context: RunContext, ml_config: MLConfig) -> MLAgent:
    return MLAgent(
        ctx=run_context,
        model_runner=ModelRunner(cfg=ml_config),
        figure_plotter=FigurePlotter(),
        result_exporter=ResultExporter(),
    )


# ---------------------------------------------------------------------------
# ModelRunner
# ---------------------------------------------------------------------------

class TestModelRunner:

    def test_fit_returns_correct_metadata(self, fitted_model: tuple) -> None:
        model_result, _ = fitted_model
        assert model_result.target == TARGET
        assert model_result.library == "outboxml"
        assert model_result.model == "glm"
        assert len(model_result.features) > 0

    def test_coefficients_non_empty_and_typed(self, fitted_model: tuple) -> None:
        model_result, _ = fitted_model
        assert len(model_result.coefficients) > 0
        assert all(isinstance(v, float) for v in model_result.coefficients.values())

    def test_coefficients_no_nan(self, fitted_model: tuple) -> None:
        model_result, _ = fitted_model
        for name, value in model_result.coefficients.items():
            assert not math.isnan(value), f"NaN coefficient for '{name}'"

    def test_metrics_keys_present(self, fitted_model: tuple) -> None:
        model_result, _ = fitted_model
        assert set(model_result.metrics) >= {"aic", "bic", "deviance", "r2"}

    def test_diagnostics_keys_present(self, fitted_model: tuple) -> None:
        model_result, _ = fitted_model
        assert "p_values" in model_result.diagnostics
        assert "confidence_intervals" in model_result.diagnostics
        assert "residuals_normality_p" in model_result.diagnostics

    def test_missing_target_raises(
        self, housing_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        cfg = MLConfig(model="glm", target_variable=None, features=FEATURES)
        runner = ModelRunner(cfg=cfg)
        with pytest.raises(ModelFitError, match="target_variable"):
            runner.fit(housing_df, tmp_path / "ml_configs")

    def test_config_json_files_created(
        self, housing_df: pd.DataFrame, ml_config: MLConfig, tmp_path: Path
    ) -> None:
        configs_dir = tmp_path / "ml_configs"
        ModelRunner(cfg=ml_config).fit(housing_df, configs_dir)
        assert (configs_dir / "models_config.json").exists()
        assert (configs_dir / "auto_ml_config.json").exists()

    def test_reserved_hyperparameter_keys_ignored(
        self, housing_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """wrapper/name in hyperparameters must not override internal model setup."""
        cfg = MLConfig(
            model="glm",
            target_variable=TARGET,
            features=FEATURES,
            hyperparameters={"objective": "gamma", "wrapper": "catboost", "name": "bad"},
        )
        model_result, _ = ModelRunner(cfg=cfg).fit(housing_df, tmp_path / "ml_configs")
        assert model_result.model == "glm"


# ---------------------------------------------------------------------------
# FigurePlotter
# ---------------------------------------------------------------------------

class TestFigurePlotter:

    def test_all_four_figures_created(self, fitted_model: tuple, tmp_path: Path) -> None:
        _, glm_result = fitted_model
        FigurePlotter().plot(glm_result, tmp_path / "figures", fmt="pdf")
        figures_dir = tmp_path / "figures"
        for name in ("coef_plot.pdf", "residuals.pdf", "qq_plot.pdf", "feature_importance.pdf"):
            assert (figures_dir / name).exists(), f"Missing figure: {name}"

    def test_figures_are_non_empty(self, fitted_model: tuple, tmp_path: Path) -> None:
        _, glm_result = fitted_model
        FigurePlotter().plot(glm_result, tmp_path / "figures", fmt="pdf")
        for path in (tmp_path / "figures").iterdir():
            assert path.stat().st_size > 0, f"Empty figure: {path.name}"


# ---------------------------------------------------------------------------
# ResultExporter
# ---------------------------------------------------------------------------

class TestResultExporter:

    def test_model_results_json_created(self, fitted_model: tuple, tmp_path: Path) -> None:
        model_result, _ = fitted_model
        path = ResultExporter().model_results_path(model_result, tmp_path)
        assert path.exists()
        assert path.name == "model_results.json"

    def test_json_has_required_keys(self, fitted_model: tuple, tmp_path: Path) -> None:
        model_result, _ = fitted_model
        path = ResultExporter().model_results_path(model_result, tmp_path)
        data = json.loads(path.read_text())
        for key in ("model", "library", "target", "features", "coefficients", "metrics", "diagnostics"):
            assert key in data, f"Missing key: {key}"

    def test_json_metrics_no_nan(self, fitted_model: tuple, tmp_path: Path) -> None:
        model_result, _ = fitted_model
        path = ResultExporter().model_results_path(model_result, tmp_path)
        data = json.loads(path.read_text())
        for k, v in data["metrics"].items():
            if v is not None:
                assert not math.isnan(v), f"NaN in metrics['{k}']"


# ---------------------------------------------------------------------------
# MLAgent (full integration)
# ---------------------------------------------------------------------------

class TestMLAgent:

    def test_artifacts_set_after_execute(self, ml_agent: MLAgent, run_context: RunContext) -> None:
        ml_agent.execute()
        assert run_context.artifact_path("model_results") is not None
        assert run_context.artifact_path("figures_dir") is not None

    def test_status_completed(self, ml_agent: MLAgent, run_context: RunContext) -> None:
        ml_agent.execute()
        assert run_context.agent_status.ml == "completed"

    def test_idempotent(self, ml_agent: MLAgent, run_context: RunContext) -> None:
        """Second execute() must skip without re-training."""
        ml_agent.execute()
        first_path = run_context.artifact_path("model_results")
        ml_agent.execute()
        assert run_context.artifact_path("model_results") == first_path

    def test_model_results_file_exists(self, ml_agent: MLAgent, run_context: RunContext) -> None:
        ml_agent.execute()
        results_path = run_context.artifact_path("model_results")
        assert results_path is not None and results_path.exists()

    def test_four_figures_exist(self, ml_agent: MLAgent, run_context: RunContext) -> None:
        ml_agent.execute()
        figures_dir = run_context.artifact_path("figures_dir")
        assert figures_dir is not None
        pdfs = list(figures_dir.glob("*.pdf"))
        assert len(pdfs) == 4, f"Expected 4 PDF figures, got {len(pdfs)}: {[p.name for p in pdfs]}"
