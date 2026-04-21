"""
FigurePlotter — generates 4 diagnostic plots from a fitted statsmodels GLM result.

Plots:
  coef_plot         — coefficient bar chart with 95% confidence intervals
  residuals         — Pearson residuals vs fitted values
  qq_plot           — QQ plot of Pearson residuals
  feature_importance — absolute coefficient magnitudes (excluding intercept)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy import stats


class FigurePlotter:

    def plot(self, glm_result: Any, figures_dir: Path, fmt: str = "pdf") -> None:
        """Generate all 4 diagnostic figures into figures_dir."""
        figures_dir.mkdir(parents=True, exist_ok=True)

        self._coef_plot(glm_result, figures_dir / f"coef_plot.{fmt}")
        self._residuals_plot(glm_result, figures_dir / f"residuals.{fmt}")
        self._qq_plot(glm_result, figures_dir / f"qq_plot.{fmt}")
        self._feature_importance(glm_result, figures_dir / f"feature_importance.{fmt}")

    # ------------------------------------------------------------------

    def _coef_plot(self, glm_result: Any, path: Path) -> None:
        params = glm_result.params
        try:
            ci = glm_result.conf_int()
            errors_low  = params.values - ci.iloc[:, 0].values
            errors_high = ci.iloc[:, 1].values - params.values
            errors = np.array([errors_low, errors_high])
        except Exception:
            errors = None

        fig, ax = plt.subplots(figsize=(8, max(4, len(params) * 0.4)))
        y_pos = np.arange(len(params))
        ax.barh(y_pos, params.values, xerr=errors, align="center", alpha=0.7, capsize=4)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(params.index.tolist())
        ax.set_xlabel("Coefficient")
        ax.set_title("GLM Coefficients with 95% CI")
        plt.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        logger.info("[FigurePlotter] saved {}", path.name)

    def _residuals_plot(self, glm_result: Any, path: Path) -> None:
        try:
            fitted = glm_result.fittedvalues
            resid  = glm_result.resid_pearson
        except AttributeError as exc:
            logger.warning("[FigurePlotter] cannot generate residuals plot: {}", exc)
            return

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(fitted, resid, alpha=0.5, s=20)
        ax.axhline(0, color="red", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Pearson residuals")
        ax.set_title("Residuals vs Fitted")
        plt.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        logger.info("[FigurePlotter] saved {}", path.name)

    def _qq_plot(self, glm_result: Any, path: Path) -> None:
        try:
            resid = glm_result.resid_pearson
        except AttributeError as exc:
            logger.warning("[FigurePlotter] cannot generate QQ plot: {}", exc)
            return

        fig, ax = plt.subplots(figsize=(6, 6))
        (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist="norm")
        ax.scatter(osm, osr, alpha=0.5, s=20)
        x_line = np.array([min(osm), max(osm)])
        ax.plot(x_line, slope * x_line + intercept, color="red", linewidth=1)
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Sample quantiles")
        ax.set_title("QQ Plot of Pearson Residuals")
        plt.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        logger.info("[FigurePlotter] saved {}", path.name)

    def _feature_importance(self, glm_result: Any, path: Path) -> None:
        params = glm_result.params.drop("Intercept", errors="ignore")
        importance = params.abs().sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(8, max(4, len(importance) * 0.4)))
        y_pos = np.arange(len(importance))
        ax.barh(y_pos, importance.values, align="center", alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(importance.index.tolist())
        ax.set_xlabel("|Coefficient|")
        ax.set_title("Feature Importance (Absolute Coefficients)")
        plt.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        logger.info("[FigurePlotter] saved {}", path.name)
