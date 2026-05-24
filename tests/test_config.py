"""
Unit tests for config loading and validation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from research_agents.config import (
    AgentConfig,
    ConfigError,
    DataConfig,
    MLConfig,
    ResearchConfig,
    agent_config,
    research_goal,
)


class TestResearchGoal:
    """Test research_goal() function."""

    def test_reads_valid_goal(self, tmp_path: Path) -> None:
        goal_file = tmp_path / "research_goal.txt"
        goal_text = "This is a test research goal with enough words to be valid."
        goal_file.write_text(goal_text, encoding="utf-8")

        result = research_goal(str(goal_file))
        assert result == goal_text

    def test_handles_unicode_text(self, tmp_path: Path) -> None:
        goal_file = tmp_path / "research_goal.txt"
        goal_text = "Исследование износа зубчатых передач опорно-поворотного устройства с минимум пятьдесят слов в цели."
        goal_file.write_text(goal_text, encoding="utf-8")

        result = research_goal(str(goal_file))
        assert result == goal_text

    def test_nonexistent_file_raises_config_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            research_goal(str(tmp_path / "nonexistent.txt"))

    def test_empty_file_raises_config_error(self, tmp_path: Path) -> None:
        goal_file = tmp_path / "research_goal.txt"
        goal_file.write_text("", encoding="utf-8")

        with pytest.raises(ConfigError, match="empty"):
            research_goal(str(goal_file))


class TestResearchConfig:
    """Test ResearchConfig pydantic model."""

    def test_create_with_defaults(self) -> None:
        cfg = ResearchConfig()
        assert cfg.sources == ["arxiv"]
        assert cfg.max_papers == 25
        assert cfg.citation_format == "APA"
        assert cfg.language == "en"

    def test_create_with_custom_sources(self) -> None:
        cfg = ResearchConfig(sources=["arxiv", "semantic_scholar"])
        assert cfg.sources == ["arxiv", "semantic_scholar"]

    def test_validate_max_papers_range(self) -> None:
        cfg = ResearchConfig(max_papers=50)
        assert cfg.max_papers == 50

        cfg = ResearchConfig(max_papers=1)
        assert cfg.max_papers == 1


class TestDataConfig:
    """Test DataConfig pydantic model."""

    def test_create_empty(self) -> None:
        cfg = DataConfig()
        assert cfg.output_format == "csv"
        assert cfg.extraction_rules == []
        assert cfg.calculations == []
        assert cfg.engineering_calculations == []
        assert cfg.user_data is None

    def test_create_with_extraction_rules(self) -> None:
        cfg = DataConfig(
            extraction_rules=[
                {
                    "name": "Parameter 1",
                    "type": "numeric",
                    "description": "Test parameter",
                    "unit": "kg",
                }
            ]
        )
        assert len(cfg.extraction_rules) == 1
        assert cfg.extraction_rules[0]["name"] == "Parameter 1"


class TestMLConfig:
    """Test MLConfig pydantic model."""

    def test_create_with_target(self) -> None:
        cfg = MLConfig(target_variable="M_кр", features=["F1", "F2"])
        assert cfg.model == "GLM"
        assert cfg.target_variable == "M_кр"
        assert cfg.features == ["F1", "F2"]

    def test_hyperparameters_default_empty(self) -> None:
        cfg = MLConfig(target_variable="target")
        assert cfg.hyperparameters == {}

    def test_hyperparameters_custom(self) -> None:
        cfg = MLConfig(
            target_variable="target",
            hyperparameters={"objective": "gamma"},
        )
        assert cfg.hyperparameters == {"objective": "gamma"}


class TestAgentConfig:
    """Test AgentConfig pydantic model."""

    def test_create_full_config(self) -> None:
        cfg = AgentConfig(
            research=ResearchConfig(sources=["arxiv"], max_papers=10),
            data=DataConfig(output_format="csv"),
            ml=MLConfig(target_variable="target"),
        )
        assert cfg.research.max_papers == 10
        assert cfg.data.output_format == "csv"
        assert cfg.ml.target_variable == "target"

    def test_create_with_defaults(self) -> None:
        cfg = AgentConfig()
        assert isinstance(cfg.research, ResearchConfig)
        assert isinstance(cfg.data, DataConfig)
        assert isinstance(cfg.ml, MLConfig)


class TestAgentConfigLoader:
    """Test agent_config() function."""

    def test_loads_yaml_config(self, tmp_path: Path) -> None:
        config_file = tmp_path / "agent_config.yaml"
        config_file.write_text(
            """
research:
  sources: [arxiv, semantic_scholar]
  max_papers: 20

data:
  output_format: csv

ml:
  model: GLM
  target_variable: M_кр
  features: [F1, F2]
""",
            encoding="utf-8",
        )

        cfg = agent_config(str(config_file))
        assert cfg.research.sources == ["arxiv", "semantic_scholar"]
        assert cfg.research.max_papers == 20
        assert cfg.ml.target_variable == "M_кр"

    def test_default_config_if_none_provided(self) -> None:
        cfg = agent_config(None)
        assert isinstance(cfg, AgentConfig)
        assert cfg.research.max_papers == 25

    def test_nonexistent_file_raises_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            agent_config(str(tmp_path / "nonexistent.yaml"))

    def test_invalid_yaml_raises_error(self, tmp_path: Path) -> None:
        config_file = tmp_path / "agent_config.yaml"
        config_file.write_text("invalid: yaml: content:", encoding="utf-8")

        with pytest.raises(ConfigError, match="YAML"):
            agent_config(str(config_file))
