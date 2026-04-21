"""
DatasetAssembler — outer-joins rows from all sources into a single DataFrame
and writes dataset.csv + dataset_metadata.json.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

from research_agents.config import DataConfig


class EmptyDatasetError(Exception):
    pass


class DatasetAssembler:

    def assembled_dataset(
        self,
        paper_rows: list[dict],
        calculation_rows: list[dict],
        user_rows: list[dict],
        cfg: DataConfig,
        output_dir: Path,
    ) -> tuple[Path, Path]:
        """
        Outer-join all rows, write dataset + metadata.
        Returns (dataset_path, metadata_path).
        """
        frames: list[pd.DataFrame] = []

        if paper_rows:
            frames.append(pd.DataFrame(paper_rows))
            logger.info("[DatasetAssembler] paper rows: {}", len(paper_rows))

        if calculation_rows:
            frames.append(pd.DataFrame(calculation_rows))
            logger.info("[DatasetAssembler] calculation rows: {}", len(calculation_rows))

        if user_rows:
            frames.append(pd.DataFrame(user_rows))
            logger.info("[DatasetAssembler] user rows: {}", len(user_rows))

        if not frames:
            raise EmptyDatasetError("All sources returned 0 rows.")

        df = pd.concat(frames, join="outer", ignore_index=True)
        logger.info("[DatasetAssembler] total: {} rows × {} cols", len(df), len(df.columns))

        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_path  = output_dir / f"dataset.{cfg.output_format}"
        metadata_path = output_dir / "dataset_metadata.json"

        if cfg.output_format == "csv":
            df.to_csv(dataset_path, index=False, encoding="utf-8")
        else:
            df.to_json(dataset_path, orient="records", force_ascii=False, indent=2)

        metadata_path.write_text(
            json.dumps(self._metadata(df, paper_rows, calculation_rows, user_rows), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return dataset_path, metadata_path

    def _metadata(
        self,
        df: pd.DataFrame,
        paper_rows: list[dict],
        calculation_rows: list[dict],
        user_rows: list[dict],
    ) -> dict:
        columns = []
        for col in df.columns:
            series = df[col]
            info: dict = {
                "name": col,
                "type": str(series.dtype),
                "n_missing": int(series.isna().sum()),
            }
            if pd.api.types.is_numeric_dtype(series):
                info["mean"] = round(float(series.mean()), 4) if series.notna().any() else None
                info["std"]  = round(float(series.std()),  4) if series.notna().any() else None
                info["min"]  = round(float(series.min()),  4) if series.notna().any() else None
                info["max"]  = round(float(series.max()),  4) if series.notna().any() else None
            columns.append(info)

        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "sources": {
                "papers":       len(paper_rows),
                "calculations": len(calculation_rows),
                "user":         len(user_rows),
            },
            "columns": columns,
        }
