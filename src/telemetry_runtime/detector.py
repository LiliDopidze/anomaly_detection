"""Small deterministic detector used as the Week 1 isolation executable.

It is intentionally a baseline, not the final model portfolio. Its purpose in Week 1 is
to exercise the runtime boundary with history-only scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DetectorConfig:
    history_window: int = 48
    minimum_history: int = 12
    threshold: float = 6.0
    scale_floor: float = 1e-9


class RobustHistoryDetector:
    detector_id = "robust-history-v0.1"

    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()

    def _score_group(self, group: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        ordered = group.sort_values("event_ts", kind="stable").copy()
        values = pd.to_numeric(ordered["value"], errors="coerce")
        history = values.shift(1)
        rolling = history.rolling(
            window=cfg.history_window, min_periods=cfg.minimum_history
        )
        center = rolling.median()
        mad = rolling.apply(
            lambda sample: float(
                np.median(np.abs(sample - np.median(sample)))
            ),
            raw=True,
        )
        scale = (1.4826 * mad).clip(lower=cfg.scale_floor)
        ordered["expected_value"] = center
        ordered["surprise_score"] = (values - center).abs() / scale
        ordered["is_anomaly"] = ordered["surprise_score"].ge(cfg.threshold).fillna(False)
        ordered["model_id"] = self.detector_id
        return ordered[
            [
                "event_ts",
                "entity_id",
                "metric_id",
                "expected_value",
                "surprise_score",
                "is_anomaly",
                "model_id",
            ]
        ]

    def score(self, telemetry: pd.DataFrame) -> pd.DataFrame:
        required = {"event_ts", "entity_id", "metric_id", "value", "quality_code"}
        missing = sorted(required - set(telemetry.columns))
        if missing:
            raise ValueError(f"telemetry is missing runtime fields: {missing}")
        measured = telemetry.loc[telemetry["quality_code"].eq("measured")].copy()
        measured["event_ts"] = pd.to_datetime(measured["event_ts"], utc=True)
        parts = [
            self._score_group(group)
            for _, group in measured.groupby(["entity_id", "metric_id"], sort=True)
        ]
        if not parts:
            return pd.DataFrame(
                columns=[
                    "event_ts",
                    "entity_id",
                    "metric_id",
                    "expected_value",
                    "surprise_score",
                    "is_anomaly",
                    "model_id",
                ]
            )
        output = pd.concat(parts, ignore_index=True)
        return output.sort_values(
            ["event_ts", "entity_id", "metric_id"], kind="stable"
        ).reset_index(drop=True)


def score_core_directory(
    core_directory: str | Path,
    detector: RobustHistoryDetector | None = None,
) -> pd.DataFrame:
    """Read only the canonical telemetry table from a SPEC-CORE directory."""

    root = Path(core_directory)
    parquet_path = root / "telemetry.parquet"
    parquet_directory = root / "telemetry"
    csv_path = root / "telemetry.csv"
    if parquet_path.exists():
        telemetry = pd.read_parquet(parquet_path)
    elif parquet_directory.is_dir():
        parts = sorted(parquet_directory.glob("part-*.parquet"))
        if not parts:
            raise FileNotFoundError(f"no telemetry parts in {parquet_directory}")
        telemetry = pd.concat(
            [pd.read_parquet(path) for path in parts], ignore_index=True
        )
    elif csv_path.exists():
        telemetry = pd.read_csv(csv_path, parse_dates=["event_ts", "ingested_at"])
    else:
        raise FileNotFoundError(f"no telemetry table in {root}")
    return (detector or RobustHistoryDetector()).score(telemetry)
