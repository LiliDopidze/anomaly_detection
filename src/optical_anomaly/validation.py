"""Right-labelled, past-only aggregation; no interpolation or forward filling."""

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class DataValidator:
    interval: str = "5min"

    def transform(self, telemetry: pd.DataFrame) -> pd.DataFrame:
        required = {"timestamp", "entity_id", "metric_name", "value"}
        if set(telemetry.columns) != required:
            raise ValueError(f"Expected exactly {sorted(required)}")
        keys = ["entity_id", "metric_name", "timestamp"]
        if telemetry.duplicated(keys).any():
            raise ValueError("Duplicate telemetry")
        if pd.Timedelta(self.interval) <= pd.Timedelta(0):
            raise ValueError("Interval must be positive")
        frames = []
        for (entity, metric), group in telemetry.groupby(keys[:2], sort=True):
            series = group.sort_values("timestamp").set_index("timestamp").value
            sampled = series.resample(
                self.interval, closed="right", label="right"
            ).mean()
            frame = sampled.rename("value").reset_index()
            frame["entity_id"], frame["metric_name"] = entity, metric
            frame["observed"] = sampled.notna().to_numpy()
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)
