"""Right-labelled, past-only aggregation; no interpolation or forward filling."""

from dataclasses import dataclass
import pandas as pd
import numpy as np
from .adapter import CANONICAL_METRICS


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
            if metric not in CANONICAL_METRICS:
                raise ValueError(f"Unsupported metric: {metric}")
            series = group.sort_values("timestamp").set_index("timestamp").value
            bins = series.resample(self.interval, closed="right", label="right")
            sampled = (
                bins.sum(min_count=1)
                if CANONICAL_METRICS[metric].kind == "interval_count"
                else bins.mean()
            )
            frame = sampled.rename("value").reset_index()
            frame["entity_id"], frame["metric_name"] = entity, metric
            frame["observed"] = sampled.notna().to_numpy()
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)


def require_downstream_rx(telemetry: pd.DataFrame) -> None:
    """Check model inputs after adaptation; history sufficiency is checked at fit."""
    entities = set(telemetry.entity_id)
    rx = telemetry.loc[telemetry.metric_name.eq("rx_power_dbm")]
    usable = set(rx.loc[np.isfinite(rx.value), "entity_id"])
    missing = entities - usable
    if not entities or missing:
        raise ValueError(
            "Downstream Rx baseline requires finite rx_power_dbm observations "
            f"for each entity; unavailable: {sorted(missing)}"
        )
