"""Explicit source mapping into the four-column long telemetry schema."""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class TelemetryAdapter:
    timestamp_column: str = "time"
    entity_column: str = "device"
    metrics: dict[str, str] = field(
        default_factory=lambda: {"rx_dbm": "rx_power_dbm", "ber": "ber"}
    )
    units: dict[str, str] = field(
        default_factory=lambda: {"rx_power_dbm": "dBm", "ber": "ratio"}
    )
    timezone: str = "UTC"

    def transform(self, native: pd.DataFrame) -> pd.DataFrame:
        times = pd.to_datetime(native[self.timestamp_column], errors="raise")
        if times.isna().any():
            raise ValueError("Missing timestamps")
        if times.dt.tz is None:
            times = times.dt.tz_localize(
                self.timezone, ambiguous="raise", nonexistent="raise"
            )
        entities = native[self.entity_column]
        if entities.isna().any() or entities.astype(str).str.strip().eq("").any():
            raise ValueError("Missing entity identifiers")
        frames = []
        for source, metric in self.metrics.items():
            if metric not in {"rx_power_dbm", "ber"}:
                raise ValueError(f"Unsupported metric: {metric}")
            values = pd.to_numeric(native[source], errors="raise").astype(float)
            unit = self.units.get(metric)
            if metric == "rx_power_dbm" and unit in {"mW", "W"}:
                values = 10 * np.log10(
                    values.where(values > 0) * (1000 if unit == "W" else 1)
                )
            elif unit != {"rx_power_dbm": "dBm", "ber": "ratio"}[metric]:
                raise ValueError(f"Declare supported units for {metric}")
            values = values.where(np.isfinite(values))
            if metric == "ber":
                values = values.where(values.between(0, 1))
            frames.append(
                pd.DataFrame(
                    {
                        "timestamp": times.dt.tz_convert("UTC"),
                        "entity_id": entities.astype(str),
                        "metric_name": metric,
                        "value": values,
                    }
                )
            )
        result = pd.concat(frames, ignore_index=True)
        keys = ["entity_id", "metric_name", "timestamp"]
        if result.duplicated(keys).any():
            raise ValueError("Duplicate entity/metric/timestamp records")
        return result.sort_values(keys).reset_index(drop=True)
