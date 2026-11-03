"""Explicit source mapping into the four-column long telemetry schema."""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from .contracts import MetricDefinition
from .pon import CANONICAL_METRICS, CANONICAL_UNITS  # Backward-compatible exports.


@dataclass
class TelemetryAdapter:
    """No source-specific defaults or inference of units/counter semantics.

    Units and kinds are keyed by canonical name. All mapped columns are required;
    omit unavailable optional measurements from the mapping. Invalid values become
    missing, never imputed. Interval counts must already cover known intervals.
    """

    metrics: dict[str, str]
    units: dict[str, str]
    kinds: dict[str, str]
    timestamp_column: str = "timestamp"
    entity_column: str = "entity_id"
    timezone: str = "UTC"
    definitions: dict[str, MetricDefinition] | None = None

    def transform(self, native: pd.DataFrame) -> pd.DataFrame:
        definitions = (
            CANONICAL_METRICS if self.definitions is None else self.definitions
        )
        if not self.metrics:
            raise ValueError("Declare at least one source measurement")
        required = {self.timestamp_column, self.entity_column, *self.metrics}
        missing = required.difference(native.columns)
        if missing:
            raise ValueError(f"Missing mapped columns: {sorted(missing)}")
        if len(set(self.metrics.values())) != len(self.metrics):
            raise ValueError("Map each canonical measurement only once")
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
            if metric not in definitions:
                raise ValueError(f"Unsupported metric: {metric}")
            definition = definitions[metric]
            if self.kinds.get(metric) != definition.kind:
                raise ValueError(
                    f"Declare {definition.kind} for {metric}; cumulative counters "
                    "need explicit reset-aware conversion before adaptation"
                )
            values = pd.to_numeric(native[source], errors="raise").astype(float)
            unit = self.units.get(metric)
            if definition.unit == "dBm" and unit in {"mW", "W"}:
                values = 10 * np.log10(
                    values.where(values > 0) * (1000 if unit == "W" else 1)
                )
            elif unit != definition.unit:
                raise ValueError(f"Declare supported units for {metric}")
            values = values.where(np.isfinite(values))
            if definition.lower is not None:
                values = values.where(values >= definition.lower)
            if definition.upper is not None:
                values = values.where(values <= definition.upper)
            if definition.kind == "interval_count":
                values = values.where((values >= 0) & (values == np.floor(values)))
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
