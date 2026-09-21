"""Explicit source mapping into the four-column long telemetry schema."""

from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricDefinition:
    unit: str
    kind: Literal["gauge", "interval_count"]
    description: str


# Source: ETSI GS F5G 011 §§8.3–8.4; definitions, not synthetic distributions.
CANONICAL_METRICS = {
    "rx_power_dbm": MetricDefinition("dBm", "gauge", "Downstream Rx at ONT"),
    "upstream_rx_power_dbm": MetricDefinition(
        "dBm", "gauge", "Upstream Rx at OLT per ONT"
    ),
    "ont_tx_power_dbm": MetricDefinition("dBm", "gauge", "ONT transmit power"),
    "olt_tx_power_dbm": MetricDefinition(
        "dBm", "gauge", "Shared OLT port transmit power"
    ),
    "ont_temperature_c": MetricDefinition("C", "gauge", "ONT temperature"),
    "olt_temperature_c": MetricDefinition("C", "gauge", "Shared OLT port temperature"),
    "ber": MetricDefinition("ratio", "gauge", "Downstream pre-FEC BER"),
    "upstream_ber": MetricDefinition("ratio", "gauge", "Upstream pre-FEC BER"),
}
for direction in ("downstream", "upstream"):
    for kind in ("corrected", "uncorrectable", "total"):
        # Source: ITU-T G.989.3 Table 14-1. Total means ALL received codewords.
        # ETSI's upstream-FEC-total-block instead counts errored blocks only.
        name = f"{direction}_fec_{kind}_codewords"
        CANONICAL_METRICS[name] = MetricDefinition(
            "interval_count",
            "interval_count",
            f"{direction.title()} {kind} codewords in preceding reporting interval",
        )
CANONICAL_UNITS = {name: item.unit for name, item in CANONICAL_METRICS.items()}


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

    def transform(self, native: pd.DataFrame) -> pd.DataFrame:
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
            if metric not in CANONICAL_UNITS:
                raise ValueError(f"Unsupported metric: {metric}")
            definition = CANONICAL_METRICS[metric]
            if self.kinds.get(metric) != definition.kind:
                raise ValueError(
                    f"Declare {definition.kind} for {metric}; cumulative counters "
                    "need explicit reset-aware conversion before adaptation"
                )
            values = pd.to_numeric(native[source], errors="raise").astype(float)
            unit = self.units.get(metric)
            if CANONICAL_UNITS[metric] == "dBm" and unit in {"mW", "W"}:
                values = 10 * np.log10(
                    values.where(values > 0) * (1000 if unit == "W" else 1)
                )
            elif unit != CANONICAL_UNITS[metric]:
                raise ValueError(f"Declare supported units for {metric}")
            values = values.where(np.isfinite(values))
            if CANONICAL_UNITS[metric] == "ratio":
                values = values.where(values.between(0, 1))
            if CANONICAL_UNITS[metric] == "interval_count":
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
