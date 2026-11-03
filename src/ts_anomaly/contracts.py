"""Measurement definitions and the four-column observation representation."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class MetricDefinition:
    """Semantics, not a fitted model. None means the unit is undocumented."""

    description: str
    unit: str | None
    kind: Literal["gauge", "interval_count"] = "gauge"
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("gauge", "interval_count"):
            raise ValueError("Cumulative counters need an explicit conversion first")
        bounds = (self.lower, self.upper)
        if any(bound is not None and not isfinite(bound) for bound in bounds):
            raise ValueError("Specified bounds must be finite")
        if None not in bounds and self.lower > self.upper:
            raise ValueError("Lower bound exceeds upper bound")


def validate_schema(
    observations: pd.DataFrame, definitions: dict[str, MetricDefinition]
) -> None:
    """Check representation; duplicates, order and value quality are audited later.

    Timestamps may be naive when their timezone is unknown. This does not permit
    aligning unrelated sources as if they shared an absolute time coordinate.
    """
    expected = {"timestamp", "entity_id", "metric_name", "value"}
    if set(observations.columns) != expected or len(observations.columns) != 4:
        raise ValueError("Expected timestamp, entity_id, metric_name and value only")
    if observations.empty:
        raise ValueError("No observations supplied")
    if not pd.api.types.is_datetime64_any_dtype(observations.timestamp):
        raise ValueError("Parse timestamps explicitly before validation")
    if observations.timestamp.isna().any():
        raise ValueError("Missing timestamps")
    for key in ("entity_id", "metric_name"):
        values = observations[key]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing {key}")
    unknown = set(observations.metric_name).difference(definitions)
    if unknown:
        raise ValueError(f"Missing metric definitions: {sorted(unknown)}")
    if not pd.api.types.is_numeric_dtype(observations.value):
        raise ValueError("Values must be numeric; unknown values should be NaN")
