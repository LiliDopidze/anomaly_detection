"""Measurement semantics shared by adapters and temporal validation."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal


@dataclass(frozen=True)
class MetricDefinition:
    unit: str
    kind: Literal["gauge", "interval_count"]
    description: str
    lower: float | None = None
    upper: float | None = None
    count_group: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("gauge", "interval_count"):
            raise ValueError("Use gauges or explicitly converted interval counts")
        if any(
            bound is not None and not isfinite(bound)
            for bound in (self.lower, self.upper)
        ):
            raise ValueError("Measurement bounds must be finite when specified")
        if (
            self.lower is not None
            and self.upper is not None
            and self.lower > self.upper
        ):
            raise ValueError("Lower bound exceeds upper bound")
        if self.count_group is not None and self.kind != "interval_count":
            raise ValueError("Count groups apply only to interval counts")
