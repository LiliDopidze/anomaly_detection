"""Small measurement contract for aligned, wide sensor observations."""

from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class MetricDefinition:
    """Documented semantics; unknown units and encodings remain explicit."""

    description: str
    unit: str | None
    kind: Literal["gauge", "state", "pulse"]
    role: str
    allowed_values: tuple | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("gauge", "state", "pulse"):
            raise ValueError("Unsupported measurement kind")


def validate_schema(data: pd.DataFrame, definitions: dict[str, MetricDefinition]) -> None:
    """Check structure; duplicates and value problems remain for the audit.

    One row is one entity at one recorded time. Naive timestamps are permitted
    when the source timezone is unknown; they must not be interpreted as UTC.
    """
    expected = {"timestamp", "entity_id", *definitions}
    if set(data.columns) != expected or not data.columns.is_unique:
        raise ValueError("Columns must be timestamp, entity_id and declared measurements only")
    if not pd.api.types.is_datetime64_any_dtype(data.timestamp):
        raise ValueError("Parse timestamps explicitly")
    if data.timestamp.isna().any():
        raise ValueError("Missing timestamps")
    if data.entity_id.isna().any() or data.entity_id.astype(str).str.strip().eq("").any():
        raise ValueError("Missing entity identifiers")
    for name, definition in definitions.items():
        if definition.kind != "state" and not pd.api.types.is_numeric_dtype(data[name]):
            raise ValueError(f"Expected numeric measurements: {name}")
