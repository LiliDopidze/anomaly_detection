"""Sector-neutral SPEC-CORE vocabulary, interfaces, validation, and I/O.

This module is deliberately self-contained. It contains no sector names and has no
dependency on evaluation truth. Production runtime code may safely import it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

CONTRACT_VERSION = "0.3.0"

SPEC_CORE_TABLES = (
    "telemetry",
    "metric_catalogue",
    "entity_registry",
    "entity_relations",
    "operational_events",
    "collection_gaps",
    "context_values",
)


# ---------------------------------------------------------------------------
# Canonical vocabulary and interfaces
# ---------------------------------------------------------------------------


class QualityCode(str, Enum):
    MEASURED = "measured"
    CLIPPED = "clipped"
    INVALID = "invalid"


class AnomalyDirection(str, Enum):
    DECREASE = "decrease"
    INCREASE = "increase"
    BOTH = "both"
    CHANGE = "change"


class MeasurementKind(str, Enum):
    GAUGE = "gauge"
    RATE = "rate"
    INTERVAL_COUNT = "interval_count"
    CUMULATIVE_COUNTER = "cumulative_counter"
    BOUNDED_FRACTION = "bounded_fraction"
    DISCRETE_STATE = "discrete_state"
    EVENT = "event"


@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    entity_type: str
    measurement_kind: MeasurementKind
    unit: str
    anomaly_direction: AnomalyDirection
    sampling_semantics: str
    aggregation_semantics: str
    value_nullable: bool = True
    expected_cadence_seconds: float | None = None
    censoring_type: str = "none"
    expected_behaviour_profile: str = "stable_continuous"
    lower_bound: float | None = None
    upper_bound: float | None = None
    candidate_periods: tuple[str, ...] = ()
    exposure_metric_id: str | None = None
    exposure_semantics: str | None = None
    exposure_unit: str | None = None
    exposure_formula_id: str | None = None
    exposure_source: str | None = None
    context_keys: tuple[str, ...] = ()
    native_field: str | None = None


@dataclass(frozen=True)
class RelationSpec:
    relation_type: str
    parent_entity_type: str
    child_entity_type: str
    relation_family: str
    transitive: bool = True
    acyclic: bool = True
    propagates_fault_scope: bool = False


@dataclass(frozen=True)
class BehaviourProfile:
    profile_id: str
    model_family: str
    challenger: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)


ValidationRule = Callable[[Mapping[str, Sequence[Mapping[str, Any]]]], Sequence[str]]


@runtime_checkable
class SectorPack(Protocol):
    pack_id: str
    pack_version: str
    compatible_contract_versions: tuple[str, ...]

    def entity_types(self) -> tuple[str, ...]: ...
    def metric_specs(self) -> Mapping[str, MetricSpec]: ...
    def relation_specs(self) -> Mapping[str, RelationSpec]: ...
    def behaviour_profiles(self) -> Mapping[str, BehaviourProfile]: ...
    def parameters(self) -> Mapping[str, Any]: ...
    def validators(self) -> Sequence[ValidationRule]: ...


@dataclass(frozen=True)
class DatasetInventory:
    source_format: str
    source_version: str
    tables: tuple[str, ...]
    core_ready: bool
    evaluation_ready: bool
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class MaterialisationReport:
    adapter_id: str
    adapter_version: str
    source_format: str
    core_path: Path
    evaluation_path: Path | None
    row_counts: Mapping[str, int]
    truth_columns_removed: tuple[str, ...]


@runtime_checkable
class Adapter(Protocol):
    adapter_id: str
    adapter_version: str
    supported_source_formats: tuple[str, ...]

    def discover(self, source: Path) -> DatasetInventory: ...

    def materialise(
        self,
        source: Path,
        core_destination: Path,
        evaluation_destination: Path | None = None,
    ) -> MaterialisationReport: ...


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


CORE_REQUIRED_FIELDS = {
    "telemetry": {
        "event_ts",
        "ingested_at",
        "entity_id",
        "metric_id",
        "value",
        "quality_code",
        "quality_detail",
        "exposure",
        "source_id",
    },
    "metric_catalogue": {
        "metric_id",
        "entity_type",
        "measurement_kind",
        "unit",
        "anomaly_direction",
        "sampling_semantics",
        "aggregation_semantics",
        "censoring_type",
        "expected_behaviour_profile",
        "value_nullable",
        "expected_cadence_seconds",
        "exposure_semantics",
        "exposure_unit",
        "exposure_formula_id",
        "exposure_source",
    },
    "entity_registry": {
        "entity_id",
        "entity_type",
        "valid_from",
        "valid_to",
        "source_id",
    },
    "entity_relations": {
        "parent_entity_id",
        "child_entity_id",
        "relation_type",
        "relation_family",
        "valid_from",
        "valid_to",
    },
}

TRUTH_FIELD_NAMES = {
    "fault_event_id",
    "fault_id",
    "cause_group_id",
    "true_onset_ts",
    "first_observable_ts",
    "impact_ts",
    "repair_ts",
    "left_censored",
}


class ContractViolation(ValueError):
    """Raised when observable runtime data violates the neutral contract."""


def _columns(table: Any) -> set[str]:
    if hasattr(table, "columns"):
        return {str(item) for item in table.columns}
    if isinstance(table, Mapping):
        return {str(item) for item in table}
    rows = iter(table)
    try:
        first = next(rows)
    except StopIteration:
        return set()
    return {str(item) for item in first}


def assert_no_truth_fields(fields: Iterable[str], context: str = "SPEC-CORE") -> None:
    names = {str(field) for field in fields}
    leaked = sorted(
        field
        for field in names
        if field.startswith("gt_") or field.lower() in TRUTH_FIELD_NAMES
    )
    if leaked:
        raise ContractViolation(f"{context} contains evaluation-truth fields: {leaked}")


def assert_event_as_of(
    event: Mapping[str, Any],
    scoring_time: datetime,
    *,
    known_at_field: str = "known_at",
) -> None:
    known_at = event.get(known_at_field)
    if known_at is None:
        raise ContractViolation(f"observable event is missing {known_at_field}")
    if hasattr(known_at, "to_pydatetime"):
        known_at = known_at.to_pydatetime()
    if hasattr(scoring_time, "to_pydatetime"):
        scoring_time = scoring_time.to_pydatetime()
    if known_at > scoring_time:
        raise ContractViolation(
            f"event known at {known_at!s} cannot be consumed at {scoring_time!s}"
        )


def validate_core_bundle(bundle: Mapping[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    for table_name, required in CORE_REQUIRED_FIELDS.items():
        if table_name not in bundle:
            errors.append(f"missing required table: {table_name}")
            continue
        columns = _columns(bundle[table_name])
        missing = sorted(required - columns)
        if missing:
            errors.append(f"{table_name} missing fields: {missing}")
        try:
            assert_no_truth_fields(columns, context=table_name)
        except ContractViolation as exc:
            errors.append(str(exc))

    for table_name, table in bundle.items():
        try:
            assert_no_truth_fields(_columns(table), context=table_name)
        except ContractViolation as exc:
            errors.append(str(exc))

    catalogue = bundle.get("metric_catalogue")
    if catalogue is not None and hasattr(catalogue, "columns"):
        allowed_directions = {"decrease", "increase", "both", "change"}
        observed = set(catalogue["anomaly_direction"].dropna().astype(str))
        invalid = sorted(observed - allowed_directions)
        if invalid:
            errors.append(f"metric_catalogue has invalid anomaly directions: {invalid}")

    telemetry = bundle.get("telemetry")
    if telemetry is not None and hasattr(telemetry, "columns"):
        allowed_quality = {"measured", "clipped", "invalid"}
        observed = set(telemetry["quality_code"].dropna().astype(str))
        invalid = sorted(observed - allowed_quality)
        if invalid:
            errors.append(f"telemetry has invalid quality codes: {invalid}")
    return tuple(errors)


# ---------------------------------------------------------------------------
# Deterministic I/O
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_frame_hash(frame, *, sort_by: Iterable[str] | None = None) -> str:
    """Hash logical table content without depending on Parquet metadata."""

    import pandas as pd

    def normalise_object(value):
        if isinstance(value, (dict, list, tuple, set)):
            serialisable = sorted(value) if isinstance(value, set) else value
            return json.dumps(serialisable, sort_keys=True, default=str)
        if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
            converted = value.tolist()
            if isinstance(converted, (dict, list, tuple)):
                return json.dumps(converted, sort_keys=True, default=str)
        return value

    canonical = frame.copy()
    if sort_by:
        columns = [column for column in sort_by if column in canonical.columns]
        if columns:
            canonical = canonical.sort_values(columns, kind="stable")
    canonical = canonical.reset_index(drop=True)
    for column in canonical.columns:
        if pd.api.types.is_datetime64_any_dtype(canonical[column]):
            canonical[column] = canonical[column].astype("string")
        elif canonical[column].dtype == "object":
            canonical[column] = canonical[column].map(normalise_object)
    row_hashes = pd.util.hash_pandas_object(canonical, index=False).to_numpy()
    schema = "|".join(f"{name}:{dtype}" for name, dtype in canonical.dtypes.items())
    digest = hashlib.sha256(schema.encode("utf-8"))
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def write_json_immutable(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {destination}")
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def schema_root() -> Path:
    """Return the packaged neutral schema directory."""

    return Path(__file__).with_name("data") / "schemas" / "spec_core"
