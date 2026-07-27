"""Universal SPEC-CORE checks with no sector or fixture knowledge."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Mapping

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
    "entity_registry": {"entity_id", "entity_type", "valid_from", "valid_to", "source_id"},
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
    pass


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
