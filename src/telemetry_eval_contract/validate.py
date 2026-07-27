"""Offline SPEC-EVAL validation. This package is never imported by runtime code."""

from __future__ import annotations

from typing import Any, Mapping

EVAL_REQUIRED_FIELDS = {
    "gt_fault_events": {
        "fault_event_id",
        "fault_type",
        "fault_family",
        "fault_domain_type",
        "fault_domain_id",
        "onset_ts",
        "first_observable_ts",
        "impact_ts",
        "resolution_ts",
        "cause_group_id",
        "left_censored",
        "label_source",
        "source_instance_id",
    },
    "gt_fault_entity_intervals": {
        "fault_event_id",
        "affected_entity_id",
        "active_start_ts",
        "active_end_ts",
    },
    "gt_cause_groups": {
        "cause_group_id",
        "cause_type",
        "start_ts",
        "end_ts",
        "footprint_json",
    },
    "gt_condition_states": {
        "entity_id",
        "condition_start_ts",
        "condition_end_ts",
        "condition_code",
        "condition_label",
        "label_source",
        "source_instance_id",
    },
}

EVAL_OPTIONAL_FIELDS = {
    "gt_benign_anomalies": {"entity_id", "event_ts", "benign_type", "n_samples"},
    "gt_collection_gaps": {"entity_id", "event_ts", "gap_reason"},
    "gt_ticket_links": {
        "ticket_id",
        "entity_id",
        "fault_event_id",
        "fault_type_label",
        "is_no_fault_found",
        "is_misattributed",
        "reported_ts",
        "resolved_ts",
    },
}


def _columns(table: Any) -> set[str]:
    if hasattr(table, "columns"):
        return {str(item) for item in table.columns}
    if isinstance(table, Mapping):
        return {str(item) for item in table}
    iterator = iter(table)
    try:
        return {str(item) for item in next(iterator)}
    except StopIteration:
        return set()


def validate_eval_bundle(bundle: Mapping[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    for table_name, required in EVAL_REQUIRED_FIELDS.items():
        if table_name not in bundle:
            errors.append(f"missing required table: {table_name}")
            continue
        missing = sorted(required - _columns(bundle[table_name]))
        if missing:
            errors.append(f"{table_name} missing fields: {missing}")
    for table_name, required in EVAL_OPTIONAL_FIELDS.items():
        if table_name not in bundle:
            continue
        missing = sorted(required - _columns(bundle[table_name]))
        if missing:
            errors.append(f"{table_name} missing fields: {missing}")
    return tuple(errors)
