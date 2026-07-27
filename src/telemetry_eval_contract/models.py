"""SPEC-EVAL capability vocabulary."""

from __future__ import annotations

from dataclasses import dataclass

EVAL_CONTRACT_VERSION = "0.3.0"

EVALUATION_TABLES = (
    "gt_fault_events",
    "gt_fault_entity_intervals",
    "gt_cause_groups",
    "gt_condition_states",
    "gt_benign_anomalies",
    "gt_collection_gaps",
    "gt_ticket_links",
)


@dataclass(frozen=True)
class EvaluationCapability:
    capability_id: str
    available: bool
    overall_coverage: float
    label_source: str
    recording_delay: str | None
    known_selection_mechanism: str
    quality_status: str
    coverage_notes: str = ""
