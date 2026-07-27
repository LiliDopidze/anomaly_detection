"""Stable, sector-neutral Python interfaces for SPEC-CORE and sector packs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

CONTRACT_VERSION = "0.3.0"


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
    def validators(self) -> Sequence[ValidationRule]: ...
