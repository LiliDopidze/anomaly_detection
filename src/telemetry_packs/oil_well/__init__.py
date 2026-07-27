"""Minimal Petrobras 3W OilWell Pack used to challenge SPEC-CORE v0.3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from telemetry_contract import (
    CONTRACT_VERSION,
    AnomalyDirection,
    BehaviourProfile,
    MeasurementKind,
    MetricSpec,
    RelationSpec,
    ValidationRule,
)

PACK_ID = "oil_well"
PACK_VERSION = "0.1.0"

STATE_FIELDS = (
    "ESTADO-DHSV",
    "ESTADO-M1",
    "ESTADO-M2",
    "ESTADO-PXO",
    "ESTADO-SDV-GL",
    "ESTADO-SDV-P",
    "ESTADO-W1",
    "ESTADO-W2",
    "ESTADO-XO",
)
OPENING_FIELDS = ("ABER-CKGL", "ABER-CKP")
PRESSURE_FIELDS = (
    "P-ANULAR",
    "P-JUS-BS",
    "P-JUS-CKGL",
    "P-JUS-CKP",
    "P-MON-CKGL",
    "P-MON-CKP",
    "P-MON-SDV-P",
    "P-PDG",
    "PT-P",
    "P-TPT",
)
FLOW_FIELDS = ("QBS", "QGL")
TEMPERATURE_FIELDS = ("T-JUS-CKP", "T-MON-CKP", "T-PDG", "T-TPT")
NATIVE_METRIC_FIELDS = (
    *OPENING_FIELDS,
    *STATE_FIELDS,
    *PRESSURE_FIELDS,
    *FLOW_FIELDS,
    *TEMPERATURE_FIELDS,
)


def _slug(native_field: str) -> str:
    return native_field.lower().replace("-", "_")


def _metric(native_field: str) -> MetricSpec:
    if native_field in STATE_FIELDS:
        kind = MeasurementKind.DISCRETE_STATE
        unit = "state"
        direction = AnomalyDirection.CHANGE
        aggregation = "last"
        profile = "discrete_state"
        lower, upper = 0.0, 1.0
    elif native_field in OPENING_FIELDS:
        kind = MeasurementKind.GAUGE
        unit = "percent"
        direction = AnomalyDirection.BOTH
        aggregation = "mean"
        profile = "stable_continuous"
        lower, upper = 0.0, 100.0
    elif native_field in PRESSURE_FIELDS:
        kind = MeasurementKind.GAUGE
        unit = "Pa"
        direction = AnomalyDirection.BOTH
        aggregation = "mean"
        profile = "stable_continuous"
        lower, upper = 0.0, None
    elif native_field in FLOW_FIELDS:
        kind = MeasurementKind.GAUGE
        unit = "m3/s"
        direction = AnomalyDirection.BOTH
        aggregation = "mean"
        profile = "stable_continuous"
        lower, upper = 0.0, None
    else:
        kind = MeasurementKind.GAUGE
        unit = "degC"
        direction = AnomalyDirection.BOTH
        aggregation = "mean"
        profile = "stable_continuous"
        lower, upper = None, None
    return MetricSpec(
        metric_id=f"oil_well.{_slug(native_field)}",
        native_field=native_field,
        entity_type="oil_well",
        measurement_kind=kind,
        unit=unit,
        anomaly_direction=direction,
        sampling_semantics="one_second_observation",
        aggregation_semantics=aggregation,
        value_nullable=True,
        expected_cadence_seconds=1.0,
        censoring_type="none",
        expected_behaviour_profile=profile,
        lower_bound=lower,
        upper_bound=upper,
        candidate_periods=(),
        context_keys=("source_kind",),
    )


@dataclass(frozen=True)
class OilWellPack:
    pack_id: str = PACK_ID
    pack_version: str = PACK_VERSION
    compatible_contract_versions: tuple[str, ...] = (CONTRACT_VERSION,)

    def entity_types(self) -> tuple[str, ...]:
        return ("oil_well",)

    def metric_specs(self) -> Mapping[str, MetricSpec]:
        metrics = [_metric(field) for field in NATIVE_METRIC_FIELDS]
        return {metric.metric_id: metric for metric in metrics}

    def relation_specs(self) -> Mapping[str, RelationSpec]:
        return {}

    def behaviour_profiles(self) -> Mapping[str, BehaviourProfile]:
        profiles = (
            BehaviourProfile(
                profile_id="stable_continuous",
                model_family="rolling_robust_quantiles",
            ),
            BehaviourProfile(
                profile_id="discrete_state",
                model_family="transition_frequency",
            ),
        )
        return {profile.profile_id: profile for profile in profiles}

    def validators(self) -> Sequence[ValidationRule]:
        return ()


def load_pack() -> OilWellPack:
    return OilWellPack()


__all__ = [
    "NATIVE_METRIC_FIELDS",
    "PACK_ID",
    "PACK_VERSION",
    "OilWellPack",
    "load_pack",
]
