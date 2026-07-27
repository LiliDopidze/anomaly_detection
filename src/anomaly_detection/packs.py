"""Generic pack loader and the two Milestone 1 sector phrasebooks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .core import (
    CONTRACT_VERSION,
    AnomalyDirection,
    BehaviourProfile,
    MeasurementKind,
    MetricSpec,
    RelationSpec,
    ValidationRule,
)

TELECOM_PACK_ID = "telecom"
TELECOM_PACK_VERSION = "0.2.0"
OIL_WELL_PACK_ID = "oil_well"
OIL_WELL_PACK_VERSION = "0.1.0"

PACK_DATA_ROOT = Path(__file__).with_name("data") / "packs"


def _read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class DeclarativeSectorPack:
    pack_id: str
    pack_version: str
    compatible_contract_versions: tuple[str, ...]
    _entity_types: tuple[str, ...]
    _metrics: Mapping[str, MetricSpec]
    _relations: Mapping[str, RelationSpec]
    _profiles: Mapping[str, BehaviourProfile]
    _parameters: Mapping[str, object]
    _validators: tuple[ValidationRule, ...] = ()

    def entity_types(self) -> tuple[str, ...]:
        return self._entity_types

    def metric_specs(self) -> Mapping[str, MetricSpec]:
        return self._metrics

    def relation_specs(self) -> Mapping[str, RelationSpec]:
        return self._relations

    def behaviour_profiles(self) -> Mapping[str, BehaviourProfile]:
        return self._profiles

    def parameters(self) -> Mapping[str, object]:
        return self._parameters

    def validators(self) -> Sequence[ValidationRule]:
        return self._validators


def _metric(row: dict) -> MetricSpec:
    values = dict(row)
    values["measurement_kind"] = MeasurementKind(values["measurement_kind"])
    values["anomaly_direction"] = AnomalyDirection(values["anomaly_direction"])
    values["candidate_periods"] = tuple(values.get("candidate_periods", ()))
    values["context_keys"] = tuple(values.get("context_keys", ()))
    return MetricSpec(**values)


def _records(table):
    if hasattr(table, "to_dict"):
        return table.to_dict(orient="records")
    return list(table)


def _compile_validator(rule: dict) -> ValidationRule:
    rule_id = rule["rule_id"]
    kind = rule["kind"]

    if kind == "metric_id_prefix":
        prefix = str(rule["prefix"])

        def validate_prefix(bundle):
            rows = _records(bundle.get("metric_catalogue", ()))
            invalid = sorted(
                str(row.get("metric_id"))
                for row in rows
                if not str(row.get("metric_id", "")).startswith(prefix)
            )
            return (
                [f"{rule_id}: metric ids outside {prefix!r}: {invalid}"]
                if invalid
                else []
            )

        return validate_prefix

    if kind == "required_relation_types":
        required = set(rule["values"])

        def validate_relations(bundle):
            rows = _records(bundle.get("entity_relations", ()))
            present = {str(row.get("relation_type")) for row in rows}
            missing = sorted(required - present)
            return [f"{rule_id}: missing relation types: {missing}"] if missing else []

        return validate_relations

    raise ValueError(f"unsupported declarative validation rule kind: {kind}")


def load_declarative_pack(pack_dir: str | Path) -> DeclarativeSectorPack:
    root = Path(pack_dir)
    metadata = _read_json(root / "pack.json")
    compatible = tuple(metadata["compatible_contract_versions"])
    if CONTRACT_VERSION not in compatible:
        raise ValueError(
            f"pack {metadata['pack_id']} {metadata['pack_version']} does not declare "
            f"compatibility with SPEC-CORE {CONTRACT_VERSION}"
        )

    metrics = [_metric(row) for row in _read_json(root / "metrics.json")]
    relations = [RelationSpec(**row) for row in _read_json(root / "relations.json")]
    profile_path = root / "behaviour_profiles.json"
    profiles = [
        BehaviourProfile(
            profile_id=row["profile_id"],
            model_family=row["model_family"],
            challenger=row.get("challenger"),
            parameters=row.get("parameters", {}),
        )
        for row in (_read_json(profile_path) if profile_path.exists() else [])
    ]
    validation_path = root / "validation_rules.json"
    validators = (
        tuple(_compile_validator(rule) for rule in _read_json(validation_path))
        if validation_path.exists()
        else ()
    )
    return DeclarativeSectorPack(
        pack_id=metadata["pack_id"],
        pack_version=metadata["pack_version"],
        compatible_contract_versions=compatible,
        _entity_types=tuple(metadata["entity_types"]),
        _metrics={item.metric_id: item for item in metrics},
        _relations={item.relation_type: item for item in relations},
        _profiles={item.profile_id: item for item in profiles},
        _parameters=metadata.get("parameters", {}),
        _validators=validators,
    )


def load_telecom_pack() -> DeclarativeSectorPack:
    return load_declarative_pack(PACK_DATA_ROOT / "telecom")


# ---------------------------------------------------------------------------
# Minimal Petrobras 3W pack
# ---------------------------------------------------------------------------

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


def _oil_metric(native_field: str) -> MetricSpec:
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
        context_keys=("source_kind",),
    )


@dataclass(frozen=True)
class OilWellPack:
    pack_id: str = OIL_WELL_PACK_ID
    pack_version: str = OIL_WELL_PACK_VERSION
    compatible_contract_versions: tuple[str, ...] = (CONTRACT_VERSION,)

    def entity_types(self) -> tuple[str, ...]:
        return ("oil_well",)

    def metric_specs(self) -> Mapping[str, MetricSpec]:
        metrics = [_oil_metric(field) for field in NATIVE_METRIC_FIELDS]
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

    def parameters(self) -> Mapping[str, object]:
        return {}

    def validators(self) -> Sequence[ValidationRule]:
        return ()


def load_oil_well_pack() -> OilWellPack:
    return OilWellPack()
