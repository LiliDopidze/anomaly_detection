"""Loader for declarative JSON sector packs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .models import (
    CONTRACT_VERSION,
    AnomalyDirection,
    BehaviourProfile,
    MeasurementKind,
    MetricSpec,
    RelationSpec,
    ValidationRule,
)


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
    _validators: tuple[ValidationRule, ...] = ()

    def entity_types(self) -> tuple[str, ...]:
        return self._entity_types

    def metric_specs(self) -> Mapping[str, MetricSpec]:
        return self._metrics

    def relation_specs(self) -> Mapping[str, RelationSpec]:
        return self._relations

    def behaviour_profiles(self) -> Mapping[str, BehaviourProfile]:
        return self._profiles

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
        _validators=validators,
    )
