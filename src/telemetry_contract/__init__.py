"""Dependency-light SPEC-CORE contract.

This package is safe for production runtime imports. Evaluation truth lives in the
separate :mod:`telemetry_eval_contract` distribution boundary.
"""

from .adapters import Adapter, DatasetInventory, MaterialisationReport
from .models import (
    CONTRACT_VERSION,
    AnomalyDirection,
    BehaviourProfile,
    MeasurementKind,
    MetricSpec,
    QualityCode,
    RelationSpec,
    SectorPack,
    ValidationRule,
)
from .pack import DeclarativeSectorPack, load_declarative_pack
from .io import canonical_frame_hash, sha256_file, write_json_immutable
from .validate import (
    ContractViolation,
    assert_event_as_of,
    assert_no_truth_fields,
    validate_core_bundle,
)

__all__ = [
    "CONTRACT_VERSION",
    "AnomalyDirection",
    "Adapter",
    "BehaviourProfile",
    "ContractViolation",
    "DatasetInventory",
    "DeclarativeSectorPack",
    "MaterialisationReport",
    "MeasurementKind",
    "MetricSpec",
    "QualityCode",
    "RelationSpec",
    "SectorPack",
    "ValidationRule",
    "assert_event_as_of",
    "assert_no_truth_fields",
    "load_declarative_pack",
    "validate_core_bundle",
    "canonical_frame_hash",
    "sha256_file",
    "write_json_immutable",
]
