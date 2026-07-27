"""telemetry-synth: reproducible synthetic telco telemetry.

Emits the generator's own NATIVE form, versions it, and proves the run reproduces.
Mapping to a sector-neutral canonical schema is deliberately NOT here: it comes later,
as a separate adapter layer, once there are several sources to generalise over.
"""
from .generator import (
    GENERATOR_VERSION, NATIVE_FORMAT_VERSION as _GEN_NATIVE,
    TelecomSimulationSettings, FAULT_TYPES, DEVICE_MODELS, ENCLOSURES, SHARED_SCOPES,
    build_network_topology, build_shared_hierarchy, simulate_healthy_ont_signals,
    sample_storms, sample_fault_events, build_fault_degradation_curve,
    apply_repair_to_fault_curve, simulate_error_cascade,
    generate_telecom_reference_data, parameter_provenance,
    build_reference_configs, build_scenario_grid,
)
from .validate import validate_reference_data
from .manifest import (
    NATIVE_FORMAT_VERSION, NATIVE_TABLES, capture_environment, config_fingerprint,
    dataset_id, build_manifest, publish_dataset, native_format_doc,
    register_dataset, load_dataset_index, verify_manifest, verify_reproducibility,
)

__version__ = GENERATOR_VERSION

__all__ = [
    "GENERATOR_VERSION", "NATIVE_FORMAT_VERSION", "NATIVE_TABLES",
    "TelecomSimulationSettings", "FAULT_TYPES", "DEVICE_MODELS", "ENCLOSURES",
    "SHARED_SCOPES", "build_network_topology", "build_shared_hierarchy",
    "simulate_healthy_ont_signals", "sample_storms", "sample_fault_events",
    "build_fault_degradation_curve", "apply_repair_to_fault_curve",
    "simulate_error_cascade", "generate_telecom_reference_data",
    "parameter_provenance", "build_reference_configs", "build_scenario_grid",
    "validate_reference_data", "capture_environment", "config_fingerprint",
    "dataset_id", "build_manifest", "publish_dataset", "native_format_doc",
    "register_dataset", "load_dataset_index", "verify_manifest",
    "verify_reproducibility",
]
