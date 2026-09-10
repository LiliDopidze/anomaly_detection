"""Adapter for Microsoft's public optical-backbone telemetry.

The download is useful for real optical behaviour and alert-rate stability.
Microsoft states that outage days were removed, so this adapter intentionally
creates no evaluation truth and must not be used to claim fault recall.
"""

from __future__ import annotations

from copy import deepcopy

from ._tabular import build_tabular_pack, inspect_tables


SOURCE_URL = "https://www.microsoft.com/en-us/download/details.aspx?id=54267"


def inspect_microsoft_optical(source, *, file_patterns=("**/*.csv", "**/*.parquet"), csv_options=None):
    """Inventory the extracted download without guessing source columns."""

    return inspect_tables(
        source, file_patterns, csv_options=csv_options, sample_rows=5
    )


def microsoft_mapping_template(*, file_patterns=("**/*.csv", "**/*.parquet")):
    """Return a review template; native fields deliberately remain blank."""

    return {
        "file_patterns": list(file_patterns),
        "csv_options": {},
        "timestamp_column": None,
        "timestamp_unit": None,
        "entity_column": None,
        "entity_type": "optical_channel",
        "source_system": "microsoft_optical_backbone_v1",
        "episode_basis": "continuous_public_monitoring_window",
        "topology": [
            {
                "native_field": None,
                "group_type": "optical_path",
                "hierarchy_level": 0,
                "group_family": "physical_topology",
            }
        ],
        "metrics": [
            _metric(None, "optical.q_factor", "unitless", "low_bad", 0.05),
            _metric(None, "optical.tx_power", "dBm", "two_sided", 0.05),
            _metric(None, "optical.chromatic_dispersion", "ps/nm", "two_sided", 0.10),
            _metric(None, "optical.polarization_mode_dispersion", "ps", "high_bad", 0.01),
        ],
    }


def _metric(native_field, metric_id, unit, direction, minimum_scale):
    return {
        "native_field": native_field,
        "metric_id": metric_id,
        "measurement_kind": "gauge",
        "unit": unit,
        "sampling_mode": "periodic",
        "aggregation_semantics": "instantaneous",
        "expected_cadence_seconds": 900,
        "direction": direction,
        "transform": "identity",
        "minimum_scale": minimum_scale,
        "seasonality_candidate": False,
        "peer_eligible": True,
    }


def build_microsoft_optical_pack(
    source,
    destination,
    mapping,
    *,
    pack_version="microsoft_optical_v1",
    batch_rows=100_000,
):
    """Build a real-behaviour pack from a completed, reviewed mapping."""

    mapping = deepcopy(mapping)
    mapping.setdefault("source_system", "microsoft_optical_backbone_v1")
    return build_tabular_pack(
        source,
        destination,
        mapping,
        dataset_name="microsoft_optical_data",
        pack_version=pack_version,
        source_url=SOURCE_URL,
        evidence_role="real_optical_behaviour_and_alert_stability",
        licence="Microsoft Research Data License Agreement in download archive",
        licence_note=(
            "Research/non-commercial use is restricted by the supplied agreement; "
            "it also limits modification and distribution. Obtain legal approval "
            "before external or commercial use. This project redistributes neither "
            "raw nor derived data."
        ),
        batch_rows=batch_rows,
    )


__all__ = [
    "SOURCE_URL",
    "inspect_microsoft_optical",
    "microsoft_mapping_template",
    "build_microsoft_optical_pack",
]
