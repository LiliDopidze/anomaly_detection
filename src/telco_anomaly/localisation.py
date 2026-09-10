"""Topology-aware localisation views over consolidated incidents.

Localisation is deliberately downstream of anomaly detection.  The current
incident builder supplies a primary and alternative observable scope; these
helpers expose that result and audit topology-identifiability limitations.
"""

from __future__ import annotations

import pandas as pd


LOCALISATION_COLUMNS = [
    "case_id",
    "predicted_scope_type",
    "predicted_scope_id",
    "alternative_scope_type",
    "alternative_scope_id",
    "affected_fraction_estimate",
    "footprint_size",
    "identifiability_status",
    "location_explanation",
]


def localisation_view(cases: pd.DataFrame) -> pd.DataFrame:
    """Return the product-facing localisation fields from incident cases."""

    required = {
        "case_id", "scope_type", "scope_id", "scope_type_2", "scope_id_2",
        "affected_fraction_estimate", "footprint_size",
        "identifiability_status", "location_explanation",
    }
    missing = required - set(cases.columns)
    if missing:
        raise ValueError(f"Cases are missing localisation fields: {sorted(missing)}")
    renamed = cases[list(required)].rename(columns={
        "scope_type": "predicted_scope_type",
        "scope_id": "predicted_scope_id",
        "scope_type_2": "alternative_scope_type",
        "scope_id_2": "alternative_scope_id",
    })
    return renamed[LOCALISATION_COLUMNS].reset_index(drop=True)


def topology_identifiability(topology: pd.DataFrame) -> pd.DataFrame:
    """List scopes that are indistinguishable from observable descendants."""

    required = {"entity_id", "group_type", "group_id", "group_family"}
    missing = required - set(topology.columns)
    if missing:
        raise ValueError(f"Topology is missing columns: {sorted(missing)}")
    physical = topology.loc[topology["group_family"].eq("physical_topology")]
    footprints = (
        physical.groupby(["group_type", "group_id"])["entity_id"]
        .agg(lambda values: tuple(sorted(set(map(str, values)))))
        .rename("descendants")
        .reset_index()
    )
    footprints["equivalence_size"] = footprints.groupby(
        "descendants"
    )["group_id"].transform("size")
    footprints["identifiable"] = footprints["equivalence_size"].eq(1)
    return footprints


__all__ = ["LOCALISATION_COLUMNS", "localisation_view", "topology_identifiability"]
