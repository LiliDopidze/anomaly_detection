"""Mapping-driven adapter for the public live commercial RAN PM dataset.

PM counter names vary by generation and vendor export.  The adapter therefore
discovers structure but refuses to infer units, aggregation or metric meaning.
Those choices must be reviewed once and supplied in ``mapping``.
"""

from __future__ import annotations

from copy import deepcopy

import pandas as pd

from ._tabular import build_tabular_pack, inspect_tables


SOURCE_URL = "https://zenodo.org/records/17815388"


def inspect_ran_pm(source, *, file_patterns=("**/*.csv", "**/*.parquet"), csv_options=None):
    """Return files, columns and a small sample without semantic guesses."""

    result = inspect_tables(
        source, file_patterns, csv_options=csv_options, sample_rows=20
    )
    sample = result["sample"]
    result["sample_dtypes"] = {name: str(dtype) for name, dtype in sample.dtypes.items()}
    result["numeric_columns_for_review"] = [
        name for name in sample if pd.api.types.is_numeric_dtype(sample[name])
    ]
    return result


def ran_mapping_template(*, file_patterns=("**/*.csv", "**/*.parquet")):
    """Return an intentionally unresolved template for human review."""

    return {
        "file_patterns": list(file_patterns),
        "csv_options": {},
        "timestamp_column": None,
        "timestamp_unit": None,
        "entity_column": None,
        "entity_type": "ran_cell",
        "source_system": "live_commercial_ran_pm_17815388",
        "episode_basis": "continuous_public_pm_window",
        "topology": [],
        "metrics": [],
        "review_note": (
            "Add only counters whose units, cadence and aggregation semantics "
            "have been verified from the dataset documentation. File patterns "
            "should select a schema-compatible subset; counters that never "
            "report for one cell are treated as unavailable for that cell."
        ),
    }


def build_ran_pm_pack(
    source,
    destination,
    mapping,
    *,
    pack_version="ran_pm_v1",
    batch_rows=100_000,
):
    """Build a company-neutral RAN pack from an approved mapping."""

    mapping = deepcopy(mapping)
    mapping.setdefault("source_system", "live_commercial_ran_pm_17815388")
    return build_tabular_pack(
        source,
        destination,
        mapping,
        dataset_name="live_commercial_ran_pm_counters",
        pack_version=pack_version,
        source_url=SOURCE_URL,
        evidence_role="real_operator_telemetry_portability_and_workload",
        licence="CC BY 4.0",
        licence_note="Retain dataset attribution; raw files are not redistributed here.",
        batch_rows=batch_rows,
    )


__all__ = [
    "SOURCE_URL", "inspect_ran_pm", "ran_mapping_template", "build_ran_pm_pack"
]
