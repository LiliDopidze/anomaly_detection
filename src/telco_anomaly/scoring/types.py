"""Static contracts for persisted fitted scoring artefacts.

The runtime representation remains a plain dictionary for joblib and notebook
compatibility. TypedDict makes the supported fields reviewable without a
high-risk migration of existing model artefacts.
"""

from __future__ import annotations

from typing import Any, TypedDict


class ResidualBundle(TypedDict, total=False):
    model_core_version: str
    feature_columns: list[str]
    global_centre: Any
    global_scale: Any
    entity_centre: Any
    entity_scale: Any
    entity_reference_counts: Any
    entity_reference_days: Any
    training_rows: int
    training_sample_strategy: str
    selected_row_hash: str
    maximum_rows_per_entity_day: int | None
    random_seed: int
    feature_directions: dict[str, str]
    minimum_scales: dict[str, float]
    pca: Any
    isolation_base_features: list[str]
    isolation_temporal_features: list[str]
    isolation_fill_values: Any
    isolation_forest_base: Any
    isolation_forest_temporal: Any
    isolation_temporal_tail_reference: Any
    isolation_entity_score_reference: Any


class ContextualIsolationBundle(TypedDict, total=False):
    feature_columns: list[str]
    minimum_observed_inputs: int
    imputer: Any
    scaler: Any
    isolation_forest: Any
    scaled_feature_cap: float
