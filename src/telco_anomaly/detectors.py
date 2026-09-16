"""Backward-compatible public model API.

Implementation lives in focused ``pipeline`` and ``scoring`` modules. Existing
notebooks may continue importing from :mod:`telco_anomaly.detectors`.
"""

from .pipeline.materialize import (
    materialize_features, materialize_measurement_features,
    materialize_wide_partition, measurement_features, partition_definition,
    partition_exposure, raw_features, resample_wide_panel,
    split_feature_file_by_time,
)
from .scoring._common import (
    IDENTITY_COLUMNS, LEGACY_MODEL_IDS, MODEL_CORE_VERSION, MODEL_IDS,
    duration_to_observations, feature_columns, iter_episode_frames,
)
from .scoring.alerts import (
    alert_grid_from_score_file, alerts_from_score_file, candidate_key,
    materialize_candidate_alert_grid,
)
from .scoring.isolation_forest import (
    append_contextual_isolation_scores, fit_contextual_isolation_forest,
    fit_residual_bundle, score_residual_episode, score_residual_file,
)
from .scoring.orchestration import (
    case_score_trace, merge_score_files, score_partition_file,
)
from .scoring.sensitivity import run_sampling_sensitivity
from .scoring.statistical import (
    calibration_sample, fit_isolation_feature_subset,
    fit_isolation_seed_models, fit_model_bundle, score_feature_file,
    score_isolation_feature_subset_file, score_isolation_seed_file,
    score_matrix, score_percentiles,
)
from .scoring.thresholds import calibration_thresholds
from .scoring.topology import (
    choose_peer_level, eligible_peer_levels, fit_topology_reference,
    physical_hierarchy, score_topology_file,
)

# Private helpers remain importable here for the existing regression suite and
# for downstream code that adopted the former monolithic module before the
# package was split. New code should import only public functions.
from .scoring.alerts import _deduplicate_scope_alerts, _recovery_threshold
from .scoring.isolation_forest import (
    _balanced_isolation_features,
    _cusum_scores,
    _fit_isolation_forest,
    _is_temporal_feature,
    _isolation_feature_priority,
    _row_max,
)
from .scoring.reference import (
    _directional_model_frame,
    _empirical_vector_evidence,
    _entity_adjusted_score_evidence,
    _fit_entity_score_reference,
    _full_entity_reference,
    _reference_components,
    _reference_sample,
    _residual_frame,
    _robust_reference,
    _second_metric_evidence,
    _top_two_metric_mean_evidence,
)
from .scoring.topology import (
    _calibrated_affected_gate,
    _chosen_value_case,
    _group_entity_best_sql,
    _group_wide_sql,
    _model_topology,
    _normalised_topology_score,
    _peer_wide_sql,
    _reference_summary_sql,
    _scope_value_case,
    _size_band_condition,
    _size_band_sql,
    _topology_feature_specs,
)
