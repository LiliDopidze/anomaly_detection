"""Isolation Forest fitting and residual time-series scoring."""

from __future__ import annotations

import hashlib
import math
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from ..features import (
    directional_cusum, empirical_tail_evidence, feature_policy,
    fit_empirical_tail_reference, orient_residuals,
)
from ._common import (
    IDENTITY_COLUMNS, MODEL_CORE_VERSION, SCALED_FEATURE_CAP,
    _model_duckdb, _sql_identifier, _sql_literal, feature_columns,
    iter_episode_frames,
)
from .reference import (
    _directional_model_frame, _empirical_vector_evidence,
    _entity_adjusted_score_evidence, _fit_entity_score_reference,
    _full_entity_reference, _reference_sample, _residual_frame,
    _robust_reference, _second_metric_evidence,
    _top_two_metric_mean_evidence,
)
from .types import ContextualIsolationBundle, ResidualBundle

def _is_temporal_feature(name):
    """Identify features added by the multi-timescale temporal layer."""

    return any(
        marker in str(name)
        for marker in ("__history_", "__lag_", "__rate_", "__sum_")
    )

def _isolation_feature_priority(name):
    """Order compact Isolation Forest inputs within each metric."""

    markers = (
        "__nonzero", "__positive_log10", "__positive_log1p",
        "__increment", "__reset", "__state", "__level",
        "__history_24h_z", "__lag_1h", "__history_7d_z", "__lag_6h",
        "__rate_6h", "__sum_6h", "__difference",
        "__rate_24h", "__sum_24h", "__seasonal_difference",
    )
    return next(
        (rank for rank, marker in enumerate(markers) if str(name).endswith(marker)),
        len(markers),
    )

def _balanced_isolation_features(
    policy,
    feature_columns,
    maximum_features_per_metric,
    contextual_metrics=(),
):
    """Choose a deterministic, bounded number of features per metric."""

    maximum = int(maximum_features_per_metric)
    if maximum < 1:
        raise ValueError("maximum_features_per_metric must be positive")
    audit = policy.loc[list(feature_columns), ["metric_id"]].copy()
    audit["feature"] = audit.index.astype(str)
    audit = audit.reset_index(drop=True)
    audit["priority"] = audit["feature"].map(_isolation_feature_priority)
    audit["contextual_metric"] = audit["metric_id"].astype(str).isin(
        set(map(str, contextual_metrics))
    )
    audit = audit.sort_values(
        ["metric_id", "priority", "feature"], kind="stable"
    )
    audit["metric_rank"] = audit.groupby("metric_id").cumcount() + 1
    audit["retained"] = (
        ~audit["contextual_metric"] & audit["metric_rank"].le(maximum)
    )
    selected = audit.loc[audit["retained"], "feature"].tolist()
    return selected, audit.reset_index(drop=True)

def _fit_isolation_forest(
    values,
    *,
    trees,
    maximum_samples,
    maximum_features,
    random_seed,
):
    """Fit one reproducible Isolation Forest with bounded tree samples."""

    maximum_samples = min(int(maximum_samples), len(values))
    if maximum_samples < 2:
        raise ValueError("Isolation Forest requires at least two rows")
    return IsolationForest(
        n_estimators=int(trees),
        max_samples=maximum_samples,
        max_features=maximum_features,
        contamination="auto",
        random_state=int(random_seed),
        n_jobs=-1,
    ).fit(values)

def fit_residual_bundle(
    calibration_features,
    *,
    use_entity_reference,
    catalogue=None,
    feature_directions=None,
    minimum_scales=None,
    reference_exclusions=None,
    maximum_training_rows=150_000,
    maximum_rows_per_entity_day=8,
    random_seed=42,
    isolation_trees=200,
    isolation_max_samples=1024,
    isolation_max_features=1.0,
    maximum_isolation_features_per_metric=5,
    entity_reference_minimum_rows=30,
    entity_reference_minimum_days=2,
    entity_reference_shrinkage_days=7,
    isolation_entity_minimum_rows=30,
    isolation_entity_scale_floor_fraction=0.25,
    fit_multivariate=False,
) -> ResidualBundle:
    """Fit frozen robust references and optional residual ML models."""

    if float(entity_reference_shrinkage_days) <= 0:
        raise ValueError("entity_reference_shrinkage_days must be positive")

    sample = _reference_sample(
        calibration_features,
        maximum_training_rows,
        random_seed,
        maximum_rows_per_entity_day=maximum_rows_per_entity_day,
    )
    sample["entity_id"] = sample["entity_id"].astype(str)
    sample_keys = sample[["entity_id", "episode_id", "event_ts"]].sort_values(
        ["entity_id", "episode_id", "event_ts"]
    )
    selected_row_hash = hashlib.sha256(
        pd.util.hash_pandas_object(sample_keys, index=False).to_numpy().tobytes()
    ).hexdigest()
    candidates = [
        column for column in sample
        if column not in IDENTITY_COLUMNS and not column.endswith("__clipped")
    ]
    feature_audit = pd.DataFrame({
        "feature": candidates,
        "available_fraction": [sample[name].notna().mean() for name in candidates],
        "unique_values": [sample[name].dropna().nunique() for name in candidates],
    })
    feature_audit["retained"] = (
        feature_audit["available_fraction"].ge(0.20)
        & feature_audit["unique_values"].gt(1)
    )
    feature_audit["reason"] = np.select(
        [
            feature_audit["available_fraction"].lt(0.20),
            feature_audit["unique_values"].le(1),
        ],
        ["less_than_20_percent_available", "constant_or_empty"],
        default="retained",
    )
    usable = feature_audit.loc[feature_audit["retained"], "feature"].tolist()
    if not usable:
        raise ValueError("No usable calibration features")

    requested = pd.DataFrame(
        reference_exclusions,
        columns=["entity_id", "metric_id"],
    ).dropna().astype(str).drop_duplicates()
    reference_sample = sample[usable].copy()
    applied = []
    for entity_id, metric_id in requested.itertuples(index=False):
        columns = [
            name for name in usable
            if name.startswith(f"{metric_id}__")
        ]
        rows = sample["entity_id"].eq(entity_id)
        if rows.any() and columns:
            reference_sample.loc[rows, columns] = np.nan
            applied.append((entity_id, metric_id))
    exclusions = pd.DataFrame(
        applied, columns=["entity_id", "metric_id"]
    )

    if catalogue is not None:
        policy = feature_policy(catalogue, usable).set_index("feature")
        catalogue_directions = policy["direction"].to_dict()
        catalogue_scales = policy["minimum_scale"].astype(float).to_dict()
        catalogue_index = catalogue.assign(
            metric_id=catalogue["metric_id"].astype(str)
        ).set_index("metric_id")
        contextual_metrics = set(
            catalogue_index.index[
                catalogue_index.get(
                    "direction", pd.Series("two_sided", index=catalogue_index.index)
                ).astype(str).eq("contextual")
            ]
        )
    else:
        policy = pd.DataFrame({
            "feature": usable,
            "metric_id": [name.split("__", 1)[0] for name in usable],
        }).set_index("feature")
        catalogue_directions = {}
        catalogue_scales = {}
        contextual_metrics = set()
    directions = {
        name: (feature_directions or {}).get(
            name, catalogue_directions.get(name, "two_sided")
        )
        for name in usable
    }
    scale_floors = {
        name: float((minimum_scales or {}).get(
            name, catalogue_scales.get(name, 1e-6)
        ))
        for name in usable
    }

    fallback_centre, fallback_scale = _robust_reference(
        sample[usable], scale_floors
    )
    global_centre, global_scale = _robust_reference(
        reference_sample, scale_floors
    )
    global_centre = global_centre.combine_first(fallback_centre)
    global_scale = global_scale.combine_first(fallback_scale)
    entity_centre = entity_scale = None
    entity_reference_counts = None
    entity_reference_days = None
    if use_entity_reference:
        entity_centre, entity_scale, entity_reference_counts, entity_reference_days = (
            _full_entity_reference(
                calibration_features,
                usable,
                scale_floors,
                minimum_rows=entity_reference_minimum_rows,
                minimum_days=entity_reference_minimum_days,
            )
        )

        # A short local history should influence the pooled reference, not
        # replace it abruptly. Count and elapsed span both control the weight.
        row_weight = entity_reference_counts / (
            entity_reference_counts + float(entity_reference_minimum_rows)
        )
        day_weight = entity_reference_days / (
            entity_reference_days + float(entity_reference_shrinkage_days)
        )
        weight = (row_weight * day_weight).where(entity_centre.notna())
        entity_centre = global_centre + weight * (entity_centre - global_centre)
        entity_scale = global_scale + weight * (entity_scale - global_scale)

        # EDA may flag an entity-metric calibration baseline as atypical or
        # unstable. Keep the entity monitored, but use the pooled reference
        # for that metric instead of freezing the suspect entity baseline.
        for entity_id, metric_id in exclusions.itertuples(index=False):
            columns = [
                name for name in usable
                if name.startswith(f"{metric_id}__")
            ]
            if entity_id in entity_centre.index and columns:
                entity_centre.loc[entity_id, columns] = np.nan
                entity_scale.loc[entity_id, columns] = np.nan

    bundle = {
        "model_core_version": MODEL_CORE_VERSION,
        "feature_columns": usable,
        "global_centre": global_centre,
        "global_scale": global_scale,
        "entity_centre": entity_centre,
        "entity_scale": entity_scale,
        "entity_reference_counts": entity_reference_counts,
        "entity_reference_days": entity_reference_days,
        "use_entity_reference": bool(use_entity_reference),
        "reference_exclusions": exclusions.to_dict("records"),
        "training_rows": len(sample),
        "training_sample_strategy": (
            "all_rows_then_hash_cap"
            if maximum_rows_per_entity_day is None
            else "time_stratified_entity_day_then_hash_cap"
        ),
        "selected_row_hash": selected_row_hash,
        "maximum_rows_per_entity_day": (
            None if maximum_rows_per_entity_day is None
            else int(maximum_rows_per_entity_day)
        ),
        "random_seed": int(random_seed),
        "isolation_trees": int(isolation_trees),
        "isolation_max_samples": int(isolation_max_samples),
        "isolation_max_features": isolation_max_features,
        "maximum_isolation_features_per_metric": int(
            maximum_isolation_features_per_metric
        ),
        "entity_reference_minimum_rows": int(
            entity_reference_minimum_rows
        ),
        "entity_reference_minimum_days": float(
            entity_reference_minimum_days
        ),
        "entity_reference_shrinkage_days": float(
            entity_reference_shrinkage_days
        ),
        "isolation_entity_minimum_rows": int(
            isolation_entity_minimum_rows
        ),
        "isolation_entity_scale_floor_fraction": float(
            isolation_entity_scale_floor_fraction
        ),
        "fit_multivariate": bool(fit_multivariate),
        "feature_directions": directions,
        "minimum_scales": scale_floors,
        "feature_audit": feature_audit,
    }
    residuals = _residual_frame(bundle, sample[[*IDENTITY_COLUMNS, *usable]])
    clean = residuals.fillna(0).clip(-50, 50)
    bundle["residual_scale"] = clean.std().replace(0, 1).fillna(1)
    tail_minimum = min(30, max(3, len(sample) // 2))
    bundle["tail_reference"] = fit_empirical_tail_reference(
        residuals, directions, minimum_observations=tail_minimum
    )
    bundle["tail_reference_minimum_observations"] = tail_minimum

    if fit_multivariate and len(usable) >= 2:
        component_limit = min(len(usable) - 1, 12)
        pca = PCA(n_components=component_limit, svd_solver="full").fit(clean)
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        keep = min(int(np.searchsorted(cumulative, 0.90) + 1), component_limit)
        bundle["pca"] = PCA(n_components=keep, svd_solver="full").fit(clean)
        isolation_features, isolation_audit = _balanced_isolation_features(
            policy,
            usable,
            maximum_isolation_features_per_metric,
            contextual_metrics=contextual_metrics,
        )
        base_features = [
            name for name in isolation_features
            if not _is_temporal_feature(name)
        ]
        if len(base_features) < 2:
            raise ValueError("The base Isolation Forest needs at least two features")
        if len(isolation_features) < 2:
            raise ValueError("The temporal Isolation Forest needs at least two features")
        bundle["isolation_base_features"] = base_features
        bundle["isolation_temporal_features"] = isolation_features
        bundle["isolation_feature_audit"] = isolation_audit
        bundle["feature_metric_ids"] = policy["metric_id"].astype(str).to_dict()
        oriented = _directional_model_frame(residuals, directions).fillna(0)
        bundle["isolation_forest_base"] = _fit_isolation_forest(
            oriented[base_features],
            trees=isolation_trees,
            maximum_samples=isolation_max_samples,
            maximum_features=isolation_max_features,
            random_seed=random_seed,
        )
        bundle["isolation_forest_temporal"] = _fit_isolation_forest(
            oriented[isolation_features],
            trees=isolation_trees,
            maximum_samples=isolation_max_samples,
            maximum_features=isolation_max_features,
            random_seed=random_seed,
        )
        temporal_scores = -bundle[
            "isolation_forest_temporal"
        ].decision_function(oriented[isolation_features])
        bundle["isolation_temporal_tail_reference"] = np.sort(
            temporal_scores[np.isfinite(temporal_scores)]
        )
        bundle["isolation_entity_score_reference"] = (
            _fit_entity_score_reference(
                temporal_scores,
                sample["entity_id"],
                minimum_rows=isolation_entity_minimum_rows,
                scale_floor_fraction_of_global=(
                    isolation_entity_scale_floor_fraction
                ),
            )
        )
        # Compatibility for model cards written before the variants were named.
        bundle["isolation_forest"] = bundle["isolation_forest_temporal"]
    else:
        bundle["pca"] = None
        bundle["isolation_base_features"] = []
        bundle["isolation_temporal_features"] = []
        bundle["isolation_feature_audit"] = pd.DataFrame()
        bundle["feature_metric_ids"] = policy["metric_id"].astype(str).to_dict()
        bundle["isolation_forest_base"] = None
        bundle["isolation_forest_temporal"] = None
        bundle["isolation_temporal_tail_reference"] = np.array([])
        bundle["isolation_entity_score_reference"] = None
        bundle["isolation_forest"] = None
    return bundle

def _row_max(values, names):
    array = values.to_numpy(dtype=float)
    available = np.isfinite(array)
    safe = np.where(available, array, -np.inf)
    positions = safe.argmax(axis=1)
    maximum = safe[np.arange(len(safe)), positions]
    maximum[~available.any(axis=1)] = np.nan
    leading = np.asarray(names, dtype=object)[positions]
    leading[~available.any(axis=1)] = None
    return maximum, leading

def _cusum_scores(
    residuals,
    names,
    allowance=0.5,
    *,
    directions=None,
    reset_before=None,
):
    """Compatibility wrapper around the directional, gap-safe CUSUM."""

    directions = directions or {name: "two_sided" for name in names}
    return directional_cusum(
        residuals[names],
        directions,
        allowance=float(allowance),
        reset_before=reset_before,
    )

def score_residual_episode(
    bundle,
    features,
    *,
    cadence_seconds,
    dispersion_window_seconds,
    cusum_allowance,
    residuals=None,
):
    """Score one episode with rapid, drift, dispersion and residual ML channels."""

    if float(cadence_seconds) <= 0:
        raise ValueError("cadence_seconds must be positive")

    if residuals is None:
        residuals = _residual_frame(bundle, features)
    else:
        residuals = residuals[bundle["feature_columns"]]
    available = residuals.notna().sum(axis=1)
    minimum = max(1, int(np.ceil(len(bundle["feature_columns"]) * 0.20)))
    ready = available.ge(minimum)

    directions = bundle.get("feature_directions", {
        name: "two_sided" for name in bundle["feature_columns"]
    })
    if bundle.get("tail_reference"):
        rapid_evidence = empirical_tail_evidence(
            residuals, bundle["tail_reference"], directions
        )
    else:
        # Compatibility with bundles created before empirical calibration.
        rapid_evidence = residuals.abs()
    rapid_values, rapid_leading = _row_max(
        rapid_evidence, bundle["feature_columns"]
    )
    multimetric_values = _second_metric_evidence(
        rapid_evidence,
        bundle.get(
            "feature_metric_ids",
            {name: name.split("__", 1)[0] for name in bundle["feature_columns"]},
        ),
    )
    multimetric_mean_values = _top_two_metric_mean_evidence(
        rapid_evidence,
        bundle.get(
            "feature_metric_ids",
            {name: name.split("__", 1)[0] for name in bundle["feature_columns"]},
        ),
    )
    level_features = [
        name for name in bundle["feature_columns"]
        if name.endswith((
            "__level", "__increment", "__nonzero",
            "__positive_log10", "__positive_log1p", "__state",
            "__history_z",
        ))
    ] or bundle["feature_columns"]
    timestamps = pd.to_datetime(features["event_ts"], utc=True)
    elapsed = timestamps.diff().dt.total_seconds()
    reset_before = (
        elapsed.isna()
        | elapsed.le(0)
        | elapsed.gt(float(cadence_seconds) * 1.5)
    ).to_numpy()
    drift_values, drift_leading = _cusum_scores(
        residuals,
        level_features,
        allowance=float(cusum_allowance),
        directions={name: directions.get(name, "two_sided") for name in level_features},
        reset_before=reset_before,
    )

    window = max(4, round(float(dispersion_window_seconds) / float(cadence_seconds)))
    spread = pd.DataFrame(index=residuals.index, dtype=float)
    entity_restart = pd.Series(reset_before, index=residuals.index)
    for name in level_features:
        valid = residuals[name].notna()
        # A later valid reading starts a new volatility window after this
        # metric went invalid, even if other metrics kept the entity grid full.
        metric_restart = valid & ~valid.shift(fill_value=False)
        segment_id = (entity_restart | metric_restart).cumsum()
        spread[name] = (
            residuals[name]
            .groupby(segment_id)
            .rolling(window, min_periods=max(3, window // 2))
            .std()
            .reset_index(level=0, drop=True)
            .reindex(residuals.index)
        )
    reference_spread = bundle["residual_scale"].reindex(level_features).replace(0, 1)
    dispersion = np.abs(np.log(spread.div(reference_spread).clip(lower=0.05)))
    dispersion_values, dispersion_leading = _row_max(dispersion, level_features)

    clean = residuals.fillna(0).clip(-50, 50)
    pca_values = np.full(len(features), np.nan)
    isolation_base_values = np.full(len(features), np.nan)
    isolation_temporal_values = np.full(len(features), np.nan)
    isolation_confirmed_values = np.full(len(features), np.nan)
    isolation_soft_confirmed_values = np.full(len(features), np.nan)
    isolation_entity_values = np.full(len(features), np.nan)
    pca_leading = np.full(len(features), None, dtype=object)
    isolation_base_leading = rapid_leading.copy()
    isolation_temporal_leading = rapid_leading.copy()
    if bundle["pca"] is not None:
        projected = bundle["pca"].transform(clean)
        reconstruction = bundle["pca"].inverse_transform(projected)
        error = (clean.to_numpy() - reconstruction) ** 2
        pca_values = error.mean(axis=1)
        pca_leading = np.asarray(bundle["feature_columns"], dtype=object)[error.argmax(axis=1)]
        temporal_model = bundle.get(
            "isolation_forest_temporal", bundle.get("isolation_forest")
        )
        directional = _directional_model_frame(residuals, directions).fillna(0)
        if temporal_model is not None:
            temporal_features = bundle.get(
                "isolation_temporal_features", bundle["feature_columns"]
            )
            temporal_values = directional[temporal_features]
            isolation_temporal_values = -temporal_model.decision_function(
                temporal_values
            )
            isolation_temporal_leading = np.asarray(
                temporal_features, dtype=object
            )[temporal_values.to_numpy().argmax(axis=1)]
            temporal_evidence = _empirical_vector_evidence(
                isolation_temporal_values,
                bundle.get("isolation_temporal_tail_reference", []),
            )
            isolation_confirmed_values = np.minimum(
                temporal_evidence, rapid_values
            )
            # Both inputs are empirical tail evidence, so their equal-weight
            # mean is on the same scale. Unlike the strict minimum, moderate
            # self-history evidence does not erase a strong temporal signal.
            isolation_soft_confirmed_values = np.where(
                np.isfinite(temporal_evidence) & np.isfinite(rapid_values),
                (temporal_evidence + rapid_values) / 2,
                np.nan,
            )
            entity_reference = bundle.get("isolation_entity_score_reference")
            if entity_reference is not None:
                isolation_entity_values = _entity_adjusted_score_evidence(
                    isolation_temporal_values,
                    features["entity_id"],
                    entity_reference,
                )
        base_model = bundle.get("isolation_forest_base")
        base_features = bundle.get("isolation_base_features", [])
        if base_model is not None and base_features:
            base_values = directional[base_features]
            isolation_base_values = -base_model.decision_function(base_values)
            isolation_base_leading = np.asarray(base_features, dtype=object)[
                base_values.to_numpy().argmax(axis=1)
            ]

    output = features[IDENTITY_COLUMNS].copy()
    output["available_features"] = available.to_numpy()
    output["readiness"] = np.where(ready, "monitored", "temporarily_unscoreable")
    channels = {
        "rapid_residual": (rapid_values, rapid_leading),
        "multimetric_residual": (multimetric_values, rapid_leading),
        "multimetric_tail_mean": (multimetric_mean_values, rapid_leading),
        "drift_cusum": (drift_values, drift_leading),
        "dispersion_change": (dispersion_values, dispersion_leading),
        "pca_spe": (pca_values, pca_leading),
        "isolation_forest_base": (
            isolation_base_values, isolation_base_leading,
        ),
        "isolation_forest_temporal": (
            isolation_temporal_values, isolation_temporal_leading,
        ),
        "isolation_forest_confirmed": (
            isolation_confirmed_values, isolation_temporal_leading,
        ),
        "isolation_forest_soft_confirmed": (
            isolation_soft_confirmed_values, isolation_temporal_leading,
        ),
        "isolation_forest_entity_calibrated": (
            isolation_entity_values, isolation_temporal_leading,
        ),
    }
    for channel, (values, leading) in channels.items():
        output[channel] = np.where(ready, values, np.nan)
        output[f"{channel}__leading_feature"] = pd.Series(
            leading, index=output.index, dtype="string"
        )
    return output

def score_residual_file(
    bundle,
    features_path,
    destination,
    *,
    cadence_seconds,
    dispersion_window_seconds,
    cusum_allowance,
    residual_destination=None,
    residual_features=None,
    score_start=None,
    score_end=None,
    progress_every=25,
):
    """Score complete episodes, then keep only the requested partition.

    The ordering is important: lookback rows initialise dispersion and CUSUM
    state, but are removed before thresholds, alerts or evaluation see them.
    """

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    residual_writer = None
    residual_destination = (
        Path(residual_destination) if residual_destination is not None else None
    )
    residual_features = list(
        bundle["feature_columns"]
        if residual_features is None else residual_features
    )
    unknown = set(residual_features) - set(bundle["feature_columns"])
    if unknown:
        raise ValueError(f"Unknown residual features: {sorted(unknown)}")
    started = time.perf_counter()
    written_rows = 0
    episode_count = 0
    try:
        for episode_count, episode in enumerate(
            iter_episode_frames(features_path), start=1
        ):
            residuals = _residual_frame(bundle, episode)
            residual_frame = None
            if residual_destination is not None:
                residual_frame = episode[IDENTITY_COLUMNS].copy()
                # Topology compares harmful-direction evidence, not signed
                # deviations. A cooler device or improved optical power must
                # not become a peer anomaly merely because it is unusual.
                residual_frame[residual_features] = _directional_model_frame(
                    residuals[residual_features],
                    bundle.get("feature_directions", {}),
                )
            scores = score_residual_episode(
                bundle,
                episode,
                cadence_seconds=cadence_seconds,
                dispersion_window_seconds=dispersion_window_seconds,
                cusum_allowance=cusum_allowance,
                residuals=residuals,
            )
            timestamps = pd.to_datetime(scores["event_ts"], utc=True)
            if score_start is not None:
                scores = scores.loc[timestamps.ge(score_start)]
                if residual_frame is not None:
                    residual_frame = residual_frame.loc[
                        timestamps.ge(score_start)
                    ]
                timestamps = pd.to_datetime(scores["event_ts"], utc=True)
            if score_end is not None:
                scores = scores.loc[timestamps.lt(score_end)]
                if residual_frame is not None:
                    residual_frame = residual_frame.loc[
                        timestamps.lt(score_end)
                    ]
            if scores.empty:
                continue
            written_rows += len(scores)
            table = pa.Table.from_pandas(scores, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            writer.write_table(table)
            if residual_frame is not None:
                residual_table = pa.Table.from_pandas(
                    residual_frame.reset_index(drop=True), preserve_index=False
                )
                if residual_writer is None:
                    residual_destination.parent.mkdir(parents=True, exist_ok=True)
                    residual_writer = pq.ParquetWriter(
                        residual_destination, residual_table.schema, compression="zstd"
                    )
                residual_writer.write_table(residual_table)
            if progress_every and (
                episode_count == 1
                or episode_count % int(progress_every) == 0
            ):
                print(
                    f"    scoring: {episode_count} episodes, "
                    f"{written_rows:,} rows, "
                    f"{(time.perf_counter() - started) / 60:.1f} minutes",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()
        if residual_writer is not None:
            residual_writer.close()
    if writer is None:
        raise ValueError("Feature file produced no residual scores")
    if residual_destination is not None and residual_writer is None:
        raise ValueError("Feature file produced no residual rows")
    print(
        f"    scoring complete — {episode_count} episodes, "
        f"{written_rows:,} rows in "
        f"{(time.perf_counter() - started) / 60:.1f} minutes",
        flush=True,
    )
    return destination

def fit_contextual_isolation_forest(
    score_path,
    candidate_features,
    *,
    maximum_training_rows=150_000,
    random_seed=42,
    trees=300,
    maximum_samples=2048,
    maximum_features=0.75,
) -> ContextualIsolationBundle:
    """Fit an Isolation Forest to self and topology evidence together."""

    available = set(pq.ParquetFile(score_path).schema_arrow.names)
    candidates = [name for name in candidate_features if name in available]
    if len(candidates) < 2:
        raise ValueError("Contextual Isolation Forest needs at least two inputs")
    columns = ", ".join(_sql_identifier(name) for name in candidates)
    source = _sql_literal(str(Path(score_path)))
    with _model_duckdb(Path(score_path).parent) as connection:
        sample = connection.execute(f"""
            SELECT {columns}
            FROM read_parquet({source})
            USING SAMPLE reservoir({int(maximum_training_rows)} ROWS)
            REPEATABLE ({int(random_seed)})
        """).df().replace([np.inf, -np.inf], np.nan)

    audit = pd.DataFrame({
        "feature": candidates,
        "available_fraction": [sample[name].notna().mean() for name in candidates],
        "unique_values": [sample[name].dropna().nunique() for name in candidates],
    })
    audit["retained"] = (
        audit["available_fraction"].ge(0.50)
        & audit["unique_values"].gt(1)
    )
    retained = audit.loc[audit["retained"], "feature"].tolist()
    if len(retained) < 2:
        raise ValueError("Contextual evidence has fewer than two usable inputs")

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler(quantile_range=(25, 75))
    values = imputer.fit_transform(sample[retained])
    scaled = np.clip(
        scaler.fit_transform(values), -SCALED_FEATURE_CAP, SCALED_FEATURE_CAP
    )
    model = _fit_isolation_forest(
        scaled,
        trees=trees,
        maximum_samples=maximum_samples,
        maximum_features=maximum_features,
        random_seed=random_seed,
    )
    return {
        "feature_columns": retained,
        "feature_audit": audit,
        "minimum_observed_inputs": max(2, math.ceil(len(retained) / 2)),
        "imputer": imputer,
        "scaler": scaler,
        "model": model,
        "scaled_feature_cap": SCALED_FEATURE_CAP,
        "training_rows": len(sample),
        "random_seed": int(random_seed),
    }

def append_contextual_isolation_scores(
    bundle,
    score_path,
    destination,
    *,
    batch_rows=100_000,
):
    """Append contextual scores while preserving the source Arrow schema."""

    score_path, destination = Path(score_path), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Score destination already exists: {destination}")

    parquet = pq.ParquetFile(score_path)
    names = np.asarray(bundle["feature_columns"], dtype=object)
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix="contextual-score-"
    ) as temporary:
        staged = Path(temporary) / "scores.parquet"
        writer = None
        try:
            for batch in parquet.iter_batches(batch_size=int(batch_rows)):
                frame = batch.to_pandas()
                clean = frame[bundle["feature_columns"]].replace(
                    [np.inf, -np.inf], np.nan
                )
                observed = clean.notna().sum(axis=1).to_numpy()
                ready = observed >= int(bundle["minimum_observed_inputs"])
                values = bundle["imputer"].transform(clean)
                scaled = bundle["scaler"].transform(values)
                scaled = np.clip(
                    scaled,
                    -float(bundle["scaled_feature_cap"]),
                    float(bundle["scaled_feature_cap"]),
                )
                scores = -bundle["model"].decision_function(scaled)
                leading = names[np.abs(scaled).argmax(axis=1)]

                # Keep every existing Arrow field exactly as stored. Converting
                # the full batch back from pandas can turn nullable integers
                # into floats in one batch and integers in the next.
                table = pa.Table.from_batches([batch])
                table = table.append_column(
                    "isolation_forest_contextual",
                    pa.array(scores, mask=~ready, type=pa.float64()),
                )
                table = table.append_column(
                    "isolation_forest_contextual__leading_feature",
                    pa.array(np.where(ready, leading, None), type=pa.string()),
                )
                if writer is None:
                    writer = pq.ParquetWriter(
                        staged, table.schema, compression="zstd"
                    )
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
        if writer is None:
            raise ValueError("Score file produced no contextual model rows")
        staged.replace(destination)
    return destination
