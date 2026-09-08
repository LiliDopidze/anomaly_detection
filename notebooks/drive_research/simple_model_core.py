"""Small, shared mechanics for the final anomaly-model notebooks.

The notebooks keep every scientific choice visible.  This module contains only
the repetitive operations that must be identical during development and
holdout scoring: partition loading, causal feature calculation and model
application.
"""

from __future__ import annotations

import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


MODEL_CORE_VERSION = "2.3.0"
MODEL_IDS = (
    "rapid_residual",
    "drift_cusum",
    "dispersion_change",
    "pca_spe",
    "isolation_forest",
)
IDENTITY_COLUMNS = ["event_ts", "entity_id", "episode_id"]
SCALED_FEATURE_CAP = 50.0
PCA_VARIANCE_TARGET = 0.90

# A short-window robust scale may not claim more precision than a quarter of
# the metric's own long-window variability.  This replaces the previous 1e-9
# floor, under which a flat history made every change a maximal anomaly.
SHORT_SCALE_FLOOR_FRACTION = 0.25
RELATIVE_SCALE_FLOOR = 1e-6
ABSOLUTE_SCALE_FLOOR = 1e-12
PCA_EIGENVALUE_FLOOR = 1e-12


def duration_to_observations(duration_seconds, cadence_seconds):
    """Convert an elapsed-duration rule to a whole number of observations."""

    duration_seconds = float(duration_seconds)
    cadence_seconds = float(cadence_seconds)
    if duration_seconds <= 0 or cadence_seconds <= 0:
        raise ValueError("Duration and cadence must be positive")
    return max(1, math.ceil(duration_seconds / cadence_seconds))


def _sql_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def _sql_identifier(value):
    return '"' + str(value).replace('"', '""') + '"'


def partition_definition(split_root, partition):
    """Return the split type and rows defining one partition."""

    split_root = Path(split_root)
    time_path = split_root / "time_partitions.parquet"
    entity_path = split_root / "entity_partitions.parquet"
    if time_path.is_file():
        rows = pd.read_parquet(time_path)
        rows = rows.loc[rows["partition"].eq(partition)].copy()
        if len(rows) != 1:
            raise ValueError(f"Expected one {partition!r} time partition")
        for column in ("start_ts", "end_ts"):
            rows[column] = pd.to_datetime(rows[column], utc=True)
        return "time", rows
    if entity_path.is_file():
        rows = pd.read_parquet(entity_path)
        rows = rows.loc[rows["partition"].eq(partition), ["entity_id"]].copy()
        rows["entity_id"] = rows["entity_id"].astype(str)
        if rows.empty:
            raise ValueError(f"No entities assigned to {partition!r}")
        return "entity", rows
    raise FileNotFoundError("No time or entity split manifest found")


def materialize_wide_partition(
    core_root,
    split_root,
    partition,
    catalogue,
    destination,
    *,
    lookback_seconds=0,
    memory_limit="3GB",
    threads=2,
):
    """Create one ordered wide Parquet panel for a canonical partition."""

    core_root, destination = Path(core_root), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metrics = catalogue["metric_id"].astype(str).tolist()
    metric_columns = []
    for metric_id in metrics:
        metric = _sql_literal(metric_id)
        metric_columns.extend([
            (
                "max(CASE WHEN metric_id = {metric} "
                "AND quality_code <> 'invalid' "
                "AND value IS NOT NULL AND isfinite(value) "
                "THEN value END) AS {alias}"
            ).format(metric=metric, alias=_sql_identifier(metric_id)),
            (
                "max(CASE WHEN metric_id = {metric} "
                "AND quality_code = 'clipped' THEN 1 ELSE 0 END) AS {alias}"
            ).format(
                metric=metric,
                alias=_sql_identifier(f"{metric_id}__clipped"),
            ),
        ])
    metric_columns = ",\n".join(metric_columns)
    telemetry_glob = str(core_root / "telemetry" / "*.parquet")
    split_type, rows = partition_definition(split_root, partition)
    connection = duckdb.connect()
    connection.execute("SET memory_limit = ?", [memory_limit])
    connection.execute("SET threads = ?", [int(threads)])

    if split_type == "time":
        start = rows.iloc[0]["start_ts"]
        end = rows.iloc[0]["end_ts"]
        read_start = start - pd.Timedelta(seconds=lookback_seconds)
        source = (
            f"read_parquet({_sql_literal(telemetry_glob)}) AS t "
            f"WHERE event_ts >= TIMESTAMPTZ {_sql_literal(read_start.isoformat())} "
            f"AND event_ts < TIMESTAMPTZ {_sql_literal(end.isoformat())}"
        )
        score_start, score_end = start, end
    else:
        connection.register("partition_entities", rows)
        source = (
            f"read_parquet({_sql_literal(telemetry_glob)}) AS t "
            "JOIN partition_entities AS p "
            "ON CAST(t.entity_id AS VARCHAR) = p.entity_id"
        )
        score_start, score_end = None, None

    query = f"""
        COPY (
            SELECT t.event_ts,
                   CAST(t.entity_id AS VARCHAR) AS entity_id,
                   CAST(t.episode_id AS VARCHAR) AS episode_id,
                   {metric_columns}
            FROM {source}
            GROUP BY t.event_ts, t.entity_id, t.episode_id
            ORDER BY entity_id, episode_id, event_ts
        ) TO {_sql_literal(destination)}
          (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """
    connection.execute(query)
    connection.close()
    return {
        "split_type": split_type,
        "score_start": score_start,
        "score_end": score_end,
    }


def iter_episode_frames(path, columns=None, batch_rows=100_000):
    """Yield complete episodes from an ordered Parquet file."""

    parquet = pq.ParquetFile(path)
    carry = pd.DataFrame()
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=columns):
        frame = batch.to_pandas()
        if not carry.empty:
            frame = pd.concat([carry, frame], ignore_index=True)
        if frame.empty:
            continue
        last = frame.iloc[-1][["entity_id", "episode_id"]].astype(str).tolist()
        is_last = (
            frame["entity_id"].astype(str).eq(last[0])
            & frame["episode_id"].astype(str).eq(last[1])
        )
        complete, carry = frame.loc[~is_last], frame.loc[is_last].copy()
        for _, episode in complete.groupby(
            ["entity_id", "episode_id"], sort=False
        ):
            yield episode.reset_index(drop=True)
    if not carry.empty:
        yield carry.reset_index(drop=True)


def _trailing_centre_and_scale(base, window_seconds, minimum_history):
    """Trailing median and robust scale, excluding the current observation."""

    trailing = base.rolling(
        pd.Timedelta(seconds=float(window_seconds)),
        closed="left",
        min_periods=minimum_history,
    )
    centre = trailing.median()
    scale = (trailing.quantile(0.75) - trailing.quantile(0.25)) / 1.349
    return centre, scale


def raw_features(
    panel,
    catalogue,
    *,
    rolling_windows_seconds,
    minimum_history_seconds,
):
    """Create causal features using each metric's declared cadence.

    ``rolling_windows_seconds`` maps a window name to its length in seconds,
    for example ``{"short": 120, "long": 1800}``.  Each continuous metric
    receives one robust deviation per window, so a change the short window
    absorbs into its own reference stays visible against the long one.  The
    longest window also supplies the scale for the change feature and the
    floor for every shorter scale.
    """

    windows = {
        name: float(seconds)
        for name, seconds in sorted(
            dict(rolling_windows_seconds).items(), key=lambda item: float(item[1])
        )
    }
    if not windows:
        raise ValueError("At least one rolling window is required")
    reference_window = list(windows)[-1]

    panel = panel.sort_values("event_ts").reset_index(drop=True)
    output = panel[IDENTITY_COLUMNS].copy()
    metadata = catalogue.set_index("metric_id")
    timestamps = pd.DatetimeIndex(pd.to_datetime(panel["event_ts"], utc=True))

    for metric_id in catalogue["metric_id"].astype(str):
        values = pd.to_numeric(panel[metric_id], errors="coerce")
        base = pd.Series(values.to_numpy(), index=timestamps, dtype=float)
        kind = metadata.loc[metric_id, "measurement_kind"]
        cadence_seconds = pd.to_numeric(
            metadata.loc[metric_id, "expected_cadence_seconds"],
            errors="coerce",
        )
        minimum_history = (
            max(3, round(minimum_history_seconds / cadence_seconds))
            if pd.notna(cadence_seconds) and cadence_seconds > 0 else 3
        )

        if kind == "interval_count":
            base = np.log1p(base.clip(lower=0))

        observed = base.dropna()
        change = observed.diff()
        if pd.notna(cadence_seconds) and cadence_seconds > 0:
            elapsed = observed.index.to_series().diff().dt.total_seconds()
            change = change.mask(elapsed.gt(float(cadence_seconds) * 1.5))
        change = change.reindex(timestamps)

        if kind == "cumulative_counter":
            output[f"{metric_id}__increment"] = change.mask(change.lt(0)).to_numpy()
            output[f"{metric_id}__reset"] = (
                change.lt(0).astype(float).where(change.notna()).to_numpy()
            )
            continue

        if kind == "discrete_state":
            output[f"{metric_id}__level"] = base.to_numpy()
            state_change = observed.ne(observed.shift()).astype(float)
            if len(state_change):
                state_change.iloc[0] = np.nan
            if pd.notna(cadence_seconds) and cadence_seconds > 0:
                state_change = state_change.mask(
                    elapsed.gt(float(cadence_seconds) * 1.5)
                )
            output[f"{metric_id}__change"] = (
                state_change.reindex(timestamps).to_numpy()
            )
            continue

        # Raw levels in physical units are intentionally omitted for gauges
        # and counts: they often differ legitimately between assets. A
        # bounded fraction remains comparable across entities by definition.
        if kind == "bounded_fraction":
            output[f"{metric_id}__level"] = base.to_numpy()

        statistics = {
            name: _trailing_centre_and_scale(base, seconds, minimum_history)
            for name, seconds in windows.items()
        }
        reference_scale = statistics[reference_window][1]

        for name in windows:
            centre, scale = statistics[name]
            floor = np.maximum(
                reference_scale * SHORT_SCALE_FLOOR_FRACTION
                if name != reference_window else 0.0,
                centre.abs() * RELATIVE_SCALE_FLOOR,
            ).clip(lower=ABSOLUTE_SCALE_FLOOR)
            scale = scale.where(scale.gt(floor), floor)
            output[f"{metric_id}__z_{name}"] = ((base - centre) / scale).to_numpy()

        # The change is expressed against the metric's own long-run scale, so
        # a quiet entity is not compressed by a noisy one during the pooled
        # scaling that follows.
        change_scale = reference_scale.where(
            reference_scale.gt(ABSOLUTE_SCALE_FLOOR), ABSOLUTE_SCALE_FLOOR
        )
        output[f"{metric_id}__change_z"] = (change / change_scale).to_numpy()
    return output


def materialize_features(
    wide_path,
    catalogue,
    feature_settings,
    destination,
    *,
    score_start=None,
    score_end=None,
):
    """Write causal features episode by episode with bounded memory."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for panel in iter_episode_frames(wide_path):
            features = raw_features(
                panel,
                catalogue,
                rolling_windows_seconds=feature_settings[
                    "rolling_windows_seconds"
                ],
                minimum_history_seconds=float(
                    feature_settings["minimum_history_seconds"]
                ),
            )
            timestamps = pd.to_datetime(features["event_ts"], utc=True)
            if score_start is not None:
                features = features.loc[timestamps.ge(score_start)]
                timestamps = pd.to_datetime(features["event_ts"], utc=True)
            if score_end is not None:
                features = features.loc[timestamps.lt(score_end)]
            if features.empty:
                continue
            table = pa.Table.from_pandas(features, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("Partition produced no feature rows")
    return destination


def feature_columns(path):
    names = pq.ParquetFile(path).schema_arrow.names
    return [name for name in names if name not in IDENTITY_COLUMNS]


def calibration_sample(path, maximum_rows, random_seed):
    """Return a reproducible reservoir sample without loading the full file."""

    columns = feature_columns(path)
    select = ", ".join(_sql_identifier(column) for column in columns)
    source = _sql_literal(str(Path(path)))
    connection = duckdb.connect()
    frame = connection.execute(f"""
        SELECT {select}
        FROM read_parquet({source})
        USING SAMPLE reservoir({int(maximum_rows)} ROWS)
        REPEATABLE ({int(random_seed)})
    """).df()
    connection.close()
    return frame


def fit_model_bundle(
    calibration_features,
    feature_settings,
    *,
    maximum_training_rows=200_000,
    random_seed=42,
):
    """Fit preprocessing, Isolation Forest and PCA on calibration only."""

    sample = calibration_sample(
        calibration_features, maximum_training_rows, random_seed
    ).replace([np.inf, -np.inf], np.nan)
    usable = [
        column for column in sample
        if sample[column].notna().mean() >= 0.50
        and sample[column].dropna().nunique() > 1
    ]
    if not usable:
        raise ValueError("No usable calibration features")

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler(quantile_range=(25, 75))
    values = imputer.fit_transform(sample[usable])
    scaled = scaler.fit_transform(values)
    scaled = np.clip(scaled, -SCALED_FEATURE_CAP, SCALED_FEATURE_CAP)
    if not np.isfinite(scaled).all() or np.var(scaled, axis=0).sum() == 0:
        raise ValueError("Calibration features have no finite model variance")

    isolation_forest = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination="auto",
        random_state=random_seed,
        n_jobs=-1,
    ).fit(scaled)
    if scaled.shape[1] < 2:
        raise ValueError("PCA baseline requires at least two usable features")
    full_pca = PCA(svd_solver="full").fit(scaled)
    explained = np.cumsum(full_pca.explained_variance_ratio_)
    pca_components = int(np.searchsorted(
        explained, PCA_VARIANCE_TARGET
    ) + 1)
    pca_components = min(pca_components, scaled.shape[1] - 1)
    pca = PCA(n_components=pca_components, svd_solver="full").fit(scaled)

    bundle = {
        "model_core_version": MODEL_CORE_VERSION,
        "feature_settings": dict(feature_settings),
        "feature_columns": usable,
        "imputer": imputer,
        "scaler": scaler,
        "isolation_forest": isolation_forest,
        "pca": pca,
        "pca_variance_target": PCA_VARIANCE_TARGET,
        "scaled_feature_cap": SCALED_FEATURE_CAP,
        "training_rows": len(sample),
        "random_seed": random_seed,
    }
    reference = score_matrix(bundle, sample[usable])
    bundle["score_reference"] = {
        model_id: np.sort(reference[model_id].to_numpy())
        for model_id in MODEL_IDS
    }
    # A PCA that retains almost every direction reconstructs everything, so
    # SPE collapses towards zero and the baseline silently stops alarming.
    spe = reference["pca"].to_numpy()
    upper = float(np.quantile(spe, 0.999))
    if not np.isfinite(upper) or upper <= 0:
        raise ValueError(
            "PCA reconstruction error is degenerate on calibration data; "
            "reduce the retained variance target or the feature count"
        )
    bundle["pca_spe_tail_ratio"] = upper / max(float(np.median(spe)), 1e-12)
    return bundle


def score_matrix(bundle, features):
    """Apply all four models to one feature frame."""

    clean = features[bundle["feature_columns"]].replace(
        [np.inf, -np.inf], np.nan
    )
    values = bundle["imputer"].transform(clean)
    scaled = bundle["scaler"].transform(values)
    if not np.isfinite(scaled).all():
        raise ValueError("Preprocessed model features must be finite")

    # The statistical baseline keeps the full ordering of extreme finite
    # deviations. log1p makes its scale readable without changing ranks.
    statistical = np.max(np.log1p(np.abs(scaled)), axis=1)
    feature_names = np.asarray(bundle["feature_columns"])
    statistical_leading = feature_names[
        np.argmax(np.abs(scaled), axis=1)
    ]

    cap = float(bundle["scaled_feature_cap"])
    model_values = np.clip(scaled, -cap, cap)
    isolation = -bundle["isolation_forest"].decision_function(model_values)

    pca = bundle["pca"]
    projected = pca.transform(model_values)
    reconstructed = pca.inverse_transform(projected)
    residual = model_values - reconstructed
    pca_error = np.mean(residual ** 2, axis=1)
    pca_leading = feature_names[np.argmax(residual ** 2, axis=1)]

    # SPE above answers "has the correlation structure broken?".  T-squared
    # answers "how far along the retained directions has the point moved?".
    # An excursion inside the model subspace is invisible to the first and
    # obvious to the second, so process monitoring reports both.
    eigenvalues = np.maximum(pca.explained_variance_, PCA_EIGENVALUE_FLOOR)
    t_squared = np.sum(projected ** 2 / eigenvalues, axis=1)
    # Exact additive decomposition: contributions sum to T-squared.
    contributions = (projected / eigenvalues) @ pca.components_ * (
        model_values - pca.mean_
    )
    t_squared_leading = feature_names[np.argmax(np.abs(contributions), axis=1)]

    return pd.DataFrame({
        "statistical": statistical,
        "isolation_forest": isolation,
        "pca": pca_error,
        "pca_t2": t_squared,
        "statistical__leading_feature": statistical_leading,
        # Isolation Forest has no faithful built-in local explanation. The
        # largest robust deviation is retained as an explicitly named proxy.
        "isolation_forest__leading_feature": statistical_leading,
        "pca__leading_feature": pca_leading,
        "pca_t2__leading_feature": t_squared_leading,
    })


def score_feature_file(bundle, features_path, destination, batch_rows=100_000):
    """Score a feature file in batches and write one compact score table."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(features_path)
    columns = [*IDENTITY_COLUMNS, *bundle["feature_columns"]]
    writer = None
    try:
        for batch in parquet.iter_batches(batch_size=batch_rows, columns=columns):
            frame = batch.to_pandas()
            scores = score_matrix(bundle, frame[bundle["feature_columns"]])
            if not np.isfinite(scores[list(MODEL_IDS)].to_numpy()).all():
                raise ValueError("A model produced a non-finite anomaly score")
            output = pd.concat(
                [frame[IDENTITY_COLUMNS].reset_index(drop=True), scores], axis=1
            )
            table = pa.Table.from_pandas(output, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("Feature file produced no scores")
    return destination


def fit_isolation_seed_models(
    bundle,
    calibration_features,
    seeds,
    *,
    maximum_training_rows=100_000,
    sample_seed=42,
):
    """Refit Isolation Forest seeds on one fixed calibration sample."""

    sample = calibration_sample(
        calibration_features, maximum_training_rows, sample_seed
    )[bundle["feature_columns"]].replace([np.inf, -np.inf], np.nan)
    values = bundle["imputer"].transform(sample)
    scaled = np.clip(
        bundle["scaler"].transform(values),
        -float(bundle["scaled_feature_cap"]),
        float(bundle["scaled_feature_cap"]),
    )
    return {
        f"if_seed_{int(seed)}": IsolationForest(
            n_estimators=200,
            max_samples="auto",
            contamination="auto",
            random_state=int(seed),
            n_jobs=-1,
        ).fit(scaled)
        for seed in seeds
    }


def fit_isolation_feature_subset(
    bundle,
    calibration_features,
    excluded_features,
    *,
    maximum_training_rows=100_000,
    random_seed=42,
    model_id="isolation_forest_feature_subset",
):
    """Fit one diagnostic Isolation Forest after removing named features."""

    excluded = set(excluded_features)
    kept = [
        name for name in bundle["feature_columns"]
        if name not in excluded
    ]
    if not excluded or len(kept) == len(bundle["feature_columns"]):
        raise ValueError("No fitted feature was excluded")
    if not kept:
        raise ValueError("Feature sensitivity removed every fitted feature")

    sample = calibration_sample(
        calibration_features, maximum_training_rows, random_seed
    )[bundle["feature_columns"]].replace([np.inf, -np.inf], np.nan)
    values = bundle["imputer"].transform(sample)
    scaled = bundle["scaler"].transform(values)
    indices = [bundle["feature_columns"].index(name) for name in kept]
    model_values = np.clip(
        scaled[:, indices],
        -float(bundle["scaled_feature_cap"]),
        float(bundle["scaled_feature_cap"]),
    )
    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination="auto",
        random_state=int(random_seed),
        n_jobs=-1,
    ).fit(model_values)
    return {
        "model_id": str(model_id),
        "model": model,
        "feature_columns": kept,
        "feature_indices": indices,
        "excluded_features": sorted(excluded),
    }


def score_isolation_feature_subset_file(
    bundle,
    sensitivity,
    features_path,
    destination,
    batch_rows=100_000,
):
    """Score a diagnostic feature-subset Isolation Forest in batches."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(features_path)
    columns = [*IDENTITY_COLUMNS, *bundle["feature_columns"]]
    writer = None
    try:
        for batch in parquet.iter_batches(batch_size=batch_rows, columns=columns):
            frame = batch.to_pandas()
            clean = frame[bundle["feature_columns"]].replace(
                [np.inf, -np.inf], np.nan
            )
            values = bundle["imputer"].transform(clean)
            scaled = bundle["scaler"].transform(values)
            subset = scaled[:, sensitivity["feature_indices"]]
            subset = np.clip(
                subset,
                -float(bundle["scaled_feature_cap"]),
                float(bundle["scaled_feature_cap"]),
            )
            score = -sensitivity["model"].decision_function(subset)
            names = np.asarray(sensitivity["feature_columns"])
            leading = names[np.argmax(np.abs(subset), axis=1)]
            output = frame[IDENTITY_COLUMNS].reset_index(drop=True)
            output[sensitivity["model_id"]] = score
            output[f"{sensitivity['model_id']}__leading_feature"] = leading
            table = pa.Table.from_pandas(output, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    destination, table.schema, compression="zstd"
                )
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("Feature file produced no sensitivity scores")
    return destination


def score_isolation_seed_file(
    bundle,
    seed_models,
    features_path,
    destination,
    batch_rows=100_000,
):
    """Score several Isolation Forest seeds in one feature-file pass."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(features_path)
    columns = [*IDENTITY_COLUMNS, *bundle["feature_columns"]]
    feature_names = np.asarray(bundle["feature_columns"])
    writer = None
    try:
        for batch in parquet.iter_batches(batch_size=batch_rows, columns=columns):
            frame = batch.to_pandas()
            clean = frame[bundle["feature_columns"]].replace(
                [np.inf, -np.inf], np.nan
            )
            values = bundle["imputer"].transform(clean)
            scaled = bundle["scaler"].transform(values)
            model_values = np.clip(
                scaled,
                -float(bundle["scaled_feature_cap"]),
                float(bundle["scaled_feature_cap"]),
            )
            leading = feature_names[np.argmax(np.abs(scaled), axis=1)]
            output = frame[IDENTITY_COLUMNS].reset_index(drop=True)
            for model_id, model in seed_models.items():
                output[model_id] = -model.decision_function(model_values)
                output[f"{model_id}__leading_feature"] = leading
            table = pa.Table.from_pandas(output, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    destination, table.schema, compression="zstd"
                )
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("Feature file produced no seed-stability scores")
    return destination


def calibration_thresholds(
    score_path,
    quantiles,
    *,
    block_column,
    minimum_block_rows=1_000,
    model_ids=MODEL_IDS,
):
    """Estimate calibration thresholds from the median block quantile."""

    connection = duckdb.connect()
    rows = []
    source = _sql_literal(str(Path(score_path)))
    block = _sql_identifier(block_column)
    block_counts = connection.execute(f"""
        SELECT count(*) AS total_blocks,
               count(*) FILTER (WHERE rows >= {int(minimum_block_rows)})
                   AS used_blocks
        FROM (
            SELECT {block}, count(*) AS rows
            FROM read_parquet({source})
            GROUP BY {block}
        )
    """).fetchone()
    total_blocks, used_blocks = map(int, block_counts)
    if used_blocks == 0:
        raise ValueError(
            f"No calibration {block_column} blocks contain at least "
            f"{minimum_block_rows} rows"
        )

    for model_id in model_ids:
        for quantile in quantiles:
            threshold = connection.execute(
                f"""
                SELECT median(block_threshold)
                FROM (
                    SELECT {block},
                           approx_quantile(
                               {_sql_identifier(model_id)}, CAST(? AS FLOAT)
                           ) AS block_threshold
                    FROM read_parquet({source})
                    GROUP BY {block}
                    HAVING count(*) >= {int(minimum_block_rows)}
                )
                """,
                [float(quantile)],
            ).fetchone()[0]
            if threshold is None or not np.isfinite(threshold):
                raise ValueError(
                    f"{model_id} has no finite calibration threshold at "
                    f"quantile {quantile}"
                )
            rows.append({
                "model_id": model_id,
                "threshold_quantile": float(quantile),
                "threshold": float(threshold),
                "threshold_block": block_column,
                "minimum_block_rows": int(minimum_block_rows),
                "blocks_total": total_blocks,
                "blocks_used": used_blocks,
                "blocks_excluded": total_blocks - used_blocks,
            })
    connection.close()
    return pd.DataFrame(rows)


def alerts_from_score_file(
    score_path,
    model_id,
    threshold,
    *,
    min_consecutive,
    recovery_consecutive,
):
    """Create alerts episode by episode without materialising all scores."""

    from evaluation_core import SCORE_COLUMNS, scores_to_alerts

    explanation_column = f"{model_id}__leading_feature"
    columns = [*IDENTITY_COLUMNS, model_id, explanation_column]
    alerts = []
    for episode in iter_episode_frames(score_path, columns=columns):
        scores = episode.rename(columns={
            model_id: "anomaly_score",
            explanation_column: "leading_feature",
        })
        scores["model_id"] = model_id
        produced = scores_to_alerts(
            scores[[*SCORE_COLUMNS, "leading_feature"]],
            threshold,
            min_consecutive=int(min_consecutive),
            recovery_consecutive=int(recovery_consecutive),
        )
        if not produced.empty:
            alerts.append(produced)
    if not alerts:
        from evaluation_core import ALERT_COLUMNS
        return pd.DataFrame(columns=ALERT_COLUMNS)
    output = pd.concat(alerts, ignore_index=True).sort_values("alert_start")
    output = output.reset_index(drop=True)
    output["alert_id"] = [
        f"A-{number:06d}" for number in range(1, len(output) + 1)
    ]
    return output


def alert_grid_from_score_file(
    score_path,
    thresholds,
    *,
    persistence,
    recovery_consecutive,
):
    """Create all channel/threshold alert sets with one scan per channel."""

    from evaluation_core import ALERT_COLUMNS, SCORE_COLUMNS, scores_to_alerts

    result = {}
    for model_id, model_thresholds in thresholds.groupby("model_id", sort=False):
        candidate_rows = list(model_thresholds.itertuples(index=False))
        collected = {float(row.threshold_quantile): [] for row in candidate_rows}
        columns = [*IDENTITY_COLUMNS, model_id, f"{model_id}__leading_feature"]
        for episode in iter_episode_frames(score_path, columns=columns):
            scores = episode[IDENTITY_COLUMNS].copy()
            scores["anomaly_score"] = episode[model_id].to_numpy()
            scores["model_id"] = model_id
            scores["leading_feature"] = episode[
                f"{model_id}__leading_feature"
            ].to_numpy()
            for row in candidate_rows:
                produced = scores_to_alerts(
                    scores[[*SCORE_COLUMNS, "leading_feature"]],
                    float(row.threshold),
                    min_consecutive=int(persistence[model_id]),
                    recovery_consecutive=int(recovery_consecutive),
                )
                if not produced.empty:
                    collected[float(row.threshold_quantile)].append(produced)

        for quantile, frames in collected.items():
            if frames:
                alerts = pd.concat(frames, ignore_index=True).sort_values(
                    ["alert_start", "entity_id", "episode_id"]
                ).reset_index(drop=True)
                alerts["alert_id"] = [
                    f"A-{number:09d}" for number in range(1, len(alerts) + 1)
                ]
            else:
                alerts = pd.DataFrame(columns=ALERT_COLUMNS)
            result[(str(model_id), float(quantile))] = alerts
    return result


def candidate_key(model_id, threshold_quantile, persistence_observations):
    """Return a stable key for one development configuration."""

    return (
        f"{model_id}|{float(threshold_quantile):.8g}|"
        f"{int(persistence_observations)}"
    )


def materialize_candidate_alert_grid(
    score_path,
    thresholds,
    persistence_values,
    destination,
    *,
    recovery_consecutive,
):
    """Write candidate alerts to disk with bounded memory.

    The score table is scanned once per model. Candidate alerts are written as
    they are produced, so a large development grid is never retained in RAM.
    """

    from evaluation_core import ALERT_COLUMNS, SCORE_COLUMNS, scores_to_alerts

    required = {"model_id", "threshold_quantile", "threshold"}
    missing = required - set(thresholds.columns)
    if missing:
        raise ValueError(f"Missing threshold columns: {sorted(missing)}")
    persistence_values = sorted({int(value) for value in persistence_values})
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(f"Candidate directory is not empty: {destination}")

    manifest = []
    for model_id in MODEL_IDS:
        model_thresholds = thresholds.loc[
            thresholds["model_id"].eq(model_id)
        ]
        candidates = []
        for threshold_number, row in enumerate(
            model_thresholds.itertuples(index=False), start=1
        ):
            for persistence in persistence_values:
                purpose = getattr(row, "purpose", "operating")
                supplied_key = getattr(row, "candidate_key", None)
                key = supplied_key or (
                    f"{purpose}|"
                    f"{candidate_key(row.model_id, row.threshold_quantile, persistence)}"
                )
                path = destination / (
                    f"{model_id}-q{threshold_number}-p{persistence}.parquet"
                )
                candidates.append({
                    "candidate_key": key,
                    "purpose": purpose,
                    "model_id": model_id,
                    "threshold_quantile": float(row.threshold_quantile),
                    "threshold": float(row.threshold),
                    "persistence_observations": persistence,
                    "alert_path": path,
                    "alert_count": 0,
                    "writer": None,
                })

        columns = [
            *IDENTITY_COLUMNS,
            model_id,
            f"{model_id}__leading_feature",
        ]
        try:
            for episode in iter_episode_frames(score_path, columns=columns):
                scores = episode[IDENTITY_COLUMNS].copy()
                scores["anomaly_score"] = episode[model_id].to_numpy()
                scores["model_id"] = model_id
                scores["leading_feature"] = episode[
                    f"{model_id}__leading_feature"
                ].to_numpy()

                for candidate in candidates:
                    produced = scores_to_alerts(
                        scores[[*SCORE_COLUMNS, "leading_feature"]],
                        candidate["threshold"],
                        min_consecutive=candidate[
                            "persistence_observations"
                        ],
                        recovery_consecutive=int(recovery_consecutive),
                    )
                    if produced.empty:
                        continue
                    first = candidate["alert_count"] + 1
                    candidate["alert_count"] += len(produced)
                    produced["alert_id"] = [
                        f"A-{number:09d}"
                        for number in range(
                            first, candidate["alert_count"] + 1
                        )
                    ]
                    table = pa.Table.from_pandas(
                        produced[ALERT_COLUMNS], preserve_index=False
                    )
                    if candidate["writer"] is None:
                        candidate["writer"] = pq.ParquetWriter(
                            candidate["alert_path"],
                            table.schema,
                            compression="zstd",
                        )
                    candidate["writer"].write_table(table)
        finally:
            for candidate in candidates:
                if candidate["writer"] is not None:
                    candidate["writer"].close()

        for candidate in candidates:
            if not candidate["alert_path"].is_file():
                pd.DataFrame(columns=ALERT_COLUMNS).to_parquet(
                    candidate["alert_path"], index=False
                )
            manifest.append({
                name: candidate[name]
                for name in (
                    "candidate_key", "purpose", "model_id", "threshold_quantile",
                    "threshold", "persistence_observations", "alert_path",
                    "alert_count",
                )
            })
        print(
            f"Prepared {len(candidates)} candidate configurations "
            f"for {model_id}"
        )

    return pd.DataFrame(manifest)


def partition_exposure(score_path, exposure_unit, cadence_seconds):
    """Calculate the monitoring exposure represented by one score file."""

    connection = duckdb.connect()
    source = _sql_literal(str(Path(score_path)))
    episode_count, observed_seconds = connection.execute(f"""
        WITH episodes AS (
            SELECT entity_id, episode_id,
                   min(event_ts) AS first_ts, max(event_ts) AS last_ts
            FROM read_parquet({source})
            GROUP BY entity_id, episode_id
        )
        SELECT count(*),
               sum(date_diff('second', first_ts, last_ts) + {float(cadence_seconds)})
        FROM episodes
    """).fetchone()
    connection.close()
    if exposure_unit == "episode":
        return float(episode_count)
    if exposure_unit == "entity_day":
        return float(observed_seconds) / 86_400
    if exposure_unit == "observed_hour":
        return float(observed_seconds) / 3_600
    raise ValueError("Unsupported exposure unit")


def score_percentiles(bundle, model_id, values):
    reference = np.asarray(bundle["score_reference"][model_id])
    return np.searchsorted(reference, np.asarray(values), side="right") / len(reference)


# ---------------------------------------------------------------------------
# Frozen-residual modelling API
# ---------------------------------------------------------------------------

def measurement_features(panel, catalogue):
    """Apply the declared measurement-kind transformations to one episode.

    The function is causal.  Clipped values are withheld from asset-health
    features and retained as explicit data-quality indicators.
    """

    panel = panel.sort_values("event_ts").reset_index(drop=True)
    output = panel[IDENTITY_COLUMNS].copy()
    metadata = catalogue.set_index("metric_id")
    timestamps = pd.to_datetime(panel["event_ts"], utc=True)

    for metric_id in catalogue["metric_id"].astype(str):
        values = pd.to_numeric(panel.get(metric_id), errors="coerce")
        clipped_name = f"{metric_id}__clipped"
        clipped = pd.to_numeric(
            panel.get(clipped_name, pd.Series(0, index=panel.index)),
            errors="coerce",
        ).fillna(0).astype(bool)
        values = values.mask(clipped)
        kind = str(metadata.loc[metric_id, "measurement_kind"])
        cadence = pd.to_numeric(
            metadata.loc[metric_id, "expected_cadence_seconds"],
            errors="coerce",
        )

        if kind == "bounded_fraction":
            level = np.log10(values.clip(lower=1e-12))
            output[f"{metric_id}__zero"] = values.eq(0).astype(float)
        elif kind == "interval_count":
            level = np.log1p(values.clip(lower=0))
        else:
            level = values.astype(float)

        previous = level.shift()
        difference = level - previous
        if pd.notna(cadence) and cadence > 0:
            elapsed = timestamps.diff().dt.total_seconds()
            difference = difference.mask(elapsed.gt(float(cadence) * 1.5))

        if kind == "cumulative_counter":
            output[f"{metric_id}__increment"] = difference.mask(difference.lt(0))
            output[f"{metric_id}__reset"] = difference.lt(0).where(difference.notna()).astype(float)
        elif kind == "discrete_state":
            output[f"{metric_id}__state"] = level
            output[f"{metric_id}__transition"] = difference.ne(0).where(difference.notna()).astype(float)
        else:
            output[f"{metric_id}__level"] = level
            output[f"{metric_id}__difference"] = difference

        output[clipped_name] = clipped.astype(float)
    return output


def resample_wide_panel(panel, catalogue, cadence_seconds):
    """Aggregate a common-cadence wide panel without changing the source data.

    Gauges use a median, interval counts add within the bin, and counters or
    discrete states keep the last observation.  Clipping flags use a maximum.
    The current research sectors have one native cadence per panel; a future
    mixed-cadence source should be resampled before it is made wide.
    """

    cadence_seconds = float(cadence_seconds)
    native = pd.to_numeric(
        catalogue["expected_cadence_seconds"], errors="coerce"
    ).dropna().unique()
    if len(native) != 1:
        raise ValueError("Wide-panel resampling requires one native cadence")
    if cadence_seconds < float(native[0]):
        raise ValueError("Model cadence cannot be faster than the source cadence")
    if cadence_seconds == float(native[0]):
        return panel

    metadata = catalogue.set_index("metric_id")
    aggregations = {}

    def interval_sum(values):
        return values.sum(min_count=1)

    for metric_id in catalogue["metric_id"].astype(str):
        kind = str(metadata.loc[metric_id, "measurement_kind"])
        aggregations[metric_id] = (
            interval_sum if kind == "interval_count"
            else "last" if kind in {"cumulative_counter", "discrete_state"}
            else "median"
        )
        aggregations[f"{metric_id}__clipped"] = "max"

    rule = pd.Timedelta(seconds=cadence_seconds)
    indexed = panel.set_index(pd.to_datetime(panel["event_ts"], utc=True))
    values = indexed[list(aggregations)].resample(
        rule, origin="start"
    ).agg(aggregations)
    values = values.reset_index(names="event_ts")
    values.insert(1, "entity_id", str(panel["entity_id"].iloc[0]))
    values.insert(2, "episode_id", str(panel["episode_id"].iloc[0]))
    return values


def materialize_measurement_features(
    wide_path,
    catalogue,
    destination,
    *,
    target_cadence_seconds=None,
    score_start=None,
    score_end=None,
):
    """Write transformed features one complete episode at a time."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for panel in iter_episode_frames(wide_path):
            feature_catalogue = catalogue
            if target_cadence_seconds is not None:
                panel = resample_wide_panel(
                    panel, catalogue, target_cadence_seconds
                )
                feature_catalogue = catalogue.copy()
                feature_catalogue["expected_cadence_seconds"] = (
                    target_cadence_seconds
                )
            features = measurement_features(panel, feature_catalogue)
            times = pd.to_datetime(features["event_ts"], utc=True)
            if score_start is not None:
                features = features.loc[times.ge(score_start)]
                times = pd.to_datetime(features["event_ts"], utc=True)
            if score_end is not None:
                features = features.loc[times.lt(score_end)]
            if features.empty:
                continue
            table = pa.Table.from_pandas(features, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("Partition produced no transformed feature rows")
    return destination


def _robust_reference(frame):
    centre = frame.median()
    scale = (frame.quantile(0.75) - frame.quantile(0.25)) / 1.349

    # Quantised and zero-inflated features often have an IQR of zero even
    # though their occasional non-zero changes are perfectly valid. In that
    # case use the typical non-zero deviation instead of an arbitrary tiny
    # denominator, which would manufacture enormous residual scores.
    deviation = frame.sub(centre).abs()
    fallback = deviation.mask(deviation.eq(0)).median()
    floor = np.maximum(centre.abs() * 1e-6, 1e-6)
    scale = scale.where(scale.gt(floor), fallback)
    scale = scale.where(scale.gt(floor), 1.0)
    return centre, scale


def _reference_sample(path, maximum_rows, random_seed):
    source = _sql_literal(str(Path(path)))
    with duckdb.connect() as connection:
        return connection.execute(f"""
            SELECT * FROM read_parquet({source})
            USING SAMPLE reservoir({int(maximum_rows)} ROWS)
            REPEATABLE ({int(random_seed)})
        """).df()


def _reference_components(bundle, frame):
    """Return transformed values and their frozen centre and scale."""

    features = bundle["feature_columns"]
    values = frame[features].replace([np.inf, -np.inf], np.nan).astype(float)
    centre = pd.DataFrame(
        np.tile(bundle["global_centre"].to_numpy(), (len(frame), 1)),
        columns=features,
        index=frame.index,
    )
    scale = pd.DataFrame(
        np.tile(bundle["global_scale"].to_numpy(), (len(frame), 1)),
        columns=features,
        index=frame.index,
    )
    if bundle["entity_centre"] is not None:
        entity_ids = frame["entity_id"].astype(str)
        entity_centre = bundle["entity_centre"].reindex(entity_ids).set_axis(frame.index)
        entity_scale = bundle["entity_scale"].reindex(entity_ids).set_axis(frame.index)
        centre = entity_centre.combine_first(centre)
        scale = entity_scale.combine_first(scale)
    return values, centre, scale


def _residual_frame(bundle, frame):
    """Standardise against calibration-frozen references."""

    values, centre, scale = _reference_components(bundle, frame)
    return (values - centre) / scale


def fit_residual_bundle(
    calibration_features,
    *,
    use_entity_reference,
    reference_exclusions=None,
    maximum_training_rows=150_000,
    random_seed=42,
    isolation_trees=200,
    fit_multivariate=False,
):
    """Fit frozen robust references and optional residual ML models."""

    sample = _reference_sample(
        calibration_features, maximum_training_rows, random_seed
    )
    sample["entity_id"] = sample["entity_id"].astype(str)
    candidates = [
        column for column in sample
        if column not in IDENTITY_COLUMNS and not column.endswith("__clipped")
    ]
    usable = [
        column for column in candidates
        if sample[column].notna().mean() >= 0.20
        and sample[column].dropna().nunique() > 1
    ]
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

    fallback_centre, fallback_scale = _robust_reference(sample[usable])
    global_centre, global_scale = _robust_reference(reference_sample)
    global_centre = global_centre.combine_first(fallback_centre)
    global_scale = global_scale.combine_first(fallback_scale)
    entity_centre = entity_scale = None
    if use_entity_reference:
        counts = sample.groupby("entity_id")[usable].count()
        centre = sample.groupby("entity_id")[usable].median()
        q25 = sample.groupby("entity_id")[usable].quantile(0.25)
        q75 = sample.groupby("entity_id")[usable].quantile(0.75)
        scales = (q75 - q25) / 1.349
        valid = counts.ge(30)
        entity_centre = centre.where(valid)
        floors = np.maximum(entity_centre.abs() * 1e-6, 1e-6)
        entity_scale = scales.where(valid & scales.gt(floors))

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
        "use_entity_reference": bool(use_entity_reference),
        "reference_exclusions": exclusions.to_dict("records"),
        "training_rows": len(sample),
        "random_seed": int(random_seed),
        "isolation_trees": int(isolation_trees),
        "fit_multivariate": bool(fit_multivariate),
    }
    residuals = _residual_frame(bundle, sample[[*IDENTITY_COLUMNS, *usable]])
    clean = residuals.fillna(0).clip(-50, 50)
    bundle["residual_scale"] = clean.std().replace(0, 1).fillna(1)

    if fit_multivariate and len(usable) >= 2:
        component_limit = min(len(usable) - 1, 12)
        pca = PCA(n_components=component_limit, svd_solver="full").fit(clean)
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        keep = min(int(np.searchsorted(cumulative, 0.90) + 1), component_limit)
        bundle["pca"] = PCA(n_components=keep, svd_solver="full").fit(clean)
        bundle["isolation_forest"] = IsolationForest(
            n_estimators=int(isolation_trees),
            max_samples=min(1024, len(clean)),
            contamination="auto",
            random_state=int(random_seed),
            n_jobs=-1,
        ).fit(clean)
    else:
        bundle["pca"] = None
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


def _cusum_scores(residuals, names, allowance=0.5):
    values = residuals[names].to_numpy(dtype=float)
    positive = np.zeros(values.shape[1])
    negative = np.zeros(values.shape[1])
    scores = np.full(len(values), np.nan)
    leading = np.full(len(values), None, dtype=object)
    for row_number, row in enumerate(values):
        valid = np.isfinite(row)
        positive[~valid] = 0
        negative[~valid] = 0
        positive[valid] = np.maximum(0, positive[valid] + row[valid] - allowance)
        negative[valid] = np.maximum(0, negative[valid] - row[valid] - allowance)
        combined = np.maximum(positive, negative)
        if valid.any():
            position = int(np.argmax(combined))
            scores[row_number] = combined[position]
            leading[row_number] = names[position]
    return scores, leading


def score_residual_episode(
    bundle,
    features,
    *,
    cadence_seconds,
    dispersion_window_seconds,
    cusum_allowance,
):
    """Score one episode with rapid, drift, dispersion and residual ML channels."""

    residuals = _residual_frame(bundle, features)
    available = residuals.notna().sum(axis=1)
    minimum = max(1, int(np.ceil(len(bundle["feature_columns"]) * 0.20)))
    ready = available.ge(minimum)

    rapid_values, rapid_leading = _row_max(residuals.abs(), bundle["feature_columns"])
    level_features = [
        name for name in bundle["feature_columns"]
        if name.endswith(("__level", "__increment"))
    ] or bundle["feature_columns"]
    drift_values, drift_leading = _cusum_scores(
        residuals, level_features, allowance=float(cusum_allowance)
    )

    window = max(4, round(float(dispersion_window_seconds) / float(cadence_seconds)))
    spread = residuals[level_features].rolling(window, min_periods=max(3, window // 2)).std()
    reference_spread = bundle["residual_scale"].reindex(level_features).replace(0, 1)
    dispersion = np.abs(np.log(spread.div(reference_spread).clip(lower=0.05)))
    dispersion_values, dispersion_leading = _row_max(dispersion, level_features)

    clean = residuals.fillna(0).clip(-50, 50)
    pca_values = np.full(len(features), np.nan)
    isolation_values = np.full(len(features), np.nan)
    pca_leading = np.full(len(features), None, dtype=object)
    isolation_leading = rapid_leading.copy()
    if bundle["pca"] is not None:
        projected = bundle["pca"].transform(clean)
        reconstruction = bundle["pca"].inverse_transform(projected)
        error = (clean.to_numpy() - reconstruction) ** 2
        pca_values = error.mean(axis=1)
        pca_leading = np.asarray(bundle["feature_columns"], dtype=object)[error.argmax(axis=1)]
        isolation_values = -bundle["isolation_forest"].decision_function(clean)

    output = features[IDENTITY_COLUMNS].copy()
    output["available_features"] = available.to_numpy()
    output["readiness"] = np.where(ready, "monitored", "temporarily_unscoreable")
    channels = {
        "rapid_residual": (rapid_values, rapid_leading),
        "drift_cusum": (drift_values, drift_leading),
        "dispersion_change": (dispersion_values, dispersion_leading),
        "pca_spe": (pca_values, pca_leading),
        "isolation_forest": (isolation_values, isolation_leading),
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
    score_start=None,
    score_end=None,
):
    """Score complete episodes, then keep only the requested partition.

    The ordering is important: lookback rows initialise dispersion and CUSUM
    state, but are removed before thresholds, alerts or evaluation see them.
    """

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for episode in iter_episode_frames(features_path):
            scores = score_residual_episode(
                bundle,
                episode,
                cadence_seconds=cadence_seconds,
                dispersion_window_seconds=dispersion_window_seconds,
                cusum_allowance=cusum_allowance,
            )
            timestamps = pd.to_datetime(scores["event_ts"], utc=True)
            if score_start is not None:
                scores = scores.loc[timestamps.ge(score_start)]
                timestamps = pd.to_datetime(scores["event_ts"], utc=True)
            if score_end is not None:
                scores = scores.loc[timestamps.lt(score_end)]
            if scores.empty:
                continue
            table = pa.Table.from_pandas(scores, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("Feature file produced no residual scores")
    return destination


def case_score_trace(
    score_path,
    feature_path,
    cases,
    members,
    alerts,
    bundle,
    thresholds,
    channels,
    *,
    window_seconds,
    maximum_cases=20,
):
    """Return compact score and reference evidence around top cases."""

    columns = [
        "case_id", "event_ts", "entity_id", "episode_id", "channel",
        "anomaly_score", "threshold", "leading_feature",
        "observed_transformed", "expected_transformed",
        "standardized_residual",
    ]
    if cases.empty or members.empty or alerts.empty:
        return pd.DataFrame(columns=columns)

    selected = cases.nlargest(
        int(maximum_cases), "anomaly_evidence_score"
    )[["case_id", "peak_ts"]]
    windows = (
        members.merge(
            alerts[["alert_id", "episode_id"]],
            on="alert_id", how="left", validate="many_to_one",
        )
        .merge(selected, on="case_id", how="inner", validate="many_to_one")
        [["case_id", "entity_id", "episode_id", "peak_ts"]]
        .drop_duplicates()
    )
    radius = pd.Timedelta(seconds=float(window_seconds))
    windows["trace_start"] = pd.to_datetime(windows["peak_ts"], utc=True) - radius
    windows["trace_end"] = pd.to_datetime(windows["peak_ts"], utc=True) + radius
    windows = windows.drop(columns="peak_ts")

    feature_columns = bundle["feature_columns"]
    score_select = []
    for channel in channels:
        score_select.extend([
            f"s.{_sql_identifier(channel)}",
            f"s.{_sql_identifier(channel + '__leading_feature')}",
        ])
    feature_select = [f"f.{_sql_identifier(name)}" for name in feature_columns]
    with duckdb.connect() as connection:
        connection.register("trace_windows", windows)
        data = connection.execute(f"""
            SELECT w.case_id, s.event_ts,
                   CAST(s.entity_id AS VARCHAR) AS entity_id,
                   CAST(s.episode_id AS VARCHAR) AS episode_id,
                   {', '.join(score_select + feature_select)}
            FROM read_parquet({_sql_literal(str(Path(score_path)))}) AS s
            JOIN trace_windows AS w
              ON CAST(s.entity_id AS VARCHAR) = w.entity_id
             AND CAST(s.episode_id AS VARCHAR) = w.episode_id
             AND s.event_ts BETWEEN w.trace_start AND w.trace_end
            JOIN read_parquet({_sql_literal(str(Path(feature_path)))}) AS f
              ON s.event_ts = f.event_ts
             AND CAST(s.entity_id AS VARCHAR) = CAST(f.entity_id AS VARCHAR)
             AND CAST(s.episode_id AS VARCHAR) = CAST(f.episode_id AS VARCHAR)
            ORDER BY w.case_id, s.entity_id, s.event_ts
        """).df()

    global_centre = bundle["global_centre"]
    global_scale = bundle["global_scale"]
    entity_centre = bundle["entity_centre"]
    entity_scale = bundle["entity_scale"]

    def frozen_reference(entity_id, feature):
        centre = scale = np.nan
        if entity_centre is not None and entity_id in entity_centre.index:
            centre = entity_centre.at[entity_id, feature]
            scale = entity_scale.at[entity_id, feature]
        if pd.isna(centre):
            centre = global_centre[feature]
        if pd.isna(scale):
            scale = global_scale[feature]
        return float(centre), float(scale)

    rows = []
    for channel in channels:
        leading_column = f"{channel}__leading_feature"
        for _, record in data.iterrows():
            feature = record.get(leading_column)
            score = record.get(channel)
            if pd.isna(score) or pd.isna(feature) or feature not in feature_columns:
                continue
            observed = record.get(feature)
            centre, scale = frozen_reference(str(record["entity_id"]), feature)
            residual = (
                (float(observed) - centre) / scale
                if pd.notna(observed) and np.isfinite(scale) and scale > 0
                else np.nan
            )
            rows.append({
                "case_id": record["case_id"],
                "event_ts": record["event_ts"],
                "entity_id": str(record["entity_id"]),
                "episode_id": str(record["episode_id"]),
                "channel": channel,
                "anomaly_score": float(score),
                "threshold": float(thresholds[channel]),
                "leading_feature": feature,
                "observed_transformed": observed,
                "expected_transformed": centre,
                "standardized_residual": residual,
            })
    return pd.DataFrame(rows, columns=columns)
