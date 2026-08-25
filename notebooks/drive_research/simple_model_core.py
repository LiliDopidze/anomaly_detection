"""Small, shared mechanics for the final anomaly-model notebooks.

The notebooks keep every scientific choice visible.  This module contains only
the repetitive operations that must be identical during development and
holdout scoring: partition loading, causal feature calculation and model
application.
"""

from __future__ import annotations

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


MODEL_CORE_VERSION = "1.3.0"
MODEL_IDS = ("statistical", "isolation_forest", "pca", "pca_t2")
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
    metric_columns = ",\n".join(
        "max(CASE WHEN metric_id = {metric} "
        "AND quality_code <> 'invalid' "
        "AND value IS NOT NULL AND isfinite(value) "
        "THEN value END) AS {alias}".format(
            metric=_sql_literal(metric_id), alias=_sql_identifier(metric_id)
        )
        for metric_id in metrics
    )
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


def score_quantile_thresholds(score_path, quantiles):
    """Return pooled score quantiles for a threshold sweep."""

    connection = duckdb.connect()
    source = _sql_literal(str(Path(score_path)))
    rows = []
    for model_id in MODEL_IDS:
        for quantile in quantiles:
            threshold = connection.execute(
                f"SELECT approx_quantile({_sql_identifier(model_id)}, "
                "CAST(? AS FLOAT)) "
                f"FROM read_parquet({source})",
                [float(quantile)],
            ).fetchone()[0]
            if threshold is None or not np.isfinite(threshold):
                raise ValueError(
                    f"{model_id} has no finite development threshold at "
                    f"quantile {quantile}"
                )
            rows.append({
                "model_id": model_id,
                "threshold_quantile": float(quantile),
                "threshold": float(threshold),
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
