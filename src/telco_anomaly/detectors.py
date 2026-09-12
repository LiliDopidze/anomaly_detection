"""Small, shared mechanics for the final anomaly-model notebooks.

The notebooks keep every scientific choice visible.  This module contains only
the repetitive operations that must be identical during development and
holdout scoring: partition loading, causal feature calculation and model
application.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
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

from .features import (
    add_causal_history,
    add_causal_seasonal_differences,
    add_causal_temporal_features,
    directional_cusum,
    empirical_tail_evidence,
    feature_policy,
    fit_empirical_tail_reference,
    transform_episode,
)


MODEL_CORE_VERSION = "4.3.2"
MODEL_IDS = (
    "rapid_residual",
    "drift_cusum",
    "peer_deviation",
    "group_common_mode",
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


@contextmanager
def _model_duckdb(work_directory=None):
    """Use bounded memory and disposable local spill space for model scans."""

    parent = Path(
        work_directory
        or os.getenv("TELCO_WORK_ROOT", tempfile.gettempdir())
    )
    parent.mkdir(parents=True, exist_ok=True)
    memory_limit = os.getenv("TELCO_MODEL_DUCKDB_MEMORY_LIMIT", "1GB")
    threads = int(os.getenv("TELCO_MODEL_DUCKDB_THREADS", "1"))
    with tempfile.TemporaryDirectory(
        dir=parent, prefix="telco-model-duckdb-"
    ) as spill_directory:
        with duckdb.connect() as connection:
            connection.execute("SET memory_limit = ?", [memory_limit])
            connection.execute("SET threads = ?", [threads])
            connection.execute("SET temp_directory = ?", [spill_directory])
            connection.execute("SET preserve_insertion_order = false")
            yield connection


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
    spill_directory = destination.parent / f".{destination.stem}-duckdb"
    spill_directory.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("SET memory_limit = ?", [memory_limit])
    connection.execute("SET threads = ?", [int(threads)])
    connection.execute("SET temp_directory = ?", [str(spill_directory)])
    connection.execute("SET preserve_insertion_order = false")

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
    started = time.perf_counter()
    print(
        f"  {partition}: pivoting canonical telemetry to a local wide panel",
        flush=True,
    )
    try:
        connection.execute(query)
    finally:
        connection.close()
        shutil.rmtree(spill_directory, ignore_errors=True)
    rows = pq.ParquetFile(destination).metadata.num_rows
    print(
        f"  {partition}: wide panel complete — {rows:,} rows "
        f"in {(time.perf_counter() - started) / 60:.1f} minutes",
        flush=True,
    )
    return {
        "split_type": split_type,
        "score_start": score_start,
        "score_end": score_end,
    }


def split_feature_file_by_time(
    source,
    fit_destination,
    threshold_destination,
    *,
    fit_fraction=0.70,
):
    """Split calibration features chronologically into fit and threshold slices."""

    if not 0 < float(fit_fraction) < 1:
        raise ValueError("fit_fraction must be between zero and one")
    source = Path(source)
    fit_destination = Path(fit_destination)
    threshold_destination = Path(threshold_destination)
    fit_destination.parent.mkdir(parents=True, exist_ok=True)
    threshold_destination.parent.mkdir(parents=True, exist_ok=True)
    source_sql = _sql_literal(str(source))

    with duckdb.connect() as connection:
        bounds = connection.execute(f"""
            SELECT min(event_ts), max(event_ts)
            FROM read_parquet({source_sql})
        """).fetchone()
        start, end = map(lambda value: pd.to_datetime(value, utc=True), bounds)
        if pd.isna(start) or pd.isna(end) or start >= end:
            raise ValueError("Calibration features need a non-empty time span")
        cutoff = start + (end - start) * float(fit_fraction)
        for destination, operator in (
            (fit_destination, "<"),
            (threshold_destination, ">="),
        ):
            connection.execute(f"""
                COPY (
                    SELECT * FROM read_parquet({source_sql})
                    WHERE event_ts {operator} TIMESTAMPTZ {_sql_literal(cutoff.isoformat())}
                ) TO {_sql_literal(str(destination))}
                (FORMAT PARQUET, COMPRESSION ZSTD)
            """)

    fit_rows = pq.ParquetFile(fit_destination).metadata.num_rows
    threshold_rows = pq.ParquetFile(threshold_destination).metadata.num_rows
    if fit_rows == 0 or threshold_rows == 0:
        raise ValueError("Calibration split produced an empty slice")
    return {
        "cutoff": cutoff,
        "fit_rows": fit_rows,
        "threshold_rows": threshold_rows,
        "fit_fraction": float(fit_fraction),
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
    block_duration_seconds=None,
    minimum_block_rows=1_000,
    model_ids=MODEL_IDS,
):
    """Estimate thresholds from calibration block maxima.

    One maximum per independent entity or episode absorbs the many chances to
    fire across timestamps, metrics and topology levels inside a channel.
    """

    rows = []
    source = _sql_literal(str(Path(score_path)))
    block = _sql_identifier(block_column)
    duration = None
    if block_duration_seconds is not None:
        duration = float(block_duration_seconds)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("block_duration_seconds must be positive")
    score_columns = set(pq.ParquetFile(score_path).schema_arrow.names)
    with _model_duckdb(Path(score_path).parent) as connection:
        for model_id in model_ids:
            started = time.perf_counter()
            print(f"    threshold calibration: {model_id}", flush=True)
            model_column = _sql_identifier(model_id)
            scope_columns = {
                f"{model_id}__scope_type", f"{model_id}__scope_id",
            }
            if scope_columns <= score_columns:
                base_key = (
                    f"coalesce(CAST({_sql_identifier(model_id + '__scope_type')} AS VARCHAR), '') "
                    f"|| ':' || coalesce(CAST({_sql_identifier(model_id + '__scope_id')} AS VARCHAR), '')"
                )
                block_label = f"{model_id}_scope"
                # A scope score is repeated on descendant rows. Completeness
                # therefore counts distinct observation times, not children.
                block_rows = "count(DISTINCT event_ts)"
            else:
                base_key = f"CAST({block} AS VARCHAR)"
                block_label = block_column
                block_rows = f"count({model_column})"
            if duration is not None:
                block_key = (
                    f"({base_key}) || ':' || "
                    f"CAST(floor(epoch(event_ts) / {duration}) AS BIGINT)"
                )
                block_label = f"{block_label}+{duration:g}s"
            else:
                block_key = base_key

            maxima = connection.execute(f"""
                SELECT {block_key} AS calibration_block,
                       max({model_column}) AS block_maximum,
                       {block_rows} AS rows
                FROM read_parquet({source})
                WHERE {model_column} IS NOT NULL
                GROUP BY calibration_block
            """).df()
            total_blocks = len(maxima)
            usable = maxima.loc[
                maxima["rows"].ge(int(minimum_block_rows)), "block_maximum"
            ]
            used_blocks = len(usable)
            if used_blocks == 0:
                raise ValueError(
                    f"No calibration blocks for {model_id} contain at least "
                    f"{minimum_block_rows} rows"
                )

            connection.register("calibration_block_maxima", maxima)
            requested_quantiles = list(map(float, quantiles))
            quantile_expressions = ", ".join(
                "quantile_cont(block_maximum, CAST(? AS FLOAT))"
                for _ in requested_quantiles
            )
            threshold_values = connection.execute(
                f"""
                SELECT {quantile_expressions}
                FROM calibration_block_maxima
                WHERE rows >= ?
                """,
                [*requested_quantiles, int(minimum_block_rows)],
            ).fetchone()
            connection.unregister("calibration_block_maxima")

            for quantile, threshold in zip(
                requested_quantiles, threshold_values
            ):
                if threshold is None or not np.isfinite(threshold):
                    raise ValueError(
                        f"{model_id} has no finite calibration threshold at "
                        f"quantile {quantile}"
                    )
                rows.append({
                    "model_id": model_id,
                    "threshold_quantile": quantile,
                    "threshold": float(threshold),
                    "threshold_block": block_label,
                    "threshold_method": "empirical_quantile_of_block_maxima",
                    "minimum_block_rows": int(minimum_block_rows),
                    "blocks_total": total_blocks,
                    "blocks_used": used_blocks,
                    "blocks_excluded": total_blocks - used_blocks,
                })
            print(
                f"    threshold calibration: {model_id} complete "
                f"({used_blocks}/{total_blocks} blocks used, "
                f"{time.perf_counter() - started:.1f}s)",
                flush=True,
            )
    return pd.DataFrame(rows)


def _deduplicate_scope_alerts(alerts):
    """Keep one overlapping alert for each physical scope.

    Group scores are repeated on descendant entity rows so they can share the
    ordinary alert interface. Descendants with slightly different coverage can
    produce intervals whose endpoints do not match exactly. Consolidating
    overlapping copies here prevents one physical signal being counted more
    than once before incident formation.
    """

    if alerts.empty or "evidence_scope_id" not in alerts:
        return alerts
    scoped = alerts["evidence_scope_id"].notna()
    ordinary = alerts.loc[~scoped].to_dict("records")
    consolidated = []
    keys = ["model_id", "evidence_scope_type", "evidence_scope_id"]
    for _, group in alerts.loc[scoped].groupby(keys, sort=True):
        current = None
        for alert in group.sort_values(["alert_start", "alert_end"]).to_dict("records"):
            if current is None:
                current = alert
                continue
            if alert["alert_start"] <= current["alert_end"]:
                current["alert_end"] = max(current["alert_end"], alert["alert_end"])
                current["n_scores"] = max(current["n_scores"], alert["n_scores"])
                if alert["peak_score"] > current["peak_score"]:
                    for name in (
                        "entity_id", "episode_id", "peak_ts", "peak_score",
                        "leading_feature", "evidence_affected_fraction",
                    ):
                        current[name] = alert[name]
            else:
                consolidated.append(current)
                current = alert
        if current is not None:
            consolidated.append(current)
    return pd.DataFrame(ordinary + consolidated, columns=alerts.columns)


def alerts_from_score_file(
    score_path,
    model_id,
    threshold,
    *,
    min_consecutive,
    recovery_consecutive,
):
    """Create alerts episode by episode without materialising all scores."""

    from .evaluation import SCORE_COLUMNS, scores_to_alerts

    explanation_column = f"{model_id}__leading_feature"
    score_columns = set(pq.ParquetFile(score_path).schema_arrow.names)
    scope_type_column = f"{model_id}__scope_type"
    scope_id_column = f"{model_id}__scope_id"
    affected_fraction_column = f"{model_id}__affected_fraction"
    scope_available = {
        scope_type_column, scope_id_column,
    } <= score_columns
    columns = [*IDENTITY_COLUMNS, model_id, explanation_column]
    if scope_available:
        columns += [scope_type_column, scope_id_column]
    fraction_available = affected_fraction_column in score_columns
    if fraction_available:
        columns.append(affected_fraction_column)
    alerts = []
    for episode in iter_episode_frames(score_path, columns=columns):
        scores = episode.rename(columns={
            model_id: "anomaly_score",
            explanation_column: "leading_feature",
        })
        scores["model_id"] = model_id
        scores["evidence_scope_type"] = (
            episode[scope_type_column] if scope_available else pd.NA
        )
        scores["evidence_scope_id"] = (
            episode[scope_id_column] if scope_available else pd.NA
        )
        scores["evidence_affected_fraction"] = (
            episode[affected_fraction_column]
            if fraction_available else pd.NA
        )
        produced = scores_to_alerts(
            scores[[
                *SCORE_COLUMNS, "leading_feature",
                "evidence_scope_type", "evidence_scope_id",
                "evidence_affected_fraction",
            ]],
            threshold,
            min_consecutive=int(min_consecutive),
            recovery_consecutive=int(recovery_consecutive),
        )
        if not produced.empty:
            alerts.append(produced)
    if not alerts:
        from .evaluation import ALERT_COLUMNS
        return pd.DataFrame(columns=ALERT_COLUMNS)
    output = _deduplicate_scope_alerts(
        pd.concat(alerts, ignore_index=True)
    ).sort_values("alert_start")
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

    from .evaluation import ALERT_COLUMNS, SCORE_COLUMNS, scores_to_alerts

    result = {}
    score_columns = set(pq.ParquetFile(score_path).schema_arrow.names)
    for model_id, model_thresholds in thresholds.groupby("model_id", sort=False):
        candidate_rows = list(model_thresholds.itertuples(index=False))
        collected = {float(row.threshold_quantile): [] for row in candidate_rows}
        scope_type_column = f"{model_id}__scope_type"
        scope_id_column = f"{model_id}__scope_id"
        affected_fraction_column = f"{model_id}__affected_fraction"
        scope_available = {
            scope_type_column, scope_id_column,
        } <= score_columns
        columns = [*IDENTITY_COLUMNS, model_id, f"{model_id}__leading_feature"]
        if scope_available:
            columns += [scope_type_column, scope_id_column]
        fraction_available = affected_fraction_column in score_columns
        if fraction_available:
            columns.append(affected_fraction_column)
        for episode in iter_episode_frames(score_path, columns=columns):
            scores = episode[IDENTITY_COLUMNS].copy()
            scores["anomaly_score"] = episode[model_id].to_numpy()
            scores["model_id"] = model_id
            scores["leading_feature"] = episode[
                f"{model_id}__leading_feature"
            ].to_numpy()
            scores["evidence_scope_type"] = (
                episode[scope_type_column].to_numpy()
                if scope_available else pd.NA
            )
            scores["evidence_scope_id"] = (
                episode[scope_id_column].to_numpy()
                if scope_available else pd.NA
            )
            scores["evidence_affected_fraction"] = (
                episode[affected_fraction_column].to_numpy()
                if fraction_available else pd.NA
            )
            for row in candidate_rows:
                produced = scores_to_alerts(
                    scores[[
                        *SCORE_COLUMNS, "leading_feature",
                        "evidence_scope_type", "evidence_scope_id",
                        "evidence_affected_fraction",
                    ]],
                    float(row.threshold),
                    min_consecutive=int(persistence[model_id]),
                    recovery_consecutive=int(recovery_consecutive),
                )
                if not produced.empty:
                    collected[float(row.threshold_quantile)].append(produced)

        for quantile, frames in collected.items():
            if frames:
                alerts = _deduplicate_scope_alerts(
                    pd.concat(frames, ignore_index=True)
                ).sort_values(
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

    from .evaluation import ALERT_COLUMNS, SCORE_COLUMNS, scores_to_alerts

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

def measurement_features(
    panel,
    catalogue,
    *,
    history_window_seconds=None,
    minimum_history_seconds=None,
    gap_tolerance=1.5,
    seasonal_periods=None,
    history_windows_seconds=None,
    lag_windows_seconds=None,
    activity_windows_seconds=None,
    history_metric_ids=None,
    lag_metric_ids=None,
    activity_metric_ids=None,
    minimum_window_fraction=0.50,
):
    """Apply metric transforms and optional causal temporal features."""

    output = transform_episode(
        panel, catalogue, gap_tolerance=float(gap_tolerance)
    )
    output = add_causal_seasonal_differences(
        output, catalogue, seasonal_periods or {}
    )
    if any((
        history_windows_seconds,
        lag_windows_seconds,
        activity_windows_seconds,
    )):
        return add_causal_temporal_features(
            output,
            catalogue,
            history_windows_seconds=history_windows_seconds,
            lag_windows_seconds=lag_windows_seconds,
            activity_windows_seconds=activity_windows_seconds,
            history_metric_ids=history_metric_ids,
            lag_metric_ids=lag_metric_ids,
            activity_metric_ids=activity_metric_ids,
            minimum_window_fraction=float(minimum_window_fraction),
            gap_tolerance=float(gap_tolerance),
        )
    if history_window_seconds is None:
        return output
    if minimum_history_seconds is None:
        raise ValueError(
            "minimum_history_seconds is required with a history window"
        )
    return add_causal_history(
        output,
        catalogue,
        history_seconds=float(history_window_seconds),
        minimum_history_seconds=float(minimum_history_seconds),
        gap_tolerance=float(gap_tolerance),
    )


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
    history_window_seconds=None,
    minimum_history_seconds=None,
    gap_tolerance=1.5,
    seasonal_periods=None,
    history_windows_seconds=None,
    lag_windows_seconds=None,
    activity_windows_seconds=None,
    history_metric_ids=None,
    lag_metric_ids=None,
    activity_metric_ids=None,
    minimum_window_fraction=0.50,
    score_start=None,
    score_end=None,
    progress_every=25,
):
    """Write transformed features one complete episode at a time."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        started = time.perf_counter()
        episodes = 0
        written_rows = 0
        for episodes, panel in enumerate(iter_episode_frames(wide_path), start=1):
            feature_catalogue = catalogue
            if target_cadence_seconds is not None:
                panel = resample_wide_panel(
                    panel, catalogue, target_cadence_seconds
                )
                feature_catalogue = catalogue.copy()
                feature_catalogue["expected_cadence_seconds"] = (
                    target_cadence_seconds
                )
            features = measurement_features(
                panel,
                feature_catalogue,
                history_window_seconds=history_window_seconds,
                minimum_history_seconds=minimum_history_seconds,
                gap_tolerance=gap_tolerance,
                seasonal_periods=seasonal_periods,
                history_windows_seconds=history_windows_seconds,
                lag_windows_seconds=lag_windows_seconds,
                activity_windows_seconds=activity_windows_seconds,
                history_metric_ids=history_metric_ids,
                lag_metric_ids=lag_metric_ids,
                activity_metric_ids=activity_metric_ids,
                minimum_window_fraction=minimum_window_fraction,
            )
            times = pd.to_datetime(features["event_ts"], utc=True)
            if score_start is not None:
                features = features.loc[times.ge(score_start)]
                times = pd.to_datetime(features["event_ts"], utc=True)
            if score_end is not None:
                features = features.loc[times.lt(score_end)]
            if features.empty:
                continue
            written_rows += len(features)
            table = pa.Table.from_pandas(features, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            writer.write_table(table)
            if progress_every and (
                episodes == 1 or episodes % int(progress_every) == 0
            ):
                print(
                    f"    features: {episodes} episodes, "
                    f"{written_rows:,} rows, "
                    f"{(time.perf_counter() - started) / 60:.1f} minutes",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("Partition produced no transformed feature rows")
    print(
        f"    features complete — {episodes} episodes, {written_rows:,} rows "
        f"in {(time.perf_counter() - started) / 60:.1f} minutes",
        flush=True,
    )
    return destination


def _robust_reference(frame, minimum_scales=None):
    centre = frame.median()
    scale = (frame.quantile(0.75) - frame.quantile(0.25)) / 1.349

    # Quantised and zero-inflated features often have an IQR of zero even
    # though their occasional non-zero changes are perfectly valid. In that
    # case use the typical non-zero deviation instead of an arbitrary tiny
    # denominator, which would manufacture enormous residual scores.
    deviation = frame.sub(centre).abs()
    fallback = deviation.mask(deviation.eq(0)).median()
    declared = pd.Series(minimum_scales or {}, dtype=float).reindex(
        frame.columns
    ).fillna(1e-6)
    floor = pd.Series(
        np.maximum(centre.abs() * 1e-6, declared), index=frame.columns
    )
    scale = scale.where(scale.gt(floor), fallback)
    scale = scale.where(scale.gt(floor), floor)
    return centre, scale


def _reference_sample(path, maximum_rows, random_seed):
    source = _sql_literal(str(Path(path)))
    with _model_duckdb() as connection:
        return connection.execute(f"""
            SELECT * FROM read_parquet({source})
            USING SAMPLE reservoir({int(maximum_rows)} ROWS)
            REPEATABLE ({int(random_seed)})
        """).df()


def _full_entity_reference(path, features, scale_floors, minimum_rows=30):
    """Robust entity references from the full calibration-fit slice.

    The training-row cap is for pooled PCA and Isolation Forest only. Entity
    baselines are small group summaries, so DuckDB can calculate them over the
    complete early-calibration file without materialising all rows in Python.
    """

    expressions = []
    for feature in features:
        column = _sql_identifier(feature)
        expressions.extend([
            f"count({column}) AS {_sql_identifier(feature + '__count')}",
            f"approx_quantile({column}, 0.50) AS {_sql_identifier(feature + '__centre')}",
            f"approx_quantile({column}, 0.25) AS {_sql_identifier(feature + '__q25')}",
            f"approx_quantile({column}, 0.75) AS {_sql_identifier(feature + '__q75')}",
        ])
    source = _sql_literal(str(Path(path)))
    with _model_duckdb() as connection:
        summary = connection.execute(f"""
            SELECT CAST(entity_id AS VARCHAR) AS entity_id,
                   {', '.join(expressions)}
            FROM read_parquet({source})
            GROUP BY entity_id
            ORDER BY entity_id
        """).df().set_index("entity_id")

    counts = pd.DataFrame(
        {name: summary[f"{name}__count"] for name in features}
    ).astype(float)
    centre = pd.DataFrame(
        {name: summary[f"{name}__centre"] for name in features}
    ).astype(float)
    q25 = pd.DataFrame(
        {name: summary[f"{name}__q25"] for name in features}
    ).astype(float)
    q75 = pd.DataFrame(
        {name: summary[f"{name}__q75"] for name in features}
    ).astype(float)

    valid = counts.ge(int(minimum_rows))
    centre = centre.where(valid)
    scale = (q75 - q25) / 1.349
    floors = centre.abs() * 1e-6
    for name in features:
        floors[name] = floors[name].clip(lower=float(scale_floors[name]))
    scale = scale.where(valid).where(scale.gt(floors), floors.where(valid))
    return centre, scale, counts


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


def _is_temporal_feature(name):
    """Identify features added by the multi-timescale temporal layer."""

    return any(
        marker in str(name)
        for marker in ("__history_", "__lag_", "__rate_", "__sum_")
    )


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
    random_seed=42,
    isolation_trees=200,
    isolation_max_samples=1024,
    isolation_max_features=1.0,
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
    else:
        catalogue_directions = {}
        catalogue_scales = {}
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
    if use_entity_reference:
        entity_centre, entity_scale, entity_reference_counts = (
            _full_entity_reference(
                calibration_features,
                usable,
                scale_floors,
                minimum_rows=30,
            )
        )

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
        "use_entity_reference": bool(use_entity_reference),
        "reference_exclusions": exclusions.to_dict("records"),
        "training_rows": len(sample),
        "random_seed": int(random_seed),
        "isolation_trees": int(isolation_trees),
        "isolation_max_samples": int(isolation_max_samples),
        "isolation_max_features": isolation_max_features,
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
        base_features = [name for name in usable if not _is_temporal_feature(name)]
        if len(base_features) < 2:
            raise ValueError("The base Isolation Forest needs at least two features")
        bundle["isolation_base_features"] = base_features
        bundle["isolation_forest_base"] = _fit_isolation_forest(
            clean[base_features],
            trees=isolation_trees,
            maximum_samples=isolation_max_samples,
            maximum_features=isolation_max_features,
            random_seed=random_seed,
        )
        bundle["isolation_forest_temporal"] = _fit_isolation_forest(
            clean,
            trees=isolation_trees,
            maximum_samples=isolation_max_samples,
            maximum_features=isolation_max_features,
            random_seed=random_seed,
        )
        # Compatibility for model cards written before the variants were named.
        bundle["isolation_forest"] = bundle["isolation_forest_temporal"]
    else:
        bundle["pca"] = None
        bundle["isolation_base_features"] = []
        bundle["isolation_forest_base"] = None
        bundle["isolation_forest_temporal"] = None
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
    spread = residuals[level_features].rolling(window, min_periods=max(3, window // 2)).std()
    reference_spread = bundle["residual_scale"].reindex(level_features).replace(0, 1)
    dispersion = np.abs(np.log(spread.div(reference_spread).clip(lower=0.05)))
    dispersion_values, dispersion_leading = _row_max(dispersion, level_features)

    clean = residuals.fillna(0).clip(-50, 50)
    pca_values = np.full(len(features), np.nan)
    isolation_base_values = np.full(len(features), np.nan)
    isolation_temporal_values = np.full(len(features), np.nan)
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
        if temporal_model is not None:
            isolation_temporal_values = -temporal_model.decision_function(clean)
            isolation_temporal_leading = np.asarray(
                bundle["feature_columns"], dtype=object
            )[np.abs(clean.to_numpy()).argmax(axis=1)]
        base_model = bundle.get("isolation_forest_base")
        base_features = bundle.get("isolation_base_features", [])
        if base_model is not None and base_features:
            base_values = clean[base_features]
            isolation_base_values = -base_model.decision_function(base_values)
            isolation_base_leading = np.asarray(base_features, dtype=object)[
                np.abs(base_values.to_numpy()).argmax(axis=1)
            ]

    output = features[IDENTITY_COLUMNS].copy()
    output["available_features"] = available.to_numpy()
    output["readiness"] = np.where(ready, "monitored", "temporarily_unscoreable")
    channels = {
        "rapid_residual": (rapid_values, rapid_leading),
        "drift_cusum": (drift_values, drift_leading),
        "dispersion_change": (dispersion_values, dispersion_leading),
        "pca_spe": (pca_values, pca_leading),
        "isolation_forest_base": (
            isolation_base_values, isolation_base_leading,
        ),
        "isolation_forest_temporal": (
            isolation_temporal_values, isolation_temporal_leading,
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
                residual_frame[residual_features] = residuals[residual_features]
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


TOPOLOGY_REFERENCE_COLUMNS = [
    "channel", "group_type", "size_band", "leading_feature",
    "reference_rows", "centre", "upper", "scale",
]


def _size_band_sql(count_expression):
    return (
        f"CASE WHEN {count_expression} < 7 THEN 'lt_7' "
        f"WHEN {count_expression} < 15 THEN '7_14' "
        f"WHEN {count_expression} < 30 THEN '15_29' ELSE '30_plus' END"
    )


def _model_topology(topology):
    required = {
        "entity_id", "group_type", "group_id", "hierarchy_level", "group_family",
    }
    missing = required - set(topology.columns)
    if missing:
        raise ValueError(f"Topology is missing columns: {sorted(missing)}")
    physical = topology.loc[
        topology["group_family"].eq("physical_topology"), list(required)
    ].copy()
    if physical.empty:
        raise ValueError("No physical topology memberships are available")
    physical[["entity_id", "group_type", "group_id"]] = physical[
        ["entity_id", "group_type", "group_id"]
    ].astype(str)
    physical = physical.drop_duplicates()
    membership_rows = physical.groupby(
        ["entity_id", "group_type"], sort=False
    ).size()
    if membership_rows.gt(1).any():
        raise ValueError(
            "Topology scoring requires one effective membership per entity "
            "and group type. Materialise time-valid memberships before scoring."
        )
    physical["group_size"] = physical.groupby(
        ["group_type", "group_id"]
    )["entity_id"].transform("nunique")
    return physical


def _topology_feature_specs(feature_columns):
    """Give canonical feature names short, safe SQL aliases."""

    features = sorted(dict.fromkeys(map(str, feature_columns)))
    if not features:
        raise ValueError("Topology scoring requires at least one feature")
    return [(feature, f"feature_{index:02d}") for index, feature in enumerate(features)]


def _peer_wide_sql(residual_source, feature_specs, peer_group_type):
    """Calculate peer counts and deviations without expanding rows by feature."""

    window = (
        "PARTITION BY r.event_ts, t.group_id "
        "ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING "
        "EXCLUDE CURRENT ROW"
    )
    statistics = []
    for feature, alias in feature_specs:
        value = f"r.{_sql_identifier(feature)}"
        statistics.extend([
            f"count({value}) OVER ({window}) AS {_sql_identifier(alias + '__count')}",
            f"abs({value} - median({value}) OVER ({window})) "
            f"AS {_sql_identifier(alias + '__raw')}",
        ])
    return f"""
        SELECT r.event_ts, CAST(r.entity_id AS VARCHAR) AS entity_id,
               CAST(r.episode_id AS VARCHAR) AS episode_id,
               t.group_type, t.group_id, {', '.join(statistics)}
        FROM read_parquet({residual_source}) AS r
        JOIN model_topology AS t
          ON CAST(r.entity_id AS VARCHAR) = t.entity_id
        WHERE t.group_type = {_sql_literal(peer_group_type)}
    """


def _group_wide_sql(residual_source, feature_specs, group_types):
    """Calculate common-mode statistics once per timestamp and physical group."""

    allowed = ", ".join(_sql_literal(value) for value in group_types)
    statistics = []
    for feature, alias in feature_specs:
        value = f"r.{_sql_identifier(feature)}"
        statistics.extend([
            f"count({value}) AS {_sql_identifier(alias + '__count')}",
            f"abs(median({value})) * sqrt(count({value})) "
            f"AS {_sql_identifier(alias + '__raw')}",
            "avg(CASE "
            f"WHEN {value} IS NULL THEN NULL "
            f"WHEN abs({value}) >= 3 THEN 1.0 ELSE 0.0 END) "
            f"AS {_sql_identifier(alias + '__affected')}",
        ])
    return f"""
        SELECT r.event_ts, t.group_type, t.group_id,
               max(t.group_size) AS group_size, {', '.join(statistics)}
        FROM read_parquet({residual_source}) AS r
        JOIN model_topology AS t
          ON CAST(r.entity_id AS VARCHAR) = t.entity_id
        WHERE t.group_type IN ({allowed})
        GROUP BY r.event_ts, t.group_type, t.group_id
    """


def _reference_summary_sql(
    table_name,
    feature_specs,
    channel,
    *,
    minimum_count,
    minimum_fraction=None,
    upper_quantile,
):
    """Aggregate a wide topology table into calibration reference rows."""

    queries = []
    for feature, alias in feature_specs:
        count_column = _sql_identifier(alias + "__count")
        raw_column = _sql_identifier(alias + "__raw")
        filters = [
            f"{raw_column} IS NOT NULL",
            f"{count_column} >= {int(minimum_count)}",
        ]
        if minimum_fraction is not None:
            filters.append(
                f"{count_column} * 1.0 / group_size >= {float(minimum_fraction)}"
            )
        size_band = _size_band_sql(count_column)
        queries.append(f"""
            SELECT {_sql_literal(channel)} AS channel, group_type,
                   {size_band} AS size_band,
                   {_sql_literal(feature)} AS leading_feature,
                   count(*) AS reference_rows,
                   median({raw_column}) AS centre,
                   quantile_cont({raw_column}, {float(upper_quantile)}) AS upper
            FROM {table_name}
            WHERE {' AND '.join(filters)}
            GROUP BY group_type, {size_band}
        """)
    return " UNION ALL ".join(queries)


def _size_band_condition(count_column, size_band):
    if size_band == "lt_7":
        return f"{count_column} < 7"
    if size_band == "7_14":
        return f"{count_column} >= 7 AND {count_column} < 15"
    if size_band == "15_29":
        return f"{count_column} >= 15 AND {count_column} < 30"
    if size_band == "30_plus":
        return f"{count_column} >= 30"
    raise ValueError(f"Unknown topology size band: {size_band}")


def _normalised_topology_score(reference, channel, feature, alias):
    """Create a CASE expression from the small frozen reference table."""

    rows = reference.loc[
        reference["channel"].eq(channel)
        & reference["leading_feature"].astype(str).eq(str(feature))
    ].sort_values(["group_type", "size_band"])
    count_column = _sql_identifier(alias + "__count")
    raw_column = _sql_identifier(alias + "__raw")
    cases = []
    for row in rows.itertuples(index=False):
        condition = (
            f"group_type = {_sql_literal(row.group_type)} AND "
            f"{_size_band_condition(count_column, row.size_band)}"
        )
        score = (
            f"greatest(0.0, ({raw_column} - {float(row.centre)!r}) "
            f"/ {float(row.scale)!r})"
        )
        cases.append(f"WHEN {condition} THEN {score}")
    if not cases:
        return "CAST(NULL AS DOUBLE)"
    return "CASE " + " ".join(cases) + " ELSE NULL END"


def _chosen_value_case(feature_specs, value_suffix, leading_column="score"):
    """Select metadata belonging to the first maximum-scoring feature."""

    cases = []
    for feature, alias in feature_specs:
        score = _sql_identifier(alias + "__score")
        value = (
            _sql_literal(feature)
            if value_suffix is None
            else _sql_identifier(alias + value_suffix)
        )
        cases.append(f"WHEN {score} = {leading_column} THEN {value}")
    return "CASE " + " ".join(cases) + " ELSE NULL END"


def _scope_value_case(scope_specs, value_suffix, leading_column="score"):
    """Select metadata from the first maximum-scoring topology scope."""

    cases = []
    for group_type, alias in scope_specs:
        score = _sql_identifier(alias + "__score")
        value = (
            _sql_literal(group_type)
            if value_suffix is None
            else _sql_identifier(alias + value_suffix)
        )
        cases.append(f"WHEN {score} = {leading_column} THEN {value}")
    return "CASE " + " ".join(cases) + " ELSE NULL END"


def _group_entity_best_sql(residual_source, scope_specs):
    """Choose the strongest physical scope without a long entity expansion."""

    joins = []
    values = []
    for group_type, alias in scope_specs:
        joins.append(f"""
            LEFT JOIN topology_group_feature_best AS {alias}
              ON {alias}.group_type = {_sql_literal(group_type)}
             AND {alias}.group_id = m.{_sql_identifier(alias + '__group_id')}
             AND {alias}.event_ts = i.event_ts
        """)
        for source_name, suffix in (
            ("score", "__score"),
            ("leading_feature", "__leading_feature"),
            ("group_id", "__group_id"),
            ("available_count", "__available_count"),
            ("available_fraction", "__available_fraction"),
            ("affected_fraction", "__affected_fraction"),
        ):
            values.append(
                f"{alias}.{source_name} AS {_sql_identifier(alias + suffix)}"
            )

    scores = [
        _sql_identifier(alias + "__score") for _, alias in scope_specs
    ]
    best_score = f"greatest({', '.join(scores)})"
    return f"""
        WITH identities AS (
            SELECT event_ts, CAST(entity_id AS VARCHAR) AS entity_id,
                   CAST(episode_id AS VARCHAR) AS episode_id
            FROM read_parquet({residual_source})
        ), joined AS (
            SELECT i.*, {', '.join(values)}
            FROM identities AS i
            LEFT JOIN model_entity_topology AS m USING (entity_id)
            {' '.join(joins)}
        ), best AS (
            SELECT *, {best_score} AS score
            FROM joined
        )
        SELECT event_ts, entity_id, episode_id, score,
               {_scope_value_case(scope_specs, '__leading_feature')}
                   AS leading_feature,
               {_scope_value_case(scope_specs, None)} AS group_type,
               {_scope_value_case(scope_specs, '__group_id')} AS group_id,
               {_scope_value_case(scope_specs, '__available_count')}
                   AS available_count,
               {_scope_value_case(scope_specs, '__available_fraction')}
                   AS available_fraction,
               {_scope_value_case(scope_specs, '__affected_fraction')}
                   AS affected_fraction
        FROM best
        WHERE score IS NOT NULL
    """


def fit_topology_reference(
    calibration_residuals,
    topology,
    feature_columns,
    *,
    peer_group_type,
    group_types,
    min_peers=7,
    min_group_entities=3,
    min_group_fraction=0.50,
    minimum_reference_rows=100,
    upper_quantile=0.995,
):
    """Fit calibration-only null scales for peer and group evidence.

    Features remain in columns while peer and group statistics are calculated.
    This avoids multiplying a large telemetry table by the feature count.
    """

    physical = _model_topology(topology)
    known_types = set(physical["group_type"])
    if peer_group_type not in known_types:
        raise ValueError(f"Unknown peer group type: {peer_group_type}")
    group_types = [name for name in group_types if name in known_types]
    if not group_types:
        raise ValueError("No requested common-mode topology level is available")

    source = _sql_literal(str(Path(calibration_residuals)))
    feature_specs = _topology_feature_specs(feature_columns)
    peer_sql = _peer_wide_sql(source, feature_specs, peer_group_type)
    group_sql = _group_wide_sql(source, feature_specs, group_types)
    peer_reference_sql = _reference_summary_sql(
        "topology_peer_wide",
        feature_specs,
        "peer_deviation",
        minimum_count=min_peers,
        upper_quantile=upper_quantile,
    )
    group_reference_sql = _reference_summary_sql(
        "topology_group_wide",
        feature_specs,
        "group_common_mode",
        minimum_count=min_group_entities,
        minimum_fraction=min_group_fraction,
        upper_quantile=upper_quantile,
    )

    with _model_duckdb(Path(calibration_residuals).parent) as connection:
        connection.register("model_topology", physical)
        started = time.perf_counter()
        print(
            f"    topology reference: peer statistics "
            f"({len(feature_specs)} features)",
            flush=True,
        )
        connection.execute(
            f"CREATE TEMP TABLE topology_peer_wide AS {peer_sql}"
        )
        peer_reference = connection.execute(peer_reference_sql).df()
        connection.execute("DROP TABLE topology_peer_wide")
        print(
            f"    topology reference: peer complete in "
            f"{(time.perf_counter() - started) / 60:.1f} minutes",
            flush=True,
        )

        started = time.perf_counter()
        print("    topology reference: group statistics", flush=True)
        connection.execute(
            f"CREATE TEMP TABLE topology_group_wide AS {group_sql}"
        )
        group_reference = connection.execute(group_reference_sql).df()
        connection.execute("DROP TABLE topology_group_wide")
        print(
            f"    topology reference: group complete in "
            f"{(time.perf_counter() - started) / 60:.1f} minutes",
            flush=True,
        )

    reference = pd.concat(
        [peer_reference, group_reference], ignore_index=True
    ).sort_values(
        ["channel", "group_type", "size_band", "leading_feature"]
    ).reset_index(drop=True)
    reference["scale"] = reference["upper"] - reference["centre"]
    reference = reference.loc[
        reference["reference_rows"].ge(int(minimum_reference_rows))
        & reference["scale"].gt(1e-9)
    ].reset_index(drop=True)
    if reference.empty:
        raise ValueError("Calibration contains no stable topology reference")
    return reference[TOPOLOGY_REFERENCE_COLUMNS]


def score_topology_file(
    residuals_path,
    topology,
    reference,
    destination,
    *,
    peer_group_type,
    group_types,
    min_peers=7,
    min_group_entities=3,
    min_group_fraction=0.50,
):
    """Score eligible peer and common-mode evidence from frozen residuals.

    Wide intermediate tables keep memory proportional to source rows rather
    than source rows multiplied by features or topology levels. Compact peer
    and group evidence is written between stages so each DuckDB connection can
    release its working memory before the next join.
    """

    physical = _model_topology(topology)
    known_types = set(physical["group_type"])
    if peer_group_type not in known_types:
        raise ValueError(f"Unknown peer group type: {peer_group_type}")
    group_types = sorted({
        name for name in group_types if name in known_types
    })
    if not group_types:
        raise ValueError("No requested common-mode topology level is available")
    features = sorted(reference["leading_feature"].astype(str).unique())
    feature_specs = _topology_feature_specs(features)
    scope_specs = [
        (group_type, f"scope_{index:02d}")
        for index, group_type in enumerate(group_types)
    ]
    membership_columns = {
        group_type: alias + "__group_id"
        for group_type, alias in scope_specs
    }
    entity_topology = (
        physical.loc[
            physical["group_type"].isin(group_types),
            ["entity_id", "group_type", "group_id"],
        ]
        .pivot(index="entity_id", columns="group_type", values="group_id")
        .reindex(columns=group_types)
        .rename(columns=membership_columns)
        .reset_index()
    )
    source = _sql_literal(str(Path(residuals_path)))
    peer_sql = _peer_wide_sql(source, feature_specs, peer_group_type)
    group_sql = _group_wide_sql(source, feature_specs, group_types)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    peer_scores = []
    group_scores = []
    for feature, alias in feature_specs:
        peer_score = _normalised_topology_score(
            reference, "peer_deviation", feature, alias
        )
        peer_count = _sql_identifier(alias + "__count")
        peer_scores.append(
            f"CASE WHEN {peer_count} >= {int(min_peers)} "
            f"THEN ({peer_score}) ELSE NULL END "
            f"AS {_sql_identifier(alias + '__score')}"
        )

        group_score = _normalised_topology_score(
            reference, "group_common_mode", feature, alias
        )
        group_count = _sql_identifier(alias + "__count")
        group_scores.append(
            f"CASE WHEN {group_count} >= {int(min_group_entities)} "
            f"AND {group_count} * 1.0 / group_size >= {float(min_group_fraction)} "
            f"THEN ({group_score}) ELSE NULL END "
            f"AS {_sql_identifier(alias + '__score')}"
        )

    score_columns = [
        _sql_identifier(alias + "__score") for _, alias in feature_specs
    ]
    best_score = f"greatest({', '.join(score_columns)})"
    leading_feature = _chosen_value_case(feature_specs, None)
    selected_count = _chosen_value_case(feature_specs, "__count")
    selected_affected = _chosen_value_case(feature_specs, "__affected")

    peer_best_sql = f"""
        WITH normalised AS (
            SELECT *, {', '.join(peer_scores)}
            FROM topology_peer_wide
        ), best AS (
            SELECT *, {best_score} AS score
            FROM normalised
        ), selected AS (
            SELECT event_ts, entity_id, episode_id, score,
                   {leading_feature} AS leading_feature,
                   {selected_count} AS available_count
            FROM best
            WHERE score IS NOT NULL
        )
        SELECT *, {_size_band_sql('available_count')} AS size_band
        FROM selected
    """
    group_feature_best_sql = f"""
        WITH normalised AS (
            SELECT *, {', '.join(group_scores)}
            FROM topology_group_wide
        ), best AS (
            SELECT *, {best_score} AS score
            FROM normalised
        )
        SELECT event_ts, group_type, group_id, score,
               {leading_feature} AS leading_feature,
               {selected_count} AS available_count,
               {selected_count} * 1.0 / group_size AS available_fraction,
               {selected_affected} AS affected_fraction
        FROM best
        WHERE score IS NOT NULL
    """

    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix="topology-evidence-"
    ) as evidence_name:
        evidence_root = Path(evidence_name)
        peer_path = evidence_root / "peer.parquet"
        group_path = evidence_root / "group.parquet"
        peer_joined_path = evidence_root / "identities_with_peer.parquet"

        with _model_duckdb(destination.parent) as connection:
            connection.register("model_topology", physical)

            started = time.perf_counter()
            print("    topology scoring: peer evidence", flush=True)
            connection.execute(
                f"CREATE TEMP TABLE topology_peer_wide AS {peer_sql}"
            )
            connection.execute(
                f"CREATE TEMP TABLE topology_peer_best AS {peer_best_sql}"
            )
            connection.execute("DROP TABLE topology_peer_wide")
            connection.execute(
                f"COPY topology_peer_best TO {_sql_literal(str(peer_path))} "
                "(FORMAT PARQUET, COMPRESSION ZSTD)"
            )
            connection.execute("DROP TABLE topology_peer_best")
            print(
                f"    topology scoring: peer complete in "
                f"{(time.perf_counter() - started) / 60:.1f} minutes",
                flush=True,
            )

            started = time.perf_counter()
            print("    topology scoring: group evidence", flush=True)
            connection.execute(
                f"CREATE TEMP TABLE topology_group_wide AS {group_sql}"
            )
            connection.execute(
                f"CREATE TEMP TABLE topology_group_feature_best "
                f"AS {group_feature_best_sql}"
            )
            connection.execute("DROP TABLE topology_group_wide")
            connection.register("model_entity_topology", entity_topology)
            group_best_sql = _group_entity_best_sql(source, scope_specs)
            connection.execute(
                f"CREATE TEMP TABLE topology_group_best AS {group_best_sql}"
            )
            connection.execute("DROP TABLE topology_group_feature_best")
            connection.execute(
                f"COPY topology_group_best TO {_sql_literal(str(group_path))} "
                "(FORMAT PARQUET, COMPRESSION ZSTD)"
            )
            print(
                f"    topology scoring: group complete in "
                f"{(time.perf_counter() - started) / 60:.1f} minutes",
                flush=True,
            )

        # Each merge has one small hash side. Closing the calculation
        # connection first prevents earlier window and aggregate state from
        # competing with the final Parquet write for memory.
        peer_source = _sql_literal(str(peer_path))
        group_source = _sql_literal(str(group_path))
        peer_joined = _sql_literal(str(peer_joined_path))
        with _model_duckdb(destination.parent) as connection:
            started = time.perf_counter()
            print("    topology scoring: bounded final merge", flush=True)
            connection.execute(f"""
                COPY (
                    SELECT i.event_ts,
                           CAST(i.entity_id AS VARCHAR) AS entity_id,
                           CAST(i.episode_id AS VARCHAR) AS episode_id,
                           p.score AS peer_deviation,
                           p.leading_feature
                               AS peer_deviation__leading_feature,
                           p.available_count AS peer_valid_peers,
                           p.size_band AS peer_size_band
                    FROM read_parquet({source}) AS i
                    LEFT JOIN read_parquet({peer_source}) AS p
                      ON i.event_ts = p.event_ts
                     AND CAST(i.entity_id AS VARCHAR) = p.entity_id
                     AND CAST(i.episode_id AS VARCHAR) = p.episode_id
                ) TO {peer_joined}
                (FORMAT PARQUET, COMPRESSION ZSTD)
            """)
            connection.execute(f"""
                COPY (
                    SELECT p.*,
                           g.score AS group_common_mode,
                           g.leading_feature
                               AS group_common_mode__leading_feature,
                           g.group_type AS group_common_mode__scope_type,
                           g.group_id AS group_common_mode__scope_id,
                           g.available_count AS group_valid_entities,
                           g.available_fraction AS group_available_fraction,
                           g.affected_fraction
                               AS group_common_mode__affected_fraction
                    FROM read_parquet({peer_joined}) AS p
                    LEFT JOIN read_parquet({group_source}) AS g
                      ON p.event_ts = g.event_ts
                     AND CAST(p.entity_id AS VARCHAR) = g.entity_id
                     AND CAST(p.episode_id AS VARCHAR) = g.episode_id
                ) TO {_sql_literal(str(destination))}
                (FORMAT PARQUET, COMPRESSION ZSTD)
            """)
            print(
                f"    topology scoring: merge complete in "
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
):
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
    """Append the contextual Isolation Forest channel to a score file."""

    score_path, destination = Path(score_path), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(score_path)
    writer = None
    names = np.asarray(bundle["feature_columns"], dtype=object)
    try:
        for batch in parquet.iter_batches(batch_size=int(batch_rows)):
            frame = batch.to_pandas()
            clean = frame[bundle["feature_columns"]].replace(
                [np.inf, -np.inf], np.nan
            )
            values = bundle["imputer"].transform(clean)
            scaled = bundle["scaler"].transform(values)
            scaled = np.clip(
                scaled,
                -float(bundle["scaled_feature_cap"]),
                float(bundle["scaled_feature_cap"]),
            )
            frame["isolation_forest_contextual"] = -bundle[
                "model"
            ].decision_function(scaled)
            frame["isolation_forest_contextual__leading_feature"] = names[
                np.abs(scaled).argmax(axis=1)
            ]
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    destination, table.schema, compression="zstd"
                )
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("Score file produced no contextual model rows")
    return destination


def merge_score_files(self_scores, topology_scores, destination):
    """Join self and topology scores in two bounded, disk-backed stages.

    Keeping the hash join and global sort in one DuckDB query forces both
    blocking operators to compete for memory.  Materialising the unsorted join
    first releases its state before the ordered Parquet file is written.
    """

    self_scores = Path(self_scores)
    topology_scores = Path(topology_scores)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Score destination already exists: {destination}")

    key_columns = ["event_ts", "entity_id", "episode_id"]
    topology_columns = [
        "peer_deviation", "peer_deviation__leading_feature",
        "peer_valid_peers", "peer_size_band",
        "group_common_mode", "group_common_mode__leading_feature",
        "group_common_mode__scope_type", "group_common_mode__scope_id",
        "group_valid_entities", "group_available_fraction",
        "group_common_mode__affected_fraction",
    ]
    for path, required in (
        (self_scores, key_columns),
        (topology_scores, [*key_columns, *topology_columns]),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        available = set(pq.ParquetFile(path).schema_arrow.names)
        missing = set(required) - available
        if missing:
            raise ValueError(
                f"{path.name} is missing score columns: {sorted(missing)}"
            )

    expected_rows = pq.ParquetFile(self_scores).metadata.num_rows
    topology_rows = pq.ParquetFile(topology_scores).metadata.num_rows
    if expected_rows == 0:
        raise ValueError("Self score file is empty")
    if topology_rows != expected_rows:
        raise ValueError(
            "Topology score keys are not one-to-one: "
            f"{expected_rows:,} self rows and {topology_rows:,} topology rows"
        )

    self_source = _sql_literal(str(self_scores))
    topology_source = _sql_literal(str(topology_scores))
    selected = ", ".join(
        f"t.{_sql_identifier(name)}" for name in topology_columns
    )

    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix="score-merge-"
    ) as merge_name:
        merge_root = Path(merge_name)
        joined = merge_root / "joined.parquet"
        staged = merge_root / "combined.parquet"

        print("    score merge: joining self and topology evidence", flush=True)
        with _model_duckdb(destination.parent) as connection:
            connection.execute(f"""
                COPY (
                    SELECT s.*, {selected},
                           t.event_ts IS NOT NULL AS __topology_match
                    FROM read_parquet({self_source}) AS s
                    LEFT JOIN read_parquet({topology_source}) AS t
                    USING (event_ts, entity_id, episode_id)
                ) TO {_sql_literal(str(joined))}
                (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 16384)
            """)

        joined_rows = pq.ParquetFile(joined).metadata.num_rows
        if joined_rows != expected_rows:
            raise ValueError(
                "Topology score keys are not one-to-one: "
                f"{expected_rows:,} self rows produced {joined_rows:,} rows"
            )

        joined_columns = pq.ParquetFile(joined).schema_arrow.names
        output_columns = ", ".join(
            _sql_identifier(name)
            for name in joined_columns
            if name != "__topology_match"
        )

        print("    score merge: validating and ordering merged evidence", flush=True)
        joined_source = _sql_literal(str(joined))
        with _model_duckdb(destination.parent) as connection:
            missing_matches = connection.execute(f"""
                SELECT count(*)
                FROM read_parquet({joined_source})
                WHERE NOT __topology_match
            """).fetchone()[0]
            if missing_matches:
                raise ValueError(
                    "Topology score keys are not one-to-one: "
                    f"{missing_matches:,} self rows have no topology match"
                )
            connection.execute(f"""
                COPY (
                    SELECT {output_columns}
                    FROM read_parquet({joined_source})
                    ORDER BY entity_id, episode_id, event_ts
                ) TO {_sql_literal(str(staged))}
                (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 16384)
            """)

        staged.replace(destination)
        print(f"    score merge: {expected_rows:,} rows complete", flush=True)
    return destination


def score_partition_file(
    bundle,
    feature_path,
    destination,
    workspace,
    *,
    cadence_seconds,
    dispersion_window_seconds,
    resolved_policy,
    topology=None,
    topology_reference=None,
    contextual_isolation_bundle=None,
):
    """Apply one frozen scoring procedure to any feature partition.

    Development and holdout must pass through this function with the same
    fitted bundle and resolved policy. Intermediate residual and topology
    files stay in ``workspace``; only the combined score file is published.
    """

    destination = Path(destination)
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    stem = destination.stem
    self_scores = workspace / f"{stem}__self.parquet"
    residuals = workspace / f"{stem}__residuals.parquet"
    topology_enabled = resolved_policy.get("topology_enabled", False)
    topology_features = (
        resolved_policy.get("topology_features", bundle["feature_columns"])
        if topology_enabled else None
    )

    started = time.perf_counter()
    score_residual_file(
        bundle,
        feature_path,
        self_scores,
        cadence_seconds=cadence_seconds,
        dispersion_window_seconds=dispersion_window_seconds,
        cusum_allowance=resolved_policy["cusum_allowance"],
        residual_destination=residuals if topology_enabled else None,
        residual_features=topology_features,
    )

    if topology_enabled:
        if topology is None or topology_reference is None:
            raise ValueError("Frozen topology scoring requires topology and its reference")
        topology_scores = workspace / f"{stem}__topology.parquet"
        score_topology_file(
            residuals,
            topology,
            topology_reference,
            topology_scores,
            peer_group_type=resolved_policy["peer_group_type"],
            group_types=resolved_policy["group_types"],
            min_peers=resolved_policy["min_peers"],
            min_group_entities=resolved_policy["min_group_entities"],
            min_group_fraction=resolved_policy["min_group_fraction"],
        )
        residuals.unlink(missing_ok=True)
        combined = (
            workspace / f"{stem}__combined.parquet"
            if contextual_isolation_bundle is not None else destination
        )
        merge_score_files(self_scores, topology_scores, combined)
        self_scores.unlink(missing_ok=True)
        topology_scores.unlink(missing_ok=True)
        if contextual_isolation_bundle is not None:
            append_contextual_isolation_scores(
                contextual_isolation_bundle, combined, destination
            )
            combined.unlink(missing_ok=True)
    else:
        if contextual_isolation_bundle is not None:
            raise ValueError(
                "Contextual Isolation Forest requires frozen topology evidence"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self_scores, destination)
        self_scores.unlink(missing_ok=True)

    print(
        f"  scored {Path(feature_path).name} in "
        f"{(time.perf_counter() - started) / 60:.1f} minutes",
        flush=True,
    )
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
