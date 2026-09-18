"""Causal feature and partition materialisation."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..features import (
    add_causal_history, add_causal_seasonal_differences,
    add_causal_temporal_features, transform_episode,
)
from ..scoring._common import (
    ABSOLUTE_SCALE_FLOOR, IDENTITY_COLUMNS, RELATIVE_SCALE_FLOOR,
    SHORT_SCALE_FLOOR_FRACTION, _sql_identifier, _sql_literal,
    iter_episode_frames,
)

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
    cutoff=None,
):
    """Split calibration features chronologically into fit and threshold slices.

    ``cutoff`` lets calibration EDA and model fitting share one exact boundary.
    When it is omitted, the boundary is derived from ``fit_fraction``.
    """

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
        if cutoff is None:
            cutoff = start + (end - start) * float(fit_fraction)
        else:
            cutoff = pd.to_datetime(cutoff, utc=True)
            if not start < cutoff <= end:
                raise ValueError("cutoff must fall inside the calibration span")
        for destination, operator in (
            (fit_destination, "<"),
            (threshold_destination, ">="),
        ):
            connection.execute(f"""
                COPY (
                    SELECT * FROM read_parquet({source_sql})
                    WHERE event_ts {operator} TIMESTAMPTZ {_sql_literal(cutoff.isoformat())}
                    ORDER BY entity_id, episode_id, event_ts
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
        "fit_fraction": float((cutoff - start) / (end - start)),
    }

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
    history_max_gap_seconds=None,
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
            history_max_gap_seconds=history_max_gap_seconds,
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
    history_max_gap_seconds=None,
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
                history_max_gap_seconds=history_max_gap_seconds,
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
