"""Streaming score-file joins and partition scoring orchestration."""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ._common import _model_duckdb, _sql_identifier, _sql_literal, feature_columns
from .isolation_forest import append_contextual_isolation_scores, score_residual_file
from .topology import score_topology_file

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
