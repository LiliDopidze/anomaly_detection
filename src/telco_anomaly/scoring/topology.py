"""Topology-aware peer and common-mode scoring."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pandas as pd

from ._common import (
    _model_duckdb, _sql_identifier, _sql_literal, feature_columns,
)

TOPOLOGY_REFERENCE_COLUMNS = [
    "channel", "group_type", "size_band", "leading_feature",
    "reference_rows", "centre", "upper", "scale", "affected_upper",
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

def physical_hierarchy(topology):
    """Return physical topology levels from deepest to coarsest.

    The hierarchy already belongs to the canonical topology contract. Deriving
    the order here avoids a second hand-written preference list that can drift.
    """

    physical = _model_topology(topology)
    levels = physical.groupby("group_type")["hierarchy_level"].agg(
        ["nunique", "first"]
    )
    if levels["first"].isna().any() or levels["nunique"].ne(1).any():
        raise ValueError(
            "Every physical group type needs one numeric hierarchy level"
        )
    return levels.sort_values("first", ascending=False).index.tolist()

def eligible_peer_levels(
    topology,
    *,
    minimum_valid_peers=7,
    minimum_entity_coverage=0.80,
):
    """Return physical peer levels with adequate membership coverage."""

    physical = _model_topology(topology)
    total_entities = physical["entity_id"].nunique()
    minimum_group_size = int(minimum_valid_peers) + 1
    eligible_levels = []
    for group_type in physical_hierarchy(physical):
        level = physical.loc[physical["group_type"].eq(group_type)]
        sizes = level.groupby("group_id")["entity_id"].nunique()
        eligible = set(sizes.loc[sizes.ge(minimum_group_size)].index)
        covered = level.loc[
            level["group_id"].isin(eligible), "entity_id"
        ].nunique()
        if total_entities and covered / total_entities >= float(
            minimum_entity_coverage
        ):
            eligible_levels.append(group_type)
    return eligible_levels

def choose_peer_level(
    topology,
    *,
    minimum_valid_peers=7,
    minimum_entity_coverage=0.80,
):
    """Choose the deepest physical level with adequate membership coverage."""

    eligible = eligible_peer_levels(
        topology,
        minimum_valid_peers=minimum_valid_peers,
        minimum_entity_coverage=minimum_entity_coverage,
    )
    if eligible:
        return eligible[0]
    raise ValueError(
        "No physical topology level gives enough peers to the required "
        "share of entities"
    )

def _topology_feature_specs(feature_columns):
    """Give canonical feature names short, safe SQL aliases."""

    features = sorted(dict.fromkeys(map(str, feature_columns)))
    if not features:
        raise ValueError("Topology scoring requires at least one feature")
    return [(feature, f"feature_{index:02d}") for index, feature in enumerate(features)]

def _peer_wide_sql(residual_source, feature_specs, peer_group_type):
    """Calculate one-sided, contemporaneously scaled peer deviations."""

    window = (
        "PARTITION BY r.event_ts, t.group_id "
        "ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING "
        "EXCLUDE CURRENT ROW"
    )
    statistics = []
    for feature, alias in feature_specs:
        value = f"r.{_sql_identifier(feature)}"
        median = f"median({value}) OVER ({window})"
        q25 = f"quantile_cont({value}, 0.25) OVER ({window})"
        q75 = f"quantile_cont({value}, 0.75) OVER ({window})"
        statistics.extend([
            f"count({value}) OVER ({window}) AS {_sql_identifier(alias + '__count')}",
            f"greatest(0.0, {value} - {median}) "
            f"/ greatest(({q75} - {q25}) / 1.349, 0.25) "
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
            f"greatest(0.0, median({value})) "
            f"AS {_sql_identifier(alias + '__raw')}",
            "avg(CASE "
            f"WHEN {value} IS NULL THEN NULL "
            f"WHEN {value} >= 3 THEN 1.0 ELSE 0.0 END) "
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
    affected_quantile=None,
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
        affected_upper = (
            f"quantile_cont({_sql_identifier(alias + '__affected')}, "
            f"{float(affected_quantile)})"
            if affected_quantile is not None
            else "CAST(NULL AS DOUBLE)"
        )
        queries.append(f"""
            SELECT {_sql_literal(channel)} AS channel, group_type,
                   {size_band} AS size_band,
                   {_sql_literal(feature)} AS leading_feature,
                   count(*) AS reference_rows,
                   median({raw_column}) AS centre,
                   quantile_cont({raw_column}, {float(upper_quantile)}) AS upper,
                   {affected_upper} AS affected_upper
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

def _calibrated_affected_gate(reference, feature, alias):
    """Create the group-size-aware affected-fraction calibration gate."""

    rows = reference.loc[
        reference["channel"].eq("group_common_mode")
        & reference["leading_feature"].astype(str).eq(str(feature))
    ].sort_values(["group_type", "size_band"])
    count_column = _sql_identifier(alias + "__count")
    affected_column = _sql_identifier(alias + "__affected")
    cases = []
    for row in rows.itertuples(index=False):
        if pd.isna(row.affected_upper):
            continue
        condition = (
            f"group_type = {_sql_literal(row.group_type)} AND "
            f"{_size_band_condition(count_column, row.size_band)}"
        )
        cases.append(
            f"WHEN {condition} THEN {affected_column} > "
            f"{float(row.affected_upper)!r}"
        )
    if not cases:
        return "FALSE"
    return "CASE " + " ".join(cases) + " ELSE FALSE END"

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
    affected_fraction_quantile=0.99,
):
    """Fit calibration-only null scales for peer and group evidence.

    Features remain in columns while peer and group statistics are calculated.
    This avoids multiplying a large telemetry table by the feature count. If
    ``peer_group_type`` is an ordered list, the first level with a stable
    calibration reference is used.
    """

    physical = _model_topology(topology)
    known_types = set(physical["group_type"])
    peer_candidates = (
        [peer_group_type]
        if isinstance(peer_group_type, str)
        else list(peer_group_type)
    )
    unknown_peer_types = set(peer_candidates) - known_types
    if unknown_peer_types:
        raise ValueError(f"Unknown peer group types: {sorted(unknown_peer_types)}")
    if not peer_candidates:
        raise ValueError("At least one peer group type is required")
    group_types = [name for name in group_types if name in known_types]
    if not group_types:
        raise ValueError("No requested common-mode topology level is available")

    source = _sql_literal(str(Path(calibration_residuals)))
    feature_specs = _topology_feature_specs(feature_columns)
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
        affected_quantile=affected_fraction_quantile,
        upper_quantile=upper_quantile,
    )

    with _model_duckdb(Path(calibration_residuals).parent) as connection:
        connection.register("model_topology", physical)
        peer_reference = pd.DataFrame()
        for candidate in peer_candidates:
            started = time.perf_counter()
            print(
                f"    topology reference: peer={candidate} "
                f"({len(feature_specs)} features)",
                flush=True,
            )
            peer_sql = _peer_wide_sql(source, feature_specs, candidate)
            connection.execute(
                f"CREATE TEMP TABLE topology_peer_wide AS {peer_sql}"
            )
            candidate_reference = connection.execute(peer_reference_sql).df()
            connection.execute("DROP TABLE topology_peer_wide")
            candidate_reference["scale"] = (
                candidate_reference["upper"] - candidate_reference["centre"]
            )
            candidate_reference = candidate_reference.loc[
                candidate_reference["reference_rows"].ge(
                    int(minimum_reference_rows)
                )
                & candidate_reference["scale"].gt(1e-9)
            ].reset_index(drop=True)
            print(
                f"    topology reference: peer={candidate} complete in "
                f"{(time.perf_counter() - started) / 60:.1f} minutes",
                flush=True,
            )
            if not candidate_reference.empty:
                peer_reference = candidate_reference
                break
        if peer_reference.empty:
            raise ValueError(
                "Calibration contains no stable peer reference at any "
                "eligible physical level"
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

    group_reference["scale"] = (
        group_reference["upper"] - group_reference["centre"]
    )
    group_reference = group_reference.loc[
        group_reference["reference_rows"].ge(int(minimum_reference_rows))
        & group_reference["scale"].gt(1e-9)
    ]
    reference = pd.concat(
        [peer_reference, group_reference], ignore_index=True
    ).sort_values(
        ["channel", "group_type", "size_band", "leading_feature"]
    ).reset_index(drop=True)
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
        affected_gate = _calibrated_affected_gate(reference, feature, alias)
        group_scores.append(
            f"CASE WHEN {group_count} >= {int(min_group_entities)} "
            f"AND {group_count} * 1.0 / group_size >= {float(min_group_fraction)} "
            f"AND ({affected_gate}) "
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
