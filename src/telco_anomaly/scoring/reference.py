"""Robust pooled/entity references and residual evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..features import empirical_tail_evidence, orient_residuals
from ._common import _model_duckdb, _sql_identifier, _sql_literal

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

def _reference_sample(
    path,
    maximum_rows,
    random_seed,
    maximum_rows_per_entity_day=8,
):
    """Spread fit rows across each entity-day, then apply the global cap.

    Adjacent telemetry rows are strongly correlated. Taking a small,
    deterministic sample from every entity-day gives the multivariate models
    broader regime coverage than a reservoir sample over raw rows alone.
    """

    rows_per_block = (
        None if maximum_rows_per_entity_day is None
        else int(maximum_rows_per_entity_day)
    )
    if rows_per_block is not None and rows_per_block < 1:
        raise ValueError("maximum_rows_per_entity_day must be positive")
    if int(maximum_rows) < 1:
        raise ValueError("maximum_rows must be positive")
    source = _sql_literal(str(Path(path)))
    hash_sql = (
        "hash(CAST(entity_id AS VARCHAR), "
        "CAST(episode_id AS VARCHAR), event_ts, "
        f"{int(random_seed)})"
    )
    with _model_duckdb() as connection:
        if rows_per_block is None:
            # Rank only the narrow identity key. Sorting every wide feature
            # column for the uncapped sensitivity case can exceed Colab's
            # memory even though the final sample is small.
            connection.execute(f"""
                CREATE TEMP TABLE selected_reference_rows AS
                WITH candidates AS (
                    SELECT file_row_number AS sample_row_number,
                           CAST(entity_id AS VARCHAR) AS entity_id,
                           CAST(episode_id AS VARCHAR) AS episode_id,
                           event_ts,
                           {hash_sql} AS sample_hash
                    FROM read_parquet({source}, file_row_number=true)
                ), selected AS (
                    SELECT *
                    FROM candidates
                    ORDER BY sample_hash, entity_id, episode_id, event_ts,
                             sample_row_number
                    LIMIT {int(maximum_rows)}
                )
                SELECT sample_row_number,
                       row_number() OVER (
                           ORDER BY sample_hash, entity_id, episode_id,
                                    event_ts, sample_row_number
                       ) AS sample_rank
                FROM selected
            """)
        else:
            connection.execute(f"""
                CREATE TEMP TABLE selected_reference_rows AS
                WITH candidates AS (
                    SELECT file_row_number AS sample_row_number,
                           CAST(entity_id AS VARCHAR) AS entity_id,
                           CAST(episode_id AS VARCHAR) AS episode_id,
                           event_ts,
                           CAST(event_ts AT TIME ZONE 'UTC' AS DATE) AS sample_day,
                           floor((extract(hour FROM event_ts AT TIME ZONE 'UTC') * 60
                               + extract(minute FROM event_ts AT TIME ZONE 'UTC'))
                               / (1440.0 / {rows_per_block})) AS time_bin,
                           {hash_sql} AS sample_hash
                    FROM read_parquet({source}, file_row_number=true)
                ), balanced AS (
                    SELECT sample_row_number, entity_id, episode_id,
                           event_ts, sample_hash
                    FROM candidates
                    QUALIFY row_number() OVER (
                        PARTITION BY entity_id, sample_day, time_bin
                        ORDER BY sample_hash, event_ts, episode_id,
                                 sample_row_number
                    ) = 1
                ), selected AS (
                    SELECT *
                    FROM balanced
                    ORDER BY sample_hash, entity_id, episode_id, event_ts,
                             sample_row_number
                    LIMIT {int(maximum_rows)}
                )
                SELECT sample_row_number,
                       row_number() OVER (
                           ORDER BY sample_hash, entity_id, episode_id,
                                    event_ts, sample_row_number
                       ) AS sample_rank
                FROM selected
            """)

        # Joining the selected keys back to the wide file avoids a wide
        # full-source sort. The bounded result is ordered in Pandas so model
        # fitting remains deterministic even with insertion-order disabled.
        sample = connection.execute(f"""
            SELECT keys.sample_rank AS __sample_rank,
                   source.* EXCLUDE (file_row_number)
            FROM read_parquet({source}, file_row_number=true) AS source
            JOIN selected_reference_rows AS keys
              ON source.file_row_number = keys.sample_row_number
        """).df()
    sample = sample.sort_values(
        "__sample_rank",
        kind="mergesort",
    ).drop(columns="__sample_rank").reset_index(drop=True)
    return sample

def _full_entity_reference(
    path, features, scale_floors, minimum_rows=30, minimum_days=2,
):
    """Robust entity references from the full calibration-fit slice.

    The training-row cap is for pooled PCA and Isolation Forest only. Entity
    baselines are small group summaries, so DuckDB can calculate them over the
    complete early-calibration file without materialising all rows in Python.
    """

    if int(minimum_rows) < 1 or float(minimum_days) < 0:
        raise ValueError("Entity reference row and duration limits are invalid")

    expressions = []
    for feature in features:
        column = _sql_identifier(feature)
        expressions.extend([
            f"count({column}) AS {_sql_identifier(feature + '__count')}",
            f"approx_quantile({column}, 0.50) AS {_sql_identifier(feature + '__centre')}",
            f"approx_quantile({column}, 0.25) AS {_sql_identifier(feature + '__q25')}",
            f"approx_quantile({column}, 0.75) AS {_sql_identifier(feature + '__q75')}",
            f"min(event_ts) FILTER (WHERE {column} IS NOT NULL) "
            f"AS {_sql_identifier(feature + '__first_ts')}",
            f"max(event_ts) FILTER (WHERE {column} IS NOT NULL) "
            f"AS {_sql_identifier(feature + '__last_ts')}",
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

    days = pd.DataFrame({
        name: (
            pd.to_datetime(summary[f"{name}__last_ts"], utc=True)
            - pd.to_datetime(summary[f"{name}__first_ts"], utc=True)
        ).dt.total_seconds() / 86400
        for name in features
    }).fillna(0.0)
    valid = counts.ge(int(minimum_rows)) & days.ge(float(minimum_days))
    centre = centre.where(valid)
    scale = (q75 - q25) / 1.349
    floors = centre.abs() * 1e-6
    for name in features:
        floors[name] = floors[name].clip(lower=float(scale_floors[name]))
    scale = scale.where(valid).where(scale.gt(floors), floors.where(valid))
    return centre, scale, counts, days

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

def _directional_model_frame(residuals, directions):
    """Orient model inputs so only declared harmful directions are large."""

    return orient_residuals(residuals, directions).clip(lower=0, upper=50)

def _empirical_vector_evidence(values, calibration):
    """Convert a one-dimensional score to calibration-tail evidence."""

    values = np.asarray(values, dtype=float)
    calibration = np.sort(np.asarray(calibration, dtype=float))
    result = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    if not len(calibration):
        return result
    exceedances = len(calibration) - np.searchsorted(
        calibration, values[valid], side="left"
    )
    result[valid] = -np.log10(
        (exceedances + 1.0) / (len(calibration) + 1.0)
    )
    return result

def _second_metric_evidence(evidence, feature_metric_ids):
    """Return the second-largest tail score across distinct metrics."""

    metric_scores = {}
    for metric_id in sorted(set(feature_metric_ids.values())):
        columns = [
            name for name in evidence
            if feature_metric_ids.get(name) == metric_id
        ]
        if columns:
            metric_scores[metric_id] = evidence[columns].max(
                axis=1, skipna=True
            )
    if len(metric_scores) < 2:
        return np.full(len(evidence), np.nan)
    values = pd.DataFrame(metric_scores, index=evidence.index).to_numpy(float)
    finite_count = np.isfinite(values).sum(axis=1)
    safe = np.where(np.isfinite(values), values, -np.inf)
    second = np.partition(safe, -2, axis=1)[:, -2]
    second[finite_count < 2] = np.nan
    return second

def _top_two_metric_mean_evidence(evidence, feature_metric_ids):
    """Mean the two strongest distinct-metric tail scores.

    This retains the corroboration requirement of ``multimetric_residual``
    while allowing one very strong and one moderate metric to contribute.
    It is metric-name agnostic and is calibrated like every other channel.
    """

    metric_scores = {}
    for metric_id in sorted(set(feature_metric_ids.values())):
        columns = [
            name for name in evidence
            if feature_metric_ids.get(name) == metric_id
        ]
        if columns:
            metric_scores[metric_id] = evidence[columns].max(
                axis=1, skipna=True
            )
    if len(metric_scores) < 2:
        return np.full(len(evidence), np.nan)
    values = pd.DataFrame(metric_scores, index=evidence.index).to_numpy(float)
    finite_count = np.isfinite(values).sum(axis=1)
    safe = np.where(np.isfinite(values), values, -np.inf)
    strongest = np.partition(safe, -2, axis=1)[:, -2:]
    result = strongest.mean(axis=1)
    result[finite_count < 2] = np.nan
    return result

def _fit_entity_score_reference(
    scores,
    entity_ids,
    *,
    minimum_rows=30,
    scale_floor_fraction_of_global=0.25,
):
    """Fit frozen robust references for an entity's Isolation Forest score."""

    frame = pd.DataFrame({
        "entity_id": pd.Series(entity_ids, dtype="string").to_numpy(),
        "score": np.asarray(scores, dtype=float),
    }).replace([np.inf, -np.inf], np.nan).dropna(subset=["score"])
    if frame.empty:
        raise ValueError("Isolation Forest produced no finite calibration scores")

    global_centre = float(frame["score"].median())
    global_scale = float(
        (frame["score"].quantile(0.75) - frame["score"].quantile(0.25))
        / 1.349
    )
    global_mad = float((frame["score"] - global_centre).abs().median() * 1.4826)
    global_scale = max(global_scale, global_mad, 1e-9)

    grouped = frame.groupby("entity_id")["score"]
    summary = grouped.agg(count="count", centre="median")
    summary["q25"] = grouped.quantile(0.25)
    summary["q75"] = grouped.quantile(0.75)
    summary["scale"] = (summary["q75"] - summary["q25"]) / 1.349
    scale_floor = global_scale * float(scale_floor_fraction_of_global)
    summary["scale"] = summary["scale"].clip(lower=scale_floor)
    summary.loc[summary["count"].lt(int(minimum_rows)), ["centre", "scale"]] = np.nan

    centres = frame["entity_id"].map(summary["centre"]).fillna(global_centre)
    scales = frame["entity_id"].map(summary["scale"]).fillna(global_scale)
    adjusted = np.maximum(0.0, (frame["score"] - centres) / scales)
    return {
        "global_centre": global_centre,
        "global_scale": global_scale,
        "entity_centre": summary["centre"],
        "entity_scale": summary["scale"],
        "minimum_rows": int(minimum_rows),
        "scale_floor_fraction_of_global": float(
            scale_floor_fraction_of_global
        ),
        "tail_reference": np.sort(adjusted.to_numpy(float)),
    }

def _entity_adjusted_score_evidence(scores, entity_ids, reference):
    """Apply a frozen entity score reference, with a pooled fallback."""

    ids = pd.Series(entity_ids, dtype="string")
    centre = ids.map(reference["entity_centre"]).fillna(
        reference["global_centre"]
    )
    scale = ids.map(reference["entity_scale"]).fillna(
        reference["global_scale"]
    )
    adjusted = np.maximum(
        0.0,
        (np.asarray(scores, dtype=float) - centre.to_numpy(float))
        / scale.to_numpy(float),
    )
    return _empirical_vector_evidence(
        adjusted, reference["tail_reference"]
    )
