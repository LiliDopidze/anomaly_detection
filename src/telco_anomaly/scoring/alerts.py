"""Score-to-alert conversion and label-free candidate grids."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ._common import IDENTITY_COLUMNS, MODEL_IDS, iter_episode_frames

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

def _recovery_threshold(alert_threshold, threshold_fraction):
    """Return a lower off-threshold while preserving the score sign."""

    fraction = float(threshold_fraction)
    if not 0 < fraction <= 1:
        raise ValueError("recovery_threshold_fraction must be in (0, 1]")
    threshold = float(alert_threshold)
    return threshold * fraction if threshold >= 0 else threshold / fraction

def alerts_from_score_file(
    score_path,
    model_id,
    threshold,
    *,
    min_consecutive,
    recovery_consecutive,
    cadence_seconds=None,
    recovery_threshold_fraction=1.0,
):
    """Create alerts episode by episode without materialising all scores."""

    from ..evaluation import SCORE_COLUMNS, scores_to_alerts

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
            cadence_seconds=cadence_seconds,
            recovery_threshold=_recovery_threshold(
                threshold, recovery_threshold_fraction
            ),
        )
        if not produced.empty:
            alerts.append(produced)
    if not alerts:
        from ..evaluation import ALERT_COLUMNS
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
    cadence_seconds=None,
    recovery_threshold_fraction=1.0,
):
    """Create all channel/threshold alert sets with one scan per channel."""

    from ..evaluation import ALERT_COLUMNS, SCORE_COLUMNS, scores_to_alerts

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
                    cadence_seconds=cadence_seconds,
                    recovery_threshold=_recovery_threshold(
                        float(row.threshold), recovery_threshold_fraction
                    ),
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

    from ..evaluation import ALERT_COLUMNS, SCORE_COLUMNS, scores_to_alerts

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
