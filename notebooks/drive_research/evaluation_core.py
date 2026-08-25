"""Tested mechanics for the evaluation-harness notebook.

Notebook 03 owns policy choices and adversarial controls.  This flat module
owns the repetitive mechanics whose silent failure would invalidate results:
truth partitioning, score-to-alert conversion, one-to-one event matching and
metric calculation.  It contains no sector-specific logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2


EVALUATION_CORE_VERSION = "1.3.0"
PARTITIONS = ("calibration", "development", "holdout")
SCORE_COLUMNS = [
    "event_ts", "entity_id", "episode_id", "anomaly_score", "model_id",
]
ALERT_COLUMNS = [
    "alert_id", "model_id", "entity_id", "episode_id", "alert_start",
    "alert_end", "peak_ts", "peak_score", "n_scores", "leading_feature",
]
AUDIT_COLUMNS = [
    "fault_id", "fault_type", "assigned_partition", "status",
    "scoreable", "cross_partition", "declared_entity_count",
    "scoreable_entity_count", "invalid_window_entity_count",
]


def _interval_scoreability(events, intervals, registry):
    """Identify affected-entity intervals with a usable matching window."""

    entity_windows = registry[
        ["entity_id", "observed_from", "observed_to"]
    ].copy()
    event_windows = events[
        ["fault_id", "observable_ts", "onset_ts", "end_ts"]
    ].rename(columns={"end_ts": "event_end"})

    joined = intervals.copy()
    joined["entity_id"] = joined["entity_id"].astype(str)
    joined = joined.merge(
        entity_windows, on="entity_id", how="left", validate="many_to_one"
    ).merge(
        event_windows, on="fault_id", how="left", validate="many_to_one"
    )

    for column in [
        "start_ts", "end_ts", "observed_from", "observed_to",
        "observable_ts", "onset_ts", "event_end",
    ]:
        joined[column] = pd.to_datetime(
            joined[column], utc=True, errors="coerce"
        )

    reference = joined["observable_ts"].fillna(joined["onset_ts"])
    joined["match_start"] = pd.concat(
        [joined["start_ts"], joined["observed_from"], reference], axis=1
    ).max(axis=1)
    joined["match_end"] = pd.concat(
        [joined["end_ts"], joined["observed_to"], joined["event_end"]],
        axis=1,
    ).min(axis=1)
    joined["scoreable_interval"] = (
        joined["observed_from"].notna()
        & joined["match_start"].notna()
        & joined["match_end"].notna()
        & joined["match_end"].gt(joined["match_start"])
    )
    return joined


def _temporal_assignment(events, intervals, partitions, registry):
    events = events.copy()
    interval_audit = _interval_scoreability(events, intervals, registry)
    interval_audit["scoreable_for_partition"] = False
    events["reference_ts"] = events["observable_ts"].fillna(events["onset_ts"])

    rows = []
    for event in events.itertuples(index=False):
        assigned = partitions.loc[
            partitions["start_ts"].le(event.reference_ts)
            & (event.reference_ts < partitions["end_ts"])
        ]
        assigned_name = assigned.iloc[0]["partition"] if len(assigned) == 1 else None
        event_end = event.end_ts if pd.notna(event.end_ts) else event.reference_ts
        touched = partitions.loc[
            partitions["start_ts"].lt(event_end)
            & partitions["end_ts"].gt(event.reference_ts)
        ]
        cross_partition = len(touched) > 1

        fault_intervals = interval_audit.loc[
            interval_audit["fault_id"].astype(str).eq(str(event.fault_id))
        ].copy()
        if assigned_name is not None:
            part = assigned.iloc[0]
            fault_intervals["scoreable_for_partition"] = (
                fault_intervals["scoreable_interval"]
                & fault_intervals["match_start"].lt(part["end_ts"])
                & fault_intervals["match_end"].gt(part["start_ts"])
            )
        else:
            fault_intervals["scoreable_for_partition"] = False

        interval_audit.loc[
            fault_intervals.index, "scoreable_for_partition"
        ] = fault_intervals["scoreable_for_partition"]
        n_scoreable = fault_intervals.loc[
            fault_intervals["scoreable_for_partition"], "entity_id"
        ].nunique()
        if assigned_name is None:
            status = "unassigned"
        elif cross_partition:
            status = "cross_partition"
        elif n_scoreable == 0:
            status = "unscoreable"
        else:
            status = "scoreable"

        rows.append({
            "fault_id": str(event.fault_id),
            "fault_type": event.fault_type,
            "assigned_partition": assigned_name,
            "status": status,
            "scoreable": status == "scoreable",
            "cross_partition": cross_partition,
            "declared_entity_count": fault_intervals["entity_id"].nunique(),
            "scoreable_entity_count": n_scoreable,
            "invalid_window_entity_count": fault_intervals.loc[
                ~fault_intervals["scoreable_interval"], "entity_id"
            ].nunique(),
        })

    audit = pd.DataFrame(rows, columns=AUDIT_COLUMNS)
    interval_audit = interval_audit.merge(
        audit[["fault_id", "assigned_partition", "status"]],
        on="fault_id", how="left",
    )
    interval_audit["scoreable_for_partition"] &= interval_audit["status"].eq(
        "scoreable"
    )
    return audit, interval_audit


def _entity_assignment(events, intervals, partitions, registry):
    interval_audit = _interval_scoreability(events, intervals, registry)
    membership = partitions[["entity_id", "partition"]].copy()
    membership["entity_id"] = membership["entity_id"].astype(str)
    if membership["entity_id"].duplicated().any():
        raise ValueError("Entity partitions must assign each entity once")
    interval_audit = interval_audit.merge(membership, on="entity_id", how="left")

    rows = []
    for event in events.itertuples(index=False):
        fault_intervals = interval_audit.loc[
            interval_audit["fault_id"].astype(str).eq(str(event.fault_id))
        ]
        scoreable = fault_intervals.loc[fault_intervals["scoreable_interval"]]
        assigned = sorted(scoreable["partition"].dropna().unique())
        cross_partition = len(assigned) > 1
        if scoreable.empty:
            partition, status = None, "unscoreable"
        elif cross_partition:
            partition, status = None, "cross_partition"
        elif len(assigned) != 1:
            partition, status = None, "unassigned"
        else:
            partition, status = assigned[0], "scoreable"

        rows.append({
            "fault_id": str(event.fault_id),
            "fault_type": event.fault_type,
            "assigned_partition": partition,
            "status": status,
            "scoreable": status == "scoreable",
            "cross_partition": cross_partition,
            "declared_entity_count": fault_intervals["entity_id"].nunique(),
            "scoreable_entity_count": (
                scoreable["entity_id"].nunique() if status == "scoreable" else 0
            ),
            "invalid_window_entity_count": fault_intervals.loc[
                ~fault_intervals["scoreable_interval"], "entity_id"
            ].nunique(),
        })

    audit = pd.DataFrame(rows, columns=AUDIT_COLUMNS)
    interval_audit = interval_audit.merge(
        audit[["fault_id", "assigned_partition", "status"]],
        on="fault_id", how="left",
    )
    interval_audit["scoreable_for_partition"] = (
        interval_audit["scoreable_interval"]
        & interval_audit["status"].eq("scoreable")
    )
    return audit, interval_audit


def _assign_conditions(conditions, primary_split, time_partitions, entity_partitions):
    if conditions.empty:
        return conditions.assign(assigned_partition=pd.Series(dtype="string"))
    assigned = conditions.copy()
    if primary_split == "entity":
        membership = entity_partitions[["entity_id", "partition"]].copy()
        membership["entity_id"] = membership["entity_id"].astype(str)
        assigned["entity_id"] = assigned["entity_id"].astype(str)
        return assigned.merge(
            membership.rename(columns={"partition": "assigned_partition"}),
            on="entity_id", how="left",
        )

    # A state may cross a time boundary.  Split and clip it so a development
    # row can never reveal a holdout end time.
    pieces = []
    for part in time_partitions.itertuples(index=False):
        overlaps = assigned["start_ts"].lt(part.end_ts) & (
            assigned["end_ts"].isna()
            | assigned["end_ts"].gt(part.start_ts)
        )
        piece = assigned.loc[overlaps].copy()
        if piece.empty:
            continue
        piece["start_ts"] = piece["start_ts"].clip(lower=part.start_ts)
        piece["end_ts"] = piece["end_ts"].fillna(part.end_ts).clip(
            upper=part.end_ts
        )
        piece["assigned_partition"] = part.partition
        pieces.append(piece)
    if not pieces:
        return assigned.iloc[:0].assign(
            assigned_partition=pd.Series(dtype="string")
        )
    return pd.concat(pieces, ignore_index=True)


def partition_truth(
    events,
    intervals,
    conditions,
    registry,
    *,
    primary_split,
    time_partitions,
    entity_partitions,
):
    """Return physically writable truth partitions and their audit."""

    if primary_split == "time":
        audit, interval_audit = _temporal_assignment(
            events, intervals, time_partitions, registry
        )
    elif primary_split == "entity":
        audit, interval_audit = _entity_assignment(
            events, intervals, entity_partitions, registry
        )
    else:
        raise ValueError("primary_split must be 'time' or 'entity'")

    condition_assignments = _assign_conditions(
        conditions, primary_split, time_partitions, entity_partitions
    )
    output = {}
    for partition in PARTITIONS:
        fault_ids = set(audit.loc[
            audit["assigned_partition"].eq(partition)
            & audit["status"].eq("scoreable"),
            "fault_id",
        ])
        output[partition] = {
            "fault_events": events.loc[
                events["fault_id"].astype(str).isin(fault_ids)
            ].reset_index(drop=True),
            "fault_entity_intervals": interval_audit.loc[
                interval_audit["fault_id"].astype(str).isin(fault_ids)
                & interval_audit["scoreable_for_partition"],
                intervals.columns,
            ].reset_index(drop=True),
            "condition_states": condition_assignments.loc[
                condition_assignments["assigned_partition"].eq(partition),
                conditions.columns,
            ].reset_index(drop=True),
        }

    summary = (
        audit.groupby(
            ["assigned_partition", "fault_type", "status"],
            dropna=False, as_index=False,
        )
        .agg(
            faults=("fault_id", "nunique"),
            affected_entities=("scoreable_entity_count", "sum"),
        )
        .sort_values(["assigned_partition", "fault_type", "status"])
    )
    return output, audit, summary


def validate_scores(scores):
    missing = set(SCORE_COLUMNS) - set(scores.columns)
    if missing:
        raise ValueError(f"Missing score columns: {sorted(missing)}")
    columns = [*SCORE_COLUMNS]
    if "leading_feature" in scores.columns:
        columns.append("leading_feature")
    clean = scores[columns].copy()
    clean["event_ts"] = pd.to_datetime(clean["event_ts"], utc=True, errors="raise")
    clean["entity_id"] = clean["entity_id"].astype(str)
    clean["model_id"] = clean["model_id"].astype(str)
    clean["anomaly_score"] = pd.to_numeric(clean["anomaly_score"], errors="raise")
    if not np.isfinite(clean["anomaly_score"]).all():
        raise ValueError("Anomaly scores must be finite")
    clean["episode_id"] = clean["episode_id"].astype(str)
    if "leading_feature" not in clean:
        clean["leading_feature"] = pd.NA
    keys = ["event_ts", "entity_id", "episode_id", "model_id"]
    if clean.duplicated(keys).any():
        raise ValueError("A model may emit only one score per episode and timestamp")
    return clean.sort_values(
        ["model_id", "entity_id", "episode_id", "event_ts"]
    ).reset_index(drop=True)


def merge_nearby_alerts(alerts, merge_gap_seconds):
    """Merge nearby alerts within the same model, entity and episode."""

    if alerts.empty or merge_gap_seconds <= 0:
        return alerts

    maximum_gap = pd.Timedelta(seconds=merge_gap_seconds)
    merged = []
    for _, group in alerts.groupby(
        ["model_id", "entity_id", "episode_id"], sort=True
    ):
        current = None
        for alert in group.sort_values("alert_start").to_dict("records"):
            if current is None:
                current = alert
                continue
            if alert["alert_start"] - current["alert_end"] <= maximum_gap:
                current["alert_end"] = max(
                    current["alert_end"], alert["alert_end"]
                )
                current["n_scores"] += alert["n_scores"]
                if alert["peak_score"] > current["peak_score"]:
                    current["peak_ts"] = alert["peak_ts"]
                    current["peak_score"] = alert["peak_score"]
            else:
                merged.append(current)
                current = alert
        if current is not None:
            merged.append(current)

    output = pd.DataFrame(merged, columns=ALERT_COLUMNS)
    output = output.sort_values(
        ["alert_start", "model_id", "entity_id"]
    ).reset_index(drop=True)
    output["alert_id"] = [
        f"A-{number:06d}" for number in range(1, len(output) + 1)
    ]
    return output


def apply_alert_cooldown(alerts, cooldown_seconds):
    """Suppress repeat notifications within an episode after an alert fires.

    The first alert is retained, so the rule is causal. A new source episode
    always starts with a clean alert state.
    """

    if alerts.empty or cooldown_seconds <= 0:
        return alerts

    cooldown = pd.Timedelta(seconds=cooldown_seconds)
    retained = []
    for _, group in alerts.groupby(
        ["model_id", "entity_id", "episode_id"], sort=True
    ):
        next_allowed = None
        for alert in group.sort_values("alert_start").to_dict("records"):
            if next_allowed is None or alert["alert_start"] >= next_allowed:
                retained.append(alert)
                next_allowed = alert["alert_start"] + cooldown

    output = pd.DataFrame(retained, columns=ALERT_COLUMNS)
    output = output.sort_values(
        ["alert_start", "model_id", "entity_id", "episode_id"]
    ).reset_index(drop=True)
    output["alert_id"] = [
        f"A-{number:06d}" for number in range(1, len(output) + 1)
    ]
    return output


def _alert_intervals(high, min_consecutive, recovery_consecutive):
    """Return inclusive (start, end) index pairs from a boolean array."""

    n = len(high)
    if n == 0:
        return []
    boundaries = np.flatnonzero(np.diff(high)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [n]))

    intervals = []
    active = None
    for start, end in zip(starts, ends):
        length = end - start
        if high[start]:
            if active is None and length >= min_consecutive:
                active = start
        elif active is not None and length >= recovery_consecutive:
            intervals.append((active, start - 1))
            active = None
    if active is not None:
        intervals.append((active, n - 1))
    return intervals


def scores_to_alerts(
    scores,
    threshold,
    *,
    min_consecutive=2,
    gap_factor=1.5,
    recovery_consecutive=None,
):
    """Convert scores to causal alerts with a simple recovery rule.

    An alert opens after ``min_consecutive`` high scores. It closes only after
    ``recovery_consecutive`` normal scores, which prevents a short dip from
    creating repeated notifications for one continuing anomaly.
    """

    scores = validate_scores(scores)
    recovery_consecutive = (
        min_consecutive
        if recovery_consecutive is None else int(recovery_consecutive)
    )
    if min_consecutive < 1 or recovery_consecutive < 1:
        raise ValueError("Persistence and recovery must be positive")

    alerts = []
    for (model_id, entity_id, episode_id), group in scores.groupby(
        ["model_id", "entity_id", "episode_id"], sort=True
    ):
        group = group.sort_values("event_ts").reset_index(drop=True)
        timestamps = group["event_ts"]
        values = group["anomaly_score"].to_numpy()
        features = group["leading_feature"].to_numpy()

        differences = timestamps.diff()
        positive = differences.loc[differences.gt(pd.Timedelta(0))]
        cadence = positive.median() if not positive.empty else pd.Timedelta(0)

        # A long gap ends any open alert and clears the state machine, so
        # each stretch of regular observations is handled independently.
        if cadence > pd.Timedelta(0):
            breaks = np.flatnonzero(differences.gt(cadence * gap_factor).to_numpy())
        else:
            breaks = np.empty(0, dtype=int)
        segment_starts = np.concatenate(([0], breaks))
        segment_ends = np.concatenate((breaks, [len(group)]))

        high = values >= threshold
        for segment_start, segment_end in zip(segment_starts, segment_ends):
            window = high[segment_start:segment_end]
            for start, end in _alert_intervals(
                window, min_consecutive, recovery_consecutive
            ):
                start += segment_start
                end += segment_start
                peak = start + int(np.argmax(values[start:end + 1]))
                alerts.append({
                    "model_id": model_id,
                    "entity_id": entity_id,
                    "episode_id": episode_id,
                    "alert_start": timestamps.iloc[start],
                    "alert_end": timestamps.iloc[end] + cadence,
                    "peak_ts": timestamps.iloc[peak],
                    "peak_score": values[peak],
                    "n_scores": end - start + 1,
                    "leading_feature": features[peak],
                })

    output = pd.DataFrame(alerts)
    if output.empty:
        return pd.DataFrame(columns=ALERT_COLUMNS)
    output = output.sort_values(
        ["alert_start", "model_id", "entity_id", "episode_id"]
    ).reset_index(drop=True)
    output.insert(
        0, "alert_id",
        [f"A-{number:06d}" for number in range(1, len(output) + 1)],
    )
    return output[ALERT_COLUMNS]


def validate_alerts(alerts):
    """Validate the small alert interface before matching it to truth."""

    missing = set(ALERT_COLUMNS) - set(alerts.columns)
    if missing:
        raise ValueError(f"Missing alert columns: {sorted(missing)}")
    clean = alerts[ALERT_COLUMNS].copy()
    if clean.empty:
        return clean

    for column in ("alert_start", "alert_end", "peak_ts"):
        clean[column] = pd.to_datetime(
            clean[column], utc=True, errors="raise"
        )
    clean["peak_score"] = pd.to_numeric(
        clean["peak_score"], errors="raise"
    )
    clean["n_scores"] = pd.to_numeric(
        clean["n_scores"], errors="raise"
    )
    if clean["alert_id"].astype(str).duplicated().any():
        raise ValueError("alert_id must be unique")
    if not np.isfinite(clean["peak_score"]).all():
        raise ValueError("Alert peak scores must be finite")
    if clean["n_scores"].lt(1).any():
        raise ValueError("Each alert must contain at least one score")
    if clean["alert_end"].lt(clean["alert_start"]).any():
        raise ValueError("alert_end cannot be earlier than alert_start")
    if (
        clean["peak_ts"].lt(clean["alert_start"])
        | clean["peak_ts"].gt(clean["alert_end"])
    ).any():
        raise ValueError("peak_ts must fall inside the alert interval")
    return clean


def _fault_windows(events, intervals, decision_horizon_seconds):
    windows = intervals[[
        "fault_id", "entity_id", "start_ts", "end_ts"
    ]].rename(columns={"end_ts": "interval_end"}).merge(
        events[[
            "fault_id", "fault_type", "observable_ts", "onset_ts",
            "impact_ts", "end_ts", "group_id",
        ]].rename(columns={"end_ts": "event_end"}),
        on="fault_id", how="inner", validate="many_to_one",
    )
    for column in [
        "start_ts", "interval_end", "observable_ts", "onset_ts",
        "impact_ts", "event_end",
    ]:
        windows[column] = pd.to_datetime(windows[column], utc=True, errors="coerce")
    reference = windows["observable_ts"].fillna(windows["onset_ts"])
    windows["match_start"] = pd.concat(
        [reference, windows["start_ts"]], axis=1
    ).max(axis=1)
    horizon_end = windows["match_start"] + pd.to_timedelta(
        decision_horizon_seconds, unit="s"
    )
    windows["match_end"] = pd.concat(
        [windows["event_end"], windows["interval_end"], horizon_end], axis=1
    ).min(axis=1)
    invalid_end = (
        windows["match_end"].lt(windows["match_start"])
        if decision_horizon_seconds == 0
        else windows["match_end"].le(windows["match_start"])
    )
    invalid = windows["match_start"].isna() | (
        windows["match_end"].notna() & invalid_end
    )
    if invalid.any():
        examples = windows.loc[
            invalid,
            ["fault_id", "entity_id", "match_start", "match_end"],
        ].head(5).to_dict("records")
        raise ValueError(
            "Evaluation truth contains an invalid matching window. "
            "Rebuild the evaluation bundle with Notebook 03. "
            f"Examples: {examples}"
        )

    for entity_id, entity_windows in windows.groupby("entity_id", sort=False):
        ordered = entity_windows.sort_values("match_start").reset_index(drop=True)
        for index, fault in ordered.iloc[:-1].iterrows():
            overlaps = ordered.iloc[index + 1:].loc[
                ordered.iloc[index + 1:]["match_start"].lt(fault["match_end"])
                & ordered.iloc[index + 1:]["fault_id"].astype(str).ne(
                    str(fault["fault_id"])
                )
            ]
            if not overlaps.empty:
                other = overlaps.iloc[0]
                raise ValueError(
                    "Greedy matching is unsafe because scoreable faults "
                    f"{fault['fault_id']} and {other['fault_id']} have "
                    f"overlapping windows on entity {entity_id}"
                )
    return windows


def _match_alerts(alerts, events, intervals, decision_horizon_seconds):
    alerts = alerts.copy()
    windows = _fault_windows(events, intervals, decision_horizon_seconds)
    candidates = alerts.merge(windows, on="entity_id", how="inner")
    candidates = candidates.loc[
        candidates["alert_start"].ge(candidates["match_start"])
        & (
            candidates["match_end"].isna()
            | candidates["alert_start"].lt(candidates["match_end"])
        )
    ].copy()
    candidates = candidates.sort_values(["match_end", "match_start", "fault_id"])
    by_alert = {
        key: frame for key, frame in candidates.groupby("alert_id", sort=False)
    }
    empty = candidates.iloc[:0]

    matched_faults = set()
    matched_pairs = set()
    rows = []
    ordered_alerts = alerts.sort_values(
        ["alert_start", "peak_score", "alert_id"],
        ascending=[True, False, True],
    )
    for alert in ordered_alerts.itertuples(index=False):
        choices = by_alert.get(alert.alert_id, empty)
        available = choices.loc[
            ~choices["fault_id"].astype(str).isin(matched_faults)
        ]
        if not available.empty:
            selected, status = available.iloc[0], "matched"
            matched_faults.add(str(selected["fault_id"]))
            matched_pairs.add((str(selected["fault_id"]), str(alert.entity_id)))
        elif not choices.empty:
            new_pair = choices.loc[
                [
                    (str(fault_id), str(alert.entity_id)) not in matched_pairs
                    for fault_id in choices["fault_id"]
                ]
            ]
            if not new_pair.empty:
                selected, status = new_pair.iloc[0], "entity_confirmation"
                matched_pairs.add((str(selected["fault_id"]), str(alert.entity_id)))
            else:
                selected, status = choices.iloc[0], "duplicate"
        else:
            selected, status = None, "false_alert"

        record = {
            **{column: getattr(alert, column) for column in ALERT_COLUMNS},
            "match_status": status,
            "fault_id": pd.NA,
            "fault_type": pd.NA,
            "detection_delay_seconds": np.nan,
            "preimpact": False,
        }
        if selected is not None:
            reference = selected["observable_ts"]
            if pd.isna(reference):
                reference = selected["onset_ts"]
            record.update({
                "fault_id": selected["fault_id"],
                "fault_type": selected["fault_type"],
                "detection_delay_seconds": (
                    alert.alert_start - reference
                ).total_seconds(),
                "preimpact": (
                    pd.notna(selected["impact_ts"])
                    and alert.alert_start < selected["impact_ts"]
                ),
            })
        rows.append(record)

    matches = pd.DataFrame(rows, columns=[
        *ALERT_COLUMNS, "match_status", "fault_id", "fault_type",
        "detection_delay_seconds", "preimpact",
    ])
    first = (
        matches.loc[matches["match_status"].eq("matched")]
        .sort_values(["fault_id", "alert_start"])
        .drop_duplicates("fault_id")
    )
    affected = windows.groupby("fault_id")["entity_id"].nunique().rename(
        "affected_entities"
    )
    fault_results = events.copy().merge(affected, on="fault_id", how="left").merge(
        first[[
            "fault_id", "alert_id", "alert_start",
            "detection_delay_seconds", "preimpact",
        ]],
        on="fault_id", how="left",
    )
    fault_results["affected_entities"] = (
        fault_results["affected_entities"].fillna(0).astype(int)
    )
    fault_results["detected"] = fault_results["alert_id"].notna()
    fault_results["preimpact"] = fault_results["preimpact"].eq(True)

    # Event credit remains one-to-one, while a distinct affected entity may
    # confirm an already detected shared fault without becoming a duplicate.
    pair_candidates = (
        matches.loc[matches["match_status"].isin([
            "matched", "entity_confirmation"
        ]), [
            "fault_id", "entity_id",
        ]]
        .drop_duplicates()
        .assign(detected=True)
    )
    pair_results = windows[["fault_id", "entity_id"]].drop_duplicates().merge(
        pair_candidates, on=["fault_id", "entity_id"], how="left"
    )
    pair_results["detected"] = pair_results["detected"].eq(True)
    group_results = (
        fault_results.loc[fault_results["group_id"].notna()]
        .groupby("group_id", as_index=False)
        .agg(
            faults=("fault_id", "nunique"),
            affected_entities=("affected_entities", "sum"),
            detected=("detected", "max"),
        )
    )
    return matches, fault_results, pair_results, group_results


def _wilson_interval(successes, total, z=1.96):
    if total == 0:
        return np.nan, np.nan
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * np.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return centre - margin, centre + margin


def _poisson_rate_interval(count, exposure, alpha=0.05):
    """Exact Garwood interval for an event rate."""

    if exposure <= 0:
        return np.nan, np.nan
    lower = 0.0 if count == 0 else chi2.ppf(alpha / 2, 2 * count) / 2
    upper = chi2.ppf(1 - alpha / 2, 2 * (count + 1)) / 2
    return lower / exposure, upper / exposure


def _false_alert_cluster_count(false_alerts, entity_groups, window_seconds):
    """Count temporally connected false-alert clusters."""

    if false_alerts.empty or window_seconds <= 0:
        return len(false_alerts)

    alerts = false_alerts[["alert_id", "entity_id", "alert_start"]].copy()
    alerts["entity_id"] = alerts["entity_id"].astype(str)
    alerts = alerts.reset_index(drop=True).reset_index(names="alert_number")
    identity_groups = alerts[["entity_id"]].drop_duplicates().assign(
        group_type="entity", group_id=lambda frame: frame["entity_id"]
    )
    if entity_groups is None or entity_groups.empty:
        memberships = identity_groups
    else:
        required = {"entity_id", "group_type", "group_id"}
        missing = required - set(entity_groups.columns)
        if missing:
            raise ValueError(
                f"Entity groups are missing columns: {sorted(missing)}"
            )
        memberships = entity_groups[
            ["entity_id", "group_type", "group_id"]
        ].copy()
        memberships["entity_id"] = memberships["entity_id"].astype(str)
        memberships = pd.concat(
            [memberships, identity_groups], ignore_index=True
        ).drop_duplicates()

    joined = alerts.merge(memberships, on="entity_id", how="left")
    parent = np.arange(len(alerts))

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    maximum_gap = pd.Timedelta(seconds=float(window_seconds))
    for _, group in joined.groupby(["group_type", "group_id"], sort=False):
        ordered = group.sort_values("alert_start")
        previous = None
        previous_ts = None
        for row in ordered.itertuples(index=False):
            if previous is not None and row.alert_start - previous_ts <= maximum_gap:
                union(previous, row.alert_number)
            previous, previous_ts = row.alert_number, row.alert_start

    return len({find(value) for value in range(len(alerts))})


def evaluate_alerts(
    alerts,
    events,
    intervals,
    *,
    exposure_value,
    exposure_unit,
    decision_horizon_seconds,
    grouping_window_seconds=0,
    entity_groups=None,
    min_reliable_faults=5,
):
    """Match alerts and return operational event-level evaluation tables."""

    if exposure_unit not in {"entity_day", "episode", "observed_hour"}:
        raise ValueError("Unsupported exposure unit")
    alerts = validate_alerts(alerts)
    matches, faults, pairs, groups = _match_alerts(
        alerts, events, intervals, decision_horizon_seconds
    )
    metric_rows = []

    def add_ratio(name, successes, total):
        low, high = _wilson_interval(successes, total)
        metric_rows.append({
            "metric": name,
            "value": successes / total if total else np.nan,
            "numerator": successes,
            "denominator": total,
            "ci_low": low,
            "ci_high": high,
            "unit": "ratio",
        })

    add_ratio("event_recall", int(faults["detected"].sum()), len(faults))
    add_ratio(
        "alert_precision",
        int(matches["match_status"].isin([
            "matched", "entity_confirmation"
        ]).sum()),
        len(matches),
    )
    impact_known = faults["impact_ts"].notna()
    add_ratio(
        "preimpact_event_recall",
        int((faults["detected"] & faults["preimpact"] & impact_known).sum()),
        int(impact_known.sum()),
    )
    add_ratio("entity_fault_coverage", int(pairs["detected"].sum()), len(pairs))
    shared = faults["affected_entities"].gt(1)
    add_ratio(
        "shared_fault_recall",
        int((faults["detected"] & shared).sum()),
        int(shared.sum()),
    )
    add_ratio(
        "cause_group_recall",
        int(groups["detected"].sum()) if len(groups) else 0,
        len(groups),
    )

    false_alerts = int(matches["match_status"].eq("false_alert").sum())
    false_alert_clusters = _false_alert_cluster_count(
        matches.loc[matches["match_status"].eq("false_alert")],
        entity_groups,
        grouping_window_seconds,
    )
    duplicates = int(matches["match_status"].eq("duplicate").sum())
    false_low, false_high = _poisson_rate_interval(
        false_alert_clusters, exposure_value
    )
    total_low, total_high = _poisson_rate_interval(len(matches), exposure_value)
    metric_rows.extend([
        {
            "metric": "false_alert_count",
            "value": false_alerts,
            "numerator": false_alerts,
            "denominator": len(matches),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "unit": "alerts",
        },
        {
            "metric": "false_alert_cluster_count",
            "value": false_alert_clusters,
            "numerator": false_alert_clusters,
            "denominator": len(matches),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "unit": "clusters",
        },
        {
            "metric": f"false_alerts_per_{exposure_unit}",
            "value": (
                false_alerts / exposure_value
                if exposure_value > 0 else np.nan
            ),
            "numerator": false_alerts,
            "denominator": exposure_value,
            "ci_low": false_low,
            "ci_high": false_high,
            "unit": f"alerts/{exposure_unit.replace('_', '-')}",
        },
        {
            "metric": f"total_alerts_per_{exposure_unit}",
            "value": len(matches) / exposure_value if exposure_value > 0 else np.nan,
            "numerator": len(matches),
            "denominator": exposure_value,
            "ci_low": total_low,
            "ci_high": total_high,
            "unit": f"alerts/{exposure_unit.replace('_', '-')}",
        },
        {
            "metric": "duplicate_alerts",
            "value": duplicates,
            "numerator": duplicates,
            "denominator": len(matches),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "unit": "alerts",
        },
    ])
    delays = faults.loc[faults["detected"], "detection_delay_seconds"].dropna()
    for name, value in {
        "median_detection_delay_seconds": delays.median() if len(delays) else np.nan,
        "p90_detection_delay_seconds": (
            delays.quantile(0.90) if len(delays) else np.nan
        ),
    }.items():
        metric_rows.append({
            "metric": name,
            "value": value,
            "numerator": len(delays),
            "denominator": len(faults),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "unit": "seconds",
        })

    by_type = faults.groupby("fault_type", as_index=False).agg(
        scoreable_faults=("fault_id", "nunique"),
        detected_faults=("detected", "sum"),
    )
    by_type["recall"] = by_type["detected_faults"] / by_type["scoreable_faults"]
    by_type["reporting_status"] = np.where(
        by_type["scoreable_faults"].ge(min_reliable_faults),
        "estimable", "descriptive_only",
    )
    return {
        "alert_matches": matches,
        "fault_results": faults,
        "entity_fault_results": pairs,
        "group_results": groups,
        "metrics": pd.DataFrame(metric_rows),
        "fault_type_results": by_type,
    }


def evaluate_vus_pr(
    candidate_manifest,
    events,
    intervals,
    *,
    exposure_value,
    exposure_unit,
    decision_horizon_seconds,
    grouping_window_seconds=0,
    entity_groups=None,
    n_tolerances=5,
):
    """Evaluate a development threshold sweep and return VUS-PR evidence."""

    if n_tolerances < 2:
        raise ValueError("VUS-PR requires at least two tolerance values")
    required = {
        "model_id", "threshold_quantile", "threshold", "alert_path"
    }
    missing = required - set(candidate_manifest.columns)
    if missing:
        raise ValueError(
            f"Candidate manifest is missing columns: {sorted(missing)}"
        )

    tolerances = np.linspace(
        0, float(decision_horizon_seconds), int(n_tolerances)
    )
    rows = []
    for candidate in candidate_manifest.itertuples(index=False):
        alerts = pd.read_parquet(candidate.alert_path)
        for tolerance in tolerances:
            result = evaluate_alerts(
                alerts,
                events,
                intervals,
                exposure_value=exposure_value,
                exposure_unit=exposure_unit,
                decision_horizon_seconds=float(tolerance),
                grouping_window_seconds=grouping_window_seconds,
                entity_groups=entity_groups,
            )
            metrics = result["metrics"].set_index("metric")["value"]
            rows.append({
                "model_id": candidate.model_id,
                "threshold_quantile": candidate.threshold_quantile,
                "threshold": candidate.threshold,
                "tolerance_seconds": float(tolerance),
                "event_recall": metrics["event_recall"],
                "alert_precision": metrics["alert_precision"],
            })

    points = pd.DataFrame(rows)
    areas = []
    for (model_id, tolerance), curve in points.groupby(
        ["model_id", "tolerance_seconds"], sort=True
    ):
        curve = (
            curve.groupby("event_recall", as_index=False)["alert_precision"]
            .max().sort_values("event_recall")
        )
        area = (
            float(np.trapz(curve["alert_precision"], curve["event_recall"]))
            if len(curve) >= 2 else 0.0
        )
        areas.append({
            "model_id": model_id,
            "tolerance_seconds": tolerance,
            "pr_area": area,
        })
    area_table = pd.DataFrame(areas)
    summary = (
        area_table.groupby("model_id", as_index=False)["pr_area"]
        .mean().rename(columns={"pr_area": "vus_pr"})
        .sort_values("vus_pr", ascending=False)
        .reset_index(drop=True)
    )
    return points, summary
