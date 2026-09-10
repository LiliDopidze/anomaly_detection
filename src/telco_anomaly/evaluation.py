"""Label-isolated incident evaluation for Telecom anomaly detection.

Notebook 03 owns policy choices and adversarial controls.  This flat module
owns the repetitive mechanics whose silent failure would invalidate results:
truth partitioning, score-to-alert conversion, one-to-one event matching and
metric calculation.  It contains no sector-specific logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2


EVALUATION_CORE_VERSION = "4.0.0"
PARTITIONS = ("calibration", "development", "holdout")
SCORE_COLUMNS = [
    "event_ts", "entity_id", "episode_id", "anomaly_score", "model_id",
]

CASE_COLUMNS = [
    "case_id", "case_start", "case_end", "peak_ts",
    "scope_type", "scope_id", "affected_entity_count",
    "scope_type_2", "scope_id_2", "identifiability_status",
    "footprint_size", "affected_fraction_estimate",
    "anomaly_evidence_score", "channels", "leading_features",
    "location_explanation",
]
ALERT_COLUMNS = [
    "alert_id", "model_id", "entity_id", "episode_id", "alert_start",
    "alert_end", "peak_ts", "peak_score", "n_scores", "leading_feature",
    "evidence_scope_type", "evidence_scope_id",
    "evidence_affected_fraction",
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


def _assign_tickets(tickets, audit, primary_split, time_partitions, entity_partitions):
    """Assign evaluator-only tickets without exposing another split's dates."""

    if tickets is None or tickets.empty:
        return pd.DataFrame() if tickets is None else tickets.assign(
            assigned_partition=pd.Series(dtype="string")
        )

    assigned = tickets.copy()
    assigned["fault_id"] = assigned["fault_id"].astype("string")
    fault_partition = audit.loc[
        audit["status"].eq("scoreable"), ["fault_id", "assigned_partition"]
    ].drop_duplicates("fault_id")
    assigned = assigned.merge(
        fault_partition, on="fault_id", how="left", validate="many_to_one"
    )

    unresolved = assigned["assigned_partition"].isna()
    if primary_split == "entity":
        membership = entity_partitions[["entity_id", "partition"]].copy()
        membership["entity_id"] = membership["entity_id"].astype(str)
        fallback = assigned.loc[unresolved, ["entity_id"]].astype(str).merge(
            membership, on="entity_id", how="left"
        )["partition"]
        assigned.loc[unresolved, "assigned_partition"] = fallback.to_numpy()
    else:
        reported = pd.to_datetime(
            assigned.loc[unresolved, "reported_ts"], utc=True, errors="coerce"
        )
        fallback = pd.Series(pd.NA, index=reported.index, dtype="string")
        for part in time_partitions.itertuples(index=False):
            inside = reported.ge(part.start_ts) & reported.lt(part.end_ts)
            fallback.loc[inside] = part.partition
        assigned.loc[unresolved, "assigned_partition"] = fallback

    return assigned


def partition_truth(
    events,
    intervals,
    conditions,
    registry,
    *,
    primary_split,
    time_partitions,
    entity_partitions,
    tickets=None,
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
    ticket_assignments = _assign_tickets(
        tickets, audit, primary_split, time_partitions, entity_partitions
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
        if tickets is not None:
            output[partition]["tickets"] = ticket_assignments.loc[
                ticket_assignments["assigned_partition"].eq(partition),
                tickets.columns,
            ].reset_index(drop=True)

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
    for column in (
        "evidence_scope_type", "evidence_scope_id",
        "evidence_affected_fraction",
    ):
        if column in scores.columns:
            columns.append(column)
    clean = scores[columns].copy()
    clean["event_ts"] = pd.to_datetime(clean["event_ts"], utc=True, errors="raise")
    clean["entity_id"] = clean["entity_id"].astype(str)
    clean["model_id"] = clean["model_id"].astype(str)
    clean["anomaly_score"] = pd.to_numeric(clean["anomaly_score"], errors="raise")
    if np.isinf(clean["anomaly_score"]).any():
        raise ValueError("Anomaly scores must not contain infinity")

    # A detector may need a short warm-up before it can emit a score. Those
    # rows are represented by NaN and are not alerts. Removing them also lets
    # the timestamp-gap rule below reset persistence across an unscored span.
    clean = clean.loc[clean["anomaly_score"].notna()].copy()
    clean["episode_id"] = clean["episode_id"].astype(str)
    if "leading_feature" not in clean:
        clean["leading_feature"] = pd.NA
    for column in (
        "evidence_scope_type", "evidence_scope_id",
        "evidence_affected_fraction",
    ):
        if column not in clean:
            clean[column] = pd.NA
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
                    current["leading_feature"] = alert["leading_feature"]
                    current["evidence_scope_type"] = alert["evidence_scope_type"]
                    current["evidence_scope_id"] = alert["evidence_scope_id"]
                    current["evidence_affected_fraction"] = alert[
                        "evidence_affected_fraction"
                    ]
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
        scope_types = group["evidence_scope_type"].to_numpy()
        scope_ids = group["evidence_scope_id"].to_numpy()
        affected_fractions = group["evidence_affected_fraction"].to_numpy()

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
                    "evidence_scope_type": scope_types[peak],
                    "evidence_scope_id": scope_ids[peak],
                    "evidence_affected_fraction": affected_fractions[peak],
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


def _horizon_seconds(fault_types, rule):
    """Return one non-negative decision horizon per fault.

    ``rule`` may be one number for every fault type or a dictionary keyed by
    ``fault_type``.  A dictionary must be complete: silently falling back to a
    generic window would make a class-specific evaluation look more rigorous
    than it is.
    """

    if isinstance(rule, dict):
        horizons = fault_types.astype(str).map(rule)
        missing = sorted(fault_types.loc[horizons.isna()].astype(str).unique())
        if missing:
            raise ValueError(
                "Decision horizons are missing for fault types: "
                f"{missing}"
            )
    else:
        horizons = pd.Series(rule, index=fault_types.index)
    horizons = pd.to_numeric(horizons, errors="coerce")
    if horizons.isna().any() or (~np.isfinite(horizons)).any() or horizons.lt(0).any():
        raise ValueError("Decision horizons must be finite non-negative seconds")
    return horizons.astype(float)


def _fault_windows(events, intervals, decision_horizon_seconds):
    windows = intervals[[
        "fault_id", "entity_id", "start_ts", "end_ts"
    ]].rename(columns={"end_ts": "interval_end"}).merge(
        events[[
            "fault_id", "fault_type", "domain_type", "domain_id",
            "observable_ts", "onset_ts",
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
    horizons = _horizon_seconds(
        windows["fault_type"], decision_horizon_seconds
    )
    windows["decision_horizon_seconds"] = horizons
    horizon_end = windows["match_start"] + pd.to_timedelta(
        horizons, unit="s"
    )
    windows["match_end"] = pd.concat(
        [windows["event_end"], windows["interval_end"], horizon_end], axis=1
    ).min(axis=1)
    invalid_end = windows["match_end"].le(windows["match_start"])
    zero_horizon = horizons.eq(0)
    invalid_end = invalid_end & ~(
        zero_horizon & windows["match_end"].eq(windows["match_start"])
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

    return windows


def _maximum_event_matches(candidates, ordered_alerts):
    """Return a deterministic maximum-cardinality alert-to-fault match."""

    if candidates.empty:
        return {}

    ranked = candidates.copy()
    ranked["alert_key"] = ranked["alert_id"].astype(str)
    ranked["fault_key"] = ranked["fault_id"].astype(str)
    ranked = (
        ranked.sort_values([
            "alert_key", "match_end", "match_start", "fault_key",
        ])
        .drop_duplicates(["alert_key", "fault_key"])
    )
    choices = {
        alert_key: group["fault_key"].tolist()
        for alert_key, group in ranked.groupby("alert_key", sort=False)
    }

    fault_to_alert = {}
    alert_to_fault = {}

    def assign(alert_key, visited_faults):
        for fault_key in choices.get(alert_key, []):
            if fault_key in visited_faults:
                continue
            visited_faults.add(fault_key)
            previous_alert = fault_to_alert.get(fault_key)
            if previous_alert is None or assign(previous_alert, visited_faults):
                fault_to_alert[fault_key] = alert_key
                alert_to_fault[alert_key] = fault_key
                return True
        return False

    for alert_id in ordered_alerts["alert_id"].astype(str):
        assign(alert_id, set())
    return alert_to_fault


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
        str(key): frame for key, frame in candidates.groupby("alert_id", sort=False)
    }
    empty = candidates.iloc[:0]

    ordered_alerts = alerts.sort_values(
        ["alert_start", "peak_score", "alert_id"],
        ascending=[True, False, True],
    )
    primary_matches = _maximum_event_matches(candidates, ordered_alerts)
    matched_faults = set(primary_matches.values())
    entity_by_alert = dict(zip(
        alerts["alert_id"].astype(str), alerts["entity_id"].astype(str)
    ))
    matched_pairs = {
        (fault_id, entity_by_alert[alert_id])
        for alert_id, fault_id in primary_matches.items()
    }

    rows = []
    for alert in ordered_alerts.itertuples(index=False):
        alert_key = str(alert.alert_id)
        choices = by_alert.get(alert_key, empty)
        primary_fault = primary_matches.get(alert_key)
        if primary_fault is not None:
            selected = choices.loc[
                choices["fault_id"].astype(str).eq(primary_fault)
            ].iloc[0]
            status = "matched"
        elif not choices.empty:
            new_pair = choices.loc[
                [
                    str(fault_id) in matched_faults
                    and (str(fault_id), str(alert.entity_id)) not in matched_pairs
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
        false_alerts, exposure_value
    )
    cluster_low, cluster_high = _poisson_rate_interval(
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
            "metric": f"false_alert_clusters_per_{exposure_unit}",
            "value": (
                false_alert_clusters / exposure_value
                if exposure_value > 0 else np.nan
            ),
            "numerator": false_alert_clusters,
            "denominator": exposure_value,
            "ci_low": cluster_low,
            "ci_high": cluster_high,
            "unit": f"clusters/{exposure_unit.replace('_', '-')}",
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


def _topology_context(topology):
    """Return memberships, group sizes and observable equivalence classes."""

    if topology is None or topology.empty:
        return {}, {}, {}, {}
    required = {
        "entity_id", "group_type", "group_id", "hierarchy_level", "group_family",
    }
    missing = required - set(topology.columns)
    if missing:
        raise ValueError(f"Topology is missing columns: {sorted(missing)}")
    physical = topology.loc[
        topology["group_family"].eq("physical_topology")
    ].copy()
    physical[["entity_id", "group_type", "group_id"]] = physical[
        ["entity_id", "group_type", "group_id"]
    ].astype(str)
    memberships = {
        entity: set(zip(rows["group_type"], rows["group_id"]))
        for entity, rows in physical.groupby("entity_id")
    }
    descendants = {
        key: frozenset(rows["entity_id"].astype(str))
        for key, rows in physical.groupby(["group_type", "group_id"])
    }
    levels = {
        key: int(pd.to_numeric(rows["hierarchy_level"], errors="raise").iloc[0])
        for key, rows in physical.groupby(["group_type", "group_id"])
    }
    equivalent = {}
    by_footprint = {}
    for key, members in descendants.items():
        by_footprint.setdefault(members, []).append(key)
    for keys in by_footprint.values():
        for key in keys:
            equivalent[key] = sorted(keys)
    return memberships, descendants, levels, equivalent


def form_cases(
    alerts,
    entity_groups=None,
    *,
    gap_seconds,
    thresholds,
    shared_scope_models=("group_common_mode",),
):
    """Consolidate channel alerts without uncontrolled topology chaining.

    A new alert may join an existing case only when it is close in time and
    all entities in the resulting case still share a common topology group.
    This prevents A-B-C chains whose endpoints have no resolvable scope.
    """

    alerts = validate_alerts(alerts)
    if alerts.empty:
        return (
            pd.DataFrame(columns=CASE_COLUMNS),
            pd.DataFrame(columns=["case_id", "alert_id", "entity_id", "model_id"]),
        )
    thresholds = {str(key): float(value) for key, value in thresholds.items()}
    missing = set(alerts["model_id"].astype(str)) - set(thresholds)
    if missing:
        raise ValueError(f"Missing channel thresholds: {sorted(missing)}")

    memberships, descendants, levels, equivalent = _topology_context(entity_groups)
    maximum_gap = pd.Timedelta(seconds=float(gap_seconds))
    shared_scope_models = set(map(str, shared_scope_models))

    open_cases = []
    active_cases = []
    ordered = alerts.sort_values(["alert_start", "alert_end", "alert_id"])
    for alert in ordered.itertuples(index=False):
        entity = str(alert.entity_id)
        alert_groups = memberships.get(entity, set())
        if (
            str(alert.model_id) in shared_scope_models
            and pd.notna(alert.evidence_scope_type)
            and pd.notna(alert.evidence_scope_id)
        ):
            declared_scope = (
                str(alert.evidence_scope_type), str(alert.evidence_scope_id)
            )
            if declared_scope not in alert_groups:
                raise ValueError(
                    "Common-mode alert scope does not contain its entity"
                )
            alert_groups = {declared_scope}
        active_cases = [
            number for number in active_cases
            if alert.alert_start <= open_cases[number]["end"] + maximum_gap
        ]
        compatible = []
        for number in active_cases:
            case = open_cases[number]
            shared = case["common_groups"] & alert_groups
            same_entity = entity in case["entities"]
            common_mode_evidence = (
                str(alert.model_id) in shared_scope_models
                or case["has_shared_scope_evidence"]
            )
            if same_entity or (shared and common_mode_evidence):
                compatible.append((case["end"], number, shared))
        if compatible:
            selected = [max(compatible)]
            if str(alert.model_id) in shared_scope_models:
                candidates_by_group = [
                    [item for item in compatible if group in item[2]]
                    for group in alert_groups
                ]
                candidates_by_group = [items for items in candidates_by_group if items]
                if candidates_by_group:
                    selected = max(
                        candidates_by_group,
                        key=lambda items: (len(items), max(item[0] for item in items)),
                    )
            combined_shared = set(alert_groups)
            for _, selected_number, _ in selected:
                combined_shared &= open_cases[selected_number]["common_groups"]
            _, number, _ = max(selected)
            case = open_cases[number]
            for _, other_number, _ in selected:
                if other_number == number:
                    continue
                other = open_cases[other_number]
                case["alerts"].extend(other["alerts"])
                case["entities"].update(other["entities"])
                case["start"] = min(case["start"], other["start"])
                case["end"] = max(case["end"], other["end"])
                case["common_groups"] &= other["common_groups"]
                case["has_shared_scope_evidence"] |= other[
                    "has_shared_scope_evidence"
                ]
                other["merged_into"] = number
                if other_number in active_cases:
                    active_cases.remove(other_number)
            case["alerts"].append(alert)
            case["entities"].add(entity)
            case["end"] = max(case["end"], alert.alert_end)
            case["has_shared_scope_evidence"] |= (
                str(alert.model_id) in shared_scope_models
            )
            if len(case["entities"]) > 1:
                case["common_groups"] = combined_shared
        else:
            open_cases.append({
                "alerts": [alert],
                "entities": {entity},
                "start": alert.alert_start,
                "end": alert.alert_end,
                "common_groups": set(alert_groups),
                "has_shared_scope_evidence": (
                    str(alert.model_id) in shared_scope_models
                ),
                "merged_into": None,
            })
            active_cases.append(len(open_cases) - 1)

    def exceedance(row):
        threshold = thresholds[str(row.model_id)]
        return 1.0 + (float(row.peak_score) - threshold) / max(abs(threshold), 1e-12)

    case_rows, member_rows = [], []
    output_number = 0
    for case in open_cases:
        if case["merged_into"] is not None:
            continue
        output_number += 1
        number = output_number
        case_id = f"C-{number:06d}"
        members = case["alerts"]
        peak = max(
            members,
            key=exceedance,
        )
        evidence = max(exceedance(row) for row in members)
        second_type = second_id = pd.NA
        scope_evidence = [
            row for row in members
            if str(row.model_id) in shared_scope_models
            and pd.notna(row.evidence_scope_type)
            and pd.notna(row.evidence_scope_id)
        ]
        selected_scope_evidence = (
            max(scope_evidence, key=exceedance) if scope_evidence else None
        )
        if selected_scope_evidence is not None:
            scope_type = str(selected_scope_evidence.evidence_scope_type)
            scope_id = str(selected_scope_evidence.evidence_scope_id)
            footprint = set(descendants.get((scope_type, scope_id), ()))
            if not footprint:
                raise ValueError("Common-mode evidence has an unknown topology scope")
            aliases = [
                candidate
                for candidate in equivalent.get((scope_type, scope_id), [])
                if candidate != (scope_type, scope_id)
            ]
            if aliases:
                second_type, second_id = aliases[0]
                status = "topology_equivalent"
                explanation = (
                    "Common-mode evidence identifies an observable footprint "
                    "shared by two topology scopes."
                )
            else:
                status = "common_mode_scope"
                explanation = "Common-mode residual evidence identifies this physical scope."
        elif len(case["entities"]) == 1:
            scope_type, scope_id = "entity", next(iter(case["entities"]))
            footprint = {scope_id}
            status = "entity_exact"
            explanation = "One entity carries the incident evidence."
        elif case["common_groups"]:
            candidates = sorted(
                case["common_groups"],
                key=lambda group: (
                    len(descendants.get(group, ())),
                    -levels.get(group, -1), group[0], group[1],
                ),
            )
            scope_type, scope_id = candidates[0]
            footprint = set(descendants[(scope_type, scope_id)])
            aliases = [
                candidate for candidate in equivalent.get((scope_type, scope_id), [])
                if candidate != (scope_type, scope_id)
            ]
            if aliases:
                second_type, second_id = aliases[0]
                status = "topology_equivalent"
                explanation = "Two topology scopes have the same observable descendants."
            else:
                alternatives = [candidate for candidate in candidates[1:]
                                if candidate not in aliases]
                if alternatives:
                    second_type, second_id = alternatives[0]
                status = "hierarchical_candidate"
                explanation = "Most specific common physical scope of the affected entities."
        else:  # guarded by the merge rule; kept as an explicit invariant
            raise AssertionError("A multi-entity case has no common topology scope")
        affected_fraction = np.nan
        if selected_scope_evidence is not None:
            affected_fraction = pd.to_numeric(
                selected_scope_evidence.evidence_affected_fraction,
                errors="coerce",
            )
        if pd.isna(affected_fraction):
            affected_fraction = (
                len(case["entities"]) / len(footprint) if footprint else np.nan
            )
        affected_count = (
            max(
                len(case["entities"]),
                round(float(affected_fraction) * len(footprint)),
            )
            if footprint and pd.notna(affected_fraction)
            else len(case["entities"])
        )
        case_rows.append({
            "case_id": case_id,
            "case_start": case["start"],
            "case_end": case["end"],
            "peak_ts": peak.peak_ts,
            "scope_type": scope_type,
            "scope_id": scope_id,
            "affected_entity_count": affected_count,
            "scope_type_2": second_type,
            "scope_id_2": second_id,
            "identifiability_status": status,
            "footprint_size": len(footprint),
            "affected_fraction_estimate": affected_fraction,
            "anomaly_evidence_score": float(evidence),
            "channels": ", ".join(sorted({str(row.model_id) for row in members})),
            "leading_features": ", ".join(sorted({
                str(row.leading_feature) for row in members
                if pd.notna(row.leading_feature)
            })),
            "location_explanation": explanation,
        })
        member_rows.extend({
            "case_id": case_id,
            "alert_id": str(row.alert_id),
            "entity_id": str(row.entity_id),
            "model_id": str(row.model_id),
        } for row in members)
    return (
        pd.DataFrame(case_rows, columns=CASE_COLUMNS),
        pd.DataFrame(member_rows),
    )


def evaluate_cases(
    cases,
    case_members,
    events,
    intervals,
    *,
    exposure_value,
    exposure_unit,
    decision_horizon_seconds,
    topology_memberships=None,
    min_reliable_faults=5,
):
    """One-to-one case-to-fault evaluation at operational workload level."""

    missing = set(CASE_COLUMNS) - set(cases.columns)
    if missing:
        raise ValueError(f"Missing case columns: {sorted(missing)}")
    if exposure_unit not in {"entity_day", "episode", "observed_hour"}:
        raise ValueError("Unsupported exposure unit")
    cases = cases[CASE_COLUMNS].copy()
    for column in ("case_start", "case_end", "peak_ts"):
        cases[column] = pd.to_datetime(cases[column], utc=True, errors="raise")

    _, descendants, levels, equivalent = _topology_context(topology_memberships)
    windows = _fault_windows(events, intervals, decision_horizon_seconds)
    case_entities = case_members[["case_id", "entity_id"]].drop_duplicates().copy()
    footprint_rows = []
    for case in cases.itertuples(index=False):
        if str(case.scope_type) == "entity":
            continue
        for entity_id in descendants.get(
            (str(case.scope_type), str(case.scope_id)), ()
        ):
            footprint_rows.append((str(case.case_id), str(entity_id)))
    if footprint_rows:
        case_entities = pd.concat([
            case_entities,
            pd.DataFrame(footprint_rows, columns=["case_id", "entity_id"]),
        ], ignore_index=True).drop_duplicates()
    candidates = case_entities.merge(
        windows, on="entity_id", how="inner"
    ).merge(cases[["case_id", "case_start"]], on="case_id", how="inner")
    candidates = candidates.loc[
        candidates["case_start"].ge(candidates["match_start"])
        & (candidates["match_end"].isna() | candidates["case_start"].lt(candidates["match_end"]))
    ].drop_duplicates(["case_id", "fault_id"])
    matching_candidates = candidates.rename(columns={"case_id": "alert_id"})
    ordered = cases.rename(columns={
        "case_id": "alert_id", "case_start": "alert_start",
        "anomaly_evidence_score": "peak_score",
    })
    matched = _maximum_event_matches(matching_candidates, ordered)

    match_rows = []
    for case in cases.itertuples(index=False):
        fault_id = matched.get(str(case.case_id))
        choices = candidates.loc[candidates["case_id"].astype(str).eq(str(case.case_id))]
        selected = choices.loc[choices["fault_id"].astype(str).eq(fault_id)] if fault_id else choices.iloc[:0]
        status = "matched" if fault_id else ("duplicate" if not choices.empty else "false_case")
        record = {"case_id": case.case_id, "match_status": status, "fault_id": pd.NA,
                  "fault_type": pd.NA, "detection_delay_seconds": np.nan, "preimpact": False}
        if not selected.empty:
            row = selected.iloc[0]
            reference = row["observable_ts"] if pd.notna(row["observable_ts"]) else row["onset_ts"]
            record.update({
                "fault_id": row["fault_id"],
                "fault_type": row["fault_type"],
                "detection_delay_seconds": (case.case_start - reference).total_seconds(),
                "preimpact": pd.notna(row["impact_ts"]) and case.case_start < row["impact_ts"],
            })
        match_rows.append(record)
    matches = pd.DataFrame(match_rows, columns=[
        "case_id", "match_status", "fault_id", "fault_type",
        "detection_delay_seconds", "preimpact",
    ])

    detections = matches.loc[matches["match_status"].eq("matched")]
    fault_results = events.copy().merge(
        detections[["fault_id", "case_id", "detection_delay_seconds", "preimpact"]],
        on="fault_id", how="left", validate="one_to_one",
    )
    fault_results["detected"] = fault_results["case_id"].notna()
    fault_results["preimpact"] = fault_results["preimpact"].eq(True)

    interval_entities = (
        intervals.assign(
            fault_id=intervals["fault_id"].astype(str),
            entity_id=intervals["entity_id"].astype(str),
        )
        .groupby("fault_id")["entity_id"].agg(lambda values: frozenset(values))
        .to_dict()
    )
    case_index = cases.set_index("case_id") if not cases.empty else pd.DataFrame()
    location_rows = []
    entity_level = max(levels.values(), default=-1) + 1
    for fault in fault_results.itertuples(index=False):
        true_type = str(fault.domain_type)
        true_id = str(fault.domain_id)
        true_key = (true_type, true_id)
        true_entities = set(interval_entities.get(str(fault.fault_id), ()))
        record = {
            "fault_id": str(fault.fault_id),
            "case_id": fault.case_id,
            "detected": bool(fault.detected),
            "true_domain_type": true_type,
            "true_domain_id": true_id,
            "predicted_domain_type": pd.NA,
            "predicted_domain_id": pd.NA,
            "exact_scope": False,
            "top2_scope": False,
            "equivalent_scope": False,
            "truth_identifiable": True,
            "hierarchy_distance": np.nan,
            "hierarchy_comparable": False,
            "different_branch": False,
            "footprint_precision": np.nan,
            "footprint_recall": np.nan,
            "footprint_jaccard": np.nan,
        }
        if true_type != "entity":
            aliases = equivalent.get(true_key, [true_key])
            record["truth_identifiable"] = len(aliases) == 1
        if not fault.detected:
            location_rows.append(record)
            continue

        case = case_index.loc[fault.case_id]
        predicted_key = (str(case.scope_type), str(case.scope_id))
        second_key = (
            (str(case.scope_type_2), str(case.scope_id_2))
            if pd.notna(case.scope_type_2) and pd.notna(case.scope_id_2)
            else None
        )
        exact = predicted_key == true_key
        same_footprint = (
            predicted_key in descendants
            and true_key in descendants
            and descendants[predicted_key] == descendants[true_key]
        )
        record.update({
            "predicted_domain_type": predicted_key[0],
            "predicted_domain_id": predicted_key[1],
            "exact_scope": exact,
            "top2_scope": exact or second_key == true_key,
            "equivalent_scope": exact or same_footprint,
        })

        if exact or same_footprint:
            record["hierarchy_distance"] = 0.0
            record["hierarchy_comparable"] = True
        else:
            predicted_level = (
                entity_level if predicted_key[0] == "entity"
                else levels.get(predicted_key)
            )
            true_level = entity_level if true_type == "entity" else levels.get(true_key)
            predicted_footprint = (
                {predicted_key[1]} if predicted_key[0] == "entity"
                else set(descendants.get(predicted_key, ()))
            )
            true_footprint = (
                {true_key[1]} if true_type == "entity"
                else set(descendants.get(true_key, ()))
            )
            nested = (
                bool(predicted_footprint)
                and bool(true_footprint)
                and (
                    predicted_footprint <= true_footprint
                    or true_footprint <= predicted_footprint
                )
            )
            if nested and predicted_level is not None and true_level is not None:
                record["hierarchy_distance"] = abs(predicted_level - true_level)
                record["hierarchy_comparable"] = True
            else:
                record["different_branch"] = True

        predicted_entities = (
            {predicted_key[1]}
            if predicted_key[0] == "entity"
            else set(descendants.get(predicted_key, ()))
        )
        overlap = len(predicted_entities & true_entities)
        union = len(predicted_entities | true_entities)
        record["footprint_precision"] = (
            overlap / len(predicted_entities) if predicted_entities else np.nan
        )
        record["footprint_recall"] = (
            overlap / len(true_entities) if true_entities else np.nan
        )
        record["footprint_jaccard"] = overlap / union if union else np.nan
        location_rows.append(record)

    localisation = pd.DataFrame(location_rows)
    fault_results = fault_results.merge(
        localisation.drop(columns=["case_id", "detected"]),
        on="fault_id", how="left", validate="one_to_one",
    )
    detected = int(fault_results["detected"].sum())
    credited_cases = len(detections)
    nuisance_cases = len(cases) - credited_cases

    rows = []
    def ratio(name, numerator, denominator):
        low, high = _wilson_interval(numerator, denominator)
        rows.append({"metric": name, "value": numerator / denominator if denominator else np.nan,
                     "numerator": numerator, "denominator": denominator,
                     "ci_low": low, "ci_high": high, "unit": "ratio"})
    ratio("event_recall", detected, len(fault_results))
    ratio("case_precision", credited_cases, len(cases))
    impact_known = fault_results["impact_ts"].notna()
    ratio("preimpact_event_recall", int((fault_results["detected"] & fault_results["preimpact"] & impact_known).sum()), int(impact_known.sum()))
    detected_locations = localisation.loc[localisation["detected"]]
    identifiable = detected_locations["truth_identifiable"]
    ratio(
        "exact_localisation_accuracy",
        int((detected_locations["exact_scope"] & identifiable).sum()),
        int(identifiable.sum()),
    )
    ratio(
        "top2_localisation_accuracy",
        int(detected_locations["top2_scope"].sum()),
        len(detected_locations),
    )
    ratio(
        "equivalence_aware_localisation_accuracy",
        int(detected_locations["equivalent_scope"].sum()),
        len(detected_locations),
    )
    ratio(
        "joint_detection_and_localisation_recall",
        int((localisation["detected"] & localisation["equivalent_scope"]).sum()),
        len(localisation),
    )
    ratio(
        "different_branch_rate_among_detected",
        int(detected_locations["different_branch"].sum()),
        len(detected_locations),
    )
    for name, count, unit_name in (
        (f"false_cases_per_{exposure_unit}", nuisance_cases, "cases"),
        (f"total_cases_per_{exposure_unit}", len(cases), "cases"),
    ):
        low, high = _poisson_rate_interval(count, exposure_value)
        rows.append({"metric": name, "value": count / exposure_value if exposure_value else np.nan,
                     "numerator": count, "denominator": exposure_value,
                     "ci_low": low, "ci_high": high,
                     "unit": f"{unit_name}/{exposure_unit.replace('_', '-')}"})
    delays = fault_results.loc[fault_results["detected"], "detection_delay_seconds"].dropna()
    rows.append({"metric": "median_detection_delay_seconds",
                 "value": delays.median() if len(delays) else np.nan,
                 "numerator": len(delays), "denominator": len(fault_results),
                 "ci_low": np.nan, "ci_high": np.nan, "unit": "seconds"})
    raw_alerts = int(case_members["alert_id"].nunique()) if not case_members.empty else 0
    duplicate_cases = int(matches["match_status"].eq("duplicate").sum())
    rows.extend([
        {"metric": "raw_alert_count", "value": raw_alerts,
         "numerator": raw_alerts, "denominator": raw_alerts,
         "ci_low": np.nan, "ci_high": np.nan, "unit": "alerts"},
        {"metric": "consolidated_case_count", "value": len(cases),
         "numerator": len(cases), "denominator": raw_alerts,
         "ci_low": np.nan, "ci_high": np.nan, "unit": "cases"},
        {"metric": "alert_volume_reduction", "value": (
             1 - len(cases) / raw_alerts if raw_alerts else np.nan
         ), "numerator": raw_alerts - len(cases), "denominator": raw_alerts,
         "ci_low": np.nan, "ci_high": np.nan, "unit": "ratio"},
        {"metric": "mean_alerts_per_case", "value": (
             raw_alerts / len(cases) if len(cases) else np.nan
         ), "numerator": raw_alerts, "denominator": len(cases),
         "ci_low": np.nan, "ci_high": np.nan, "unit": "alerts/case"},
        {"metric": "duplicate_case_count", "value": duplicate_cases,
         "numerator": duplicate_cases, "denominator": len(cases),
         "ci_low": np.nan, "ci_high": np.nan, "unit": "cases"},
    ])
    for name in (
        "footprint_precision", "footprint_recall", "footprint_jaccard",
        "hierarchy_distance",
    ):
        values = detected_locations[name].dropna()
        summary = values.median() if name == "hierarchy_distance" else values.mean()
        rows.append({
            "metric": ("median_hierarchy_distance" if name == "hierarchy_distance"
                       else f"mean_{name}"),
            "value": summary if len(values) else np.nan,
            "numerator": len(values),
            "denominator": len(detected_locations),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "unit": "levels" if name == "hierarchy_distance" else "ratio",
        })

    by_type = fault_results.groupby("fault_type", as_index=False).agg(
        scoreable_faults=("fault_id", "nunique"),
        detected_faults=("detected", "sum"),
        detected_and_localised=("equivalent_scope", "sum"),
    )
    by_type["recall"] = by_type["detected_faults"] / by_type["scoreable_faults"]
    by_type["joint_detection_localisation_recall"] = (
        by_type["detected_and_localised"] / by_type["scoreable_faults"]
    )
    by_type["reporting_status"] = np.where(
        by_type["scoreable_faults"].ge(min_reliable_faults), "estimable", "descriptive_only"
    )
    recall_intervals = [
        _wilson_interval(detected, total)
        for detected, total in zip(by_type["detected_faults"], by_type["scoreable_faults"])
    ]
    by_type[["recall_ci_low", "recall_ci_high"]] = recall_intervals

    by_domain = fault_results.groupby("domain_type", as_index=False).agg(
        scoreable_faults=("fault_id", "nunique"),
        detected_faults=("detected", "sum"),
        detected_and_localised=("equivalent_scope", "sum"),
    )
    by_domain["recall"] = by_domain["detected_faults"] / by_domain["scoreable_faults"]
    by_domain["joint_detection_localisation_recall"] = (
        by_domain["detected_and_localised"] / by_domain["scoreable_faults"]
    )
    by_domain["reporting_status"] = np.where(
        by_domain["scoreable_faults"].ge(min_reliable_faults),
        "estimable", "descriptive_only",
    )
    return {
        "case_matches": matches,
        "fault_results": fault_results,
        "localisation_results": localisation,
        "metrics": pd.DataFrame(rows),
        "fault_type_results": by_type,
        "domain_type_results": by_domain,
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
