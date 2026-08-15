"""Tested mechanics for the evaluation-harness notebook.

Notebook 03 owns policy choices and adversarial controls.  This flat module
owns the repetitive mechanics whose silent failure would invalidate results:
truth partitioning, score-to-alert conversion, one-to-one event matching and
metric calculation.  It contains no sector-specific logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


PARTITIONS = ("calibration", "development", "holdout")
SCORE_COLUMNS = ["event_ts", "entity_id", "anomaly_score", "model_id"]
ALERT_COLUMNS = [
    "alert_id", "model_id", "entity_id", "alert_start", "alert_end",
    "peak_ts", "peak_score", "n_scores",
]
AUDIT_COLUMNS = [
    "fault_id", "fault_type", "assigned_partition", "status",
    "scoreable", "cross_partition", "declared_entity_count",
    "scoreable_entity_count",
]


def _interval_scoreability(intervals, registry):
    windows = registry[["entity_id", "observed_from", "observed_to"]].copy()
    joined = intervals.copy()
    joined["entity_id"] = joined["entity_id"].astype(str)
    joined = joined.merge(windows, on="entity_id", how="left")
    start = pd.to_datetime(joined["start_ts"], utc=True, errors="coerce")
    end = pd.to_datetime(joined["end_ts"], utc=True, errors="coerce")
    joined["scoreable_interval"] = (
        joined["observed_from"].notna()
        & start.notna()
        & start.le(joined["observed_to"])
        & (end.isna() | end.ge(joined["observed_from"]))
    )
    return joined


def _temporal_assignment(events, intervals, partitions, registry):
    events = events.copy()
    interval_audit = _interval_scoreability(intervals, registry)
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
            start = pd.to_datetime(
                fault_intervals["start_ts"], utc=True, errors="coerce"
            )
            end = pd.to_datetime(
                fault_intervals["end_ts"], utc=True, errors="coerce"
            )
            fault_intervals["scoreable_for_partition"] = (
                fault_intervals["scoreable_interval"]
                & start.lt(part["end_ts"])
                & (end.isna() | end.gt(part["start_ts"]))
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
    interval_audit = _interval_scoreability(intervals, registry)
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
    clean = scores[SCORE_COLUMNS].copy()
    clean["event_ts"] = pd.to_datetime(clean["event_ts"], utc=True, errors="raise")
    clean["entity_id"] = clean["entity_id"].astype(str)
    clean["model_id"] = clean["model_id"].astype(str)
    clean["anomaly_score"] = pd.to_numeric(clean["anomaly_score"], errors="raise")
    if not np.isfinite(clean["anomaly_score"]).all():
        raise ValueError("Anomaly scores must be finite")
    if clean.duplicated(["event_ts", "entity_id", "model_id"]).any():
        raise ValueError("A model may emit only one score per entity and timestamp")
    return clean.sort_values(["model_id", "entity_id", "event_ts"]).reset_index(
        drop=True
    )


def scores_to_alerts(scores, threshold, *, min_consecutive=2, gap_factor=1.5):
    """Collapse consecutive above-threshold scores into alerts."""

    scores = validate_scores(scores)
    alerts = []
    alert_number = 0
    for (model_id, entity_id), group in scores.groupby(
        ["model_id", "entity_id"], sort=True
    ):
        group = group.sort_values("event_ts").copy()
        differences = group["event_ts"].diff().dropna()
        positive = differences.loc[differences.gt(pd.Timedelta(0))]
        cadence = positive.median() if not positive.empty else pd.Timedelta(0)
        long_gap = (
            group["event_ts"].diff().gt(cadence * gap_factor)
            if cadence > pd.Timedelta(0)
            else pd.Series(False, index=group.index)
        )
        above = group["anomaly_score"].ge(threshold)
        group["run_id"] = (
            above.ne(above.shift(fill_value=False)) | long_gap
        ).cumsum()

        for _, run in group.loc[above].groupby("run_id", sort=True):
            if len(run) < min_consecutive:
                continue
            peak_index = run["anomaly_score"].idxmax()
            alert_number += 1
            alerts.append({
                "alert_id": f"A-{alert_number:06d}",
                "model_id": model_id,
                "entity_id": entity_id,
                "alert_start": run["event_ts"].iloc[0],
                "alert_end": run["event_ts"].iloc[-1] + cadence,
                "peak_ts": group.loc[peak_index, "event_ts"],
                "peak_score": group.loc[peak_index, "anomaly_score"],
                "n_scores": len(run),
            })
    return pd.DataFrame(alerts, columns=ALERT_COLUMNS)


def _fault_windows(events, intervals):
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
    windows["match_end"] = pd.concat(
        [windows["event_end"], windows["interval_end"]], axis=1
    ).min(axis=1)
    invalid = windows["match_start"].isna() | (
        windows["match_end"].notna()
        & windows["match_end"].le(windows["match_start"])
    )
    if invalid.any():
        raise ValueError("Evaluation truth contains an invalid matching window")
    return windows


def _match_alerts(alerts, events, intervals):
    alerts = alerts.copy()
    windows = _fault_windows(events, intervals)
    candidates = alerts.merge(windows, on="entity_id", how="inner")
    candidates = candidates.loc[
        candidates["alert_start"].ge(candidates["match_start"])
        & (
            candidates["match_end"].isna()
            | candidates["alert_start"].lt(candidates["match_end"])
        )
    ].copy()

    matched_faults = set()
    rows = []
    ordered_alerts = alerts.sort_values(
        ["alert_start", "peak_score", "alert_id"],
        ascending=[True, False, True],
    )
    for alert in ordered_alerts.itertuples(index=False):
        choices = candidates.loc[candidates["alert_id"].eq(alert.alert_id)].sort_values(
            ["match_end", "match_start", "fault_id"]
        )
        available = choices.loc[
            ~choices["fault_id"].astype(str).isin(matched_faults)
        ]
        if not available.empty:
            selected, status = available.iloc[0], "matched"
            matched_faults.add(str(selected["fault_id"]))
        elif not choices.empty:
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

    # Entity coverage follows the final one-to-one matches.  Eligible but
    # unmatched overlapping faults must not receive detection credit.
    pair_candidates = (
        matches.loc[matches["match_status"].eq("matched"), [
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


def evaluate_alerts(
    alerts,
    events,
    intervals,
    *,
    exposure_entity_days,
    min_reliable_faults=5,
):
    """Match alerts and return operational event-level evaluation tables."""

    matches, faults, pairs, groups = _match_alerts(alerts, events, intervals)
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
        int(matches["match_status"].eq("matched").sum()),
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
    duplicates = int(matches["match_status"].eq("duplicate").sum())
    metric_rows.extend([
        {
            "metric": "false_alerts_per_entity_day",
            "value": (
                false_alerts / exposure_entity_days
                if exposure_entity_days > 0 else np.nan
            ),
            "numerator": false_alerts,
            "denominator": exposure_entity_days,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "unit": "alerts/entity-day",
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
