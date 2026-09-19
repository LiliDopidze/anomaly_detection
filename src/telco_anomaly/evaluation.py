"""Event matching and uncertainty intervals used by the synthetic pipeline."""

from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import chi2, norm


def _maximum_event_matches(candidates, ordered_alerts):
    """Return a deterministic maximum-cardinality alert-to-fault match."""

    if candidates.empty:
        return {}

    ranked = candidates.copy()
    ranked["alert_key"] = ranked["alert_id"].astype(str)
    ranked["fault_key"] = ranked["fault_id"].astype(str)
    ranked = ranked.sort_values(
        [
            "alert_key",
            "match_end",
            "match_start",
            "fault_key",
        ]
    ).drop_duplicates(["alert_key", "fault_key"])
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


def _wilson_interval(successes, total, z=1.96):
    if total == 0:
        return np.nan, np.nan
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * np.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return centre - margin, centre + margin


def wilson_interval(successes, total, confidence_level=0.95):
    """Return a two-sided Wilson interval for a binomial proportion."""

    confidence_level = float(confidence_level)
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if int(successes) != successes or int(total) != total:
        raise ValueError("successes and total must be integers")
    successes, total = int(successes), int(total)
    if successes < 0 or total < 0 or successes > total:
        raise ValueError("Require 0 <= successes <= total")
    z = norm.ppf(1 - (1 - confidence_level) / 2)
    return _wilson_interval(successes, total, z=z)


def poisson_rate_interval(count, exposure, confidence_level=0.95):
    """Return an exact two-sided Garwood interval for a Poisson rate."""

    confidence_level = float(confidence_level)
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between zero and one")

    count_value = float(count)
    if not np.isfinite(count_value) or count_value < 0 or not count_value.is_integer():
        raise ValueError("count must be a non-negative integer")
    count = int(count_value)

    exposure = float(exposure)
    if not np.isfinite(exposure) or exposure <= 0:
        return np.nan, np.nan

    alpha = 1 - confidence_level
    lower = 0.0 if count == 0 else chi2.ppf(alpha / 2, 2 * count) / 2
    upper = chi2.ppf(1 - alpha / 2, 2 * (count + 1)) / 2
    return lower / exposure, upper / exposure


def evaluate(dataset, incidents, start, end, prompt_hours=24):
    """Maximum-cardinality one-to-one matching; boundary events reported separately."""
    dataset = Path(dataset)
    faults = pd.read_csv(dataset / "gt_fault_registry.csv")
    intervals = pd.read_csv(dataset / "fault_entity_intervals.csv")
    for c in ("onset_ts", "repair_ts", "first_observable_ts", "impact_ts"):
        faults[c] = pd.to_datetime(faults[c], utc=True)
    boundary = (faults.onset_ts < end) & (faults.repair_ts > start)
    eligible = (faults.onset_ts >= start) & (faults.repair_ts <= end)
    targets = faults.loc[eligible].sort_values("onset_ts")
    members = intervals.groupby("fault_id").entity_id.agg(set).to_dict()

    def belongs(entity_members):
        if "entity_ids" in incidents:
            return incidents.entity_ids.map(
                lambda values: bool(set(values) & entity_members)
            )
        return incidents.entity_id.isin(entity_members)

    candidates = []
    for fault in targets.itertuples(index=False):
        mask = (
            belongs(members.get(fault.gt_fault_id, set()))
            & (incidents.start_ts >= fault.onset_ts)
            & (incidents.start_ts < fault.repair_ts)
        )
        for incident_id in incidents.loc[mask, "incident_id"]:
            candidates.append(
                {
                    "alert_id": incident_id,
                    "fault_id": fault.gt_fault_id,
                    "match_start": fault.onset_ts,
                    "match_end": fault.repair_ts,
                }
            )
    ordered = incidents.sort_values("start_ts").rename(
        columns={"incident_id": "alert_id"}
    )
    matches = _maximum_event_matches(pd.DataFrame(candidates), ordered)
    fault_to_alert = {fault: alert for alert, fault in matches.items()}
    used, outcomes = set(matches), []
    eligible_alerts = set(pd.DataFrame(candidates).get("alert_id", []))
    by_id = incidents.set_index("incident_id")
    for fault in targets.itertuples(index=False):
        alert_id = fault_to_alert.get(fault.gt_fault_id)
        hit = by_id.loc[alert_id] if alert_id is not None else None
        delay = None
        if hit is not None and pd.notna(fault.first_observable_ts):
            delay = (hit.start_ts - fault.first_observable_ts).total_seconds() / 3600
        impact = fault.impact_ts
        lead = (
            (impact - hit.start_ts).total_seconds() / 3600
            if hit is not None and pd.notna(impact)
            else None
        )
        outcomes.append(
            {
                "fault_id": fault.gt_fault_id,
                "type": fault.gt_fault_type,
                "has_impact": pd.notna(impact),
                "lead_hours": lead,
                "early": lead is not None and lead > 0,
                "magnitude_db": getattr(fault, "magnitude_db", np.nan),
                "detected": hit is not None,
                "delay_from_onset_hours": (
                    (hit.start_ts - fault.onset_ts).total_seconds() / 3600
                    if hit is not None
                    else None
                ),
                "delay_from_visibility_proxy_hours": delay,
                "prompt": delay is not None and 0 <= delay <= prompt_hours,
                "visible": pd.notna(fault.first_observable_ts),
            }
        )
    outcomes = pd.DataFrame(
        outcomes,
        columns=[
            "fault_id",
            "type",
            "has_impact",
            "lead_hours",
            "early",
            "magnitude_db",
            "detected",
            "delay_from_onset_hours",
            "delay_from_visibility_proxy_hours",
            "prompt",
            "visible",
        ],
    )
    # Only warnings STARTING within boundary faults are excluded.
    excluded = faults.loc[boundary & ~eligible]
    boundary_ids = set()
    for fault in excluded.itertuples(index=False):
        mask = (
            belongs(members.get(fault.gt_fault_id, set()))
            & (incidents.start_ts < fault.repair_ts)
            & (incidents.start_ts >= fault.onset_ts)
        )
        boundary_ids.update(incidents.loc[mask, "incident_id"])
    nuisance = len(set(incidents.incident_id) - used - boundary_ids)
    false_alarms = len(set(incidents.incident_id) - eligible_alerts - boundary_ids)
    duplicates = len(eligible_alerts - used - boundary_ids)
    entities = pd.read_csv(dataset / "topology.csv").ont_id.nunique()
    days = entities * (end - start).total_seconds() / 86400
    detected = int(outcomes.detected.sum())
    result = {
        "faults": len(targets),
        "missed": len(targets) - detected,
        "impact_faults": int(outcomes.has_impact.sum()),
        "early_warnings": int(outcomes.early.sum()),
        "early_recall": (
            float(outcomes.early.sum() / outcomes.has_impact.sum())
            if outcomes.has_impact.sum()
            else None
        ),
        "median_positive_lead_hours": (
            float(outcomes.loc[outcomes.early, "lead_hours"].median())
            if outcomes.early.any()
            else None
        ),
        "median_delay_hours": (
            float(outcomes.loc[outcomes.detected, "delay_from_onset_hours"].median())
            if outcomes.detected.any()
            else None
        ),
        "false_alarms": false_alarms,
        "duplicate_warnings": duplicates,
        "false_alarms_per_1000_days": 1000 * false_alarms / days,
        "warnings_per_1000_days": 1000 * len(incidents) / days,
        "detected": detected,
        "event_recall": detected / len(targets) if len(targets) else None,
        "event_recall_lower": (
            wilson_interval(detected, len(targets))[0] if len(targets) else None
        ),
        "prompt_detected": int(outcomes.prompt.sum()),
        "unobservable_proxy_faults": int((~outcomes.visible.astype(bool)).sum()),
        "boundary_faults_excluded": len(excluded),
        "boundary_incidents_excluded": len(boundary_ids - used),
        "one_to_one_precision": (
            detected / (len(incidents) - len(boundary_ids - used))
            if len(incidents) - len(boundary_ids - used)
            else None
        ),
        "incidents": len(incidents),
        "nuisance_incidents": nuisance,
        "monitored_entity_days": days,
        "nuisance_per_1000_days": 1000 * nuisance / days,
        "nuisance_upper_per_1000_days": 1000 * poisson_rate_interval(nuisance, days)[1],
        "uncertainty_note": "Poisson/Wilson summaries assume independence; shared events violate it.",
    }
    return result, outcomes
