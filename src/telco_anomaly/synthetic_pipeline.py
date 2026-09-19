"""Causal baselines for the v5 synthetic stage, with a sealed final partition.

This bounded workflow intentionally excludes learned root-cause classification.
The pipeline module orchestrates these functions for every partition.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from telco_anomaly.evaluation import (
    _maximum_event_matches,
    poisson_rate_interval,
    wilson_interval,
)

SIGNALS = {
    "rx_power_dbm": (0.15, -1),
    "olt_rx_power_dbm": (0.2, -1),
    "tx_power_dbm": (0.1, 0),
    "bias_current_ma": (0.3, 0),
    "fec_log_rate": (0.2, 1),
}


def split_boundaries(manifest):
    """Fixed chronological roles. Never split individual rows at random."""
    start = pd.Timestamp(manifest["config"]["start"]).tz_convert("UTC")
    span = pd.Timedelta(days=manifest["config"]["days"])
    edges = [start + span * f for f in (0, 0.4, 0.55, 0.7, 0.85, 1)]
    names = ("train", "calibration", "verification", "development", "holdout")
    return {name: (edges[i], edges[i + 1]) for i, name in enumerate(names)}


def build_features(data, cadence_seconds=900, minimum_coverage=0.5):
    """Per-entity causal IQR-scaled deviations. Current row excluded from history."""
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be in (0, 1]")
    outputs = []
    for _, group in data.groupby("ont_id", sort=False):
        group = group.sort_values("timestamp_utc").copy()
        group["fec_log_rate"] = np.log1p(group.fec_count / cadence_seconds)
        group = group.set_index("timestamp_utc", drop=False)
        out = group[["timestamp_utc", "ont_id"]].copy()
        for metric, (floor, direction) in SIGNALS.items():
            values = group[metric]
            out[metric + "__level"] = values
            for hours in (6, 24):
                minimum = max(
                    4, int(np.ceil(hours * 3600 / cadence_seconds * minimum_coverage))
                )
                history = values.rolling(
                    f"{hours}h", closed="left", min_periods=minimum
                )
                baseline = history.median()
                scale = (history.quantile(0.75) - history.quantile(0.25)) / 1.349
                scale = scale.clip(lower=floor)
                z = (values - baseline) / scale
                evidence = z.abs() if direction == 0 else (direction * z).clip(lower=0)
                out[f"{metric}__z{hours}h"] = evidence
        # Availability is reported, not silently turned into a healthy value.
        zcols = [c for c in out if "__z" in c]
        out["feature_coverage"] = out[zcols].notna().mean(axis=1)
        outputs.append(out.reset_index(drop=True))
    if not outputs:
        raise ValueError("No observations available")
    return pd.concat(outputs, ignore_index=True)


def feature_columns(features):
    return [c for c in features if "__" in c]


def fit_baselines(features, seed=42):
    columns = [c for c in feature_columns(features) if not c.endswith("__level")]
    ready = features.feature_coverage.ge(0.8)
    train = features.loc[ready, columns]
    if len(train) < 100 or train.notna().sum().eq(0).any():
        raise ValueError("Insufficient training support; use more days/valid history")
    # Bound fitting cost; source is a balanced panel rather than a traffic-weighted log.
    train = train.sample(min(len(train), 30_000), random_state=seed)
    medians = train.median()
    model = IsolationForest(
        n_estimators=100, max_samples=256, random_state=seed, n_jobs=1
    )
    model.fit(train.fillna(medians))
    return {"columns": columns, "medians": medians, "model": model}


def score_baselines(features, fitted):
    ready = features.feature_coverage.ge(0.8)
    result = features[["timestamp_utc", "ont_id", "feature_coverage"]].copy()
    zcols = [c for c in fitted["columns"] if "__z" in c]
    result["robust"] = features[zcols].max(axis=1).where(ready)
    result["isolation_forest"] = np.nan
    if ready.any():
        inputs = features.loc[ready, fitted["columns"]].fillna(fitted["medians"])
        result.loc[ready, "isolation_forest"] = -fitted["model"].score_samples(inputs)
    return result


def make_incidents(scores, model, threshold, cadence_seconds):
    """Two consecutive exceedances; two recovery samples; never bridge gaps."""
    rows = []
    for entity, group in scores.groupby("ont_id", sort=False):
        pending = recovery = 0
        start = previous = None
        for row in group.sort_values("timestamp_utc").itertuples(index=False):
            t = row.timestamp_utc
            score = getattr(row, model)
            gap = (
                previous is not None
                and (t - previous).total_seconds() > 1.5 * cadence_seconds
            )
            if gap or not np.isfinite(score):
                if start is not None:
                    rows.append((entity, start, previous))
                start = None
                pending = recovery = 0
            if np.isfinite(score):
                if start is None:
                    pending = pending + 1 if score > threshold else 0
                    if pending >= 2:
                        start = t  # Actual decision time, not backdated onset.
                        recovery = 0
                else:
                    recovery = recovery + 1 if score <= threshold else 0
                    if recovery >= 2:
                        rows.append((entity, start, t))
                        start = None
                        pending = recovery = 0
            previous = t
        if start is not None:
            rows.append((entity, start, previous))
    result = pd.DataFrame(rows, columns=["entity_id", "start_ts", "end_ts"])
    result.insert(0, "incident_id", [f"I-{i:06d}" for i in range(len(result))])
    return result


def validate_experiment(config):
    for name in ("minimum_coverage", "minimum_score_coverage"):
        if not 0 < config[name] <= 1:
            raise ValueError(f"{name} must be in (0, 1]")
    if not 0 <= config["minimum_event_recall_lower_bound"] <= 1:
        raise ValueError("Recall bound must be in [0, 1]")
    quantiles = config["calibration_quantiles"]
    if not quantiles or quantiles != sorted(set(quantiles)):
        raise ValueError("Calibration quantiles must be nonempty, unique and sorted")
    if not all(0 < q < 1 for q in quantiles):
        raise ValueError("Calibration quantiles must be in (0, 1)")
    for name in (
        "verification_budget_per_1000_days",
        "development_budget_per_1000_days",
        "prompt_hours",
    ):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be positive and finite")


def calibration_blocks(scores, model, cadence):
    """Daily entity maxima reduce pseudo-replication, without claiming independence."""
    grouped = (
        scores.assign(day=scores.timestamp_utc.dt.floor("D"))
        .groupby(["ont_id", "day"])[model]
        .agg(["max", "count"])
    )
    return grouped.loc[grouped["count"] >= 0.8 * 86400 / cadence, "max"]


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
    by_id = incidents.set_index("incident_id")
    for fault in targets.itertuples(index=False):
        alert_id = fault_to_alert.get(fault.gt_fault_id)
        hit = by_id.loc[alert_id] if alert_id is not None else None
        delay = None
        if hit is not None and pd.notna(fault.first_observable_ts):
            delay = (hit.start_ts - fault.first_observable_ts).total_seconds() / 3600
        outcomes.append(
            {
                "fault_id": fault.gt_fault_id,
                "type": fault.gt_fault_type,
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
            "detected",
            "delay_from_onset_hours",
            "delay_from_visibility_proxy_hours",
            "prompt",
            "visible",
        ],
    )
    # Alerts overlapping boundary faults are not automatically called nuisance.
    excluded = faults.loc[boundary & ~eligible]
    boundary_ids = set()
    for fault in excluded.itertuples(index=False):
        mask = (
            belongs(members.get(fault.gt_fault_id, set()))
            & (incidents.start_ts < fault.repair_ts)
            & (incidents.end_ts >= fault.onset_ts)
        )
        boundary_ids.update(incidents.loc[mask, "incident_id"])
    nuisance = len(set(incidents.incident_id) - used - boundary_ids)
    entities = pd.read_csv(dataset / "topology.csv").ont_id.nunique()
    days = entities * (end - start).total_seconds() / 86400
    detected = int(outcomes.detected.sum())
    result = {
        "faults": len(targets),
        "detected": detected,
        "event_recall": detected / len(targets) if len(targets) else None,
        "event_recall_lower": (
            wilson_interval(detected, len(targets))[0] if len(targets) else None
        ),
        "prompt_detected": int(outcomes.prompt.sum()),
        "unobservable_proxy_faults": int((~outcomes.visible.astype(bool)).sum()),
        "boundary_faults_excluded": len(excluded),
        "boundary_incidents_excluded": len(boundary_ids - used),
        "incident_precision": (
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
