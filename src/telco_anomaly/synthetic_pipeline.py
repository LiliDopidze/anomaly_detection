"""Causal baselines for the v5 synthetic stage, with a sealed final partition.

This bounded workflow intentionally excludes learned root-cause classification.
The earlier canonical pipeline remains available for separately versioned work.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from telco_anomaly.synthetic import sha256
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


def load_observations(dataset, include_holdout=False):
    import duckdb

    dataset = Path(dataset)
    manifest = json.loads((dataset / "manifest.json").read_text())
    cutoff = split_boundaries(manifest)["holdout"][0]
    with duckdb.connect() as connection:
        query = "SELECT * FROM read_parquet(?)"
        args = [str(dataset / "reference_dataset.parquet")]
        if not include_holdout:
            query += " WHERE timestamp_utc < ?"
            args.append(cutoff)
        query += " ORDER BY ont_id, timestamp_utc"
        data = connection.execute(query, args).df()
    return data, manifest


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
    columns = feature_columns(features)
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
    candidates = []
    for fault in targets.itertuples(index=False):
        mask = (
            incidents.entity_id.isin(members.get(fault.gt_fault_id, set()))
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
            incidents.entity_id.isin(members.get(fault.gt_fault_id, set()))
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
        "incidents": len(incidents),
        "nuisance_incidents": nuisance,
        "monitored_entity_days": days,
        "nuisance_per_1000_days": 1000 * nuisance / days,
        "nuisance_upper_per_1000_days": 1000 * poisson_rate_interval(nuisance, days)[1],
        "uncertainty_note": "Poisson/Wilson summaries assume independence; shared events violate it.",
    }
    return result, outcomes


def run_development(dataset, output, config):
    """Development only. Scores/plots never load the holdout measurements."""
    validate_experiment(config)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    data, manifest = load_observations(dataset)
    cadence = manifest["config"]["sample_minutes"] * 60
    features = build_features(data, cadence, config["minimum_coverage"])
    boundaries = split_boundaries(manifest)
    train_end = boundaries["train"][1]
    fitted = fit_baselines(
        features.loc[features.timestamp_utc < train_end], config["seed"]
    )
    scores = score_baselines(features, fitted)
    comparisons, details = [], {}
    for model in ("robust", "isolation_forest"):
        a, b = boundaries["calibration"]
        calibration = scores.loc[scores.timestamp_utc.between(a, b, inclusive="left")]
        values = calibration_blocks(calibration, model, cadence)
        if not len(values):
            raise ValueError("No adequately observed daily calibration blocks")
        va, vb = boundaries["verification"]
        verification = scores.loc[
            scores.timestamp_utc.between(va, vb, inclusive="left")
        ]
        da, db = boundaries["development"]
        development = scores.loc[scores.timestamp_utc.between(da, db, inclusive="left")]
        expected_verify = (
            manifest["config"]["n_onts"] * (vb - va).total_seconds() / cadence
        )
        exposure = verification[model].notna().sum() * cadence / 86400
        coverage = verification[model].notna().sum() / expected_verify
        chosen = None
        for q in config["calibration_quantiles"]:
            threshold = float(values.quantile(q))
            check = make_incidents(verification, model, threshold, cadence)
            upper = 1000 * poisson_rate_interval(len(check), exposure)[1]
            if (
                len(values) * (1 - q) >= 5
                and upper <= config["verification_budget_per_1000_days"]
            ):
                chosen = (q, threshold, upper)
                break
        # A failed calibration still gets a clearly diagnostic development row.
        admissible = chosen is not None and coverage >= config["minimum_score_coverage"]
        q, threshold, upper = chosen or (q, threshold, upper)
        incidents = make_incidents(development, model, threshold, cadence)
        metrics, outcomes = evaluate(dataset, incidents, da, db, config["prompt_hours"])
        expected_dev = (
            manifest["config"]["n_onts"] * (db - da).total_seconds() / cadence
        )
        dev_coverage = development[model].notna().sum() / expected_dev
        lower = metrics["event_recall_lower"]
        qualified = (
            admissible
            and lower is not None
            and lower >= config["minimum_event_recall_lower_bound"]
            and dev_coverage >= config["minimum_score_coverage"]
            and metrics["nuisance_upper_per_1000_days"]
            <= config["development_budget_per_1000_days"]
        )
        comparisons.append(
            {
                "model": model,
                "quantile": q,
                "threshold": threshold,
                "verification_upper_per_1000_days": upper,
                "verification_coverage": coverage,
                "calibration_daily_blocks": len(values),
                "calibration_expected_tail_blocks": len(values) * (1 - q),
                "development_coverage": dev_coverage,
                "calibration_admissible": admissible,
                "qualified": qualified,
                **metrics,
            }
        )
        details[model] = (incidents, outcomes)
    comparison = pd.DataFrame(comparisons)
    output.mkdir(parents=True)
    comparison.to_csv(output / "comparison.csv", index=False)
    scores.to_parquet(output / "development_scores.parquet", index=False)
    for model, (incidents, outcomes) in details.items():
        incidents.to_csv(output / f"{model}_incidents.csv", index=False)
        outcomes.to_csv(output / f"{model}_faults.csv", index=False)
    joblib.dump(fitted, output / "baselines.joblib")
    receipt = {
        "dataset": str(Path(dataset).resolve()),
        "dataset_manifest_sha256": sha256(Path(dataset) / "manifest.json"),
        "pipeline_sha256": sha256(__file__),
        "config": config,
        "holdout_opened": False,
    }
    eligible = comparison.loc[comparison.qualified].sort_values(
        ["prompt_detected", "nuisance_incidents"],
        ascending=[False, True],
    )
    status = {"status": "selected" if len(eligible) else "STOP_no_qualified_candidate"}
    if len(eligible):
        selection = eligible.iloc[0][["model", "threshold"]].to_dict()
        (output / "selected_configuration.json").write_text(
            json.dumps(selection, indent=2)
        )
    receipt["frozen_files"] = {
        name: sha256(output / name)
        for name in ("baselines.joblib", "selected_configuration.json")
        if (output / name).exists()
    }
    (output / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (output / "selection_status.json").write_text(json.dumps(status, indent=2))
    return comparison


def run_holdout(run_directory):
    """Explicit one-shot evaluation of a qualified frozen selection."""
    run = Path(run_directory)
    if (run / "holdout").exists():
        raise FileExistsError("Holdout results already exist; do not retune on them")
    selection_path = run / "selected_configuration.json"
    if not selection_path.exists():
        raise ValueError("STOP: no qualified selected configuration")
    receipt = json.loads((run / "run.json").read_text())
    dataset = Path(receipt["dataset"])
    if sha256(dataset / "manifest.json") != receipt["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest changed")
    if sha256(__file__) != receipt["pipeline_sha256"]:
        raise ValueError("Pipeline changed; frozen evaluation refused")
    manifest = json.loads((dataset / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256(dataset / name) != digest:
            raise ValueError(f"Dataset changed: {name}")
    for name, digest in receipt["frozen_files"].items():
        if sha256(run / name) != digest:
            raise ValueError(f"Frozen selection artifact changed: {name}")
    selected = json.loads(selection_path.read_text())
    data, manifest = load_observations(dataset, include_holdout=True)
    cadence = manifest["config"]["sample_minutes"] * 60
    features = build_features(data, cadence, receipt["config"]["minimum_coverage"])
    fitted = joblib.load(run / "baselines.joblib")
    scores = score_baselines(features, fitted)
    start, end = split_boundaries(manifest)["holdout"]
    scores = scores.loc[scores.timestamp_utc.between(start, end, inclusive="left")]
    incidents = make_incidents(
        scores, selected["model"], selected["threshold"], cadence
    )
    metrics, outcomes = evaluate(
        dataset, incidents, start, end, receipt["config"]["prompt_hours"]
    )
    out = run / "holdout"
    out.mkdir()
    incidents.to_csv(out / "incidents.csv", index=False)
    outcomes.to_csv(out / "faults.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
