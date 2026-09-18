"""Read-only development diagnostics; never fits, selects, or opens holdout.

Run with ``python -m telco_anomaly.diagnostics --help``. Full mode reproduces
the frozen evaluation before explaining it. Findings are evidence categories,
not inferred physical root causes. Output directories are immutable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .io import file_sha256, immutable_output_directory, read_json, write_json


def _development(path):
    path = Path(path).resolve()
    if any("holdout" in part.lower() for part in path.parts):
        raise PermissionError(f"Development diagnostics refuse holdout paths: {path}")
    return path


def _child(root, relative):
    root = _development(root)
    child = _development(root / relative)
    if not child.is_relative_to(root):
        raise ValueError(f"Artifact escapes its run directory: {relative}")
    return child


def _hash_check(path, expected, label):
    if not expected or file_sha256(path) != expected:
        raise ValueError(f"Lineage mismatch: {label}: {path}")


def _save(output, name, frame):
    frame.to_parquet(output / f"{name}.parquet", index=False)
    frame.to_csv(output / f"{name}.csv", index=False)


def fault_summary(faults, horizon=172800):
    """One row per fault, preserving misses and conditional latency."""
    f = faults.copy()
    if f.fault_id.duplicated().any():
        raise ValueError("Duplicate fault IDs")
    f["delay_hours"] = f.detection_delay_seconds / 3600
    # This is the active matching assignment, not a fresh prompt matching.
    f["active_match_within_horizon"] = f.detected & f.detection_delay_seconds.lt(
        horizon
    )
    f["outcome"] = np.select(
        [~f.detected, f.preimpact, f.active_match_within_horizon],
        ["missed", "detected_preimpact", "detected_prompt_not_preimpact"],
        default="detected_late",
    )
    f["localisation_outcome"] = np.select(
        [
            ~f.detected,
            f.equivalent_scope,
            f.domain_type.ne("entity") & f.predicted_domain_type.eq("entity"),
        ],
        ["not_detected", "correct_or_equivalent", "shared_scope_reported_as_entity"],
        default="wrong_or_unresolved_scope",
    )
    return f


def candidate_frontier(comparison):
    """Descriptive Pareto frontier, never promotes diagnostic candidates."""
    d = comparison.copy()
    objectives = ["prompt_event_recall", "false_incidents_per_entity_day"]
    if not set(objectives) <= set(d):
        return d
    complete = d[objectives].notna().all(axis=1)
    d["prompt_workload_pareto"] = False
    for i, row in d.loc[complete].iterrows():
        other = d.loc[complete]
        dominates = (
            other.prompt_event_recall.ge(row.prompt_event_recall)
            & other.false_incidents_per_entity_day.le(
                row.false_incidents_per_entity_day
            )
            & (
                other.prompt_event_recall.gt(row.prompt_event_recall)
                | other.false_incidents_per_entity_day.lt(
                    row.false_incidents_per_entity_day
                )
            )
        )
        d.loc[i, "prompt_workload_pareto"] = not dominates.any()
    return d


def numeric_profile(path):
    """One bounded SQL scan of numeric columns; no wide pandas materialisation."""
    from .scoring._common import _model_duckdb, _sql_identifier

    fields = [
        f.name
        for f in pq.ParquetFile(path).schema_arrow
        if pa.types.is_floating(f.type) or pa.types.is_integer(f.type)
    ]
    expressions = ["count(*)"]
    for name in fields:
        col = _sql_identifier(name)
        finite = f"CASE WHEN isfinite({col}) THEN {col} END"
        expressions += [
            f"count({finite})",
            f"min({finite})",
            f"max({finite})",
            f"avg({finite})",
            f"stddev_pop({finite})",
        ]
    with _model_duckdb() as con:
        row = con.execute(
            f"SELECT {', '.join(expressions)} FROM read_parquet(?)", [str(path)]
        ).fetchone()
    rows = []
    for i, name in enumerate(fields):
        valid, minimum, maximum, mean, std = row[1 + i * 5 : 6 + i * 5]
        rows.append(
            {
                "feature": name,
                "rows": row[0],
                "finite_rows": valid,
                "missing_fraction": 1 - valid / row[0] if row[0] else np.nan,
                "min": minimum,
                "max": maximum,
                "mean": mean,
                "std": std,
                "constant": valid > 0 and minimum == maximum,
            }
        )
    return pd.DataFrame(rows)


def deep_data_audit(inputs, output):
    from .scoring._common import _model_duckdb

    development = numeric_profile(inputs["features"])
    calibration = numeric_profile(inputs["calibration_features"])
    quality = calibration.merge(
        development,
        on="feature",
        how="outer",
        suffixes=("_calibration", "_development"),
    )
    quality["mean_shift_in_calibration_std"] = (
        quality.mean_development - quality.mean_calibration
    ) / quality.std_calibration.where(quality.std_calibration.gt(0))
    _save(output, "feature_quality_and_shift", quality)
    with _model_duckdb() as con:
        quality = con.execute(
            """SELECT metric_id, quality_code, count(*) AS rows,
            count(value) FILTER (WHERE isfinite(value)) AS finite_rows,
            count(DISTINCT entity_id) AS entities
            FROM read_parquet(?) WHERE event_ts >= ? AND event_ts < ?
            GROUP BY metric_id, quality_code ORDER BY metric_id, quality_code""",
            [
                str(inputs["core"] / "telemetry" / "*.parquet"),
                inputs["start"],
                inputs["end"],
            ],
        ).df()
        _save(output, "raw_quality_by_metric", quality)
        events = inputs["core"] / "operational_events.parquet"
        if events.exists():
            frame = con.execute(
                """SELECT event_family, event_code, quality_code, count(*) AS events
                FROM read_parquet(?) WHERE event_ts >= ? AND event_ts < ?
                GROUP BY event_family, event_code, quality_code""",
                [str(events), inputs["start"], inputs["end"]],
            ).df()
            _save(output, "operational_event_counts", frame)
        gaps = inputs["core"] / "collection_gaps.parquet"
        if gaps.exists():
            frame = con.execute(
                """SELECT entity_id, metric_id, count(*) AS gaps,
                sum(epoch(least(gap_end, ?) - greatest(gap_start, ?))) / 3600 AS missing_hours
                FROM read_parquet(?) WHERE gap_start < ? AND gap_end > ?
                GROUP BY entity_id, metric_id""",
                [
                    inputs["end"],
                    inputs["start"],
                    str(gaps),
                    inputs["end"],
                    inputs["start"],
                ],
            ).df()
            _save(output, "collection_gap_summary", frame)


def score_evidence(score_path, windows, configuration):
    """Stream scores once; retain aggregates, never a fleet-wide score frame.

    Window ends are exclusive, as in the evaluator. Runs reset at missing
    scores, episode boundaries, and gaps >1.5 cadences. Window-local runs are
    evidence only: actual alerts are replayed independently using production
    logic (which can carry state from before a fault).
    """
    from .scoring._common import iter_episode_frames

    channels = configuration["channels"]
    cadence = float(configuration["cadence_seconds"])
    by_entity = {str(k): v for k, v in windows.groupby("entity_id")}
    rows, coverage = [], []
    for episode_number, episode in enumerate(
        iter_episode_frames(
            score_path,
            columns=[
                "event_ts",
                "entity_id",
                "episode_id",
                *channels,
            ],
        ),
        start=1,
    ):
        if episode_number % 100 == 0:
            print(f"Score evidence: {episode_number} episodes scanned", flush=True)
        episode = episode.sort_values("event_ts")
        ts = pd.to_datetime(episode.event_ts, utc=True)
        entity = str(episode.entity_id.iloc[0])
        for channel in channels:
            values = pd.to_numeric(episode[channel], errors="coerce")
            valid = np.isfinite(values)
            high = valid & values.ge(configuration["thresholds"][channel])
            daily = pd.DataFrame(
                {
                    "day": ts.dt.floor("D"),
                    "rows": 1,
                    "valid": valid.astype(int),
                    "high": high.astype(int),
                }
            )
            daily = daily.groupby("day", as_index=False).sum()
            daily["entity_id"], daily["channel"] = entity, channel
            coverage.append(daily)
            if entity not in by_entity:
                continue
            for window in by_entity[entity].itertuples(index=False):
                mask = ts.ge(window.match_start)
                if pd.notna(window.match_end):
                    mask &= ts.lt(window.match_end)
                idx = np.flatnonzero(mask.to_numpy())
                if not len(idx):
                    continue
                t, v, h = ts.iloc[idx], values.iloc[idx], high.iloc[idx]
                breaks = (~h) | t.diff().dt.total_seconds().gt(1.5 * cadence)
                # Gap at a high sample starts a new run including that sample.
                groups = breaks.cumsum()
                lengths = h.astype(int).groupby(groups).sum()
                peak = v.where(np.isfinite(v)).max()
                rows.append(
                    {
                        "fault_id": window.fault_id,
                        "entity_id": entity,
                        "episode_id": str(episode.episode_id.iloc[0]),
                        "channel": channel,
                        "rows": len(idx),
                        "valid_scores": int(np.isfinite(v).sum()),
                        "exceedances": int(h.sum()),
                        "peak_score": peak,
                        "threshold": configuration["thresholds"][channel],
                        "peak_minus_threshold": peak
                        - configuration["thresholds"][channel],
                        "max_window_local_high_run": int(lengths.max()),
                        "required_observations": configuration[
                            "persistence_observations"
                        ][channel],
                        "first_crossing": t.loc[h].min(),
                        "preimpact_exceedances": (
                            int((h & t.lt(window.impact_ts)).sum())
                            if pd.notna(window.impact_ts)
                            else np.nan
                        ),
                    }
                )
    evidence = pd.DataFrame(
        rows,
        columns=[
            "fault_id",
            "entity_id",
            "episode_id",
            "channel",
            "rows",
            "valid_scores",
            "exceedances",
            "peak_score",
            "threshold",
            "peak_minus_threshold",
            "max_window_local_high_run",
            "required_observations",
            "first_crossing",
            "preimpact_exceedances",
        ],
    )
    availability = (
        (
            pd.concat(coverage, ignore_index=True)
            .groupby(["entity_id", "channel", "day"], as_index=False)[
                ["rows", "valid", "high"]
            ]
            .sum()
        )
        if coverage
        else pd.DataFrame()
    )
    if len(availability):
        availability["valid_fraction_of_present_rows"] = (
            availability.valid / availability.rows
        )
        availability["present_score_hours"] = availability.valid * cadence / 3600
    return evidence, availability


def classify_faults(faults, evidence, alerts, windows):
    """Describe the earliest demonstrable bottleneck without guessing signal quality."""
    rows = []
    for fault in faults.itertuples(index=False):
        e = evidence.loc[evidence.fault_id.eq(fault.fault_id)]
        w = windows.loc[windows.fault_id.eq(fault.fault_id)]
        openings, overlaps = set(), set()
        for win in w.itertuples(index=False):
            a = alerts.loc[alerts.entity_id.astype(str).eq(str(win.entity_id))]
            in_window = a.alert_start.ge(win.match_start)
            overlap = a.alert_end.gt(win.match_start)
            if pd.notna(win.match_end):
                in_window &= a.alert_start.lt(win.match_end)
                overlap &= a.alert_start.lt(win.match_end)
            openings.update(a.loc[in_window, "alert_id"])
            overlaps.update(a.loc[overlap, "alert_id"])
        if fault.detected:
            reason = "detected_review_latency"
        elif len(e) == 0:
            reason = "no_score_rows_in_affected_windows"
        elif e.valid_scores.sum() == 0:
            reason = "no_finite_scores_in_affected_windows"
        elif openings:
            reason = "alert_opened_but_no_distinct_fault_credit_review_case_matching"
        elif overlaps:
            reason = "existing_alert_overlaps_fault_without_new_opening"
        elif e.exceedances.sum() == 0:
            reason = "finite_scores_never_reached_frozen_threshold"
        else:
            reason = "threshold_crossing_without_alert_review_persistence_and_gaps"
        rows.append(
            {
                "fault_id": fault.fault_id,
                "evidence_category": reason,
                "alert_openings": len(openings),
                "overlapping_alerts": len(overlaps),
                "score_rows": int(e.rows.sum()),
                "valid_score_rows": int(e.valid_scores.sum()),
                "note": "Signal absence versus feature/model weakness requires trace review.",
            }
        )
    return faults.merge(pd.DataFrame(rows), on="fault_id", validate="one_to_one")


def replay(score_path, config, topology, events, intervals, exposure, horizon=None):
    from .scoring.alerts import alerts_from_score_file
    from .evaluation import form_cases, evaluate_cases

    alerts = (
        pd.concat(
            [
                alerts_from_score_file(
                    score_path,
                    c,
                    config["thresholds"][c],
                    min_consecutive=config["persistence_observations"][c],
                    recovery_consecutive=config["recovery_observations"],
                    cadence_seconds=config["cadence_seconds"],
                    recovery_threshold_fraction=config["recovery_threshold_fraction"],
                )
                for c in config["channels"]
            ],
            ignore_index=True,
        )
        .sort_values("alert_start")
        .reset_index(drop=True)
    )
    alerts["alert_id"] = [f"A-{i:09d}" for i in range(1, len(alerts) + 1)]
    cases, members = form_cases(
        alerts,
        topology,
        gap_seconds=config["incident_quiet_period_seconds"],
        thresholds=config["thresholds"],
        shared_scope_models=("group_common_mode",),
    )
    result = evaluate_cases(
        cases,
        members,
        events,
        intervals,
        exposure_value=exposure,
        exposure_unit="entity_day",
        decision_horizon_seconds=horizon,
        topology_memberships=topology,
    )
    return alerts, cases, members, result


def _report(output, metrics, faults, manifest, full):
    values = metrics.set_index("metric").value
    lines = [
        "# Development diagnostic report",
        "",
        "Mode: " + ("full frozen replay" if full else "results only"),
        "",
        "No model was trained, selected or changed. Holdout remains sealed.",
        "",
        "## Recorded metrics",
        "",
        "| Metric | Value | Numerator | Denominator |",
        "|---|---:|---:|---:|",
    ]
    for r in metrics.itertuples(index=False):
        lines.append(
            f"| {r.metric} | {r.value:.6g} | {r.numerator} | {r.denominator:.6g} |"
        )
    lines += [
        "",
        "## Interpretation safeguards",
        "",
        "- Delay statistics condition on detected faults; misses remain in the fault ledger.",
        "- Exact, top-2 and joint localisation use different denominators.",
        "- Uncredited workload includes duplicates; it is not all unrelated false alarms.",
        "- Synthetic development evidence does not establish production performance.",
        "- Window-local threshold runs are diagnostic evidence, not simulated alerts.",
        "- Present-row score availability does not count absent timestamps.",
        "- Candidate frontiers are descriptive and do not override eligibility/calibration gates.",
        "- Small fault-type samples and correlated events limit uncertainty estimates.",
    ]
    nuisance = values.get("false_cases_per_entity_day", np.nan)
    duplicate = values.get("duplicate_case_count", np.nan)
    total = values.get("consolidated_case_count", np.nan)
    credited = metrics.loc[metrics.metric.eq("case_precision"), "numerator"]
    if len(credited):
        lines += [
            "",
            f"Incidents: {total:g}; credited: {credited.iloc[0]:g}; "
            f"duplicates: {duplicate:g}; unmatched: {total - credited.iloc[0] - duplicate:g}.",
            f"Uncredited incidents / 1,000 scoreable entity-days: {nuisance * 1000:.3f}.",
        ]
    lines += [
        "",
        "## Fault outcomes",
        "",
        faults.outcome.value_counts().to_string(),
        "",
        "## What still needs human investigation",
        "",
        "Review raw and feature traces to separate weak/no pre-impact signal from feature or "
        "model failure. Review unmatched incidents against benign changes and data-quality "
        "events. These artifacts alone cannot prove physical root causes.",
    ]
    if not full:
        lines += [
            "",
            "Scores, alerts, features, raw telemetry and per-incident matching were not "
            "available in this mode. Run full mode to diagnose pipeline bottlenecks.",
        ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _full_inputs(data_root, result_manifest, core_run, feature_run, truth_run):
    root = _development(data_root)

    def stage(name, run):
        return _child(root, f"{name}/synthetic_pon/{run}")

    selection = stage("selection", result_manifest["selection_run_id"])
    model = stage("models", result_manifest["model_run_id"])
    core = stage("core", core_run) / "SPEC-CORE"
    feature = stage("features", feature_run)
    truth = stage("evaluation", truth_run)
    files = {
        "selected_configuration_sha256": selection / "selected_configuration.json",
        "model_manifest_sha256": model / "model_manifest.json",
        "feature_manifest_sha256": feature / "feature_manifest.json",
        "truth_manifest_sha256": truth / "truth_manifest.json",
        "resolved_policy_sha256": model / "resolved_policy.json",
        "evaluation_module_sha256": Path(__file__).with_name("evaluation.py"),
        "selection_module_sha256": Path(__file__).with_name("selection.py"),
    }
    for key, path in files.items():
        _hash_check(path, result_manifest.get(key), key)
    config = read_json(files["selected_configuration_sha256"])
    mm = read_json(files["model_manifest_sha256"])
    fm = read_json(files["feature_manifest_sha256"])
    tm = read_json(files["truth_manifest_sha256"])
    fingerprint = read_json(core / "manifest.json")["fingerprint"]
    if any(
        v != fingerprint
        for v in [
            result_manifest["core_fingerprint"],
            mm["core_fingerprint"],
            fm["core_fingerprint"],
            tm["model_core_fingerprint"],
        ]
    ):
        raise ValueError("Core fingerprints differ")
    for name in ("fault_events.parquet", "fault_entity_intervals.parquet"):
        _hash_check(
            truth / "development" / name,
            tm["file_sha256"].get(f"development/{name}"),
            name,
        )
    scores = _child(model, mm["score_files"]["development"])
    features = _child(feature, fm["partitions"]["development"]["features"])
    calibration_features = _child(
        feature, fm["partitions"]["calibration_fit"]["features"]
    )
    split_path = core.parent / "SPLITS" / "time_partitions.parquet"
    _hash_check(split_path, fm["time_partitions_sha256"], "time partitions")
    splits = pd.read_parquet(split_path)
    dev = splits.loc[splits.partition.eq("development")]
    if len(dev) != 1:
        raise ValueError("Exactly one chronological development partition is required")
    start, end = (
        pd.to_datetime(dev.iloc[0][c], utc=True) for c in ["start_ts", "end_ts"]
    )
    return dict(
        selection=selection,
        model=model,
        core=core,
        scores=scores,
        features=features,
        truth=truth,
        config=config,
        start=start,
        end=end,
        calibration_features=calibration_features,
        files=files,
    )


def export_trace(
    inputs, fault_id, faults, windows, output, context_hours=24, limit=20000
):
    """Export bounded raw/feature/score context for one fault, within development.

    Earliest rows are exported deterministically. Truncation is recorded;
    traces are for manual investigation, not aggregate performance estimates.
    """
    if fault_id not in set(faults.fault_id):
        raise ValueError(f"Unknown development fault: {fault_id}")
    w = windows.loc[windows.fault_id.eq(fault_id)]
    entities = pd.DataFrame({"entity_id": sorted(w.entity_id.astype(str).unique())})
    start = max(
        inputs["start"], w.match_start.min() - pd.Timedelta(hours=context_hours)
    )
    end = min(
        inputs["end"],
        w.match_end.fillna(inputs["end"]).max() + pd.Timedelta(hours=context_hours),
    )
    _export_context(inputs, entities, start, end, output, limit, fault_id)


def _export_context(inputs, entities, start, end, output, limit, identifier):
    from .scoring._common import _model_duckdb

    stats = []
    with _model_duckdb() as con:
        con.register("trace_entities", entities)
        for label, path in [
            ("scores", inputs["scores"]),
            ("features", inputs["features"]),
            ("raw", inputs["core"] / "telemetry" / "*.parquet"),
        ]:
            order = "s.event_ts, s.entity_id, s.episode_id" + (
                ", s.metric_id" if label == "raw" else ""
            )
            frame = con.execute(
                f"""
                SELECT s.* FROM read_parquet(?) s JOIN trace_entities e
                ON cast(s.entity_id AS VARCHAR) = e.entity_id
                WHERE s.event_ts >= ? AND s.event_ts < ?
                ORDER BY {order} LIMIT ?
            """,
                [str(path), start, end, limit + 1],
            ).df()
            _save(output, f"trace_{label}", frame.head(limit))
            stats.append(
                {
                    "trace_id": identifier,
                    "source": label,
                    "rows_exported": min(len(frame), limit),
                    "truncated": len(frame) > limit,
                    "start": str(start),
                    "end": str(end),
                }
            )
    _save(output, "trace_manifest", pd.DataFrame(stats))


def run_diagnostics(
    results,
    output,
    *,
    data_root=None,
    core_run="synthetic_pon_core_v2",
    feature_run="synthetic_pon_features_v6",
    truth_run="synthetic_pon_truth_v3",
    trace_fault=None,
    trace_limit=20000,
    deep_data=False,
    trace_case=None,
):
    results, output = _development(results), _development(output)
    if trace_limit < 1:
        raise ValueError("trace_limit must be positive")
    manifest = read_json(results / "evaluation_manifest.json")
    if manifest.get("partition") != "development":
        raise PermissionError("Only development evaluation manifests are accepted")
    if (trace_fault or trace_case) and data_root is None:
        raise ValueError("A trace requires --data-root (full mode)")
    if trace_fault and trace_case:
        raise ValueError("Choose either trace_fault or trace_case per run")
    if deep_data and data_root is None:
        raise ValueError("--deep-data requires --data-root")
    metrics = pd.read_parquet(results / "metrics.parquet")
    faults = fault_summary(pd.read_parquet(results / "fault_results.parquet"))
    with immutable_output_directory(output) as out:
        _save(out, "metrics", metrics)
        for group in ["fault_type", "domain_type", "outcome", "localisation_outcome"]:
            _save(
                out,
                f"by_{group}",
                faults.groupby(group, dropna=False)
                .agg(
                    faults=("fault_id", "size"),
                    detected=("detected", "sum"),
                    preimpact=("preimpact", "sum"),
                    median_detected_delay_hours=("delay_hours", "median"),
                )
                .reset_index(),
            )
        inventory = {
            "mode": "full" if data_root else "results_only",
            "partition": "development",
            "holdout_opened": False,
            "model_modified": False,
            "results": str(results),
            "input_sha256": {
                p.name: file_sha256(p) for p in results.iterdir() if p.is_file()
            },
        }
        if data_root:
            print("Checking frozen lineage and development boundaries...", flush=True)
            inp = _full_inputs(data_root, manifest, core_run, feature_run, truth_run)
            from .scoring._common import _model_duckdb
            from .evaluation import _fault_windows, scoreable_exposure, evaluate_cases

            with _model_duckdb() as con:
                for source in [inp["scores"], inp["features"]]:
                    invalid = con.execute(
                        """SELECT count(*) FROM read_parquet(?)
                        WHERE event_ts IS NULL OR event_ts < ? OR event_ts >= ?""",
                        [str(source), inp["start"], inp["end"]],
                    ).fetchone()[0]
                    if invalid:
                        raise PermissionError(
                            f"Artifact contains rows outside development: {source}"
                        )
            events = pd.read_parquet(inp["truth"] / "development/fault_events.parquet")
            intervals = pd.read_parquet(
                inp["truth"] / "development/fault_entity_intervals.parquet"
            )
            topology = pd.read_parquet(inp["core"] / "topology_memberships.parquet")
            config = inp["config"]
            exposure = scoreable_exposure(
                inp["scores"],
                config["channels"],
                "entity_day",
                config["cadence_seconds"],
            )
            print("Replaying frozen alerts and incident matching...", flush=True)
            alerts, cases, members, result = replay(
                inp["scores"], config, topology, events, intervals, exposure
            )
            # Reject stale or mixed inputs before attributing failures.
            common = metrics.merge(
                result["metrics"], on="metric", suffixes=("_saved", "_replay")
            )
            if set(common.metric) != set(result["metrics"].metric):
                raise ValueError("Saved metrics omit frozen replay metrics")
            for column in ["value", "numerator", "denominator"]:
                if not np.allclose(
                    common[f"{column}_saved"],
                    common[f"{column}_replay"],
                    equal_nan=True,
                ):
                    raise ValueError(
                        f"Frozen replay does not reproduce saved metrics: {column}"
                    )
            saved = faults.set_index("fault_id").sort_index()
            current = result["fault_results"].set_index("fault_id").sort_index()
            columns = ["detected", "case_id", "equivalent_scope"]
            pd.testing.assert_frame_equal(
                saved[columns].fillna({"case_id": ""}),
                current[columns].fillna({"case_id": ""}),
                check_dtype=False,
            )
            horizon = config["prompt_detection_horizon_seconds"]
            prompt = evaluate_cases(
                cases,
                members,
                events,
                intervals,
                exposure_value=exposure,
                exposure_unit="entity_day",
                decision_horizon_seconds=horizon,
                topology_memberships=topology,
            )
            expected_prompt = metrics.loc[
                metrics.metric.eq("prompt_event_recall"), "value"
            ]
            actual_prompt = (
                prompt["metrics"].set_index("metric").loc["event_recall", "value"]
            )
            if len(expected_prompt) and not np.isclose(
                expected_prompt.iloc[0], actual_prompt, equal_nan=True
            ):
                raise ValueError("Prompt recall replay mismatch")
            faults = fault_summary(result["fault_results"], horizon)
            faults = faults.merge(
                prompt["fault_results"][["fault_id", "detected"]].rename(
                    columns={"detected": "prompt_match_detected"}
                ),
                on="fault_id",
                validate="one_to_one",
            )
            latency = []
            for hours in (1, 6, 24, 48, 168):
                timed = evaluate_cases(
                    cases,
                    members,
                    events,
                    intervals,
                    exposure_value=exposure,
                    exposure_unit="entity_day",
                    decision_horizon_seconds=hours * 3600,
                    topology_memberships=topology,
                )
                row = timed["metrics"].set_index("metric").loc["event_recall"].to_dict()
                latency.append({"horizon_hours": hours, **row})
            _save(out, "latency_recall_curve", pd.DataFrame(latency))
            windows = _fault_windows(events, intervals, None)
            print("Scanning score evidence and entity/day availability...", flush=True)
            evidence, availability = score_evidence(inp["scores"], windows, config)
            faults = classify_faults(faults, evidence, alerts, windows)
            incidents = cases.merge(
                result["case_matches"], on="case_id", validate="one_to_one"
            )
            for name, frame in [
                ("score_evidence", evidence),
                ("score_availability", availability),
                ("alerts", alerts),
                ("incident_members", members),
                ("incident_diagnostics", incidents),
            ]:
                _save(out, name, frame)
            _save(
                out,
                "incident_status_counts",
                incidents.groupby("match_status")
                .size()
                .rename("incidents")
                .reset_index(),
            )
            comparison_path = inp["selection"] / "development_comparison.parquet"
            if comparison_path.exists():
                _save(
                    out,
                    "candidate_frontier",
                    candidate_frontier(pd.read_parquet(comparison_path)),
                )
                inventory["candidate_comparison_sha256"] = file_sha256(comparison_path)
            inventory["lineage_verified"] = {k: str(p) for k, p in inp["files"].items()}
            inventory["score_sha256"] = file_sha256(inp["scores"])
            inventory["replay_verified"] = True
            inventory["selected_channels"] = config["channels"]
            inventory["group_common_mode_selected"] = (
                "group_common_mode" in config["channels"]
            )
            inventory["deep_data_audit"] = bool(deep_data)
            if deep_data:
                print(
                    "Auditing feature shift, raw quality and collection gaps...",
                    flush=True,
                )
                deep_data_audit(inp, out)
            if trace_fault:
                print(f"Exporting bounded trace for {trace_fault}...", flush=True)
                export_trace(inp, trace_fault, faults, windows, out, limit=trace_limit)
            if trace_case:
                selected_case = cases.loc[cases.case_id.eq(trace_case)]
                if len(selected_case) != 1:
                    raise ValueError(f"Unknown case: {trace_case}")
                row = selected_case.iloc[0]
                entities = members.loc[
                    members.case_id.eq(trace_case), ["entity_id"]
                ].drop_duplicates()
                entities["entity_id"] = entities.entity_id.astype(str)
                _export_context(
                    inp,
                    entities,
                    max(inp["start"], row.case_start - pd.Timedelta(hours=24)),
                    min(inp["end"], row.case_end + pd.Timedelta(hours=24)),
                    out,
                    trace_limit,
                    trace_case,
                )
        _save(out, "fault_diagnostics", faults)
        _save(
            out,
            "review_queue",
            faults.loc[~faults.detected | ~faults.preimpact].sort_values(
                ["detected", "delay_hours"],
                ascending=[True, False],
                na_position="first",
            ),
        )
        write_json(out / "diagnostic_manifest.json", inventory)
        _report(out, metrics, faults, manifest, bool(data_root))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Extracted development result directory",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New immutable diagnostic directory"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Enable full frozen replay from Drive/local data root",
    )
    parser.add_argument("--core-run", default="synthetic_pon_core_v2")
    parser.add_argument("--feature-run", default="synthetic_pon_features_v6")
    parser.add_argument("--truth-run", default="synthetic_pon_truth_v3")
    parser.add_argument(
        "--trace-fault", help="Optional fault ID for bounded raw/feature/score exports"
    )
    parser.add_argument(
        "--trace-case",
        help="Optional incident ID (e.g. false/duplicate case) for trace exports",
    )
    parser.add_argument("--trace-limit", type=int, default=20000)
    parser.add_argument(
        "--deep-data",
        action="store_true",
        help="Also scan raw development quality, gaps and feature shift",
    )
    args = parser.parse_args()
    print(run_diagnostics(**vars(args)))


if __name__ == "__main__":
    main()
