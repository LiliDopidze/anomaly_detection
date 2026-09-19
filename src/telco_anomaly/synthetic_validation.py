"""Evidence reports for the synthetic stage; no detector-difficulty pass gate."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from telco_anomaly.synthetic import METRICS, sha256


def validate_dataset(path):
    """Return hard invariants and descriptive diagnostics as separate tables."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    cfg = manifest["config"]
    # Inspect all rows for structural corruption, but keep holdout distributions sealed.
    cutoff = pd.Timestamp(cfg["start"]) + pd.Timedelta(days=cfg["days"] * 0.85)
    topology = pd.read_csv(path / "topology.csv")
    faults = pd.read_csv(path / "gt_fault_registry.csv")
    intervals = pd.read_csv(path / "fault_entity_intervals.csv")
    simulation = pd.read_csv(path / "simulation_audit.csv")
    checks, summaries, seasonal = [], [], []

    def check(name, ok, details):
        checks.append(
            {"check": name, "status": "pass" if ok else "fail", "details": str(details)}
        )

    check(
        "file_integrity",
        all(
            (path / name).is_file() and sha256(path / name) == digest
            for name, digest in manifest["files"].items()
        ),
        "Checksums from the immutable generation manifest",
    )
    check("unique_equipment", topology.ont_id.is_unique, len(topology))
    sizes = topology.groupby("splitter_l2").size()
    check("splitter_capacity", sizes.le(16).all(), sizes.to_dict())
    check(
        "nonnegative_path_loss",
        topology[["loss_ds_db", "loss_us_db"]].ge(0).all().all(),
        "Latent topology parameters are audit-only",
    )
    check("fault_id_unique", faults.gt_fault_id.is_unique, len(faults))
    check(
        "truth_references",
        set(intervals.fault_id) <= set(faults.gt_fault_id)
        and set(intervals.entity_id) <= set(topology.ont_id),
        "Fault/entity references",
    )
    check(
        "one_interval_per_fault_entity",
        not intervals.duplicated(["fault_id", "entity_id"]).any(),
        len(intervals),
    )
    if len(faults):
        onset = pd.to_datetime(faults.onset_ts, utc=True)
        repair = pd.to_datetime(faults.repair_ts, utc=True)
        check("positive_event_duration", (repair > onset).all(), "Half-open intervals")
    check(
        "bounded_fec_count",
        (simulation.max_corrected_codewords <= simulation.codeword_opportunities).all(),
        "Corrected codewords cannot exceed transmitted codewords",
    )
    check(
        "loss_cannot_improve_service",
        (simulation.maximum_capacity_increase_from_loss <= 1e-12).all(),
        "Paired latent no-fault comparison",
    )
    panel_file = pq.ParquetFile(path / "reference_dataset.parquet")
    allowed = {"timestamp_utc", "ont_id", *METRICS}
    check(
        "model_columns_only",
        set(panel_file.schema_arrow.names) == allowed,
        "No latent state, fault labels or topology nuisance parameters",
    )
    total = 0
    valid_entities = set(topology.ont_id)
    previous = {}
    finite_ok = True
    time_ok = True
    ranges_ok = True
    identities_ok = True
    for batch in panel_file.iter_batches(batch_size=100_000):
        data = batch.to_pandas()
        total += len(data)
        identities_ok &= set(data.ont_id) <= valid_entities
        values = data[METRICS].to_numpy()
        finite_ok &= bool((np.isfinite(values) | np.isnan(values)).all())
        for metric in (
            "ber",
            "fec_count",
            "throughput_mbps",
            "uptime_s",
            "reboot_count",
        ):
            ranges_ok &= bool(data[metric].dropna().ge(0).all())
        ranges_ok &= bool(data.ber.dropna().le(1).all())
        for metric in ("fec_count", "reboot_count"):
            v = data[metric].dropna()
            ranges_ok &= bool((v == np.floor(v)).all())
        for entity, group in data.groupby("ont_id", sort=False):
            delta = group.timestamp_utc.diff().dropna().dt.total_seconds()
            time_ok &= bool((delta > 0).all())
            if entity in previous:
                time_ok &= bool(group.timestamp_utc.iloc[0] > previous[entity])
            previous[entity] = group.timestamp_utc.iloc[-1]
        data = data.loc[data.timestamp_utc < cutoff]
        for metric in METRICS:
            v = data[metric].dropna()
            summaries.append(
                {
                    "metric": metric,
                    "rows": len(data),
                    "valid": len(v),
                    "sum": float(v.sum()),
                    "min": float(v.min()) if len(v) else None,
                    "max": float(v.max()) if len(v) else None,
                    "zero_count": int(v.eq(0).sum()),
                }
            )
        profiles = (
            data.assign(hour=data.timestamp_utc.dt.hour)
            .groupby("hour")[
                [
                    "temperature_c",
                    "throughput_mbps",
                ]
            ]
            .agg(["sum", "count"])
        )
        profiles.columns = ["_".join(c) for c in profiles.columns]
        seasonal.append(profiles)
    check("finite_or_missing", finite_ok, "Infinite values are invalid")
    check("strict_entity_time_order", time_ok, "Includes batch boundaries")
    check("valid_measurement_ranges", ranges_ok, "No healthy-range clipping")
    check("known_entities", identities_ok, "No unknown ONTs")
    check("nonempty_measurements", total > 0, total)
    expected = cfg["n_onts"] * cfg["days"] * 1440 // cfg["sample_minutes"]
    summary = (
        pd.DataFrame(summaries)
        .groupby("metric")
        .agg(
            {
                "rows": "sum",
                "valid": "sum",
                "sum": "sum",
                "min": "min",
                "max": "max",
                "zero_count": "sum",
            }
        )
    )
    summary["mean"] = summary["sum"] / summary.valid.replace(0, np.nan)
    summary["field_coverage"] = summary.valid / summary.rows
    summary["calendar_coverage"] = summary.valid / (expected * 0.85)
    profiles = pd.concat(seasonal).groupby(level=0).sum()
    for metric in ("temperature_c", "throughput_mbps"):
        profiles[metric] = profiles[f"{metric}_sum"] / profiles[f"{metric}_count"]
    faults = faults.loc[pd.to_datetime(faults.onset_ts, utc=True) < cutoff]
    intervals = intervals.loc[
        pd.to_datetime(intervals.active_end_ts, utc=True) < cutoff
    ]
    counts = faults.gt_fault_type.value_counts().rename("events").to_frame()
    counts["support_status"] = np.where(
        counts.events >= 20, "descriptive", "low_support"
    )
    if len(intervals):
        dates = {}
        for name in ("active_start_ts", "active_end_ts", "observable_ts", "impact_ts"):
            dates[name] = pd.to_datetime(intervals[name], utc=True)
        for name in ("observable_ts", "impact_ts"):
            valid = dates[name].notna()
            check(
                name + "_inside_event",
                (
                    (dates[name][valid] >= dates["active_start_ts"][valid])
                    & (dates[name][valid] < dates["active_end_ts"][valid])
                ).all(),
                "Missing time is allowed; never fabricated",
            )
    description = {
        "rows": total,
        "expected_rows": expected,
        "row_coverage": total / expected,
        "descriptive_partition": "first_85_percent_only",
        "faults": len(faults),
        "fault_entity_episodes": len(intervals),
        "unobserved_episodes": int(intervals.observable_ts.isna().sum()),
        "impact_episodes": int(intervals.impact_ts.notna().sum()),
        "distribution_status": "descriptive_only_not_field_realism_certification",
        "notes": [
            "Low-support groups are not passed statistical realism tests.",
            "No gate constrains a detector to a desired accuracy band.",
            "Seasonal profiles mix entities; inspect individual traces too.",
        ],
    }
    return {
        "checks": pd.DataFrame(checks),
        "metrics": summary,
        "seasonality": profiles,
        "fault_counts": counts,
        "observability": intervals,
        "summary": description,
    }


def save_report(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in report.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(output / f"{name}.csv", index=name != "checks")
        else:
            (output / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
    return output


def audit_event_sampling(config, repetitions=100):
    """Compare independent-seed event samples to prescribed laws, not field data.

    Monte Carlo uncertainty concerns simulator sampling only. It cannot validate
    that these assumed distributions describe a real network.
    """
    from dataclasses import replace
    from telco_anomaly.synthetic import make_topology, sample_faults

    if type(repetitions) is not int or repetitions < 2:
        raise ValueError("Use at least two independent repetitions")
    config.validate()
    topology = make_topology(config)
    counts, durations, losses, onsets = [], [], [], []
    for offset in range(repetitions):
        sample = sample_faults(replace(config, seed=config.seed + offset), topology)
        counts.append(len(sample))
        if len(sample):
            durations.extend(
                np.log((sample.repair_ts - sample.onset_ts).dt.total_seconds() / 3600)
            )
            losses.extend(np.log(sample.magnitude_db))
            onsets.extend(
                (sample.onset_ts - pd.Timestamp(config.start)).dt.total_seconds()
                / (config.days * 86400)
            )
    expected = (
        config.days
        / 365.25
        * (
            config.n_onts * config.faults_per_ont_year
            + topology.splitter_l2.nunique() * config.faults_per_splitter_year
        )
    )
    rows = [
        ("events_per_run_mean", expected, np.mean(counts)),
        ("events_per_run_variance", expected, np.var(counts, ddof=1)),
        (
            "log_duration_mean",
            np.log(config.fault_duration_median_hours),
            np.mean(durations) if durations else np.nan,
        ),
        (
            "log_duration_sd",
            config.fault_duration_log_sd,
            np.std(durations, ddof=1) if len(durations) > 1 else np.nan,
        ),
        (
            "log_loss_mean",
            np.log(config.loss_median_db),
            np.mean(losses) if losses else np.nan,
        ),
        (
            "log_loss_sd",
            config.loss_log_sd,
            np.std(losses, ddof=1) if len(losses) > 1 else np.nan,
        ),
        ("fractional_onset_mean", 0.5, np.mean(onsets) if onsets else np.nan),
    ]
    result = pd.DataFrame(rows, columns=["quantity", "prescribed", "empirical"])
    result["independent_runs"] = repetitions
    result["events_sampled"] = sum(counts)
    return result
