"""One synthetic workflow: generate -> adapt -> fit -> qualify -> frozen inference.

All modelling consumes the canonical pack. Only evaluation opens the truth root.
Filesystem separation is a guard against accidental reads, not OS access control.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile

import joblib
import numpy as np
import pandas as pd
import yaml

from .adapter import load_mapping, load_observations, verify_pack, write_pack
from .features import regularize, fit_seasonality, extended_features, channel_scores
from .operations import incident_queue
from .synthetic import generate_dataset, load_config, sha256
from .synthetic_validation import validate_dataset, save_report, audit_event_sampling
from .synthetic_pipeline import (
    build_features,
    fit_baselines,
    score_baselines,
    make_incidents,
    split_boundaries,
    calibration_blocks,
    evaluate,
    validate_experiment,
)
from .evaluation import poisson_rate_interval


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, default=str, allow_nan=False) + "\n"
    )


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def code_hashes():
    return {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def runtime(config_path="configs/pipeline.yml"):
    path = Path(config_path).resolve()
    root = path.parent.parent
    policy = yaml.safe_load(path.read_text())
    for key in ("generator", "adapter", "experiment", "pack", "truth", "run"):
        policy[key] = str((root / policy[key]).resolve())
    pack, truth = Path(policy["pack"]), Path(policy["truth"])
    if pack == truth or pack in truth.parents or truth in pack.parents:
        raise ValueError("Model pack and truth must have disjoint roots")
    if (
        not 0
        < policy["scope_background_probability"]
        < policy["scope_hit_probability"]
        < 1
    ):
        raise ValueError("Scope probabilities require 0 < background < hit < 1")
    for name in ("rest_collector_reporting", "power_share"):
        if not 0 < policy[name] <= 1:
            raise ValueError(f"{name} must be in (0, 1]")
    supported = {
        "robust",
        "drift",
        "peer",
        "common_mode",
        "silence",
        "margin",
        "isolation_forest",
        "rules",
    }
    if not policy["channels"] or not set(policy["channels"]) <= supported:
        raise ValueError("Unknown or empty evidence-channel list")
    return policy


def prepare(policy):
    """The generator's labelled output is transient; publish separate model/truth roots."""
    pack, truth = Path(policy["pack"]), Path(policy["truth"])
    config, _ = load_config(policy["generator"])
    mapping = load_mapping(policy["adapter"])
    expected = fingerprint(
        {"generator": asdict(config), "mapping": mapping, "code": code_hashes()}
    )
    receipt = truth / "preparation.json"
    if pack.exists() or truth.exists():
        if (
            not receipt.exists()
            or json.loads(receipt.read_text())["fingerprint"] != expected
        ):
            raise ValueError(
                "Existing prepared data differs; choose new pack/truth paths"
            )
        verify_pack(pack)
        verify_truth(truth)
        return pack
    truth.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pon-source-") as temporary:
        source = generate_dataset(config, Path(temporary) / "source")
        report = validate_dataset(source)
        if not report["checks"].status.eq("pass").all():
            raise ValueError("Generator validation failed")
        panel = pd.read_parquet(source / "reference_dataset.parquet")
        inventory = pd.read_csv(source / "topology.csv")
        metadata = {
            key: asdict(config)[key]
            for key in ("start", "days", "sample_minutes", "n_onts")
        }
        write_pack(panel, inventory, pack, mapping, metadata)
        truth.mkdir()
        for name in ("gt_fault_registry.csv", "fault_entity_intervals.csv"):
            shutil.copy2(source / name, truth / name)
        inventory[["ont_id"]].to_csv(truth / "topology.csv", index=False)
        save_report(report, truth / "generator_validation")
        audit_event_sampling(config).to_csv(
            truth / "generator_validation/event_sampling.csv", index=False
        )
        write_json(
            truth / "manifest.json",
            {
                "files": {
                    name: sha256(truth / name)
                    for name in (
                        "gt_fault_registry.csv",
                        "fault_entity_intervals.csv",
                        "topology.csv",
                    )
                },
                "generator_config": asdict(config),
            },
        )
        write_json(
            receipt,
            {"fingerprint": expected, "pack_manifest": sha256(pack / "manifest.json")},
        )
    return pack


def verify_truth(truth):
    truth = Path(truth)
    manifest = json.loads((truth / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if Path(name).name != name or sha256(truth / name) != digest:
            raise ValueError(f"Evaluation truth changed: {name}")
    return manifest


def ledger(truth, name, record, maximum=None):
    """Append before reading labels; exclusive lock prevents concurrent admissions."""
    directory = Path(truth)
    lock = directory / ".ledger.lock"
    # Exclusive file creation also works on Windows. A stale lock requires review.
    lock.open("x").close()
    try:
        path = directory / f"{name}.jsonl"
        entries = path.read_text().splitlines() if path.exists() else []
        if maximum is not None and len(entries) >= maximum:
            raise ValueError(f"{name} attempt limit reached")
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
        with path.open("a") as stream:
            stream.write(json.dumps(entry, default=str) + "\n")
    finally:
        lock.unlink()
    return len(entries) + 1


def observations(policy, include_holdout=False):
    data, manifest = load_observations(policy["pack"], include_holdout)
    cfg = manifest["config"]
    partitions = split_boundaries(manifest)
    end = partitions["holdout"][1 if include_holdout else 0]
    inventory = pd.read_parquet(Path(policy["pack"]) / "inventory.parquet")
    grid = regularize(
        data, inventory, partitions["train"][0], end, cfg["sample_minutes"] * 60
    )
    return grid, manifest


def featurize(grid, manifest, experiment, decisions):
    base = build_features(
        grid, manifest["config"]["sample_minutes"] * 60, experiment["minimum_coverage"]
    )
    return extended_features(grid, base, decisions)


def score(grid, manifest, experiment, bundle):
    features = featurize(grid, manifest, experiment, bundle["decisions"])
    baseline = score_baselines(features, bundle["fitted"])
    return channel_scores(grid, features, baseline, bundle["policy"])


def freeze_inputs(policy, experiment, manifest):
    return {
        "observation_window": manifest["config"],
        "policy": policy,
        "experiment": experiment,
        "code": code_hashes(),
        "pack_manifest_sha256": sha256(Path(policy["pack"]) / "manifest.json"),
        "truth_manifest_sha256": sha256(Path(policy["truth"]) / "manifest.json"),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "duckdb",
                "pyarrow",
                "joblib",
            )
        },
    }


def calibrate(scores, manifest, experiment, policy):
    partitions = split_boundaries(manifest)
    cadence = manifest["config"]["sample_minutes"] * 60
    a, b = partitions["calibration"]
    calibration = scores.loc[scores.timestamp_utc.between(a, b, inclusive="left")]
    a, b = partitions["verification"]
    verification = scores.loc[scores.timestamp_utc.between(a, b, inclusive="left")]
    results = []
    for channel in policy["channels"]:
        blocks = calibration_blocks(calibration, channel, cadence)
        if not len(blocks):
            results.append(
                {
                    "channel": channel,
                    "threshold": None,
                    "admissible": False,
                    "reason": "insufficient_daily_blocks",
                }
            )
            continue
        selected = None
        fixed = {"rules": 0.0, "margin": 0.0, "silence": 0.5}
        quantiles = [None] if channel in fixed else experiment["calibration_quantiles"]
        for q in quantiles:
            threshold = fixed[channel] if q is None else float(blocks.quantile(q))
            alerts = make_incidents(verification, channel, threshold, cadence)
            exposure = verification[channel].notna().sum() * cadence / 86400
            upper = float(1000 * poisson_rate_interval(len(alerts), exposure)[1])
            support = len(blocks) if q is None else len(blocks) * (1 - q)
            coverage = verification[channel].notna().mean()
            admissible = (
                support >= 5
                and np.isfinite(upper)
                and upper <= experiment["verification_budget_per_1000_days"]
                and coverage >= experiment["minimum_score_coverage"]
            )
            selected = {
                "channel": channel,
                "threshold": threshold,
                "quantile": q,
                "daily_blocks": len(blocks),
                "tail_support": support,
                "verification_upper": upper if np.isfinite(upper) else None,
                "coverage": coverage,
                "admissible": bool(admissible),
                "reason": (
                    "declared_fixed_rule"
                    if q is None
                    else "empirical_blocks_no_tail_extrapolation"
                ),
            }
            if admissible:
                break
        results.append(selected)
    return results


def add_portfolio(scores, choices):
    """Combine independently calibrated evidence only at the alert layer."""
    result = scores.copy()
    votes = []
    for choice in choices:
        channel = choice["channel"]
        if (
            channel in {"full", "isolation_forest", "rules"}
            or choice["threshold"] is None
        ):
            continue
        values = result[channel]
        votes.append(values.gt(choice["threshold"]).astype(float).where(values.notna()))
    result["full"] = pd.concat(votes, axis=1).max(axis=1) if votes else np.nan
    return result


def portfolio_choice(scores, manifest, experiment):
    start, end = split_boundaries(manifest)["verification"]
    subset = scores.loc[scores.timestamp_utc.between(start, end, inclusive="left")]
    cadence = manifest["config"]["sample_minutes"] * 60
    alerts = make_incidents(subset, "full", 0.5, cadence)
    exposure = subset.full.notna().sum() * cadence / 86400
    upper = float(1000 * poisson_rate_interval(len(alerts), exposure)[1])
    return {
        "channel": "full",
        "threshold": 0.5,
        "admissible": bool(
            np.isfinite(upper)
            and upper <= experiment["verification_budget_per_1000_days"]
            and subset.full.notna().mean() >= experiment["minimum_score_coverage"]
        ),
        "verification_upper": upper if np.isfinite(upper) else None,
        "reason": "union_of_independently_thresholded_channels",
    }


def evaluate_queue(truth, queue, start, end, prompt_hours):
    cases = queue.copy()
    cases["start_ts"] = cases.decision_ts
    cases["entity_ids"] = cases.affected_entities.map(json.loads)
    return evaluate(truth, cases, start, end, prompt_hours)


def write_capability(outcomes, output):
    rows = []
    from .evaluation import wilson_interval

    for kind, group in outcomes.groupby("type"):
        count, hits = len(group), int(group.detected.sum())
        low, high = wilson_interval(hits, count)
        rows.append(
            {
                "mechanism": kind,
                "faults": count,
                "detected": hits,
                "recall": hits / count,
                "lower": low,
                "upper": high,
                "status": "estimable" if (high - low) / 2 <= 0.15 else "descriptive",
            }
        )
    pd.DataFrame(
        rows,
        columns=[
            "mechanism",
            "faults",
            "detected",
            "recall",
            "lower",
            "upper",
            "status",
        ],
    ).to_csv(output, index=False)


def develop(policy):
    """Freeze label-free candidates before any development truth is opened."""
    output = Path(policy["run"])
    if output.exists():
        raise FileExistsError(
            "Run exists; inspect its receipt or choose a new run path"
        )
    experiment = yaml.safe_load(Path(policy["experiment"]).read_text())
    validate_experiment(experiment)
    grid, manifest = observations(policy)
    partitions = split_boundaries(manifest)
    decisions, seasonal_evidence = fit_seasonality(
        grid, partitions["train"][1], policy["timezone"]
    )
    features = featurize(grid, manifest, experiment, decisions)
    fitted = fit_baselines(
        features.loc[features.timestamp_utc < partitions["train"][1]],
        experiment["seed"],
    )
    bundle = {"fitted": fitted, "decisions": decisions, "policy": policy}
    scores = score(grid, manifest, experiment, bundle)
    choices = calibrate(scores, manifest, experiment, policy)
    scores = add_portfolio(scores, choices)
    choices.append(portfolio_choice(scores, manifest, experiment))
    output.mkdir(parents=True)
    receipt = freeze_inputs(policy, experiment, manifest)
    write_json(output / "resolved_policy.json", receipt)
    write_json(output / "label_free_choices.json", choices)
    write_json(output / "eda_decisions.json", decisions)
    seasonal_evidence.to_csv(output / "seasonality.csv", index=False)
    grid.groupby(["ont_id", "gap_kind"]).size().rename("rows").to_csv(
        output / "gap_audit.csv"
    )
    joblib.dump(bundle, output / "model.joblib")
    scores.to_parquet(output / "development_scores.parquet", index=False)
    # Re-score the saved artifact through the same function before opening truth.
    replay = score(grid, manifest, experiment, joblib.load(output / "model.joblib"))
    replay = add_portfolio(replay, choices)
    pd.testing.assert_frame_equal(scores, replay, check_exact=True)
    attempt = ledger(
        policy["truth"],
        "selection",
        {
            "run": str(output),
            "policy_sha256": fingerprint(receipt),
            "label_free_choices_sha256": sha256(output / "label_free_choices.json"),
            "candidates": len(choices),
        },
        policy["maximum_development_attempts"],
    )
    verify_truth(policy["truth"])
    inventory = pd.read_parquet(Path(policy["pack"]) / "inventory.parquet")
    events = pd.read_parquet(Path(policy["pack"]) / "events.parquet")
    start, end = partitions["development"]
    dev = scores.loc[scores.timestamp_utc.between(start, end, inclusive="left")]
    comparisons = []
    for choice in choices:
        channel = choice["channel"]
        if choice["threshold"] is None:
            comparisons.append(
                {"channel": channel, "qualified": False, "reason": choice["reason"]}
            )
            continue
        alerts = make_incidents(
            dev, channel, choice["threshold"], manifest["config"]["sample_minutes"] * 60
        )
        queue = incident_queue(alerts, inventory, events, policy)
        metrics, outcomes = evaluate_queue(
            policy["truth"], queue, start, end, experiment["prompt_hours"]
        )
        alerts.to_csv(output / f"{channel}_alerts.csv", index=False)
        queue.to_csv(output / f"{channel}_queue.csv", index=False)
        outcomes.to_csv(output / f"{channel}_faults.csv", index=False)
        write_capability(outcomes, output / f"{channel}_capability.csv")
        lower = metrics["event_recall_lower"]
        qualified = (
            choice["admissible"]
            and lower is not None
            and lower >= experiment["minimum_event_recall_lower_bound"]
            and metrics["nuisance_upper_per_1000_days"]
            <= experiment["development_budget_per_1000_days"]
            and dev[channel].notna().mean() >= experiment["minimum_score_coverage"]
        )
        comparisons.append(
            {
                "channel": channel,
                "qualified": bool(qualified),
                "queue_cases": len(queue),
                **metrics,
            }
        )
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(output / "comparison.csv", index=False)
    eligible = comparison.loc[comparison.qualified]
    if len(eligible):
        # Parsimony: first declared channel within the best candidate's recall interval.
        best = eligible.loc[eligible.event_recall.idxmax()]
        eligible = eligible.loc[eligible.event_recall >= best.event_recall_lower]
        channel = eligible.channel.iloc[0]
        selected = next(c for c in choices if c["channel"] == channel)
        write_json(output / "selected_configuration.json", selected)
    write_json(
        output / "selection_status.json",
        {
            "status": "selected" if len(eligible) else "STOP_no_qualified_candidate",
            "attempt": attempt,
            "holdout_opened": False,
        },
    )
    frozen = [
        "resolved_policy.json",
        "label_free_choices.json",
        "eda_decisions.json",
        "model.joblib",
        "development_scores.parquet",
    ]
    if len(eligible):
        frozen.append("selected_configuration.json")
    write_json(
        output / "manifest.json",
        {"files": {name: sha256(output / name) for name in frozen}},
    )
    (output / "model_card.md").write_text(
        "# Synthetic development model card\n\n"
        f"Selection attempt: {attempt}. Candidates: {len(choices)}.\n\n"
        f"Status: {'selected' if len(eligible) else 'STOP — no qualified candidate'}.\n\n"
        "Evidence: synthetic development only. Final test unopened.\n"
        "Scope probabilities are uncalibrated, conditional single-fault rankings.\n"
        "Power disposition has no generator-derived accuracy evidence.\n"
        "Per-mechanism support is reported; independent generator, operator transfer,\n"
        "public-data and shadow-deployment gates remain uncompleted.\n"
    )
    return comparison


def frozen_run(run_directory):
    run = Path(run_directory)
    receipt = json.loads((run / "resolved_policy.json").read_text())
    if receipt["code"] != code_hashes():
        raise ValueError("Pipeline code changed; start a new run")
    for name, digest in json.loads((run / "manifest.json").read_text())[
        "files"
    ].items():
        if sha256(run / name) != digest:
            raise ValueError(f"Frozen artifact changed: {name}")
    pack, truth = Path(receipt["policy"]["pack"]), Path(receipt["policy"]["truth"])
    if sha256(pack / "manifest.json") != receipt["pack_manifest_sha256"]:
        raise ValueError("Model pack changed")
    if sha256(truth / "manifest.json") != receipt["truth_manifest_sha256"]:
        raise ValueError("Truth manifest changed")
    verify_pack(pack)
    return receipt


def run_holdout(run_directory):
    run = Path(run_directory)
    if not (run / "selected_configuration.json").exists():
        raise ValueError("STOP: no qualified selected configuration")
    receipt = frozen_run(run)
    policy, experiment = receipt["policy"], receipt["experiment"]
    # The ledger survives deletion of the result folder and is written before truth reads.
    ledger(
        policy["truth"],
        "holdout",
        {"run": str(run), "selection": sha256(run / "selected_configuration.json")},
        maximum=1,
    )
    verify_truth(policy["truth"])
    output = run / "holdout"
    output.mkdir(exist_ok=False)
    grid, manifest = observations(policy, include_holdout=True)
    bundle = joblib.load(run / "model.joblib")
    scores = score(grid, manifest, experiment, bundle)
    scores = add_portfolio(
        scores, json.loads((run / "label_free_choices.json").read_text())
    )
    start, end = split_boundaries(manifest)["holdout"]
    selected = json.loads((run / "selected_configuration.json").read_text())
    scores = scores.loc[scores.timestamp_utc.between(start, end, inclusive="left")]
    alerts = make_incidents(
        scores,
        selected["channel"],
        selected["threshold"],
        manifest["config"]["sample_minutes"] * 60,
    )
    inventory = pd.read_parquet(Path(policy["pack"]) / "inventory.parquet")
    events = pd.read_parquet(Path(policy["pack"]) / "events.parquet")
    queue = incident_queue(alerts, inventory, events, policy)
    queue.to_csv(output / "incident_queue.csv", index=False)
    metrics, outcomes = evaluate_queue(
        policy["truth"], queue, start, end, experiment["prompt_hours"]
    )
    alerts.to_csv(output / "alerts.csv", index=False)
    outcomes.to_csv(output / "faults.csv", index=False)
    write_json(output / "metrics.json", metrics)
    return metrics


def infer(run_directory, pack_directory, output_directory):
    """Batch inference with frozen calibration. No evaluation truth is opened."""
    run, pack, output = map(Path, (run_directory, pack_directory, output_directory))
    if output.exists():
        raise FileExistsError(output)
    if not (run / "selected_configuration.json").exists():
        raise ValueError("STOP: inference requires a qualified selected configuration")
    receipt = json.loads((run / "resolved_policy.json").read_text())
    if receipt["code"] != code_hashes():
        raise ValueError("Scoring code changed")
    for name, digest in json.loads((run / "manifest.json").read_text())[
        "files"
    ].items():
        if sha256(run / name) != digest:
            raise ValueError(f"Frozen artifact changed: {name}")
    data, manifest = load_observations(pack, include_holdout=True)
    cfg = manifest["config"]
    original = receipt["observation_window"]
    end_of_experiment = pd.Timestamp(original["start"]) + pd.Timedelta(
        days=original["days"]
    )
    if pd.Timestamp(cfg["start"]) < end_of_experiment:
        raise ValueError(
            "Inference must follow the experiment window; use locked evaluation for holdout"
        )
    if cfg["sample_minutes"] != original["sample_minutes"]:
        raise ValueError("Cadence changed; recalibration is required")
    inventory = pd.read_parquet(pack / "inventory.parquet")
    events = pd.read_parquet(pack / "events.parquet")
    start = pd.Timestamp(cfg["start"])
    end = start + pd.Timedelta(days=cfg["days"])
    grid = regularize(data, inventory, start, end, cfg["sample_minutes"] * 60)
    bundle = joblib.load(run / "model.joblib")
    scores = score(grid, manifest, receipt["experiment"], bundle)
    choices = json.loads((run / "label_free_choices.json").read_text())
    scores = add_portfolio(scores, choices)
    selected = json.loads((run / "selected_configuration.json").read_text())
    alerts = make_incidents(
        scores, selected["channel"], selected["threshold"], cfg["sample_minutes"] * 60
    )
    queue = incident_queue(alerts, inventory, events, bundle["policy"])
    output.mkdir(parents=True)
    scores.to_parquet(output / "scores.parquet", index=False)
    alerts.to_csv(output / "alerts.csv", index=False)
    queue.to_csv(output / "incident_queue.csv", index=False)
    write_json(
        output / "manifest.json",
        {
            "pack_sha256": sha256(pack / "manifest.json"),
            "model_sha256": sha256(run / "model.joblib"),
            "truth_read": False,
        },
    )
    return queue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["prepare", "develop", "all", "holdout", "infer"]
    )
    parser.add_argument("--config", default="configs/pipeline.yml")
    parser.add_argument("--input-pack")
    parser.add_argument("--output")
    args = parser.parse_args()
    policy = runtime(args.config)
    if args.command in {"prepare", "all"}:
        prepare(policy)
    if args.command in {"develop", "all"}:
        print(develop(policy).to_string(index=False))
    if args.command == "infer":
        if not args.input_pack or not args.output:
            parser.error("infer requires --input-pack and --output")
        print(infer(policy["run"], args.input_pack, args.output).to_string(index=False))
    if args.command == "holdout":
        print(run_holdout(policy["run"]))


if __name__ == "__main__":
    main()
