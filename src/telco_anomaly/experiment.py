"""Small synthetic benchmark: one detector, two comparators, untouched final data."""

from pathlib import Path
import json

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .adapter import adapt
from .model import POWER, fit_reference, score, calibrate, warnings
from .evaluation import evaluate
from .synthetic import sha256


def split_times(dataset):
    config = json.loads((Path(dataset) / "manifest.json").read_text())["config"]
    start = pd.Timestamp(config["start"])
    end = lambda f: start + pd.Timedelta(days=config["days"] * f)
    return config, {
        "start": start,
        "fit_end": end(0.4),
        "calibration_end": end(0.7),
        "development_end": end(0.85),
        "end": end(1),
    }


def load_data(dataset, end):
    """Read model inputs only; never bring future rows or labels into fitting."""
    with duckdb.connect() as connection:
        data = connection.execute(
            "SELECT timestamp_utc, ont_id, rx_power_dbm, olt_rx_power_dbm "
            "FROM read_parquet(?) WHERE timestamp_utc < ? ORDER BY ont_id, timestamp_utc",
            [str(Path(dataset) / "reference_dataset.parquet"), end],
        ).df()
    return adapt(data)


def forest_scores(scores, fit_end, seed=42):
    """Small challenger on the SAME optical deviations, without entity IDs."""
    columns = [p + suffix for p in POWER for suffix in ("__z", "__ewma")]
    training = scores.loc[scores.timestamp_utc < fit_end, columns]
    medians = training.median()
    columns = medians.dropna().index.tolist()
    if not columns:
        raise ValueError("No supported features for Isolation Forest")
    medians = medians[columns]
    training = training[columns].dropna(how="all")
    forest = IsolationForest(
        n_estimators=100, max_samples=256, random_state=seed, n_jobs=1
    )
    forest.fit(
        training.sample(min(20000, len(training)), random_state=seed).fillna(medians)
    )
    result = pd.Series(np.nan, index=scores.index)
    ready = scores.score.notna()
    result.loc[ready] = -forest.score_samples(
        scores.loc[ready, columns].fillna(medians)
    )
    return result, {"forest": forest, "columns": columns, "medians": medians}


def benchmark(dataset, output, smoothing_hours=1, sensitivity=6, static_limit_dbm=-27):
    """No automatic qualification. Save evidence and limitations for human review."""
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise FileExistsError(output)
    config, times = split_times(dataset)
    data = load_data(dataset, times["development_end"])
    model = fit_reference(
        data.loc[data.timestamp_utc < times["fit_end"]],
        config["sample_minutes"],
        smoothing_hours,
    )
    scores = score(data, model)
    scores["isolation_forest"], forest = forest_scores(scores, times["fit_end"])
    fit_cal = scores.timestamp_utc.between(
        times["fit_end"], times["calibration_end"], inclusive="left"
    )
    development = scores.timestamp_utc.between(
        times["calibration_end"], times["development_end"], inclusive="left"
    )
    # Decisions are fixed before evaluation reads any fault labels.
    thresholds = {
        "ewma": calibrate(scores.loc[fit_cal, "score"], sensitivity, minimum=3),
        "isolation_forest": calibrate(
            scores.loc[fit_cal, "isolation_forest"], sensitivity, minimum=0
        ),
        "static": -static_limit_dbm,
    }
    columns = {
        "ewma": "score",
        "isolation_forest": "isolation_forest",
        "static": "static",
    }
    centres = {
        name: float(scores.loc[fit_cal, column].median())
        for name, column in columns.items()
    }
    recovery = {
        name: min(thresholds[name], (thresholds[name] + centres[name]) / 2)
        for name in thresholds
    }
    output.mkdir(parents=True)
    model.update(
        threshold=thresholds["ewma"],
        recovery_threshold=recovery["ewma"],
        fitted_until=str(times["fit_end"]),
        calibrated_until=str(times["calibration_end"]),
    )
    (output / "model.json").write_text(json.dumps(model, indent=2))
    joblib.dump(forest, output / "forest_comparator.joblib")
    settings = {
        "smoothing_hours": smoothing_hours,
        "sensitivity": sensitivity,
        "thresholds": thresholds,
        "recovery_thresholds": recovery,
        "time_splits": {key: str(value) for key, value in times.items()},
        "dataset": str(dataset.resolve()),
        "source_sha256": sha256(dataset / "reference_dataset.parquet"),
        "model_sha256": sha256(output / "model.json"),
        "code_sha256": {
            name: sha256(Path(__file__).with_name(name))
            for name in ("model.py", "adapter.py", "evaluation.py", "experiment.py")
        },
        "truth_sha256": {
            name: sha256(dataset / name)
            for name in (
                "manifest.json",
                "gt_fault_registry.csv",
                "fault_entity_intervals.csv",
                "topology.csv",
            )
        },
        "development_only": True,
    }
    (output / "settings.json").write_text(json.dumps(settings, indent=2))
    results = []
    dev = scores.loc[development].copy()
    dev.to_parquet(output / "scores.parquet", index=False)
    for name, column in columns.items():
        values = dev[["timestamp_utc", "ont_id", column]].rename(
            columns={column: "score"}
        )
        alerts = warnings(
            values,
            thresholds[name],
            config["sample_minutes"],
            recovery_fraction=recovery[name] / thresholds[name],
        )
        metrics, outcomes = evaluate(
            dataset, alerts, times["calibration_end"], times["development_end"]
        )
        expected = (
            config["n_onts"]
            * (times["development_end"] - times["calibration_end"]).total_seconds()
            / (config["sample_minutes"] * 60)
        )
        metrics["score_coverage"] = float(values.score.notna().sum() / expected)
        metrics["model"] = name
        results.append(metrics)
        alerts.to_csv(output / f"{name}_warnings.csv", index=False)
        outcomes.to_csv(output / f"{name}_faults.csv", index=False)
        breakdown = outcomes.groupby("type").agg(
            faults=("detected", "size"),
            detected=("detected", "sum"),
            impacting=("has_impact", "sum"),
            early=("early", "sum"),
        )
        breakdown.to_csv(output / f"{name}_by_type.csv")
    comparison = pd.DataFrame(results)
    comparison.to_csv(output / "comparison.csv", index=False)
    return comparison


def final_evaluation(run):
    """Explicit one-shot final test of the frozen EWMA detector, not its comparators."""
    run = Path(run)
    settings = json.loads((run / "settings.json").read_text())
    dataset = Path(settings["dataset"])
    if sha256(dataset / "reference_dataset.parquet") != settings["source_sha256"]:
        raise ValueError("Source changed")
    if sha256(run / "model.json") != settings["model_sha256"]:
        raise ValueError("Model changed")
    for name, digest in settings["code_sha256"].items():
        if sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError(f"Code changed: {name}")
    for name, digest in settings["truth_sha256"].items():
        if sha256(dataset / name) != digest:
            raise ValueError(f"Evaluation input changed: {name}")
    with (run / "FINAL_TEST_OPENED.json").open("x") as stream:
        json.dump({"opened_at": str(pd.Timestamp.now(tz="UTC"))}, stream)
    config, times = split_times(dataset)
    model = json.loads((run / "model.json").read_text())
    scores = score(load_data(dataset, times["end"]), model)
    final = scores.loc[scores.timestamp_utc >= times["development_end"]]
    alerts = warnings(
        final,
        model["threshold"],
        model["cadence_minutes"],
        recovery_fraction=model["recovery_threshold"] / model["threshold"],
    )
    metrics, outcomes = evaluate(
        dataset, alerts, times["development_end"], times["end"]
    )
    alerts.to_csv(run / "final_warnings.csv", index=False)
    outcomes.to_csv(run / "final_faults.csv", index=False)
    (run / "final_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def robustness_suite(config, output):
    """New seeds plus noise, missingness and cadence stress; same untuned model settings."""
    from dataclasses import replace
    from .synthetic import generate_dataset
    from .synthetic_validation import validate_dataset

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    scenarios = {
        "new_seed": replace(config, seed=config.seed + 1, n_onts=48),
        "noisier_missing": replace(
            config,
            seed=config.seed + 2,
            n_onts=48,
            sensor_noise_db=0.3,
            poll_loss_probability=0.1,
        ),
        "hourly": replace(config, seed=config.seed + 3, n_onts=48, sample_minutes=60),
        "healthy": replace(
            config,
            seed=config.seed + 4,
            n_onts=48,
            faults_per_ont_year=0,
            faults_per_splitter_year=0,
        ),
    }
    results = []
    for name, scenario in scenarios.items():
        dataset = generate_dataset(scenario, output / name / "data")
        checks = validate_dataset(dataset)["checks"]
        if not checks.status.eq("pass").all():
            raise ValueError(f"Invalid scenario: {name}")
        result = benchmark(dataset, output / name / "results")
        result["scenario"] = name
        results.append(result)
    result = pd.DataFrame(
        [row for frame in results for row in frame.to_dict("records")]
    )
    result.to_csv(output / "comparison.csv", index=False)
    return result
