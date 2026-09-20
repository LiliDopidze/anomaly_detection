"""Small orchestration: fit, calibrate, tune only on validation, seal final test."""

from pathlib import Path
import hashlib
import json
import joblib
import pandas as pd
import yaml
from .generator import GeneratorConfig, generate, make_topology
from .optics import validate_generated
from .multivariate import FEATURE_SETS, columns_for
from .workflow import prepare_canonical, feature_data, run_split
from .detectors import StatisticalDetector, IsolationForestDetector
from .incidents import IncidentManager
from .evaluation import Evaluator
from .splitting import TemporalSplit


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def score_detectors(
    features: pd.DataFrame,
    statistical: StatisticalDetector,
    forest: IsolationForestDetector,
) -> pd.DataFrame:
    scores = features[["timestamp", "entity_id"]].copy()
    scores["statistical"] = statistical.score(features)
    scores["isolation_forest"] = forest.score(features)
    scores["combined"] = scores[["statistical", "isolation_forest"]].max(
        axis=1, skipna=False
    )
    return scores


def incidents_for(scores: pd.DataFrame, policy: dict, interval: int) -> pd.DataFrame:
    values = scores[["timestamp", "entity_id", policy["detector"]]].rename(
        columns={policy["detector"]: "score"}
    )
    return IncidentManager(
        high=policy["high"],
        low=policy["low"],
        opening_intervals=policy["opening_intervals"],
        closing_intervals=policy["closing_intervals"],
        interval_minutes=interval,
    ).process(values)


def tune(
    scores: pd.DataFrame,
    truth: pd.DataFrame,
    telemetry: pd.DataFrame,
    split: TemporalSplit,
    settings: dict,
    interval: int,
) -> pd.DataFrame:
    rows = []
    evaluator = Evaluator(interval_minutes=interval)
    for detector in ("statistical", "isolation_forest", "combined"):
        for opening in settings["opening_candidates"]:
            for closing in settings["closing_candidates"]:
                policy = {
                    "detector": detector,
                    "high": settings["high"],
                    "low": settings["low"],
                    "opening_intervals": opening,
                    "closing_intervals": closing,
                }
                alerts = incidents_for(scores, policy, interval)
                metrics, _ = evaluator.evaluate(
                    alerts,
                    truth,
                    telemetry,
                    split.calibration_end,
                    split.validation_end,
                )
                metrics["score_coverage"] = float(scores[detector].notna().mean())
                rows.append({**policy, **metrics})
    table = pd.DataFrame(rows)
    table["meets_workload_budget"] = table.nuisance_per_1000_entity_days.le(
        settings["nuisance_budget_per_1000_days"]
    )
    # Fixed selection rule: feasible workload first, early recall, then workload.
    return table.sort_values(
        [
            "meets_workload_budget",
            "pre_impact_recall",
            "nuisance_per_1000_entity_days",
            "detector",
            "opening_intervals",
            "closing_intervals",
        ],
        ascending=[False, False, True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def prepare(config_path: str | Path) -> Path:
    """Generate once; preserve existing fixtures only when settings match."""
    settings = yaml.safe_load(Path(config_path).read_text())
    config = GeneratorConfig(**settings["generator"])
    fractions = settings["splits"]
    if len(fractions) != 4 or not (
        0 < fractions[0] < fractions[1] <= 0.55 < fractions[2] < fractions[3] == 1
    ):
        raise ValueError("Use increasing splits, calibration <=0.55 and final end=1")
    run = Path(settings["output"])
    if run.exists():
        previous = json.loads((run / "settings.json").read_text())
        if previous != settings:
            raise ValueError("Settings changed; choose a new output folder")
        return run
    run.mkdir(parents=True)
    native, truth = generate(config)
    topology = make_topology(config)
    report = validate_generated(native, truth, topology)
    topology.to_parquet(run / "topology.parquet", index=False)
    (run / "generation_checks.json").write_text(json.dumps(report, indent=2))
    native.to_parquet(run / "telemetry.parquet", index=False)
    truth.to_parquet(run / "ground_truth.parquet", index=False)
    (run / "settings.json").write_text(json.dumps(settings, indent=2))
    return run


def develop(config_path: str | Path) -> Path:
    run = prepare(config_path)
    if (run / "model.joblib").exists():
        raise FileExistsError("Model already fitted; choose a new output folder")
    settings = json.loads((run / "settings.json").read_text())
    config = GeneratorConfig(**settings["generator"])
    prepare_canonical(run)
    split = run_split(settings)
    features, telemetry, engineers = feature_data(run, settings)
    masks = split.masks(features.timestamp)
    truth = pd.read_parquet(run / "ground_truth.parquet")
    validation_truth = truth.loc[truth.onset_time < split.validation_end]
    statistical = StatisticalDetector().calibrate(features.loc[masks["calibration"]])
    comparisons, forests = [], {}
    for order, feature_set in enumerate(FEATURE_SETS):
        forest = IsolationForestDetector(feature_columns=columns_for(feature_set))
        forest.fit(features.loc[masks["train"]])
        forest.calibrate(features.loc[masks["calibration"]])
        scores = score_detectors(features, statistical, forest)
        table = tune(
            scores.loc[masks["validation"]],
            validation_truth,
            telemetry,
            split,
            settings["incidents"],
            config.interval_minutes,
        )
        # The unchanged statistical comparator belongs to the Rx-only baseline.
        if feature_set != "rx_only":
            table = table.loc[table.detector.ne("statistical")].copy()
        table["feature_set"] = feature_set
        table["feature_count"] = len(columns_for(feature_set))
        table["complexity_order"] = order
        comparisons.append(table)
        forests[feature_set] = forest
    comparison = (
        pd.concat(comparisons, ignore_index=True)
        .sort_values(
            [
                "meets_workload_budget",
                "pre_impact_recall",
                "nuisance_per_1000_entity_days",
                "complexity_order",
                "detector",
                "opening_intervals",
                "closing_intervals",
            ],
            ascending=[False, False, True, True, True, True, True],
            na_position="last",
        )
        .reset_index(drop=True)
    )
    comparison.to_csv(run / "validation_comparison.csv", index=False)
    comparison.groupby("feature_set", sort=False).head(1).to_csv(
        run / "feature_set_comparison.csv", index=False
    )
    keys = ["detector", "high", "low", "opening_intervals", "closing_intervals"]
    policy = json.loads(comparison.iloc[0][keys].to_json())
    feature_set = str(comparison.iloc[0].feature_set)
    forest = forests[feature_set]
    validation_scores = score_detectors(features, statistical, forest).loc[
        masks["validation"]
    ]
    model = {
        "engineers": engineers,
        "feature_set": feature_set,
        "statistical": statistical,
        "forest": forest,
        "policy": policy,
        "split": split,
        "interval": config.interval_minutes,
    }
    return _save_development(
        run,
        model,
        settings,
        features,
        validation_scores,
        validation_truth,
        telemetry,
        comparison,
    )


def _save_development(
    run: Path,
    model: dict,
    settings: dict,
    features: pd.DataFrame,
    validation_scores: pd.DataFrame,
    validation_truth: pd.DataFrame,
    telemetry: pd.DataFrame,
    comparison: pd.DataFrame,
) -> Path:
    policy, split = model["policy"], model["split"]
    joblib.dump(model, run / "model.joblib")
    features.to_parquet(run / "development_features.parquet", index=False)
    validation_scores.to_parquet(run / "validation_scores.parquet", index=False)
    alerts = incidents_for(validation_scores, policy, model["interval"])
    _, outcomes = Evaluator(model["interval"]).evaluate(
        alerts, validation_truth, telemetry, split.calibration_end, split.validation_end
    )
    alerts.to_csv(run / "validation_incidents.csv", index=False)
    outcomes.to_csv(run / "validation_faults.csv", index=False)
    manifest = {
        "settings": settings,
        "policy": policy,
        "feature_set": model["feature_set"],
        "feature_columns": model["forest"].feature_columns,
        "meets_validation_workload_budget": bool(
            comparison.iloc[0].meets_workload_budget
        ),
        "split_boundaries": [
            str(t)
            for t in (
                split.train_end,
                split.calibration_end,
                split.validation_end,
                split.test_end,
            )
        ],
        "files": {
            name: digest(run / name)
            for name in (
                "telemetry.parquet",
                "ground_truth.parquet",
                "topology.parquet",
                "model.joblib",
                "canonical_development.parquet",
            )
        },
        "code": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return run


def final_evaluation(run: str | Path) -> dict:
    """One explicit final opening of a frozen model; no retraining or tuning."""
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if digest(run / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    for name, expected in manifest["code"].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError(f"Code changed: {name}")
    with (run / "FINAL_OPENED.json").open("x") as stream:
        json.dump({"opened_at": str(pd.Timestamp.now(tz="UTC"))}, stream)
    # Load only trusted, locally generated joblib files.
    model = joblib.load(run / "model.joblib")
    features, telemetry, _ = feature_data(
        run, manifest["settings"], model["engineers"], final=True
    )
    scores = score_detectors(features, model["statistical"], model["forest"])
    split = model["split"]
    test = scores.loc[split.masks(scores.timestamp)["test"]]
    alerts = incidents_for(test, model["policy"], model["interval"])
    truth = pd.read_parquet(run / "ground_truth.parquet")
    metrics, faults = Evaluator(model["interval"]).evaluate(
        alerts, truth, telemetry, split.validation_end, split.test_end
    )
    metrics["score_coverage"] = float(test[model["policy"]["detector"]].notna().mean())
    alerts.to_csv(run / "test_incidents.csv", index=False)
    faults.to_csv(run / "test_faults.csv", index=False)
    (run / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
