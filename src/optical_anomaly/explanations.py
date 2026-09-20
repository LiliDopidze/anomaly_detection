"""Validation-only SHAP reports; no feature selection or detector changes."""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from .detectors import IsolationForestDetector


def explain_scores(
    detector: IsolationForestDetector,
    background: pd.DataFrame,
    observations: pd.DataFrame,
    permutations: int = 2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Explain actual negative score_samples, not tree path length or alert state.

    Independent background masking can create unlikely correlated combinations.
    Attributions describe this model and background, not physical causes.
    """
    import shap

    columns = detector.feature_columns
    reference = background[columns]
    values = observations[columns]
    if permutations < 1 or reference.empty or values.empty:
        raise ValueError("Need background, observations and positive permutations")
    if not np.isfinite(reference).all().all() or not np.isfinite(values).all().all():
        raise ValueError("SHAP requires complete finite feature vectors")

    def predict(array: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(array, columns=columns)
        return -detector.forest.score_samples(frame)

    explainer = shap.PermutationExplainer(predict, reference, seed=seed)
    result = explainer(
        values, max_evals=permutations * (2 * len(columns) + 1), silent=True
    )
    contributions = pd.DataFrame(result.values, columns=columns, index=values.index)
    predictions = predict(values.to_numpy())
    reconstructed = result.base_values + contributions.sum(axis=1).to_numpy()
    np.testing.assert_allclose(reconstructed, predictions, atol=1e-7, rtol=1e-6)
    checks = observations[["entity_id", "timestamp"]].copy()
    checks["raw_anomaly_score"] = predictions
    checks["background_score"] = result.base_values
    checks["reconstructed_score"] = reconstructed
    return contributions, checks


def explain_run(
    run: str | Path,
    feature_set: str | None = None,
    sample_size: int = 64,
    background_size: int = 16,
    local_size: int = 8,
    permutations: int = 2,
    seed: int = 42,
) -> Path:
    """Use training background and validation examples; final test stays sealed."""
    from .pipeline import digest

    if min(sample_size, background_size, local_size) < 1:
        raise ValueError("Sample sizes must be positive")
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    for name in ("model.joblib", "development_features.parquet"):
        if (
            name not in manifest["files"]
            or digest(run / name) != manifest["files"][name]
        ):
            raise ValueError(f"Missing or changed frozen explanation input: {name}")
    model = joblib.load(run / "model.joblib")  # Trusted local artifacts only.
    name = feature_set or model["feature_set"]
    detector = model["forests"][name]
    columns = detector.feature_columns
    data = pd.read_parquet(run / "development_features.parquet")
    masks = model["split"].masks(data.timestamp)
    training = data.loc[masks["train"]].dropna(subset=columns)
    validation = data.loc[masks["validation"]].dropna(subset=columns)
    if training.empty or validation.empty:
        raise ValueError("No complete training or validation feature rows")
    background = training.sample(min(background_size, len(training)), random_state=seed)
    sample = validation.sample(min(sample_size, len(validation)), random_state=seed)
    # High scores are a separate local explanation set, never global-importance data.
    raw_scores = detector.raw_score(validation)
    local = validation.loc[raw_scores.nlargest(local_size).index]
    target = run / "explanations" / name
    target.mkdir(parents=True, exist_ok=True)
    for label, rows in (("sample", sample), ("high_score", local)):
        contributions, checks = explain_scores(
            detector, background, rows, permutations=permutations, seed=seed
        )
        rows.to_parquet(target / f"{label}_features.parquet", index=False)
        checks.to_csv(target / f"{label}_scores.csv", index=False)
        pd.concat([rows[["entity_id", "timestamp"]], contributions], axis=1).to_csv(
            target / f"{label}_shap.csv", index=False
        )
        if label == "sample":
            importance = pd.DataFrame(
                {
                    "feature": columns,
                    "mean_absolute_shap": contributions.abs().mean().to_numpy(),
                    "mean_signed_shap": contributions.mean().to_numpy(),
                }
            )
            importance.sort_values("mean_absolute_shap", ascending=False).to_csv(
                target / "importance.csv", index=False
            )
    validation[columns].corr(method="spearman").to_csv(target / "correlations.csv")
    background.to_parquet(target / "background.parquet", index=False)
    metadata = {
        "feature_set": name,
        "feature_columns": columns,
        "seed": seed,
        "permutations": permutations,
        "background_rows": len(background),
        "sample_rows": len(sample),
        "high_score_rows": len(local),
        "validation_complete_fraction": len(validation)
        / int(masks["validation"].sum()),
        "explained_output": "negative IsolationForest.score_samples",
        "active_incident_detector": model["policy"]["detector"],
        "automatic_feature_removal": False,
        "interpretation": (
            "Positive SHAP raises raw anomaly score. Not a probability, causal "
            "effect or explanation of incident hysteresis. Correlated features "
            "share credit; independent masking can produce implausible combinations. "
            "Global means cover uniformly sampled complete validation rows only."
        ),
    }
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return target
