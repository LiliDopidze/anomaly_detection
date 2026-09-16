"""Legacy statistical, PCA and seed-sensitivity scorers."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from ._common import (
    IDENTITY_COLUMNS, LEGACY_MODEL_IDS, MODEL_CORE_VERSION,
    PCA_EIGENVALUE_FLOOR, PCA_VARIANCE_TARGET, SCALED_FEATURE_CAP,
    _sql_identifier, _sql_literal, feature_columns, iter_episode_frames,
)

def calibration_sample(path, maximum_rows, random_seed):
    """Return a reproducible reservoir sample without loading the full file."""

    columns = feature_columns(path)
    select = ", ".join(_sql_identifier(column) for column in columns)
    source = _sql_literal(str(Path(path)))
    connection = duckdb.connect()
    frame = connection.execute(f"""
        SELECT {select}
        FROM read_parquet({source})
        USING SAMPLE reservoir({int(maximum_rows)} ROWS)
        REPEATABLE ({int(random_seed)})
    """).df()
    connection.close()
    return frame

def fit_model_bundle(
    calibration_features,
    feature_settings,
    *,
    maximum_training_rows=200_000,
    random_seed=42,
):
    """Fit preprocessing, Isolation Forest and PCA on calibration only."""

    sample = calibration_sample(
        calibration_features, maximum_training_rows, random_seed
    ).replace([np.inf, -np.inf], np.nan)
    usable = [
        column for column in sample
        if sample[column].notna().mean() >= 0.50
        and sample[column].dropna().nunique() > 1
    ]
    if not usable:
        raise ValueError("No usable calibration features")

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler(quantile_range=(25, 75))
    values = imputer.fit_transform(sample[usable])
    scaled = scaler.fit_transform(values)
    scaled = np.clip(scaled, -SCALED_FEATURE_CAP, SCALED_FEATURE_CAP)
    if not np.isfinite(scaled).all() or np.var(scaled, axis=0).sum() == 0:
        raise ValueError("Calibration features have no finite model variance")

    isolation_forest = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination="auto",
        random_state=random_seed,
        n_jobs=-1,
    ).fit(scaled)
    if scaled.shape[1] < 2:
        raise ValueError("PCA baseline requires at least two usable features")
    full_pca = PCA(svd_solver="full").fit(scaled)
    explained = np.cumsum(full_pca.explained_variance_ratio_)
    pca_components = int(np.searchsorted(
        explained, PCA_VARIANCE_TARGET
    ) + 1)
    pca_components = min(pca_components, scaled.shape[1] - 1)
    pca = PCA(n_components=pca_components, svd_solver="full").fit(scaled)

    bundle = {
        "model_core_version": MODEL_CORE_VERSION,
        "feature_settings": dict(feature_settings),
        "feature_columns": usable,
        "imputer": imputer,
        "scaler": scaler,
        "isolation_forest": isolation_forest,
        "pca": pca,
        "pca_variance_target": PCA_VARIANCE_TARGET,
        "scaled_feature_cap": SCALED_FEATURE_CAP,
        "training_rows": len(sample),
        "random_seed": random_seed,
    }
    reference = score_matrix(bundle, sample[usable])
    bundle["score_reference"] = {
        model_id: np.sort(reference[model_id].to_numpy())
        for model_id in LEGACY_MODEL_IDS
    }
    # A PCA that retains almost every direction reconstructs everything, so
    # SPE collapses towards zero and the baseline silently stops alarming.
    spe = reference["pca"].to_numpy()
    upper = float(np.quantile(spe, 0.999))
    if not np.isfinite(upper) or upper <= 0:
        raise ValueError(
            "PCA reconstruction error is degenerate on calibration data; "
            "reduce the retained variance target or the feature count"
        )
    bundle["pca_spe_tail_ratio"] = upper / max(float(np.median(spe)), 1e-12)
    return bundle

def score_matrix(bundle, features):
    """Apply all four models to one feature frame."""

    clean = features[bundle["feature_columns"]].replace(
        [np.inf, -np.inf], np.nan
    )
    values = bundle["imputer"].transform(clean)
    scaled = bundle["scaler"].transform(values)
    if not np.isfinite(scaled).all():
        raise ValueError("Preprocessed model features must be finite")

    # The statistical baseline keeps the full ordering of extreme finite
    # deviations. log1p makes its scale readable without changing ranks.
    statistical = np.max(np.log1p(np.abs(scaled)), axis=1)
    feature_names = np.asarray(bundle["feature_columns"])
    statistical_leading = feature_names[
        np.argmax(np.abs(scaled), axis=1)
    ]

    cap = float(bundle["scaled_feature_cap"])
    model_values = np.clip(scaled, -cap, cap)
    isolation = -bundle["isolation_forest"].decision_function(model_values)

    pca = bundle["pca"]
    projected = pca.transform(model_values)
    reconstructed = pca.inverse_transform(projected)
    residual = model_values - reconstructed
    pca_error = np.mean(residual ** 2, axis=1)
    pca_leading = feature_names[np.argmax(residual ** 2, axis=1)]

    # SPE above answers "has the correlation structure broken?".  T-squared
    # answers "how far along the retained directions has the point moved?".
    # An excursion inside the model subspace is invisible to the first and
    # obvious to the second, so process monitoring reports both.
    eigenvalues = np.maximum(pca.explained_variance_, PCA_EIGENVALUE_FLOOR)
    t_squared = np.sum(projected ** 2 / eigenvalues, axis=1)
    # Exact additive decomposition: contributions sum to T-squared.
    contributions = (projected / eigenvalues) @ pca.components_ * (
        model_values - pca.mean_
    )
    t_squared_leading = feature_names[np.argmax(np.abs(contributions), axis=1)]

    return pd.DataFrame({
        "statistical": statistical,
        "isolation_forest": isolation,
        "pca": pca_error,
        "pca_t2": t_squared,
        "statistical__leading_feature": statistical_leading,
        # Isolation Forest has no faithful built-in local explanation. The
        # largest robust deviation is retained as an explicitly named proxy.
        "isolation_forest__leading_feature": statistical_leading,
        "pca__leading_feature": pca_leading,
        "pca_t2__leading_feature": t_squared_leading,
    })

def score_feature_file(bundle, features_path, destination, batch_rows=100_000):
    """Score a feature file in batches and write one compact score table."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(features_path)
    columns = [*IDENTITY_COLUMNS, *bundle["feature_columns"]]
    writer = None
    try:
        for batch in parquet.iter_batches(batch_size=batch_rows, columns=columns):
            frame = batch.to_pandas()
            scores = score_matrix(bundle, frame[bundle["feature_columns"]])
            if not np.isfinite(scores[list(LEGACY_MODEL_IDS)].to_numpy()).all():
                raise ValueError("A model produced a non-finite anomaly score")
            output = pd.concat(
                [frame[IDENTITY_COLUMNS].reset_index(drop=True), scores], axis=1
            )
            table = pa.Table.from_pandas(output, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("Feature file produced no scores")
    return destination

def fit_isolation_seed_models(
    bundle,
    calibration_features,
    seeds,
    *,
    maximum_training_rows=100_000,
    sample_seed=42,
):
    """Refit Isolation Forest seeds on one fixed calibration sample."""

    sample = calibration_sample(
        calibration_features, maximum_training_rows, sample_seed
    )[bundle["feature_columns"]].replace([np.inf, -np.inf], np.nan)
    values = bundle["imputer"].transform(sample)
    scaled = np.clip(
        bundle["scaler"].transform(values),
        -float(bundle["scaled_feature_cap"]),
        float(bundle["scaled_feature_cap"]),
    )
    return {
        f"if_seed_{int(seed)}": IsolationForest(
            n_estimators=200,
            max_samples="auto",
            contamination="auto",
            random_state=int(seed),
            n_jobs=-1,
        ).fit(scaled)
        for seed in seeds
    }

def fit_isolation_feature_subset(
    bundle,
    calibration_features,
    excluded_features,
    *,
    maximum_training_rows=100_000,
    random_seed=42,
    model_id="isolation_forest_feature_subset",
):
    """Fit one diagnostic Isolation Forest after removing named features."""

    excluded = set(excluded_features)
    kept = [
        name for name in bundle["feature_columns"]
        if name not in excluded
    ]
    if not excluded or len(kept) == len(bundle["feature_columns"]):
        raise ValueError("No fitted feature was excluded")
    if not kept:
        raise ValueError("Feature sensitivity removed every fitted feature")

    sample = calibration_sample(
        calibration_features, maximum_training_rows, random_seed
    )[bundle["feature_columns"]].replace([np.inf, -np.inf], np.nan)
    values = bundle["imputer"].transform(sample)
    scaled = bundle["scaler"].transform(values)
    indices = [bundle["feature_columns"].index(name) for name in kept]
    model_values = np.clip(
        scaled[:, indices],
        -float(bundle["scaled_feature_cap"]),
        float(bundle["scaled_feature_cap"]),
    )
    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination="auto",
        random_state=int(random_seed),
        n_jobs=-1,
    ).fit(model_values)
    return {
        "model_id": str(model_id),
        "model": model,
        "feature_columns": kept,
        "feature_indices": indices,
        "excluded_features": sorted(excluded),
    }

def score_isolation_feature_subset_file(
    bundle,
    sensitivity,
    features_path,
    destination,
    batch_rows=100_000,
):
    """Score a diagnostic feature-subset Isolation Forest in batches."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(features_path)
    columns = [*IDENTITY_COLUMNS, *bundle["feature_columns"]]
    writer = None
    try:
        for batch in parquet.iter_batches(batch_size=batch_rows, columns=columns):
            frame = batch.to_pandas()
            clean = frame[bundle["feature_columns"]].replace(
                [np.inf, -np.inf], np.nan
            )
            values = bundle["imputer"].transform(clean)
            scaled = bundle["scaler"].transform(values)
            subset = scaled[:, sensitivity["feature_indices"]]
            subset = np.clip(
                subset,
                -float(bundle["scaled_feature_cap"]),
                float(bundle["scaled_feature_cap"]),
            )
            score = -sensitivity["model"].decision_function(subset)
            names = np.asarray(sensitivity["feature_columns"])
            leading = names[np.argmax(np.abs(subset), axis=1)]
            output = frame[IDENTITY_COLUMNS].reset_index(drop=True)
            output[sensitivity["model_id"]] = score
            output[f"{sensitivity['model_id']}__leading_feature"] = leading
            table = pa.Table.from_pandas(output, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    destination, table.schema, compression="zstd"
                )
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("Feature file produced no sensitivity scores")
    return destination

def score_isolation_seed_file(
    bundle,
    seed_models,
    features_path,
    destination,
    batch_rows=100_000,
):
    """Score several Isolation Forest seeds in one feature-file pass."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(features_path)
    columns = [*IDENTITY_COLUMNS, *bundle["feature_columns"]]
    feature_names = np.asarray(bundle["feature_columns"])
    writer = None
    try:
        for batch in parquet.iter_batches(batch_size=batch_rows, columns=columns):
            frame = batch.to_pandas()
            clean = frame[bundle["feature_columns"]].replace(
                [np.inf, -np.inf], np.nan
            )
            values = bundle["imputer"].transform(clean)
            scaled = bundle["scaler"].transform(values)
            model_values = np.clip(
                scaled,
                -float(bundle["scaled_feature_cap"]),
                float(bundle["scaled_feature_cap"]),
            )
            leading = feature_names[np.argmax(np.abs(scaled), axis=1)]
            output = frame[IDENTITY_COLUMNS].reset_index(drop=True)
            for model_id, model in seed_models.items():
                output[model_id] = -model.decision_function(model_values)
                output[f"{model_id}__leading_feature"] = leading
            table = pa.Table.from_pandas(output, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    destination, table.schema, compression="zstd"
                )
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("Feature file produced no seed-stability scores")
    return destination

def score_percentiles(bundle, model_id, values):
    reference = np.asarray(bundle["score_reference"][model_id])
    return np.searchsorted(reference, np.asarray(values), side="right") / len(reference)
