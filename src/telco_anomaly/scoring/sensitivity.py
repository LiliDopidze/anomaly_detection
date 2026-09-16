"""Label-free sensitivity checks for calibration sampling choices."""

from __future__ import annotations

import gc
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..evaluation import form_cases
from ..pipeline.materialize import partition_exposure, split_feature_file_by_time
from ._common import _model_duckdb, _sql_literal
from .alerts import alerts_from_score_file
from .isolation_forest import fit_residual_bundle, score_residual_file
from .thresholds import calibration_thresholds


def _bounded_entity_panel(source, destination, maximum_rows, random_seed):
    """Materialise complete entities while keeping a diagnostic row budget."""

    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_sql = _sql_literal(str(source))
    destination_sql = _sql_literal(str(destination))
    with _model_duckdb(destination.parent) as connection:
        connection.execute(f"""
            COPY (
                WITH entity_counts AS (
                    SELECT CAST(entity_id AS VARCHAR) AS entity_id,
                           count(*) AS entity_rows
                    FROM read_parquet({source_sql})
                    GROUP BY entity_id
                ), ordered AS (
                    SELECT *,
                           row_number() OVER (
                               ORDER BY hash(entity_id, {int(random_seed)}),
                                        entity_id
                           ) AS entity_rank,
                           sum(entity_rows) OVER (
                               ORDER BY hash(entity_id, {int(random_seed)}),
                                        entity_id
                               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                           ) AS cumulative_rows
                    FROM entity_counts
                ), selected AS (
                    SELECT entity_id
                    FROM ordered
                    WHERE cumulative_rows <= {int(maximum_rows)}
                       OR entity_rank = 1
                )
                SELECT source.*
                FROM read_parquet({source_sql}) AS source
                JOIN selected
                  ON CAST(source.entity_id AS VARCHAR) = selected.entity_id
                ORDER BY source.entity_id, source.episode_id, source.event_ts
            ) TO {destination_sql} (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
    return pq.ParquetFile(destination).metadata.num_rows


def _score_keys_and_values(path, model_id):
    frame = pd.read_parquet(
        path,
        columns=["event_ts", "entity_id", "episode_id", model_id],
    )
    key_hash = pd.util.hash_pandas_object(
        frame[["event_ts", "entity_id", "episode_id"]], index=False
    ).to_numpy()
    return key_hash, pd.to_numeric(frame[model_id], errors="coerce").to_numpy()


def _rank_correlation(reference, candidate):
    valid = np.isfinite(reference) & np.isfinite(candidate)
    if valid.sum() < 3:
        return np.nan
    return pd.Series(reference[valid]).corr(
        pd.Series(candidate[valid]), method="spearman"
    )


def run_sampling_sensitivity(
    calibration_fit,
    calibration_late,
    output_directory,
    *,
    fit_kwargs,
    cadence_seconds,
    dispersion_window_seconds,
    cusum_allowance,
    rows_per_entity_day=(4, 8, 16, None),
    random_seeds=(17, 42, 73),
    baseline_rows_per_entity_day=8,
    baseline_seed=42,
    maximum_training_rows=50_000,
    maximum_scoring_rows=100_000,
    isolation_trees=100,
    threshold_quantile=0.995,
    threshold_block_seconds=86_400,
    minimum_block_rows=None,
    persistence_observations=2,
    recovery_observations=4,
    recovery_threshold_fraction=0.80,
    incident_gap_seconds=3_600,
):
    """Compare fit sampling policies without reading labels or holdout data.

    ``None`` in ``rows_per_entity_day`` means no entity-day cap. The global
    ``maximum_training_rows`` guard remains in force and is reported, so the
    diagnostic cannot silently exhaust notebook memory.
    """

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    options = list(rows_per_entity_day)
    seeds = [int(seed) for seed in random_seeds]
    configurations = [(option, seed) for option in options for seed in seeds]
    baseline_key = (baseline_rows_per_entity_day, int(baseline_seed))
    if baseline_key not in configurations:
        raise ValueError("The declared baseline is absent from the sensitivity grid")
    if maximum_training_rows < 1 or maximum_scoring_rows < 1:
        raise ValueError("Sensitivity row budgets must be positive")

    source_rows = pq.ParquetFile(calibration_fit).metadata.num_rows
    minimum_block_rows = int(
        minimum_block_rows
        if minimum_block_rows is not None
        else max(1, round(0.50 * threshold_block_seconds / cadence_seconds))
    )

    results = []
    score_vectors = {}
    key_hashes = {}
    with tempfile.TemporaryDirectory(
        dir=output_directory, prefix="sampling-sensitivity-"
    ) as temporary_name:
        temporary = Path(temporary_name)
        threshold_source = temporary / "late_threshold.parquet"
        verification_source = temporary / "late_verification.parquet"
        split_feature_file_by_time(
            calibration_late,
            threshold_source,
            verification_source,
            fit_fraction=0.50,
        )
        threshold_panel = temporary / "threshold_panel.parquet"
        verification_panel = temporary / "verification_panel.parquet"
        threshold_rows = _bounded_entity_panel(
            threshold_source, threshold_panel, maximum_scoring_rows, baseline_seed
        )
        verification_rows = _bounded_entity_panel(
            verification_source,
            verification_panel,
            maximum_scoring_rows,
            baseline_seed,
        )
        exposure = partition_exposure(
            verification_panel, "entity_day", cadence_seconds
        )

        for option, seed in configurations:
            label = "all_per_day" if option is None else str(int(option))
            key = (option, seed)
            configuration_id = f"rows={label}|seed={seed}"
            started = time.perf_counter()
            settings = dict(fit_kwargs)
            settings.update({
                "maximum_training_rows": int(maximum_training_rows),
                "maximum_rows_per_entity_day": option,
                "random_seed": int(seed),
                "isolation_trees": int(isolation_trees),
                "fit_multivariate": True,
            })
            bundle = fit_residual_bundle(calibration_fit, **settings)
            if bundle["isolation_forest_temporal"] is None:
                raise ValueError("Sensitivity requires a fitted temporal Isolation Forest")

            threshold_scores = temporary / f"threshold_{label}_{seed}.parquet"
            verification_scores = temporary / f"verification_{label}_{seed}.parquet"
            for feature_path, destination in (
                (threshold_panel, threshold_scores),
                (verification_panel, verification_scores),
            ):
                score_residual_file(
                    bundle,
                    feature_path,
                    destination,
                    cadence_seconds=cadence_seconds,
                    dispersion_window_seconds=dispersion_window_seconds,
                    cusum_allowance=cusum_allowance,
                    progress_every=0,
                )

            threshold_table = calibration_thresholds(
                threshold_scores,
                [threshold_quantile],
                block_column="entity_id",
                block_duration_seconds=threshold_block_seconds,
                minimum_block_rows=minimum_block_rows,
                model_ids=["isolation_forest_temporal"],
            )
            threshold = float(threshold_table.loc[0, "threshold"])
            alerts = alerts_from_score_file(
                verification_scores,
                "isolation_forest_temporal",
                threshold,
                min_consecutive=persistence_observations,
                recovery_consecutive=recovery_observations,
                cadence_seconds=cadence_seconds,
                recovery_threshold_fraction=recovery_threshold_fraction,
            )
            cases, _ = form_cases(
                alerts,
                gap_seconds=incident_gap_seconds,
                thresholds={"isolation_forest_temporal": threshold},
            )
            key_hash, values = _score_keys_and_values(
                verification_scores, "isolation_forest_temporal"
            )
            key_hashes[key] = key_hash
            score_vectors[key] = values
            results.append({
                "configuration_id": configuration_id,
                "rows_per_entity_day": label,
                "random_seed": seed,
                "training_rows": bundle["training_rows"],
                "training_source_rows": source_rows,
                "training_sample_fraction": bundle["training_rows"] / source_rows,
                "global_training_cap": int(maximum_training_rows),
                "global_cap_applied": bundle["training_rows"] < source_rows,
                "selected_row_hash": bundle["selected_row_hash"],
                "threshold": threshold,
                "threshold_blocks": int(threshold_table.loc[0, "blocks_used"]),
                "verification_rows": verification_rows,
                "score_availability": float(np.isfinite(values).mean()),
                "incidents": len(cases),
                "verification_entity_days": exposure,
                "incidents_per_entity_day": len(cases) / exposure,
                "elapsed_seconds": time.perf_counter() - started,
            })
            threshold_scores.unlink()
            verification_scores.unlink()
            del bundle
            gc.collect()

    baseline_keys = key_hashes[baseline_key]
    baseline_scores = score_vectors[baseline_key]
    for row in results:
        option = None if row["rows_per_entity_day"] == "all_per_day" else int(
            row["rows_per_entity_day"]
        )
        key = (option, int(row["random_seed"]))
        if not np.array_equal(key_hashes[key], baseline_keys):
            raise AssertionError("Sensitivity score rows are not aligned")
        row["score_rank_spearman_vs_baseline"] = _rank_correlation(
            baseline_scores, score_vectors[key]
        )

    pairwise_rows = []
    configuration_ids = {
        (None if row["rows_per_entity_day"] == "all_per_day" else int(
            row["rows_per_entity_day"]
        ), int(row["random_seed"])): row["configuration_id"]
        for row in results
    }
    for left_index, left in enumerate(configurations):
        for right in configurations[left_index:]:
            pairwise_rows.append({
                "left_configuration": configuration_ids[left],
                "right_configuration": configuration_ids[right],
                "score_rank_spearman": _rank_correlation(
                    score_vectors[left], score_vectors[right]
                ),
            })
    pairwise = pd.DataFrame(pairwise_rows)

    result = pd.DataFrame(results).sort_values(
        ["rows_per_entity_day", "random_seed"]
    ).reset_index(drop=True)
    result.to_parquet(output_directory / "sampling_sensitivity.parquet", index=False)
    pairwise.to_parquet(
        output_directory / "sampling_rank_stability.parquet", index=False
    )
    summary = {
        "baseline_rows_per_entity_day": baseline_rows_per_entity_day,
        "baseline_seed": baseline_seed,
        "threshold_panel_rows": threshold_rows,
        "verification_panel_rows": verification_rows,
        "labels_or_holdout_read": False,
        "pairwise_rank_comparisons": len(pairwise),
        "all_per_day_is_globally_capped": bool(
            result.loc[
                result["rows_per_entity_day"].eq("all_per_day"),
                "global_cap_applied",
            ].any()
        ),
    }
    return result, summary
