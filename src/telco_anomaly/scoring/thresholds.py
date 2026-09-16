"""Calibration-tail threshold estimation."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ._common import MODEL_IDS, _model_duckdb, _sql_identifier, _sql_literal

def calibration_thresholds(
    score_path,
    quantiles,
    *,
    block_column,
    block_duration_seconds=None,
    minimum_block_rows=1_000,
    model_ids=MODEL_IDS,
):
    """Estimate thresholds from calibration block maxima.

    One maximum per independent entity or episode absorbs the many chances to
    fire across timestamps, metrics and topology levels inside a channel.
    """

    rows = []
    source = _sql_literal(str(Path(score_path)))
    block = _sql_identifier(block_column)
    duration = None
    if block_duration_seconds is not None:
        duration = float(block_duration_seconds)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("block_duration_seconds must be positive")
    score_columns = set(pq.ParquetFile(score_path).schema_arrow.names)
    with _model_duckdb(Path(score_path).parent) as connection:
        for model_id in model_ids:
            started = time.perf_counter()
            print(f"    threshold calibration: {model_id}", flush=True)
            model_column = _sql_identifier(model_id)
            scope_columns = {
                f"{model_id}__scope_type", f"{model_id}__scope_id",
            }
            if scope_columns <= score_columns:
                base_key = (
                    f"coalesce(CAST({_sql_identifier(model_id + '__scope_type')} AS VARCHAR), '') "
                    f"|| ':' || coalesce(CAST({_sql_identifier(model_id + '__scope_id')} AS VARCHAR), '')"
                )
                block_label = f"{model_id}_scope"
                # A scope score is repeated on descendant rows. Completeness
                # therefore counts distinct observation times, not children.
                block_rows = "count(DISTINCT event_ts)"
            else:
                base_key = f"CAST({block} AS VARCHAR)"
                block_label = block_column
                block_rows = f"count({model_column})"
            if duration is not None:
                block_key = (
                    f"({base_key}) || ':' || "
                    f"CAST(floor(epoch(event_ts) / {duration}) AS BIGINT)"
                )
                block_label = f"{block_label}+{duration:g}s"
            else:
                block_key = base_key

            maxima = connection.execute(f"""
                SELECT {block_key} AS calibration_block,
                       max({model_column}) AS block_maximum,
                       {block_rows} AS rows
                FROM read_parquet({source})
                WHERE {model_column} IS NOT NULL
                GROUP BY calibration_block
            """).df()
            total_blocks = len(maxima)
            usable = maxima.loc[
                maxima["rows"].ge(int(minimum_block_rows)), "block_maximum"
            ]
            used_blocks = len(usable)
            unique_block_maxima = int(usable.nunique())
            if used_blocks == 0:
                raise ValueError(
                    f"No calibration blocks for {model_id} contain at least "
                    f"{minimum_block_rows} rows"
                )

            connection.register("calibration_block_maxima", maxima)
            requested_quantiles = list(map(float, quantiles))
            quantile_expressions = ", ".join(
                "quantile_cont(block_maximum, CAST(? AS FLOAT))"
                for _ in requested_quantiles
            )
            threshold_values = connection.execute(
                f"""
                SELECT {quantile_expressions}
                FROM calibration_block_maxima
                WHERE rows >= ?
                """,
                [*requested_quantiles, int(minimum_block_rows)],
            ).fetchone()
            connection.unregister("calibration_block_maxima")

            for quantile, threshold in zip(
                requested_quantiles, threshold_values
            ):
                if threshold is None or not np.isfinite(threshold):
                    raise ValueError(
                        f"{model_id} has no finite calibration threshold at "
                        f"quantile {quantile}"
                    )
                rows.append({
                    "model_id": model_id,
                    "threshold_quantile": quantile,
                    "threshold": float(threshold),
                    "threshold_block": block_label,
                    "threshold_method": "empirical_quantile_of_block_maxima",
                    "minimum_block_rows": int(minimum_block_rows),
                    "blocks_total": total_blocks,
                    "blocks_used": used_blocks,
                    "blocks_excluded": total_blocks - used_blocks,
                    "unique_block_maxima": unique_block_maxima,
                    "expected_tail_blocks": used_blocks * (1 - quantile),
                })
            print(
                f"    threshold calibration: {model_id} complete "
                f"({used_blocks}/{total_blocks} blocks used, "
                f"{time.perf_counter() - started:.1f}s)",
                flush=True,
            )
    result = pd.DataFrame(rows)
    for model_id, group in result.groupby("model_id", sort=False):
        ordered = group.sort_values("threshold_quantile")
        if ordered["threshold"].diff().dropna().lt(-1e-12).any():
            raise AssertionError(
                f"Calibration thresholds decrease with quantile for {model_id}"
            )
    return result
