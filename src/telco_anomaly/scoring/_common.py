"""Shared constants and small I/O helpers for model scoring."""

from __future__ import annotations

import math
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

MODEL_CORE_VERSION = "4.6.0"
MODEL_IDS = (
    "rapid_residual", "multimetric_residual", "multimetric_tail_mean",
    "drift_cusum", "peer_deviation", "group_common_mode",
    "dispersion_change", "pca_spe", "isolation_forest_base",
    "isolation_forest_temporal", "isolation_forest_confirmed",
    "isolation_forest_soft_confirmed",
    "isolation_forest_entity_calibrated", "isolation_forest_contextual",
)
LEGACY_MODEL_IDS = ("statistical", "isolation_forest", "pca", "pca_t2")
IDENTITY_COLUMNS = ["event_ts", "entity_id", "episode_id"]
SCALED_FEATURE_CAP = 50.0
PCA_VARIANCE_TARGET = 0.90
SHORT_SCALE_FLOOR_FRACTION = 0.25
RELATIVE_SCALE_FLOOR = 1e-6
ABSOLUTE_SCALE_FLOOR = 1e-12
PCA_EIGENVALUE_FLOOR = 1e-12

@contextmanager
def _model_duckdb(work_directory=None):
    """Use bounded memory and disposable local spill space for model scans."""

    parent = Path(
        work_directory
        or os.getenv("TELCO_WORK_ROOT", tempfile.gettempdir())
    )
    parent.mkdir(parents=True, exist_ok=True)
    memory_limit = os.getenv("TELCO_MODEL_DUCKDB_MEMORY_LIMIT", "1GB")
    threads = int(os.getenv("TELCO_MODEL_DUCKDB_THREADS", "1"))
    with tempfile.TemporaryDirectory(
        dir=parent, prefix="telco-model-duckdb-"
    ) as spill_directory:
        with duckdb.connect() as connection:
            connection.execute("SET memory_limit = ?", [memory_limit])
            connection.execute("SET threads = ?", [threads])
            connection.execute("SET temp_directory = ?", [spill_directory])
            connection.execute("SET preserve_insertion_order = false")
            yield connection

def duration_to_observations(duration_seconds, cadence_seconds):
    """Convert an elapsed-duration rule to a whole number of observations."""

    duration_seconds = float(duration_seconds)
    cadence_seconds = float(cadence_seconds)
    if duration_seconds <= 0 or cadence_seconds <= 0:
        raise ValueError("Duration and cadence must be positive")
    return max(1, math.ceil(duration_seconds / cadence_seconds))

def _sql_literal(value):
    return "'" + str(value).replace("'", "''") + "'"

def _sql_identifier(value):
    return '"' + str(value).replace('"', '""') + '"'

def iter_episode_frames(path, columns=None, batch_rows=100_000):
    """Yield complete episodes from an ordered Parquet file."""

    parquet = pq.ParquetFile(path)
    carry = pd.DataFrame()
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=columns):
        frame = batch.to_pandas()
        if not carry.empty:
            frame = pd.concat([carry, frame], ignore_index=True)
        if frame.empty:
            continue
        last = frame.iloc[-1][["entity_id", "episode_id"]].astype(str).tolist()
        is_last = (
            frame["entity_id"].astype(str).eq(last[0])
            & frame["episode_id"].astype(str).eq(last[1])
        )
        complete, carry = frame.loc[~is_last], frame.loc[is_last].copy()
        for _, episode in complete.groupby(
            ["entity_id", "episode_id"], sort=False
        ):
            yield episode.reset_index(drop=True)
    if not carry.empty:
        yield carry.reset_index(drop=True)

def feature_columns(path):
    names = pq.ParquetFile(path).schema_arrow.names
    return [name for name in names if name not in IDENTITY_COLUMNS]
