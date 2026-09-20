"""Memory-conscious canonical preparation and feature replay, one ONT at a time."""

from pathlib import Path
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .sources import synthetic_adapter
from .validation import DataValidator
from .features import FeatureEngineer
from .multivariate import MultivariateFeatures
from .splitting import TemporalSplit


def run_split(settings: dict) -> TemporalSplit:
    start = pd.Timestamp("2025-01-01", tz="UTC")
    return TemporalSplit(
        *[
            start + pd.Timedelta(days=settings["generator"]["days"] * fraction)
            for fraction in settings["splits"]
        ]
    )


def adapt_entity(native: pd.DataFrame, interval: int) -> pd.DataFrame:
    return DataValidator(f"{interval}min").transform(
        synthetic_adapter().transform(native)
    )


def prepare_canonical(run: str | Path) -> Path:
    """Write adapted, validated development data; no final-period rows.

    Explicit reruns rebuild this derived file before fitting. Once a model exists,
    use its frozen canonical file; choose a fresh run for changed code or settings.
    """
    run = Path(run)
    target = run / "canonical_development.parquet"
    if (run / "model.joblib").exists():
        if not target.exists():
            raise ValueError("Old run lacks canonical data; use a new output folder")
        return target
    settings = json.loads((run / "settings.json").read_text())
    split = run_split(settings)
    native = pd.read_parquet(
        run / "telemetry.parquet", filters=[("time", "<", split.validation_end)]
    )
    interval = settings["generator"]["interval_minutes"]
    temporary = target.with_suffix(".tmp")
    writer = None
    try:
        for _, group in native.groupby("device", sort=True):
            table = pa.Table.from_pandas(
                adapt_entity(group, interval), preserve_index=False
            )
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(target)
    return target


def feature_data(
    run: Path, settings: dict, engineers: dict | None = None, final: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit on training only, or replay frozen per-entity references with history."""
    fitting = engineers is None
    if fitting:
        engineers = {}
    split = run_split(settings)
    interval = settings["generator"]["interval_minutes"]
    entities = pd.read_parquet(run / "topology.parquet").entity_id
    frames, observed_rx = [], []
    for entity in entities:
        if final:
            native = pd.read_parquet(
                run / "telemetry.parquet", filters=[("device", "==", entity)]
            )
            telemetry = adapt_entity(native, interval)
        else:
            telemetry = pd.read_parquet(
                run / "canonical_development.parquet",
                filters=[("entity_id", "==", entity)],
            )
        if fitting:
            engineer = MultivariateFeatures(
                FeatureEngineer(interval_minutes=interval, **settings["features"])
            )
            engineer.fit(telemetry.loc[telemetry.timestamp < split.train_end])
            engineers[entity] = engineer
        if entity not in engineers:
            raise ValueError(f"No fitted reference for entity {entity}")
        frames.append(engineers[entity].transform(telemetry))
        observed_rx.append(telemetry.loc[telemetry.metric_name.eq("rx_power_dbm")])
    return (
        pd.concat(frames, ignore_index=True),
        pd.concat(observed_rx, ignore_index=True),
        engineers,
    )
