"""Domain-independent schedule diagnostics; no filling or model selection."""

import numpy as np
import pandas as pd


def sampling_report(telemetry: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Audit source keys before aggregation hides duplicates or irregular timing."""
    step = pd.Timedelta(interval)
    if step <= pd.Timedelta(0):
        raise ValueError("Interval must be positive")
    records = []
    for (entity, metric), group in telemetry.groupby(["entity_id", "metric_name"]):
        times = group.timestamp.sort_values()
        differences = times.diff().dropna()
        records.append(
            {
                "entity_id": entity,
                "metric_name": metric,
                "rows": len(group),
                "duplicate_timestamps": int(times.duplicated().sum()),
                "off_grid_timestamps": int(times.ne(times.dt.floor(step)).sum()),
                "gaps_longer_than_cadence": int(differences.gt(step).sum()),
                "shorter_than_cadence": int(differences.lt(step).sum()),
            }
        )
    return pd.DataFrame(records)


def coverage_report(
    telemetry: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str,
    entities: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    """Coverage on a declared UTC, midnight-aligned schedule [start, end).

    Includes leading/trailing gaps and entirely absent series. Explicit NaNs and
    absent rows are distinguished. This regular active schedule is an input,
    never inferred from first/last observations. Use separate calls for shifts or
    changing activity windows; event-driven telemetry needs another denominator.
    """
    step = pd.Timedelta(interval)
    if step <= pd.Timedelta(0) or start >= end:
        raise ValueError("Need a positive interval and start < end")
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("Schedule timestamps must be timezone-aware")
    start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
    if start != start.floor(step) or end != end.floor(step):
        raise ValueError("Schedule boundaries must align to the interval")
    if (
        not entities
        or not metrics
        or len(set(entities)) != len(entities)
        or len(set(metrics)) != len(metrics)
    ):
        raise ValueError("Declare nonempty, unique entity and metric lists")
    keys = ["entity_id", "metric_name", "timestamp"]
    if telemetry.duplicated(keys).any():
        raise ValueError("Duplicate telemetry; resolve before assessing coverage")
    window = telemetry.loc[telemetry.timestamp.ge(start) & telemetry.timestamp.lt(end)]
    if set(window.entity_id) - set(entities) or set(window.metric_name) - set(metrics):
        raise ValueError(
            "Observations fall outside the declared entity/metric population"
        )
    if window.timestamp.ne(window.timestamp.dt.floor(step)).any():
        raise ValueError(
            "Off-grid observations; audit source sampling before resampling"
        )
    grid = pd.date_range(start, end, freq=step, inclusive="left")
    indexed = window.set_index(keys)
    records = []
    for entity in entities:
        for metric in metrics:
            index = pd.MultiIndex.from_product([[entity], [metric], grid], names=keys)
            values = indexed.value.reindex(index)
            present = index.isin(indexed.index)
            observed = np.isfinite(values)
            missing = ~observed
            runs = missing.groupby(observed.cumsum()).sum()
            records.append(
                {
                    "entity_id": entity,
                    "metric_name": metric,
                    "expected_slots": len(grid),
                    "present_rows": int(present.sum()),
                    "observed_slots": int(observed.sum()),
                    "absent_rows": int((~present).sum()),
                    "invalid_or_missing_values": int((present & missing).sum()),
                    "observed_coverage": float(observed.mean()),
                    "longest_missing_hours": float(
                        runs.max() * step.total_seconds() / 3600
                    ),
                }
            )
    return pd.DataFrame(records)
