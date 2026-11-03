"""Descriptive checks without imputation, model fitting or anomaly labels."""

import numpy as np
import pandas as pd

from .contracts import MetricDefinition, validate_schema


def valid_values(values: pd.Series, definition: MetricDefinition) -> pd.Series:
    """Identify usable values without modifying the source observations."""
    valid = np.isfinite(values)
    if definition.lower is not None:
        valid &= values >= definition.lower
    if definition.upper is not None:
        valid &= values <= definition.upper
    if definition.kind == "interval_count":
        valid &= (values >= 0) & (values == np.floor(values))
    return valid.fillna(False)


def diagnostic_series(
    group: pd.DataFrame, definition: MetricDefinition, cadence: str
) -> pd.Series:
    """Return a regular diagnostic view; exclude every duplicate timestamp.

    The grid spans the observed file, anchored at its first timestamp. It is an
    assessment assumption, not an independently known monitoring schedule.
    Off-grid observations raise instead of being rounded or silently dropped.
    """
    if group.empty or len(group[["entity_id", "metric_name"]].drop_duplicates()) != 1:
        raise ValueError("Supply one nonempty entity/metric series")
    step = pd.Timedelta(cadence)
    if pd.isna(step) or step <= pd.Timedelta(0):
        raise ValueError("Cadence must be positive")
    start, end = group.timestamp.min(), group.timestamp.max()
    if ((group.timestamp - start) % step != pd.Timedelta(0)).any():
        raise ValueError("Off-grid timestamps: inspect sampling before regularising")
    unique = ~group.timestamp.duplicated(keep=False)
    usable = unique & valid_values(group.value, definition)
    values = group.loc[usable].set_index("timestamp").value
    grid = pd.date_range(start, end, freq=step, name="timestamp")
    return values.reindex(grid).rename("value")


def assessment_report(
    observations: pd.DataFrame,
    definitions: dict[str, MetricDefinition],
    cadences: dict[tuple[str, str], str],
) -> pd.DataFrame:
    """Audit source order and file-span coverage, before any sorting or cleaning."""
    validate_schema(observations, definitions)
    records = []
    for (entity, metric), group in observations.groupby(
        ["entity_id", "metric_name"], sort=False
    ):
        step = pd.Timedelta(cadences[(entity, metric)])
        if pd.isna(step) or step <= pd.Timedelta(0):
            raise ValueError("Cadence must be positive")
        times = group.timestamp
        valid = valid_values(group.value, definitions[metric])
        duplicates = times.duplicated(keep=False)
        off_grid = (times - times.min()) % step != pd.Timedelta(0)
        record = {
            "entity_id": entity, "metric_name": metric, "rows": len(group),
            "start": times.min(), "end": times.max(),
            "duplicate_timestamps": times[duplicates].nunique(),
            "duplicate_rows": int(duplicates.sum()),
            "backward_steps": int((times.diff() < pd.Timedelta(0)).sum()),
            "off_grid_rows": int(off_grid.sum()),
            "missing_values": int(group.value.isna().sum()),
            "invalid_nonmissing_values": int((~valid & group.value.notna()).sum()),
            "expected_slots_in_file_span": np.nan, "absent_slots": np.nan,
            "usable_fraction_in_file_span": np.nan, "longest_unusable_minutes": np.nan,
        }
        if not off_grid.any():
            series = diagnostic_series(group, definitions[metric], str(step))
            missing = series.isna()
            runs = missing.groupby((~missing).cumsum()).sum()
            record.update({
                "expected_slots_in_file_span": len(series),
                "absent_slots": len(series) - times.nunique(),
                "usable_fraction_in_file_span": float(series.notna().mean()),
                "longest_unusable_minutes": int(runs.max()) * step.total_seconds() / 60,
            })
        records.append(record)
    return pd.DataFrame(records)


def lag_report(series: pd.Series, lags: dict[str, str]) -> pd.DataFrame:
    """Pearson correlations between available pairs at exact elapsed-time lags.

    No IID confidence bands or significance tests: trend, regimes and events can
    all drive correlations. Pair counts are reported; gaps are never compressed.
    """
    if not isinstance(series.index, pd.DatetimeIndex) or not series.index.is_unique:
        raise ValueError("A unique DatetimeIndex is required")
    rows = []
    for label, duration in lags.items():
        delta = pd.Timedelta(duration)
        if pd.isna(delta) or delta <= pd.Timedelta(0):
            raise ValueError("Lags must be positive")
        previous = series.reindex(series.index - delta).to_numpy()
        current = series.to_numpy()
        valid = np.isfinite(current) & np.isfinite(previous)
        x, y = current[valid], previous[valid]
        correlation = np.nan
        if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
            correlation = float(np.corrcoef(x, y)[0, 1])
        rows.append({"lag": label, "pairs": int(valid.sum()), "correlation": correlation})
    return pd.DataFrame(rows)
