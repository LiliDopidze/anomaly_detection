"""Descriptive time-series assessment; no source names, labels or fitted models."""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

from .contracts import MetricDefinition


def split_development(data, start, train_end, test_start):
    """Return [start, train_end) and [train_end, test_start); never return test values."""
    start, train_end, test_start = map(pd.Timestamp, (start, train_end, test_start))
    if not start < train_end < test_start:
        raise ValueError("Require start < train_end < test_start")
    training = data.loc[data.timestamp.ge(start) & data.timestamp.lt(train_end)].copy()
    validation = data.loc[data.timestamp.ge(train_end) & data.timestamp.lt(test_start)].copy()
    return training, validation


def sampling_report(timeline: pd.DataFrame, gap_threshold: str):
    """Audit acquisition order, then intervals between unique sorted times.

    A gap exceeds the supplied threshold; it is not proof of downtime. No grid,
    rounding or missing-observation count is inferred from irregular times.
    """
    threshold = pd.Timedelta(gap_threshold).total_seconds()
    if threshold <= 0:
        raise ValueError("Gap threshold must be positive")
    summaries, intervals, gaps = [], [], []
    for entity, group in timeline.groupby("entity_id", sort=False):
        source_times = group.timestamp
        if source_times.isna().any():
            raise ValueError("Missing timestamps")
        times = source_times.drop_duplicates().sort_values().reset_index(drop=True)
        seconds = times.diff().dt.total_seconds()
        summaries.append({
            "entity_id": entity, "rows": len(group), "unique_times": len(times),
            "start": times.min(), "end": times.max(),
            "duplicate_rows": int(source_times.duplicated(keep=False).sum()),
            "backward_steps": int(source_times.diff().lt(pd.Timedelta(0)).sum()),
            "median_interval_s": seconds.median() if len(times) > 1 else np.nan,
            "p95_interval_s": seconds.quantile(.95) if len(times) > 1 else np.nan,
            "max_interval_s": seconds.max(), "gaps": int(seconds.gt(threshold).sum()),
        })
        counts = seconds.value_counts().rename_axis("interval_s").reset_index(name="count")
        intervals.append(counts.assign(entity_id=entity))
        selected = seconds.gt(threshold)
        gaps.append(pd.DataFrame({
            "entity_id": entity, "previous_time": times.shift()[selected],
            "next_time": times[selected], "elapsed_s": seconds[selected],
        }))
    return pd.DataFrame(summaries), pd.concat(intervals, ignore_index=True), pd.concat(gaps, ignore_index=True)


def bin_coverage(timestamps: pd.Series, start, end, frequency="1min") -> pd.DataFrame:
    """Occupied bins under a declared continuous [start, end) schedule.

    Occupancy is not sample completeness, and an unoccupied bin is not a zero.
    Start/end must align with the chosen bin boundaries.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    step = pd.Timedelta(frequency)
    if step <= pd.Timedelta(0) or end <= start:
        raise ValueError("Require a positive frequency and end > start")
    if start.floor(frequency) != start or end.floor(frequency) != end:
        raise ValueError("Coverage boundaries must align with bins")
    selected = timestamps[timestamps.ge(start) & timestamps.lt(end)].drop_duplicates()
    counts = selected.dt.floor(frequency).value_counts()
    grid = pd.date_range(start, end, freq=frequency, inclusive="left")
    result = counts.reindex(grid, fill_value=0).rename("observations").to_frame()
    result.index.name = "timestamp"
    result["occupied"] = result.observations.gt(0)
    return result


def duplicate_report(data: pd.DataFrame, measurements: list[str]) -> pd.DataFrame:
    """Identify conflicting duplicates without averaging or deleting them."""
    keys = ["entity_id", "timestamp"]
    duplicate = data.loc[data.duplicated(keys, keep=False)]
    records = []
    for key, group in duplicate.groupby(keys):
        conflicts = group[measurements].nunique(dropna=False).gt(1)
        records.append({"entity_id": key[0], "timestamp": key[1], "rows": len(group),
                        "conflicting_measurements": ", ".join(conflicts[conflicts].index)})
    return pd.DataFrame(records, columns=[*keys, "rows", "conflicting_measurements"])


def signal_profile(data: pd.DataFrame, definitions: dict[str, MetricDefinition]) -> pd.DataFrame:
    """Profile observed values; extreme finite values are not declared invalid."""
    records = []
    for entity, group in data.groupby("entity_id", sort=False):
        for name, definition in definitions.items():
            values = group[name]
            numeric = pd.api.types.is_numeric_dtype(values)
            finite = np.isfinite(values) if numeric else values.notna()
            observed = values[finite]
            unexpected = 0
            if definition.allowed_values is not None:
                unexpected = int((values.notna() & ~values.isin(definition.allowed_values)).sum())
            record = {
                "entity_id": entity, "measurement": name, "kind": definition.kind,
                "rows": len(values), "missing": int(values.isna().sum()),
                "nonfinite": int((values.notna() & ~finite).sum()),
                "unexpected_encoding": unexpected, "unique": observed.nunique(),
            }
            if definition.kind == "gauge":
                record.update(observed.quantile([0, .01, .25, .5, .75, .99, 1]).set_axis(
                    ["min", "p01", "q25", "median", "q75", "p99", "max"]).to_dict())
                record["sd"] = observed.std()
            records.append(record)
    return pd.DataFrame(records)


def state_runs(series: pd.Series, max_interval: str) -> pd.DataFrame:
    """Constant runs for one entity, interrupted by gaps or missing values.

    Span is last minus first observation, a lower bound on the sampled run's
    duration assuming no unobserved intermediate switches. Upper bounds require both
    adjacent transitions to be observed without a collection gap.
    """
    if not isinstance(series.index, pd.DatetimeIndex) or not series.index.is_unique:
        raise ValueError("Require a unique DatetimeIndex for one entity")
    if not series.index.is_monotonic_increasing:
        raise ValueError("Sort observations before run analysis")
    limit = pd.Timedelta(max_interval)
    if limit <= pd.Timedelta(0):
        raise ValueError("Maximum interval must be positive")
    times = pd.Series(series.index, index=series.index)
    connected = times.diff().le(limit) & series.notna() & series.shift().notna()
    change = series.ne(series.shift()) | ~connected
    run_ids = change.cumsum()
    valid = pd.DataFrame({"value": series, "time": times, "run": run_ids}).dropna(subset=["value"])
    runs = valid.groupby("run").agg(value=("value", "first"), start=("time", "first"),
                                   last=("time", "last"), samples=("value", "size"))
    left = connected & series.ne(series.shift())
    right = left.shift(-1, fill_value=False)
    runs["left_censored"] = ~left.reindex(runs.start).to_numpy()
    runs["right_censored"] = ~right.reindex(runs["last"]).to_numpy()
    runs["observed_span_s"] = (runs["last"] - runs.start).dt.total_seconds()
    previous = times.shift().reindex(runs.start).to_numpy()
    following = times.shift(-1).reindex(runs["last"]).to_numpy()
    runs["duration_upper_s"] = (pd.Series(following) - pd.Series(previous)).dt.total_seconds().to_numpy()
    runs.loc[runs.left_censored | runs.right_censored, "duration_upper_s"] = np.nan
    return runs.reset_index(drop=True)


def aggregate_gauges(data: pd.DataFrame, columns: list[str], frequency: str, min_count: int):
    """Observation means in left-labelled bins; mask insufficient support.

    No interpolation. These are descriptive aggregates, not causal features:
    a bin's value becomes available only at its right endpoint.
    """
    if min_count < 1:
        raise ValueError("Minimum count must be positive")
    if data.entity_id.nunique() != 1 or data.timestamp.duplicated().any():
        raise ValueError("Supply one entity with unique timestamps")
    values = data.set_index("timestamp")[columns].replace([np.inf, -np.inf], np.nan)
    groups = values.resample(frequency, closed="left", label="left")
    counts = groups.count()
    return groups.mean().where(counts >= min_count), counts


def longest_regular_segment(series: pd.Series, frequency: str) -> pd.Series:
    """Choose the longest finite, consecutive segment; first wins ties."""
    if not isinstance(series.index, pd.DatetimeIndex) or not series.index.is_unique:
        raise ValueError("Require a unique DatetimeIndex")
    if not series.index.is_monotonic_increasing:
        raise ValueError("Sort the series before temporal analysis")
    step = pd.Timedelta(frequency)
    if step <= pd.Timedelta(0):
        raise ValueError("Frequency must be positive")
    valid = np.isfinite(series)
    adjacent = series.index.to_series().diff().eq(step)
    starts = ~valid | ~valid.shift(fill_value=False) | ~adjacent
    groups = starts.cumsum()
    sizes = groups[valid].value_counts(sort=False)
    if sizes.empty:
        return series.iloc[:0]
    return series.loc[valid & groups.eq(sizes.idxmax())]


def segment_acf(series: pd.Series, frequency: str, max_lag: int):
    """Demeaned ACF on one regular segment, without IID significance bands."""
    segment = longest_regular_segment(series, frequency)
    if max_lag < 1:
        raise ValueError("Maximum lag must be positive")
    lags = min(max_lag, len(segment) // 4)
    if len(segment) < 8 or segment.nunique() < 2:
        correlation = np.full(lags + 1, np.nan)
    else:
        correlation = acf(segment.to_numpy(), nlags=lags, fft=True, adjusted=False, missing="raise")
    result = pd.DataFrame({"lag_steps": np.arange(lags + 1), "acf": correlation})
    result["lag_minutes"] = result.lag_steps * pd.Timedelta(frequency).total_seconds() / 60
    return result, {"start": segment.index.min(), "end": segment.index.max(), "samples": len(segment)}


def relationship_report(data: pd.DataFrame, pairs: list[tuple[str, str]], state: str | None = None):
    """One-entity finite-pair correlations, optionally within supplied states."""
    if "entity_id" in data and data.entity_id.nunique() != 1:
        raise ValueError("Supply one entity for relationship assessment")
    groups = data.groupby(state, observed=True) if state else [("all", data)]
    records = []
    for label, group in groups:
        for first, second in pairs:
            values = group[[first, second]].replace([np.inf, -np.inf], np.nan).dropna()
            pearson = spearman = np.nan
            if len(values) >= 3 and values.nunique().gt(1).all():
                pearson = values[first].corr(values[second])
                spearman = values[first].corr(values[second], method="spearman")
            records.append({"state": label, "first": first, "second": second,
                            "pairs": len(values), "pearson": pearson, "spearman": spearman})
    return pd.DataFrame(records)
