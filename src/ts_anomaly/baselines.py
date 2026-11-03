"""Explicit reference distributions, causal histories and descriptive comparisons."""

import numpy as np
import pandas as pd

from .assessment import aggregate_gauges


def fit_reference(data, measurements, start, end, *, context=None,
                  allowed_states=None, min_observations=100, min_days=3):
    """Fit medians and central 90% ranges on [start, end), separately by entity.

    Support counts are screening rules, not effective sample sizes or confidence
    guarantees. Unknown context is never silently assigned a global reference.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start >= end or min_observations < 1 or min_days < 1:
        raise ValueError("Require ordered boundaries and positive support counts")
    selected = data.loc[data.timestamp.ge(start) & data.timestamp.lt(end)].copy()
    if selected.empty:
        raise ValueError("No observations in the fitting period")
    if selected.duplicated(["entity_id", "timestamp"]).any():
        raise ValueError("Duplicate entity/timestamp observations")
    if context is not None and not allowed_states:
        raise ValueError("Declare the supported context states")
    selected["context"] = "all" if context is None else selected[context]
    records = []
    for (entity, state), group in selected.groupby(["entity_id", "context"], dropna=False):
        for measurement in measurements:
            finite = np.isfinite(group[measurement])
            values = group.loc[finite, measurement]
            days = group.loc[finite, "timestamp"].dt.normalize().nunique()
            lower, median, upper = values.quantile([.05, .5, .95])
            permitted = context is None or state in allowed_states
            supported = permitted and len(values) >= min_observations and days >= min_days
            records.append({
                "entity_id": entity, "context": state, "measurement": measurement,
                "observations": len(values), "days": days, "expected": median,
                "lower": lower, "upper": upper, "supported": supported,
                "collapsed_range": bool(len(values) and lower == upper),
                "fit_start": start, "fit_end": end,
            })
    return pd.DataFrame(records)


def apply_reference(data, reference, measurement, *, context=None):
    """Apply a frozen reference after its fitting period; retain every input row."""
    parameters = reference.loc[reference.measurement.eq(measurement)].copy()
    if parameters.empty:
        raise ValueError(f"No reference for {measurement}")
    if data.timestamp.lt(parameters.fit_end.max()).any():
        raise ValueError("Evaluation observations must follow the fitting period")
    if data.duplicated(["entity_id", "timestamp"]).any():
        raise ValueError("Duplicate entity/timestamp observations")
    result = data[["timestamp", "entity_id", measurement]].rename(columns={measurement: "observed"}).copy()
    result["context"] = "all" if context is None else data[context].to_numpy()
    result = result.merge(parameters.drop(columns="measurement"), on=["entity_id", "context"],
                          how="left", validate="many_to_one", sort=False)
    supported = result.supported.eq(True)
    finite = np.isfinite(result.observed)
    result["status"] = "available"
    result.loc[~supported, "status"] = "unsupported_reference"
    result.loc[~finite, "status"] = "missing_measurement"
    result.loc[~supported, ["expected", "lower", "upper"]] = np.nan
    result["residual"] = (result.observed - result.expected).where(finite)
    result["measurement"] = measurement
    return result


def compare_predictions(predictions, group_columns=()):
    """Errors on common finite targets; availability on all requested rows.

    Inputs must describe the same observations in the same order. Bounds are
    empirical reference ranges, not confidence intervals or alarm thresholds.
    """
    names = list(predictions)
    first = predictions[names[0]]
    keys = ["timestamp", "entity_id", "observed", *group_columns]
    common = np.isfinite(first.observed).to_numpy()
    for frame in predictions.values():
        if not frame[keys].reset_index(drop=True).equals(first[keys].reset_index(drop=True)):
            raise ValueError("Compare identical targets, ordering and grouping values")
        common &= np.isfinite(frame.expected).to_numpy()
    groups = first.groupby(list(group_columns), dropna=False, sort=True).indices if group_columns else {(): np.arange(len(first))}
    records = []
    for key, positions in groups.items():
        labels = key if isinstance(key, tuple) else (key,)
        for name, frame in predictions.items():
            requested = frame.iloc[positions]
            paired = requested.loc[common[positions]]
            error = paired.observed - paired.expected
            bounded = paired.loc[np.isfinite(paired.lower) & np.isfinite(paired.upper)]
            available = np.isfinite(requested.observed) & np.isfinite(requested.expected)
            record = dict(zip(group_columns, labels))
            record.update({
                "method": name, "requested": len(requested), "available": int(available.sum()),
                "availability": available.mean(), "common": len(paired),
                "mae": error.abs().mean(), "bias": error.mean(),
                "residual_sd": error.std(), "p90_absolute_error": error.abs().quantile(.9),
                "range_rows": len(bounded),
                "range_coverage": bounded.observed.between(bounded.lower, bounded.upper).mean(),
                "range_width": (bounded.upper - bounded.lower).mean(),
            })
            records.append(record)
    return pd.DataFrame(records)


def completed_bins(data, columns, *, frequency="1min", min_count=5, max_interval="15s",
                   complete_columns=()):
    """One-entity means available at bin end, with source-continuity identifiers.

    A bin spanning a collection break is unavailable even if its count is large.
    A break exactly between bins starts a new history segment. Values at the right
    endpoint belong to the next bin. `complete_columns` must be present at every
    observed source row (useful for context). Original times are never rounded.
    """
    if not data.timestamp.is_monotonic_increasing:
        raise ValueError("Sort observations before aggregation")
    values, counts = aggregate_gauges(data, columns, frequency, min_count)
    if complete_columns:
        rows = data.set_index("timestamp").resample(frequency, closed="left", label="left").size()
        values[list(complete_columns)] = values[list(complete_columns)].where(counts[list(complete_columns)].eq(rows, axis=0))
    step = pd.Timedelta(frequency)
    gaps = data.timestamp.diff().gt(pd.Timedelta(max_interval))
    segment = pd.Series(gaps.cumsum().to_numpy(), index=pd.DatetimeIndex(data.timestamp))
    grouped = segment.resample(frequency, closed="left", label="left")
    values.loc[grouped.nunique().gt(1), :] = np.nan
    values["segment"] = grouped.last()
    values["source_count"] = counts.min(axis=1)
    values.index += step
    values.index.name = "timestamp"
    return values


def lag_features(bins, lags, *, frequency="1min"):
    """Past-only lags within uninterrupted, fully observed input sequences.

    `lags` maps each input column to positive offsets on the declared time grid.
    Missing target/context values reset history rather than compressing time.
    """
    if not bins.index.is_unique or not bins.index.is_monotonic_increasing:
        raise ValueError("Require sorted, unique bin times")
    if any(lag < 1 or not isinstance(lag, int) for offsets in lags.values() for lag in offsets):
        raise ValueError("Lag offsets must be positive integers")
    valid = np.isfinite(bins[list(lags)]).all(axis=1)
    adjacent = bins.index.to_series().diff().eq(pd.Timedelta(frequency))
    continuous = adjacent & bins.segment.eq(bins.segment.shift()) & valid & valid.shift(fill_value=False)
    sequence = (~continuous).cumsum()
    result = bins.copy()
    for column, offsets in lags.items():
        for lag in offsets:
            result[f"{column}_lag{lag}"] = bins[column].groupby(sequence).shift(lag)
    return result
