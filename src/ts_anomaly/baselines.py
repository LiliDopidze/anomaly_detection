"""Fixed empirical references and causal persistence for one recording at a time."""

import numpy as np
import pandas as pd

from .assessment import aggregate_gauges


def fit_reference(values: pd.Series) -> pd.Series:
    """Fit explicitly selected finite values; ranges are descriptive, not calibrated."""
    observed = values[np.isfinite(values)]
    quantiles = observed.quantile([.05, .25, .5, .75, .95])
    lower, q25, centre, q75, upper = quantiles.to_numpy()
    return pd.Series({
        "observations": len(observed), "distinct": observed.nunique(),
        "expected": centre, "lower": lower, "upper": upper,
        "iqr": q75 - q25, "mad": (observed - centre).abs().median(),
        "supported": len(observed) > 0,
        "collapsed_range": bool(len(observed) and lower == upper),
    })


def calibrate_reference(series: pd.Series, start, end, minimum_occupancy=.8,
                        nominal_interval="1s") -> pd.Series:
    """Fit [start, end); count occupied nominal slots relative to the declared start.

    Failed support retains descriptive estimates for audit. Application must mask
    them. Occupancy is an availability rule, not an effective sample-size estimate.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    step = pd.Timedelta(nominal_interval)
    if end <= start or step <= pd.Timedelta(0) or not 0 <= minimum_occupancy <= 1:
        raise ValueError("Require positive duration/interval and occupancy in [0, 1]")
    if not isinstance(series.index, pd.DatetimeIndex) or not series.index.is_unique:
        raise ValueError("Supply one recording with unique timestamps")
    selected = series.loc[(series.index >= start) & (series.index < end)]
    finite = selected[np.isfinite(selected)]
    expected_slots = int(np.ceil((end - start) / step))
    occupied = len(np.unique((finite.index - start) // step))
    reference = fit_reference(selected)
    reference["fit_start"] = start
    reference["fit_end"] = end
    reference["expected_slots"] = expected_slots
    reference["occupied_slots"] = occupied
    reference["occupancy"] = occupied / expected_slots
    reference["supported"] = bool(len(finite) and reference.occupancy >= minimum_occupancy)
    return reference


def apply_reference(actual: pd.Series, reference: pd.Series, available_from=None) -> pd.DataFrame:
    """Apply a frozen centre/range; never estimate anything from application values."""
    result = pd.DataFrame(index=actual.index, columns=["expected", "lower", "upper"], dtype=float)
    eligible = np.ones(len(actual), dtype=bool)
    if available_from is not None:
        eligible &= actual.index >= pd.Timestamp(available_from)
    if reference.supported:
        result.loc[eligible, ["expected", "lower", "upper"]] = reference[
            ["expected", "lower", "upper"]].to_numpy(dtype=float)
    result["residual"] = actual.where(np.isfinite(actual)) - result.expected
    return result


def completed_gauges(data, measurements, frequency="5s", min_count=4,
                     max_gap="2s", nominal_interval="1s"):
    """Return means/counts at right endpoints and per-signal continuity flags.

    Only full windows within the observed nominal schedule are retained. A mean
    becomes available at its endpoint. Continuity requires no raw gap or missing
    measurement between the two windows, as well as two eligible means. No filling.
    """
    if not data.timestamp.is_monotonic_increasing or data.timestamp.isna().any():
        raise ValueError("Supply sorted, nonmissing timestamps")
    step = pd.Timedelta(frequency)
    gap = pd.Timedelta(max_gap)
    nominal = pd.Timedelta(nominal_interval)
    if min(step, gap, nominal) <= pd.Timedelta(0):
        raise ValueError("Intervals must be positive")
    means, counts = aggregate_gauges(data, measurements, frequency, min_count)
    # The nominal schedule ends one interval after the last observed timestamp.
    complete = (means.index >= data.timestamp.iloc[0]) & (
        means.index + step <= data.timestamp.iloc[-1] + nominal)
    means, counts = means.loc[complete].copy(), counts.loc[complete].copy()
    raw_times = pd.DatetimeIndex(data.timestamp)
    raw_gaps = raw_times.to_series().diff().gt(gap)
    connected = pd.DataFrame(False, index=means.index, columns=measurements)
    for name in measurements:
        valid = pd.Series(np.isfinite(data[name].to_numpy()), index=raw_times)
        segments = (raw_gaps | ~valid | ~valid.shift(fill_value=False)).cumsum()
        groups = segments.resample(frequency, closed="left", label="left")
        first = groups.first().reindex(means.index)
        last = groups.last().reindex(means.index)
        adjacent = means.index.to_series().diff().eq(step)
        connected[name] = (adjacent & first.eq(last) & first.shift().eq(last.shift())
                           & last.eq(last.shift())
                           & means[name].notna() & means[name].shift().notna())
    for frame in (means, counts, connected):
        frame.index = frame.index + step
        frame.index.name = "timestamp"
    return means, counts, connected


def persistence_prediction(actual: pd.Series, connected: pd.Series, frequency="5s"):
    """Predict the current completed target with the previous window's mean.

    issued_at is the target start. Expected values use past means only; eligibility
    is confirmed later, when target support and raw continuity are known. Residuals
    stay unavailable on interrupted windows. Call separately for each recording.
    """
    step = pd.Timedelta(frequency)
    adjacent = actual.index.to_series().diff().eq(step)
    expected = actual.shift().where(adjacent)
    result = pd.DataFrame({"expected": expected, "lower": np.nan, "upper": np.nan})
    result["eligible"] = connected & adjacent & np.isfinite(actual) & np.isfinite(expected)
    result["residual"] = (actual - expected).where(result.eligible)
    result["issued_at"] = (actual.index.to_series() - step).where(expected.notna())
    return result


def compare_predictions(actual: pd.Series, predictions: dict[str, pd.DataFrame], evaluation_mask=None):
    """Own availability/errors and paired errors on the intersection of methods.

    All frames must have the same ordered index. Missing targets are unavailable
    for scoring. An optional mask excludes non-independent evaluation cases without
    changing availability. Optional prediction eligibility records retrospective
    target/continuity support. Empty intersections produce NaN, never perfect scores.
    """
    for frame in predictions.values():
        if not actual.index.equals(frame.index):
            raise ValueError("Predictions must align exactly with the target index")
    observed = np.isfinite(actual)
    eligible = {name: observed & np.isfinite(frame.expected) & frame.get("eligible", True)
                for name, frame in predictions.items()}
    evaluation = pd.Series(True, index=actual.index)
    if evaluation_mask is not None:
        if not actual.index.equals(evaluation_mask.index):
            raise ValueError("Evaluation mask must align exactly with targets")
        evaluation = evaluation_mask.fillna(False).astype(bool)
    common = pd.DataFrame(eligible).all(axis=1) & observed & evaluation
    rows = []
    for name, frame in predictions.items():
        own_error = (actual - frame.expected).loc[eligible[name] & evaluation]
        error = (actual - frame.expected).loc[common]
        ranges = common & np.isfinite(frame.lower) & np.isfinite(frame.upper)
        inside = actual.loc[ranges].between(frame.lower.loc[ranges], frame.upper.loc[ranges])
        rows.append({
            "method": name, "requested": len(actual), "observed": int(observed.sum()),
            "excluded_rows": int((observed & ~evaluation).sum()),
            "available": int(eligible[name].sum()),
            "availability": eligible[name].mean() if len(actual) else np.nan,
            "own_mae": own_error.abs().mean(), "own_bias": own_error.mean(),
            "common": int(common.sum()), "mae": error.abs().mean(),
            "bias": error.mean(), "median_residual": error.median(),
            "residual_sd": error.std(), "p90_absolute_error": error.abs().quantile(.9),
            "range_rows": int(ranges.sum()),
            "range_coverage": inside.mean() if len(inside) else np.nan,
            "range_width": (frame.upper - frame.lower).loc[ranges].mean(),
        })
    return pd.DataFrame(rows)


def fit_pair_reference(predictor: pd.Series, target: pd.Series) -> pd.Series:
    """Two-parameter least-squares reference for explicitly selected paired values.

    This is a contemporaneous relationship, not a forecast or uncertainty model.
    Support and relationship stability must be assessed before calling it.
    """
    if not predictor.index.equals(target.index):
        raise ValueError("Paired measurements must align")
    valid = np.isfinite(predictor) & np.isfinite(target)
    x, y = predictor.loc[valid], target.loc[valid]
    if len(x) < 3 or x.nunique() < 2:
        raise ValueError("Require at least three finite pairs and varying predictor")
    centred = x - x.mean()
    slope = centred.dot(y - y.mean()) / centred.dot(centred)
    return pd.Series({"intercept": y.mean() - slope * x.mean(), "slope": slope,
                      "observations": len(x), "predictor_min": x.min(), "predictor_max": x.max()})
