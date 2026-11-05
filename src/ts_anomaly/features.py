"""Unscaled, completed-window features; one recording and explicit signal lists."""

import numpy as np
import pandas as pd

from .baselines import (
    calibrate_reference, completed_gauges, fit_reference, persistence_prediction,
)


def trailing_statistics(series, endpoints, window="30s", minimum_occupancy=.8,
                        max_gap="2s", nominal_interval="1s"):
    """Describe finite observations in [endpoint-window, endpoint).

    Support uses occupied nominal slots, not row count. A gap crossing the left
    edge is checked using the preceding finite observation. At the right edge,
    age is checked without consulting the next observation. No interpolation.
    Slope is ordinary least squares in units/minute; no inferential test is made.
    """
    width, gap, nominal = map(pd.Timedelta, (window, max_gap, nominal_interval))
    if min(width, gap, nominal) <= pd.Timedelta(0) or not 0 < minimum_occupancy <= 1:
        raise ValueError("Require positive intervals and occupancy in (0, 1]")
    if not isinstance(series.index, pd.DatetimeIndex) or series.empty:
        raise ValueError("Supply a nonempty series with a DatetimeIndex")
    if series.index.hasnans or not series.index.is_unique or not series.index.is_monotonic_increasing:
        raise ValueError("Supply sorted, unique, nonmissing timestamps")
    finite = series.loc[np.isfinite(series)]
    times, values = finite.index, finite.to_numpy(dtype=float)
    expected = int(np.ceil(width / nominal))
    rows = []
    for end in endpoints:
        start = end - width
        left, right = times.searchsorted([start, end], side="left")
        observed_times, observed = times[left:right], values[left:right]
        occupied = len(np.unique((observed_times - start) // nominal))
        occupancy = occupied / expected
        first_age = last_age = np.nan
        largest_gap = 0.0
        if len(observed):
            first_age = (observed_times[0] - start).total_seconds()
            last_age = (end - observed_times[-1]).total_seconds()
            if len(observed) > 1:
                largest_gap = (observed_times[1:] - observed_times[:-1]).total_seconds().max()
            if left > 0 and observed_times[0] > start:
                leading_gap = (observed_times[0] - times[left - 1]).total_seconds()
                largest_gap = max(largest_gap, leading_gap)
        reason = "available"
        if start < series.index[0]:
            reason = "warmup"
        elif not len(observed):
            reason = "no_observations"
        elif occupancy < minimum_occupancy:
            reason = "low_occupancy"
        elif largest_gap > gap.total_seconds() or first_age > gap.total_seconds():
            reason = "collection_gap"
        elif last_age > gap.total_seconds():
            reason = "stale_tail"
        slope = iqr = np.nan
        if reason == "available":
            q25, q75 = np.quantile(observed, [.25, .75])
            iqr = q75 - q25
            if len(observed) > 1:
                minutes = (observed_times - start).total_seconds().to_numpy() / 60
                centred_time = minutes - minutes.mean()
                # Remove an offset first so a constant decimal signal gives exactly zero.
                centred_value = observed - observed[0]
                centred_value -= centred_value.mean()
                slope = centred_time.dot(centred_value) / centred_time.dot(centred_time)
        rows.append({"timestamp": end, "window_start": start,
                     "observations": len(observed), "occupied_slots": occupied,
                     "expected_slots": expected, "occupancy": occupancy,
                     "first_age_s": first_age, "last_age_s": last_age,
                     "largest_gap_s": largest_gap, "reason": reason,
                     "slope": slope, "iqr": iqr})
    columns = ["timestamp", "window_start", "observations", "occupied_slots",
               "expected_slots", "occupancy", "first_age_s", "last_age_s",
               "largest_gap_s", "reason", "slope", "iqr"]
    return pd.DataFrame(rows, columns=columns).set_index("timestamp")


def engineer_recording(data, measurements, slope_measurements, variability_measurements,
                       history_measurements, calibration_seconds=60, frequency="5s",
                       min_count=4, window="30s", minimum_occupancy=.8,
                       max_gap="2s", nominal_interval="1s"):
    """Return feature values, per-feature support, and frozen calibration metadata.

    Names are <measurement>__level/slope/iqr/history. Calibration requires native
    occupancy and at least one eligible whole aggregate; aggregate counts remain
    visible in the audit. Level features start only on wholly post-calibration
    bins. Slopes/IQR/history need no fitted reference and can be available earlier.
    """
    selected = set(slope_measurements) | set(variability_measurements) | set(history_measurements)
    if not measurements or not selected.issubset(measurements) or len(set(measurements)) != len(measurements):
        raise ValueError("Supply unique measurements and subsets for temporal features")
    if calibration_seconds <= 0:
        raise ValueError("Calibration duration must be positive")
    means, counts, connected = completed_gauges(
        data, measurements, frequency, min_count, max_gap, nominal_interval)
    step = pd.Timedelta(frequency)
    start = data.timestamp.iloc[0]
    cutoff = start + pd.Timedelta(seconds=calibration_seconds)
    native = data.set_index("timestamp")
    features = pd.DataFrame(index=means.index)
    support, references = [], []
    for name in measurements:
        bins = trailing_statistics(native[name], means.index, frequency,
                                   minimum_occupancy, max_gap, nominal_interval)
        bin_reason = bins.reason.mask(counts[name].lt(min_count) & bins.reason.eq("available"),
                                      "insufficient_readings")
        actual = means[name].where(bin_reason.eq("available"))
        native_reference = calibrate_reference(
            native[name], start, cutoff, minimum_occupancy, nominal_interval)
        fit_bins = (means.index - step >= start) & (means.index <= cutoff)
        fitted = fit_reference(actual.loc[fit_bins])
        fitted.supported = bool(fitted.supported and native_reference.supported)
        references.append({"measurement": name, "calibration_seconds": calibration_seconds,
                           "fit_start": start, "fit_end": cutoff,
                           "native_observations": native_reference.observations,
                           "occupied_slots": native_reference.occupied_slots,
                           "expected_slots": native_reference.expected_slots,
                           "occupancy": native_reference.occupancy,
                           "whole_bins": int(fit_bins.sum()), **fitted.to_dict()})
        name_level = f"{name}__level"
        reason = bin_reason.copy()
        if not fitted.supported:
            reason[:] = "unsupported_calibration"
        reason.loc[means.index - step < cutoff] = "calibration_not_complete"
        features[name_level] = (actual - fitted.expected).where(reason.eq("available"))
        audit = bins.drop(columns=["slope", "iqr"]).assign(
            feature=name_level, source_reason=bin_reason, reason=reason)
        support.append(audit)
        if name in slope_measurements or name in variability_measurements:
            rolling = trailing_statistics(native[name], means.index, window,
                                          minimum_occupancy, max_gap, nominal_interval)
            for kind, names in [("slope", slope_measurements), ("iqr", variability_measurements)]:
                if name not in names:
                    continue
                feature = f"{name}__{kind}"
                reason = rolling.reason.copy()
                reason = reason.mask(reason.eq("available") & rolling[kind].isna(),
                                     "insufficient_variation_in_time")
                features[feature] = rolling[kind].where(reason.eq("available"))
                support.append(rolling.drop(columns=["slope", "iqr"]).assign(
                    feature=feature, source_reason=rolling.reason, reason=reason))
        if name in history_measurements:
            prediction = persistence_prediction(actual, connected[name], frequency)
            feature = f"{name}__history"
            reason = bin_reason.mask(bin_reason.eq("available") & ~prediction.eligible,
                                     "history_unavailable")
            features[feature] = prediction.residual
            audit = bins.drop(columns=["slope", "iqr"]).assign(
                feature=feature, source_reason=bin_reason, reason=reason)
            # Counts/occupancy describe the target bin; history has separate support.
            audit["history_start"] = audit.window_start - step
            audit["previous_bin_available"] = actual.shift().notna()
            audit["raw_history_connected"] = connected[name]
            support.append(audit)
    support = pd.concat(support).reset_index()
    support["available"] = support.reason.eq("available")
    features.index.name = "timestamp"
    return features, support, pd.DataFrame(references)


def feature_summary(values, frequency="5s"):
    """Per-feature descriptive statistics; constant spans use exact equality.

    A run's span is last minus first timestamp, not an inferred physical duration.
    Missing values and nonadjacent timestamps split runs. No IID uncertainty claim.
    """
    rows = []
    adjacent = values.index.to_series().diff().eq(pd.Timedelta(frequency))
    for name in values:
        series = values[name]
        observed = series.dropna()
        breaks = ~adjacent | series.isna() | series.shift().isna() | series.ne(series.shift())
        runs = breaks.cumsum().loc[series.notna()]
        spans = observed.index.to_series().groupby(runs).agg(["min", "max", "size"])
        longest = (spans["max"] - spans["min"]).dt.total_seconds().max()
        rows.append({"feature": name, "requested": len(series), "available": len(observed),
                     "unavailable": int(series.isna().sum()),
                     "availability": series.notna().mean(), "distinct": observed.nunique(),
                     "zeros": int(observed.eq(0).sum()), "median": observed.median(),
                     "p05": observed.quantile(.05), "p95": observed.quantile(.95),
                     "iqr": observed.quantile(.75) - observed.quantile(.25),
                     "sd": observed.std(), "constant_runs": int(spans['size'].ge(2).sum()),
                     "longest_constant_span_s": longest})
    return pd.DataFrame(rows)
