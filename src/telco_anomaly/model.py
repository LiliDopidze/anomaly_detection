"""A small, interpretable detector for sustained optical-power deterioration."""

import numpy as np
import pandas as pd

POWER = ["rx_power_dbm", "olt_rx_power_dbm"]


def fit_reference(data, cadence_minutes=15, smoothing_hours=1, noise_floor_db=0.15):
    """Fit each device's usual level/spread without fault labels.

    A majority of the reference period must describe the intended normal regime.
    The reference is frozen so gradual degradation cannot become the new normal.
    """
    settings = np.array([cadence_minutes, smoothing_hours, noise_floor_db])
    if not np.isfinite(settings).all() or (settings <= 0).any():
        raise ValueError("Cadence, smoothing and noise floor must be positive")
    references = {}
    for entity, group in data.groupby("ont_id"):
        location = group[POWER].median()
        spread = (group[POWER] - location).abs().median() * 1.4826
        supported = group[POWER].notna().sum().ge(100)
        if supported.any():
            references[str(entity)] = {
                metric: {
                    "median": float(location[metric]),
                    "scale": float(max(spread[metric], noise_floor_db)),
                }
                for metric in POWER
                if supported[metric]
            }
    if not references:
        raise ValueError(
            "Insufficient reference data: need 100 valid readings per signal"
        )
    return {
        "references": references,
        "cadence_minutes": cadence_minutes,
        "smoothing_hours": smoothing_hours,
        "noise_floor_db": noise_floor_db,
    }


def score(data, model):
    """Causal EWMA of signed drops; no predictions for an uncalibrated device."""
    output = []
    cadence = model["cadence_minutes"] * 60
    alpha = -np.expm1(-cadence / (model["smoothing_hours"] * 3600))
    for entity, group in data.groupby("ont_id", sort=True):
        group = group.sort_values("timestamp_utc").reset_index(drop=True)
        result = group[["timestamp_utc", "ont_id"]].copy()
        elapsed = group.timestamp_utc.diff().dt.total_seconds().to_numpy()
        reference = model["references"].get(str(entity), {})
        for metric in POWER:
            values = group[metric].to_numpy(dtype=float)
            if metric not in reference:
                result[metric + "__z"] = np.nan
                result[metric + "__ewma"] = np.nan
                continue
            normal = reference[metric]
            z = (normal["median"] - values) / normal["scale"]
            smoothed = np.full(len(group), np.nan)
            state = np.nan
            for i, value in enumerate(z):
                if not np.isfinite(value):
                    state = np.nan
                    continue
                if not np.isfinite(state) or elapsed[i] > 1.5 * cadence:
                    state = value
                else:
                    state = (1 - alpha) * state + alpha * value
                smoothed[i] = state
            result[metric + "__z"] = z
            result[metric + "__ewma"] = smoothed
        result["score"] = result[[p + "__ewma" for p in POWER]].max(axis=1)
        result["status"] = np.where(
            result.score.notna(), "scored", "uncalibrated_or_missing"
        )
        # A transparent static comparator, not a claimed universal optics limit.
        result["static"] = -group[POWER].min(axis=1)
        output.append(result)
    return pd.concat(output, ignore_index=True)


def calibrate(scores, sensitivity=6, minimum=3):
    """Robust operational threshold, NOT a Gaussian false-alarm probability.

    Assumes most calibration observations are normal. Validate actual workload
    on subsequent periods rather than claiming independent-observation coverage.
    """
    values = scores.loc[np.isfinite(scores)]
    if len(values) < 100 or not np.isfinite(sensitivity) or sensitivity <= 0:
        raise ValueError("Need 100 calibration scores and positive sensitivity")
    median = values.median()
    scale = 1.4826 * (values - median).abs().median()
    return float(max(minimum, median + sensitivity * scale))


def warnings(
    scores, threshold, cadence_minutes=15, recovery_fraction=0.5, max_gap_hours=6
):
    """Two readings confirm a warning; two low readings confirm recovery.

    Hysteresis avoids flicker. A short gap is unknown, not recovery: keep an open
    warning. Gaps longer than max_gap_hours close at the last observation. No
    cross-device incident merging or backdated starts.
    """
    settings = np.array([threshold, cadence_minutes, recovery_fraction, max_gap_hours])
    if not np.isfinite(settings).all() or cadence_minutes <= 0 or max_gap_hours <= 0:
        raise ValueError("Warning settings must be finite with positive time intervals")
    rows = []
    for entity, group in scores.groupby("ont_id", sort=False):
        start = previous = last_valid = None
        high = low = 0
        peak = -np.inf
        for row in group.sort_values("timestamp_utc").itertuples(index=False):
            t, value = row.timestamp_utc, row.score
            gap = (
                previous is not None
                and (t - previous).total_seconds() > cadence_minutes * 90
            )
            stale = (
                last_valid is not None
                and (t - last_valid).total_seconds() > max_gap_hours * 3600
            )
            if gap or not np.isfinite(value):
                high = low = 0
            if stale:
                if start is not None:
                    rows.append((entity, start, last_valid, peak, "gap"))
                start, high, low, peak = None, 0, 0, -np.inf
            if np.isfinite(value):
                last_valid = t
                if start is None:
                    high = high + 1 if value > threshold else 0
                    if high >= 2:
                        start, peak = t, value
                else:
                    peak = max(peak, value)
                    low = low + 1 if value <= threshold * recovery_fraction else 0
                    if low >= 2:
                        rows.append((entity, start, t, peak, "recovered"))
                        start, high, low, peak = None, 0, 0, -np.inf
            previous = t
        if start is not None:
            rows.append((entity, start, last_valid, peak, "window_end"))
    result = pd.DataFrame(
        rows, columns=["entity_id", "start_ts", "end_ts", "peak_score", "closed_by"]
    )
    result.insert(0, "incident_id", [f"W-{i:06d}" for i in range(len(result))])
    return result
