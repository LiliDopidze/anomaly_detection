"""Empirical feature rarity and causal alert decisions; no source-specific names."""

import numpy as np
import pandas as pd


def fit_rarity(values: pd.DataFrame) -> dict:
    """Freeze sorted marginal references from explicitly selected complete rows."""
    if values.empty or not np.isfinite(values.to_numpy()).all():
        raise ValueError('Fit on nonempty, finite, explicitly selected reference rows')
    return {name: np.sort(values[name].to_numpy(dtype=float)) for name in values}


def rarity_scores(reference: dict, values: pd.DataFrame):
    """Inclusive empirical tails with finite-sample smoothing; these are not p-values.

    Two-sided rarity is 1-min(1, 2*min(lower_tail, upper_tail)). Both unusually
    high and low feature values matter. Ties retain their full tail mass.
    Values beyond reference extremes share a maximum score: no tail extrapolation.
    The maximum marginal rarity is available only for a complete feature vector.
    """
    if list(values) != list(reference):
        raise ValueError('Feature names and order must match the fitted reference')
    components = pd.DataFrame(index=values.index, columns=values.columns, dtype=float)
    for name, sorted_values in reference.items():
        observed = values[name].where(np.isfinite(values[name]))
        n = len(sorted_values)
        lower = (1 + np.searchsorted(sorted_values, observed, side='right')) / (n + 1)
        upper = (1 + n - np.searchsorted(sorted_values, observed, side='left')) / (n + 1)
        tail = np.minimum(1, 2 * np.minimum(lower, upper))
        components[name] = pd.Series(1 - tail, index=values.index).where(observed.notna())
    score = components.max(axis=1).where(components.notna().all(axis=1)).rename('score')
    return score, components


def calibrate_threshold(scores: pd.Series, quantile=.99) -> pd.Series:
    """Cutoff from separate reference scores; strict exceedance handles ties visibly."""
    finite = scores[np.isfinite(scores)]
    if finite.empty or not 0 < quantile < 1:
        raise ValueError('Require finite calibration scores and quantile in (0, 1)')
    cutoff = finite.quantile(quantile)
    return pd.Series({'quantile': quantile, 'threshold': cutoff, 'rows': len(finite),
                      'distinct_scores': finite.nunique(),
                      'exceeded_fraction': finite.gt(cutoff).mean(),
                      'threshold_at_maximum': bool(cutoff == finite.max())})


def causal_alerts(scores: pd.Series, threshold: float, consecutive=1, frequency='5s'):
    """Decide at each endpoint; missing scores or nonadjacent times reset history.

    No backdating, cooldown or episode merging. NaN means an unavailable decision,
    zero an available non-alert. An alert stays active only while scores exceed
    the cutoff. A new alert can start only after the required consecutive windows.
    """
    if not isinstance(consecutive, int) or consecutive < 1 or not np.isfinite(threshold):
        raise ValueError('Require a positive integer run length and a finite threshold')
    if not scores.index.is_unique or not scores.index.is_monotonic_increasing:
        raise ValueError('Supply one recording with sorted unique timestamps')
    step = pd.Timedelta(frequency)
    streak, previous_alarm, previous_time = 0, False, None
    rows = []
    for time, score in scores.items():
        if previous_time is None or time - previous_time != step:
            streak, previous_alarm = 0, False
        above = alarm = np.nan
        onset = False
        if np.isfinite(score):
            above = float(score > threshold)
            streak = streak + 1 if above else 0
            alarm = float(streak >= consecutive)
            onset = bool(alarm and not previous_alarm)
            previous_alarm = bool(alarm)
        else:
            streak, previous_alarm = 0, False
        rows.append({'above_threshold': above, 'alarm': alarm, 'alarm_start': onset,
                     'consecutive_windows': float(streak) if np.isfinite(score) else np.nan})
        previous_time = time
    return pd.DataFrame(rows, index=scores.index)
