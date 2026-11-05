"""Median/IQR feature departures and causal alerts; no source-specific names."""

import numpy as np
import pandas as pd


def fit_departure_reference(values: pd.DataFrame) -> pd.DataFrame:
    """Freeze marginal medians and IQRs from explicitly selected complete rows.

    Zero IQR is unsupported, including tied but nonconstant references. No scale
    is inferred from application data and no epsilon or fallback scale is added.
    """
    if values.empty or not values.columns.is_unique or not np.isfinite(values.to_numpy()).all():
        raise ValueError('Fit on nonempty, finite, explicitly selected reference rows')
    quartiles = values.quantile([.25, .5, .75])
    reference = pd.DataFrame({'median': quartiles.loc[.5],
                              'iqr': quartiles.loc[.75] - quartiles.loc[.25],
                              'observations': len(values), 'distinct': values.nunique()})
    reference['supported'] = np.isfinite(reference.iqr) & reference.iqr.gt(0)
    reference.index.name = 'feature'
    return reference


def departure_scores(reference: pd.DataFrame, values: pd.DataFrame):
    """Return maximum absolute median departure / fitted IQR, and its components.

    Scores are unbounded magnitudes, not probabilities or normal-distribution
    z-scores. Every requested feature must have a finite value and supported
    fitted spread; otherwise the vector score remains unavailable.
    """
    if list(values) != list(reference.index):
        raise ValueError('Feature names and order must match the fitted reference')
    scale = reference.iqr.where(reference.supported)
    components = values.sub(reference['median'], axis='columns').abs().div(scale, axis='columns')
    components = components.where(np.isfinite(components))
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
