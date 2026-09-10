"""Small, tested time-series helpers used by calibration EDA."""

from __future__ import annotations

import pandas as pd


def regular_segment(series, cadence_seconds, max_fill=2):
    """Regularise, bridge short internal gaps, then keep the longest run.

    Interpolation is used only for gaps of at most ``max_fill`` expected
    observations. Longer gaps remain boundaries and are never bridged.
    """

    if series.empty:
        return series
    if float(cadence_seconds) <= 0:
        raise ValueError("cadence_seconds must be positive")
    if int(max_fill) < 0:
        raise ValueError("max_fill cannot be negative")

    cadence = pd.Timedelta(seconds=float(cadence_seconds))
    grid = series.sort_index().resample(cadence).mean()
    missing = grid.isna()
    gap_id = missing.ne(missing.shift(fill_value=False)).cumsum()
    gap_length = missing.groupby(gap_id).transform("sum")
    short_gap = missing & gap_length.le(int(max_fill))
    bridged = grid.interpolate(limit_area="inside").where(~missing | short_gap)

    valid = bridged.notna()
    if not valid.any():
        return bridged.iloc[:0]
    run_id = (~valid).cumsum()
    longest = run_id.loc[valid].value_counts().idxmax()
    return bridged.loc[valid & run_id.eq(longest)]
