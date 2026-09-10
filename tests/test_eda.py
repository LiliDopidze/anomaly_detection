import numpy as np
import pandas as pd

from telco_anomaly.eda import regular_segment


def _daily_series(days=14):
    index = pd.date_range(
        "2025-01-01", periods=96 * days, freq="15min", tz="UTC"
    )
    return pd.Series(
        np.sin(np.arange(len(index)) * 2 * np.pi / 96), index=index
    )


def test_regular_segment_bridges_isolated_missing_readings():
    series = _daily_series()
    thinned = series.drop(series.index[48::96])
    result = regular_segment(thinned, 900, max_fill=2)
    assert len(result) == len(series)


def test_regular_segment_does_not_bridge_long_gaps():
    series = _daily_series()
    gapped = series.drop(series.index[500:503])
    result = regular_segment(gapped, 900, max_fill=2)
    assert len(result) == len(series) - 503


def test_regular_segment_rejects_invalid_settings():
    series = _daily_series(days=1)
    try:
        regular_segment(series, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("A zero cadence was accepted")
