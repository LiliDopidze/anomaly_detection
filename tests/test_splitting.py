"""Checks for splitting."""

import pandas as pd
from optical_anomaly.splitting import TemporalSplit


def test_split_no_overlap():
    times = pd.Series(pd.date_range("2025-01-01", periods=20, freq="h", tz="UTC"))
    split = TemporalSplit(*[times.iloc[i] for i in (5, 10, 15, 19)])
    masks = split.masks(times)
    assert (sum(masks.values()) <= 1).all()
    assert masks["validation"].iloc[10]
    assert masks["test"].iloc[15]
