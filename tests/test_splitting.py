"""Checks for splitting."""

import pandas as pd
import pytest
from optical_anomaly.splitting import TemporalSplit


def test_split_no_overlap():
    times = pd.Series(pd.date_range("2025-01-01", periods=20, freq="h", tz="UTC"))
    split = TemporalSplit(*[times.iloc[i] for i in (5, 10, 15, 19)])
    masks = split.masks(times)
    assert (sum(masks.values()) <= 1).all()
    assert masks["validation"].iloc[10]
    assert masks["test"].iloc[15]


@pytest.mark.parametrize("boundary_index", [0, 1, 2, 3])
@pytest.mark.parametrize("offset_us", [-1, 0, 1])
def test_exact_half_open_boundaries(boundary_index, offset_us):
    start = pd.Timestamp("2025-01-01", tz="UTC")
    boundaries = [start + pd.Timedelta(hours=h) for h in (5, 10, 15, 19)]
    split = TemporalSplit(*boundaries)
    instant = boundaries[boundary_index] + pd.Timedelta(microseconds=offset_us)
    masks = split.masks(pd.Series([instant]))
    selected = [name for name, mask in masks.items() if mask.iloc[0]]
    names = ["train", "calibration", "validation", "test"]
    expected_index = boundary_index if offset_us < 0 else boundary_index + 1
    assert selected == ([names[expected_index]] if expected_index < len(names) else [])
