"""Checks for state machine."""

import numpy as np
import pandas as pd
import pytest
from optical_anomaly.incidents import IncidentManager


def test_incident_debounce_hysteresis_chunk_equivalence():
    times = pd.date_range("2025-01-01", periods=10, freq="5min", tz="UTC")
    scores = pd.DataFrame(
        {
            "timestamp": times,
            "entity_id": "A",
            "score": [0.99, np.nan, 0.99, 0.99, 0.99, 0.9, 0.1, np.nan, 0.1, 0.1],
        }
    )
    policy = dict(high=0.95, low=0.2, opening_intervals=3, closing_intervals=2)
    full = IncidentManager(**policy).process(scores)
    assert full.start_time.iloc[0] == times[4]
    assert full.end_time.iloc[0] == times[9]
    manager = IncidentManager(**policy)
    manager.process(scores.iloc[:6])
    chunks = manager.process(scores.iloc[6:])
    pd.testing.assert_frame_equal(full, chunks)
    with pytest.raises(ValueError, match="overlapping"):
        manager.process(scores.iloc[-1:])


def test_unknown_period_is_not_recovery_and_long_gap_is_explicit():
    times = pd.date_range("2025-01-01", periods=90, freq="5min", tz="UTC")
    scores = pd.DataFrame(
        {"timestamp": times, "entity_id": "A", "score": [1.0, 1.0, 1.0] + [np.nan] * 87}
    )
    manager = IncidentManager()
    opened = manager.process(scores.iloc[:10])
    assert opened.status.iloc[0] == "OPEN"
    assert pd.isna(opened.end_time.iloc[0])
    closed = manager.process(scores.iloc[10:])
    assert closed.reason.iloc[0] == "telemetry_gap"
    assert closed.end_time.iloc[0] == times[2]
