"""Checks for matching."""

import pandas as pd
from optical_anomaly.evaluation import Evaluator, match_events


def test_matching_and_operational_denominators(operational_tables):
    truth, alerts, telemetry = operational_tables
    start = pd.Timestamp("2025-01-01", tz="UTC")
    minute = pd.Timedelta(minutes=1)
    metrics, _ = Evaluator().evaluate(
        alerts, truth, telemetry, start, start + 360 * minute
    )
    assert metrics["pre_impact_recall"] == 0.5
    assert metrics["median_detection_delay_minutes"] == 15
    assert metrics["duplicate_incidents"] == metrics["unmatched_incidents"] == 1
    assert metrics["nuisance_per_1000_entity_days"] == 8000
    overlap = truth.copy()
    overlap.loc[1, "onset_time"] = start + 10 * minute
    overlap.loc[1, "end_time"] = start + 30 * minute
    assert len(match_events(alerts.iloc[:2], overlap)) == 2
