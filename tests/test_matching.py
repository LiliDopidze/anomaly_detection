"""Checks for matching."""

import pandas as pd
from optical_anomaly.evaluation import Evaluator, match_events


def test_matching_and_operational_denominators():
    start = pd.Timestamp("2025-01-01", tz="UTC")
    minute = pd.Timedelta(minutes=1)
    truth = pd.DataFrame(
        {
            "fault_id": ["F1", "F2"],
            "entity_id": ["A", "A"],
            "fault_type": ["drift", "drift"],
            "onset_time": [start, start + 180 * minute],
            "observable_onset_time": [start, start + 180 * minute],
            "impact_time": [start + 90 * minute, start + 270 * minute],
            "end_time": [start + 150 * minute, start + 330 * minute],
        }
    )
    alerts = pd.DataFrame(
        {
            "incident_id": ["I1", "I2", "I3"],
            "entity_id": "A",
            "start_time": [start + n * minute for n in (15, 40, 350)],
        }
    )
    telemetry = pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=72, freq="5min"),
            "entity_id": "A",
            "metric_name": "rx_power_dbm",
            "value": -20.0,
        }
    )
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
