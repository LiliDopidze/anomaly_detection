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


def test_variance_opportunity_and_delay_declare_physical_onset(
    operational_tables,
):
    truth, alerts, telemetry = operational_tables
    truth = truth.iloc[:1].copy()
    truth["fault_type"] = "variance_shift"
    truth["observable_onset_time"] = pd.NaT
    from optical_anomaly.evaluation import Evaluator

    metrics, outcomes = Evaluator().evaluate(
        alerts.iloc[:1],
        truth,
        telemetry,
        telemetry.timestamp.min(),
        telemetry.timestamp.max() + pd.Timedelta(minutes=5),
    )
    assert outcomes.iloc[0].delay_reference == "onset_time"
    assert outcomes.iloc[0].delay_minutes == 15
    assert outcomes.iloc[0].opportunity
    assert outcomes.iloc[0].opportunity_reference == "onset_time"
    assert metrics["physical_onset_warning_opportunities"] == 1
    assert metrics["physical_onset_pre_impact_recall"] == 1
    assert metrics["observable_warning_opportunities"] == 0
    assert metrics["median_detection_delay_minutes"] is None
    assert metrics["median_variance_onset_delay_minutes"] == 15


def test_variance_without_impact_has_no_warning_opportunity(operational_tables):
    truth, alerts, telemetry = operational_tables
    truth = truth.iloc[:1].copy()
    truth["fault_type"] = "variance_shift"
    truth["observable_onset_time"] = pd.NaT
    truth["impact_time"] = pd.NaT
    metrics, outcomes = Evaluator().evaluate(
        alerts, truth, telemetry, telemetry.timestamp.min(),
        telemetry.timestamp.max() + pd.Timedelta(minutes=5),
    )
    assert not outcomes.iloc[0].opportunity
    assert metrics["pre_impact_recall"] is None
