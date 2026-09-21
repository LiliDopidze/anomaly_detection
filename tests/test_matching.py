"""Checks for matching."""

import pandas as pd
import numpy as np
import pytest
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


@pytest.mark.parametrize("lead_minutes", [-1, 0, 29, 30, 31])
def test_warning_deadline_is_distinct_from_pre_impact(operational_tables, lead_minutes):
    truth, alerts, telemetry = operational_tables
    truth, alerts = truth.iloc[:1].copy(), alerts.iloc[:1].copy()
    alerts["start_time"] = truth.impact_time.iloc[0] - pd.Timedelta(
        minutes=lead_minutes
    )
    metrics, outcomes = Evaluator(minimum_lead_minutes=30).evaluate(
        alerts, truth, telemetry, telemetry.timestamp.min(),
        telemetry.timestamp.max() + pd.Timedelta(minutes=5),
    )
    assert metrics["pre_impact_recall"] == int(lead_minutes > 0)
    assert metrics["minimum_lead_recall"] == int(lead_minutes >= 30)
    assert outcomes.iloc[0].lead_minutes == lead_minutes


def test_missing_opportunity_still_counts_as_missed_impact(operational_tables):
    truth, alerts, telemetry = operational_tables
    telemetry = telemetry.copy()
    telemetry["value"] = np.nan
    metrics, outcomes = Evaluator().evaluate(
        alerts.iloc[:0], truth, telemetry, telemetry.timestamp.min(),
        telemetry.timestamp.max() + pd.Timedelta(minutes=5),
    )
    assert metrics["warning_opportunities"] == 0
    assert metrics["faults_without_warning_opportunity"] == 2
    assert metrics["pre_impact_recall"] is None
    assert metrics["pre_impact_recall_all_impacting"] == 0
    assert metrics["impacting_faults"] == metrics["missed"] == 2
    assert metrics["observation_coverage"] == 0
    assert not outcomes.detected.any()


def test_matching_is_order_independent(operational_tables):
    truth, alerts, _ = operational_tables
    assert match_events(alerts, truth) == match_events(
        alerts.iloc[::-1], truth.iloc[::-1]
    )


def test_duplicate_incident_ids_cannot_hide_workload(operational_tables):
    truth, alerts, _ = operational_tables
    with pytest.raises(ValueError, match="unique, nonmissing incident_id"):
        match_events(pd.concat([alerts, alerts.iloc[:1]]), truth)


def test_duplicate_monitoring_rows_cannot_inflate_exposure(operational_tables):
    truth, alerts, telemetry = operational_tables
    with pytest.raises(ValueError, match="Duplicate monitoring"):
        Evaluator().evaluate(
            alerts, truth, pd.concat([telemetry, telemetry.iloc[:1]]),
            telemetry.timestamp.min(),
            telemetry.timestamp.max() + pd.Timedelta(minutes=5),
        )
