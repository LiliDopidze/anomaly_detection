import pandas as pd

from telco_anomaly.contract import EVAL_SCHEMAS, SPLIT_SCHEMAS
from telco_anomaly.evaluation import (
    CASE_COLUMNS,
    evaluate_cases,
    partition_truth,
    poisson_rate_interval,
    scores_to_alerts,
)


def score_fixture(values):
    return pd.DataFrame({
        "event_ts": pd.date_range("2025-01-01", periods=len(values), freq="15min", tz="UTC"),
        "entity_id": "ONT-1",
        "episode_id": "episode-1",
        "anomaly_score": values,
        "model_id": "detector",
    })


def test_persistent_alert_fires_when_confirmation_is_observed():
    alerts = scores_to_alerts(
        score_fixture([0.0, 2.0, 2.0, 0.0, 0.0]),
        1.0,
        min_consecutive=2,
        recovery_consecutive=2,
        cadence_seconds=900,
    )

    assert len(alerts) == 1
    assert alerts.loc[0, "alert_start"] == pd.Timestamp(
        "2025-01-01 00:30:00", tz="UTC"
    )
    assert alerts.loc[0, "alert_end"] == pd.Timestamp(
        "2025-01-01 01:00:00", tz="UTC"
    )


def test_missing_score_breaks_persistence_at_declared_cadence():
    alerts = scores_to_alerts(
        score_fixture([2.0, float("nan"), 2.0]),
        1.0,
        min_consecutive=2,
        recovery_consecutive=1,
        cadence_seconds=900,
    )

    assert alerts.empty


def test_recovery_hysteresis_prevents_threshold_flicker():
    values = [2.0, 2.0, 0.9, 2.0, 2.0, 0.7]
    with_hysteresis = scores_to_alerts(
        score_fixture(values),
        1.0,
        min_consecutive=2,
        recovery_consecutive=1,
        recovery_threshold=0.8,
        cadence_seconds=900,
    )
    without_hysteresis = scores_to_alerts(
        score_fixture(values),
        1.0,
        min_consecutive=2,
        recovery_consecutive=1,
        cadence_seconds=900,
    )

    assert len(with_hysteresis) == 1
    assert len(without_hysteresis) == 2


def test_poisson_interval_uses_requested_confidence_level():
    _, upper_95 = poisson_rate_interval(20, 1_000, 0.95)
    _, upper_99 = poisson_rate_interval(20, 1_000, 0.99)

    assert upper_99 > upper_95


def test_zero_case_candidate_is_a_valid_negative_result():
    start = pd.Timestamp("2025-01-02", tz="UTC")
    events = pd.DataFrame([{
        "fault_id": "F-1",
        "fault_type": "fibre_bend",
        "domain_type": "entity",
        "domain_id": "ONT-1",
        "onset_ts": start,
        "observable_ts": start,
        "impact_ts": start + pd.Timedelta(hours=1),
        "end_ts": start + pd.Timedelta(hours=2),
        "group_id": pd.NA,
        "label_source": "fixture",
        "source_instance_id": "fixture-1",
    }])
    intervals = pd.DataFrame([{
        "fault_id": "F-1",
        "entity_id": "ONT-1",
        "start_ts": start,
        "end_ts": start + pd.Timedelta(hours=2),
        "label_source": "fixture",
        "source_instance_id": "fixture-1",
    }])
    cases = pd.DataFrame(columns=CASE_COLUMNS)
    members = pd.DataFrame(
        columns=["case_id", "alert_id", "entity_id", "model_id"]
    )

    result = evaluate_cases(
        cases,
        members,
        events,
        intervals,
        exposure_value=10,
        exposure_unit="entity_day",
        decision_horizon_seconds=24 * 3600,
    )
    metrics = result["metrics"].set_index("metric")["value"]

    assert metrics["event_recall"] == 0
    assert metrics["false_cases_per_entity_day"] == 0
    assert result["case_matches"].empty


def test_cross_partition_fault_is_audited_and_published_nowhere():
    start = pd.Timestamp("2025-01-01", tz="UTC")
    events = pd.DataFrame([[
        "F-1", "long_fault", "entity", "ONT-1", start, start,
        start + pd.Timedelta(hours=1), start + pd.Timedelta(hours=3),
        pd.NA, "fixture", "fixture-1",
    ]], columns=EVAL_SCHEMAS["fault_events"])
    intervals = pd.DataFrame([[
        "F-1", "ONT-1", start, start + pd.Timedelta(hours=3),
        "fixture", "fixture-1",
    ]], columns=EVAL_SCHEMAS["fault_entity_intervals"])
    registry = pd.DataFrame([{
        "entity_id": "ONT-1",
        "observed_from": start,
        "observed_to": start + pd.Timedelta(hours=4),
    }])
    partitions = pd.DataFrame([
        ("calibration", start, start + pd.Timedelta(hours=2), "split-v1"),
        ("development", start + pd.Timedelta(hours=2), start + pd.Timedelta(hours=3), "split-v1"),
        ("holdout", start + pd.Timedelta(hours=3), start + pd.Timedelta(hours=4), "split-v1"),
    ], columns=SPLIT_SCHEMAS["time_partitions"])
    empty_conditions = pd.DataFrame(columns=EVAL_SCHEMAS["condition_states"])
    empty_entities = pd.DataFrame(columns=SPLIT_SCHEMAS["entity_partitions"])

    output, audit, _ = partition_truth(
        events,
        intervals,
        empty_conditions,
        registry,
        primary_split="time",
        time_partitions=partitions,
        entity_partitions=empty_entities,
    )

    assert audit.loc[0, "status"] == "cross_partition"
    assert audit.loc[0, "scoreable"] == False  # noqa: E712
    assert all(tables["fault_events"].empty for tables in output.values())
