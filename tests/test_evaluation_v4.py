import pandas as pd

from telco_anomaly.contract import EVAL_SCHEMAS, SPLIT_SCHEMAS
from telco_anomaly.evaluation import CASE_COLUMNS, evaluate_cases, partition_truth


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
