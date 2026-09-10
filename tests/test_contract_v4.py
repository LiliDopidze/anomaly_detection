from pathlib import Path

import pandas as pd
import pytest
import yaml

from telco_anomaly.adapters.synthetic_pon import build_synthetic_pon_pack
from telco_anomaly.contract import (
    CORE_VERSION,
    build_canonical,
    check_core,
    core_fingerprint,
    truth_like_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _configs():
    metrics = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "metric_registry.yml").read_text()
    )
    topology = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "topology.yml").read_text()
    )
    return metrics, topology


def _pon_source(root):
    root.mkdir()
    times = pd.date_range("2025-01-01", periods=4, freq="15min", tz="UTC")
    rows = []
    for entity, offset in (("ONT-1", 0.0), ("ONT-2", 0.2)):
        for number, timestamp in enumerate(times):
            rows.append({
                "timestamp_utc": timestamp,
                "ont_id": entity,
                "rx_power_dbm": -18.0 - offset,
                "olt_rx_power_dbm": -20.0 - offset,
                "tx_power_dbm": 2.0,
                "temperature_c": 40.0 + number,
                "bias_current_ma": 8.0,
                "voltage_v": 3.3,
                "ber": 0.0,
                "fec_count": 5_000_000 if number == 2 else number,
                "crc_errors": number,
                "uptime_s": number * 900,
                "reboot_count": 0,
                "throughput_mbps": 100.0,
            })
    pd.DataFrame(rows).to_parquet(root / "reference_dataset.parquet", index=False)
    pd.DataFrame({
        "ont_id": ["ONT-1", "ONT-2"],
        "vendor": ["A", "B"],
        "device_model": ["M1", "M2"],
        "olt_id": ["OLT-1", "OLT-1"],
        "pon_port": ["PON-1", "PON-1"],
        "splitter_l1": ["L1-1", "L1-1"],
        "splitter_l2": ["L2-1", "L2-1"],
        "geo_cluster": ["GEO-1", "GEO-1"],
    }).to_csv(root / "topology.csv", index=False)
    pd.DataFrame({
        "entity_id": ["ONT-1", "ONT-2"],
        "install_ts": [times[0], times[0]],
        "decommission_ts": [pd.NaT, pd.NaT],
    }).to_csv(root / "entity_service_windows.csv", index=False)
    return root


def _add_evaluation_truth(root):
    times = pd.date_range("2025-01-01", periods=4, freq="15min", tz="UTC")
    pd.DataFrame([{
        "gt_fault_id": "F-1",
        "gt_fault_type": "fibre_bend",
        "scope": "ont",
        "target": "ONT-1",
        "onset_ts": times[1],
        "first_observable_ts": times[1],
        "impact_ts": times[2],
        "repair_ts": times[3],
        "group_id": "G-1",
    }]).to_csv(root / "gt_fault_registry.csv", index=False)
    pd.DataFrame([{
        "fault_id": "F-1",
        "entity_id": "ONT-1",
        "active_start_ts": times[1],
        "active_end_ts": times[3],
    }]).to_csv(root / "fault_entity_intervals.csv", index=False)
    pd.DataFrame([{
        "ticket_id": "T-1",
        "ont_id": "ONT-1",
        "reported_ts": times[2],
        "resolved_ts": times[3],
        "reported_symptom": "loss of service",
        "gt_fault_id": "F-1",
        "gt_is_nff": False,
        "gt_misattributed": False,
    }]).to_csv(root / "tickets.csv", index=False)
    return root


def test_truth_guard_does_not_reject_generic_sensor_state():
    assert truth_like_columns(["state", "target_power", "class"]) == ["class"]


def test_synthetic_pon_pack_and_canonical_are_valid(tmp_path):
    metrics, topology = _configs()
    source = _pon_source(tmp_path / "source")
    pack = tmp_path / "pack"
    run = tmp_path / "canonical"

    manifest = build_synthetic_pon_pack(
        source,
        pack,
        metric_registry=metrics,
        topology_config=topology,
        include_evaluation=False,
        batch_rows=5,
    )
    build_canonical(pack, run, include_evaluation=False)
    audit = check_core(run / "SPEC-CORE")

    assert manifest["capabilities"]["topology"] is True
    assert audit["fingerprint_verified"] is True
    assert audit["telemetry_rows"] == 2 * 4 * 12
    assert CORE_VERSION == "1.0.0"

    telemetry = pd.concat(
        [pd.read_parquet(path) for path in (run / "SPEC-CORE" / "telemetry").glob("*.parquet")]
    )
    fec = telemetry.loc[telemetry["metric_id"].eq("optical.fec_count")]
    assert fec["quality_code"].eq("clipped").sum() == 2


def test_eval_mount_does_not_change_core(tmp_path):
    metrics, topology = _configs()
    source = _pon_source(tmp_path / "source")
    pack = tmp_path / "pack"
    build_synthetic_pon_pack(
        source,
        pack,
        metric_registry=metrics,
        topology_config=topology,
        include_evaluation=False,
    )
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    build_canonical(pack, run_a, include_evaluation=True)
    build_canonical(pack, run_b, include_evaluation=False)
    assert core_fingerprint(run_a / "SPEC-CORE") == core_fingerprint(run_b / "SPEC-CORE")


def test_translator_core_is_invariant_when_native_truth_files_are_removed(tmp_path):
    metrics, topology = _configs()
    labelled_source = _add_evaluation_truth(_pon_source(tmp_path / "labelled"))
    redacted_source = _pon_source(tmp_path / "redacted")
    labelled_pack = tmp_path / "labelled-pack"
    redacted_pack = tmp_path / "redacted-pack"

    labelled_manifest = build_synthetic_pon_pack(
        labelled_source,
        labelled_pack,
        metric_registry=metrics,
        topology_config=topology,
        include_evaluation=True,
        batch_rows=5,
    )
    redacted_manifest = build_synthetic_pon_pack(
        redacted_source,
        redacted_pack,
        metric_registry=metrics,
        topology_config=topology,
        include_evaluation=False,
        batch_rows=5,
    )

    assert labelled_manifest["evaluation_tables"] == [
        "fault_entity_intervals", "fault_events", "tickets"
    ]
    assert redacted_manifest["evaluation_tables"] == []
    assert labelled_manifest["fingerprint"] == redacted_manifest["fingerprint"]


def test_streaming_gap_detection_keeps_metric_level_gap_semantics(tmp_path):
    metrics, topology = _configs()
    source = _pon_source(tmp_path / "source")
    panel_path = source / "reference_dataset.parquet"
    panel = pd.read_parquet(panel_path)
    missing_ts = pd.Timestamp("2025-01-01 00:15:00", tz="UTC")
    panel = panel.loc[
        ~(panel["ont_id"].eq("ONT-1") & panel["timestamp_utc"].eq(missing_ts))
    ]
    panel.to_parquet(panel_path, index=False)

    pack = tmp_path / "pack"
    run = tmp_path / "run"
    build_synthetic_pon_pack(
        source,
        pack,
        metric_registry=metrics,
        topology_config=topology,
        include_evaluation=False,
        batch_rows=3,
    )
    build_canonical(pack, run, include_evaluation=False)

    gaps = pd.read_parquet(run / "SPEC-CORE" / "collection_gaps.parquet")
    ont_gaps = gaps.loc[gaps["entity_id"].eq("ONT-1")]
    assert len(ont_gaps) == 12
    assert ont_gaps["metric_id"].nunique() == 12
    assert ont_gaps["gap_start"].eq(missing_ts).all()
    assert ont_gaps["gap_end"].eq(pd.Timestamp("2025-01-01 00:30:00", tz="UTC")).all()


def test_gap_detection_handles_unsorted_pack_parts(tmp_path, monkeypatch):
    metrics, topology = _configs()
    source = _pon_source(tmp_path / "source")
    panel_path = source / "reference_dataset.parquet"
    panel = pd.read_parquet(panel_path).sample(frac=1, random_state=7)
    panel.to_parquet(panel_path, index=False)
    monkeypatch.setenv("ANOMALY_GAP_BUCKETS", "2")

    pack = tmp_path / "pack"
    run = tmp_path / "run"
    build_synthetic_pon_pack(
        source,
        pack,
        metric_registry=metrics,
        topology_config=topology,
        include_evaluation=False,
        batch_rows=3,
    )
    build_canonical(pack, run, include_evaluation=False)

    gaps = pd.read_parquet(run / "SPEC-CORE" / "collection_gaps.parquet")
    assert gaps.empty


def test_existing_output_is_not_overwritten(tmp_path):
    metrics, topology = _configs()
    source = _pon_source(tmp_path / "source")
    pack = tmp_path / "pack"
    build_synthetic_pon_pack(
        source,
        pack,
        metric_registry=metrics,
        topology_config=topology,
        include_evaluation=False,
    )
    with pytest.raises(FileExistsError):
        build_synthetic_pon_pack(
            source,
            pack,
            metric_registry=metrics,
            topology_config=topology,
            include_evaluation=False,
        )


def test_check_core_does_not_need_a_global_duckdb_sort(tmp_path, monkeypatch):
    metrics, topology = _configs()
    source = _pon_source(tmp_path / "source")
    pack = tmp_path / "pack"
    run = tmp_path / "run"
    build_synthetic_pon_pack(
        source,
        pack,
        metric_registry=metrics,
        topology_config=topology,
        include_evaluation=False,
        batch_rows=5,
    )
    build_canonical(pack, run, include_evaluation=False)

    def fail_if_called():
        raise AssertionError("check_core attempted an unbounded DuckDB audit")

    monkeypatch.setattr("telco_anomaly.contract._duckdb_connection", fail_if_called)
    audit = check_core(run / "SPEC-CORE")
    assert audit["duplicate_keys"] == 0
    assert audit["foreign_key_failures"] == 0
    assert audit["fingerprint_verified"] is True


def test_check_core_detects_an_unknown_metric_before_accepting_content(tmp_path):
    metrics, topology = _configs()
    source = _pon_source(tmp_path / "source")
    pack = tmp_path / "pack"
    run = tmp_path / "run"
    build_synthetic_pon_pack(
        source,
        pack,
        metric_registry=metrics,
        topology_config=topology,
        include_evaluation=False,
        batch_rows=5,
    )
    build_canonical(pack, run, include_evaluation=False)

    part = sorted((run / "SPEC-CORE" / "telemetry").glob("part-*.parquet"))[0]
    frame = pd.read_parquet(part)
    frame.loc[0, "metric_id"] = "unknown.metric"
    frame.to_parquet(part, index=False)

    with pytest.raises(ValueError, match="foreign-key audit failed"):
        check_core(run / "SPEC-CORE")
