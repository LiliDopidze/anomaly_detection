from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from telco_anomaly.adapter import (
    adapt_frame,
    adapt_inventory,
    adapt_events,
    load_mapping,
    write_pack,
    verify_pack,
)
from telco_anomaly.synthetic import SyntheticConfig, generate_dataset


@pytest.fixture
def mapping():
    return load_mapping(Path(__file__).parents[1] / "configs/adapter.yml")


def native(mapping, n=6):
    data = pd.DataFrame(
        {spec["source"]: np.ones(n) for spec in mapping["columns"].values()}
    )
    data["ont_id"] = "A"
    data["timestamp_utc"] = pd.date_range(
        "2025-01-01", periods=n, freq="15min", tz="UTC"
    )
    data["uptime_s"] = np.arange(n) * 900 + 9000
    data["reboot_count"] = 0
    return data


def test_units_vendor_override_and_unknown_column(mapping):
    data = native(mapping)
    mapping["columns"]["rx_power_dbm"]["unit"] = "mW"
    data.rx_power_dbm = 0.01
    result, quality = adapt_frame(data, mapping, 900)
    np.testing.assert_allclose(result.rx_power_dbm, -20)
    data["fault_label"] = 1
    with pytest.raises(ValueError, match="Unreviewed"):
        adapt_frame(data, mapping, 900)


def test_counter_differences_never_bridge_gaps_or_resets(mapping):
    data = native(mapping)
    mapping["columns"]["fec_count"]["kind"] = "cumulative"
    data.fec_count = [100, 120, 150, 160, 10, 20]
    data.loc[3:, "timestamp_utc"] += pd.Timedelta(minutes=15)
    result, quality = adapt_frame(data, mapping, 900)
    np.testing.assert_allclose(
        result.fec_count, [np.nan, 20, 30, np.nan, np.nan, 10], equal_nan=True
    )
    assert quality.fec_count.iloc[4] == "counter_discontinuity"
    data.fec_count = [100, 120, 150, 160, 200, 220]
    data.loc[4, "reboot_count"] = 1
    result, _ = adapt_frame(data, mapping, 900)
    assert np.isnan(result.fec_count.iloc[4])


def test_invalid_power_duplicate_and_dst_refused(mapping):
    data = native(mapping)
    mapping["columns"]["rx_power_dbm"]["unit"] = "mW"
    data.loc[0, "rx_power_dbm"] = 0
    result, quality = adapt_frame(data, mapping, 900)
    assert np.isnan(result.rx_power_dbm.iloc[0])
    assert quality.rx_power_dbm.iloc[0] == "invalid"
    with pytest.raises(ValueError, match="Duplicate"):
        adapt_frame(pd.concat([data, data.iloc[[0]]], ignore_index=True), mapping, 900)
    data = native(mapping, 1)
    data["timestamp_utc"] = ["2025-10-26 01:30"]
    mapping["timezone"] = "Europe/London"
    with pytest.raises(Exception, match="[Aa]mbiguous|[Cc]annot infer"):
        adapt_frame(data, mapping, 900)


def test_vendor_counter_override(mapping):
    data = native(mapping)
    data["vendor"] = "counter_vendor"
    data.fec_count = np.arange(6) * 10
    mapping["vendor_column"] = "vendor"
    mapping["columns"]["fec_count"]["overrides"] = {
        "counter_vendor": {"kind": "cumulative"}
    }
    result, _ = adapt_frame(data, mapping, 900)
    assert result.fec_count.iloc[1:].eq(10).all()


def test_pack_blind_rebuild_and_truth_rejection(tmp_path, mapping):
    cfg = SyntheticConfig(
        days=2, n_onts=4, n_olts=1, ports_per_olt=1, splitters_per_port=1
    )
    source = generate_dataset(cfg, tmp_path / "source")
    data = pd.read_parquet(source / "reference_dataset.parquet")
    inventory = pd.read_csv(source / "topology.csv")
    metadata = {
        "start": cfg.start,
        "days": cfg.days,
        "sample_minutes": cfg.sample_minutes,
        "n_onts": cfg.n_onts,
    }
    a = write_pack(data, inventory, tmp_path / "a", mapping, metadata)
    for file in ("gt_fault_registry.csv", "fault_entity_intervals.csv"):
        (source / file).unlink()
    b = write_pack(data, inventory, tmp_path / "b", mapping, metadata)
    assert verify_pack(a)["files"] == verify_pack(b)["files"]
    (b / "labels.csv").write_text("forbidden")
    with pytest.raises(ValueError, match="Unexpected"):
        verify_pack(b)


def test_unknown_alarm_code_and_topology_overlap(mapping):
    events = pd.DataFrame(
        {"timestamp_utc": ["2025-01-01Z"], "ont_id": ["A"], "alarm_code": ["guess"]}
    )
    with pytest.raises(ValueError, match="Unreviewed alarm"):
        adapt_events(events, mapping, ["A"])
    cols = mapping["inventory"]["columns"]
    inventory = pd.DataFrame({name: ["A", "A"] for name in cols.values()})
    with pytest.raises(ValueError, match="Overlapping"):
        adapt_inventory(
            inventory,
            mapping,
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-02-01", tz="UTC"),
        )


def test_measurements_cannot_fall_outside_inventory_membership(tmp_path, mapping):
    data = native(mapping)
    mapping["inventory"]["columns"].update(valid_from="valid_from", valid_to="valid_to")
    inventory = pd.DataFrame(
        {
            "ont_id": ["A"],
            "olt_id": ["O"],
            "pon_port": ["P"],
            "splitter_l1": ["L1"],
            "splitter_l2": ["L2"],
            "rx_sensitivity_dbm": [-28.0],
            "valid_from": ["2025-01-01T12:00:00Z"],
            "valid_to": ["2025-01-02T00:00:00Z"],
        }
    )
    with pytest.raises(ValueError, match="outside effective inventory"):
        write_pack(
            data,
            inventory,
            tmp_path / "pack",
            mapping,
            {
                "start": "2025-01-01T00:00:00Z",
                "days": 1,
                "sample_minutes": 15,
                "n_onts": 1,
            },
        )
    assert not (tmp_path / "pack").exists()
