from __future__ import annotations

import pandas as pd


def native_frames():
    timestamps = pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC")
    rows = []
    for entity_index, entity_id in enumerate(("ONT-00001", "ONT-00002")):
        for step, timestamp in enumerate(timestamps):
            if entity_id == "ONT-00002" and step == 3:
                continue
            rows.append(
                {
                    "timestamp_utc": timestamp,
                    "ont_id": entity_id,
                    "rx_power_dbm": (
                        None
                        if entity_id == "ONT-00001" and step == 1
                        else -20.0 - entity_index - 0.1 * step
                    ),
                    "fec_count": (
                        5_000_000.25
                        if entity_id == "ONT-00001" and step == 4
                        else step + entity_index
                    ),
                    "crc_errors": step * 2 + entity_index,
                    "throughput_mbps": 10.0 + 5.0 * step + 2.0 * entity_index,
                    "gt_state": "degrading" if step >= 5 else "healthy",
                    "gt_fault_id": "F-00001" if step >= 5 else None,
                    "gt_margin_db": 7.0 - 0.1 * step,
                }
            )
    panel = pd.DataFrame(rows)

    topology = pd.DataFrame(
        [
            {
                "ont_id": "ONT-00001",
                "olt_id": "OLT-01",
                "pon_port": "OLT-01-PON-01",
                "splitter_l1": "L1-001",
                "splitter_l2": "L2-001",
                "device_model": "MODEL-A",
                "vendor": "VENDOR-A",
                "enclosure": "indoor",
                "firmware_version": "1.0",
                "geo_cluster": "GEO-1",
                "distance_m": 2000,
                "distance_bucket": "near",
                "splitter_ratio": "1:8",
                "l2_splitter_capacity": 8,
                "fibre_age_yr": 4.0,
                "ont_age_yr": 2.0,
                "expected_rx_power_dbm": -19.5,
                "rx_sensitivity_dbm": -27.0,
                "service_impact_weight": 1.0,
                "customer_priority_weight": 1.0,
                "gt_frailty": 1.2,
                "gt_chronic": False,
            },
            {
                "ont_id": "ONT-00002",
                "olt_id": "OLT-01",
                "pon_port": "OLT-01-PON-01",
                "splitter_l1": "L1-001",
                "splitter_l2": "L2-001",
                "device_model": "MODEL-B",
                "vendor": "VENDOR-B",
                "enclosure": "outdoor",
                "firmware_version": "2.0",
                "geo_cluster": "GEO-1",
                "distance_m": 2400,
                "distance_bucket": "near",
                "splitter_ratio": "1:8",
                "l2_splitter_capacity": 8,
                "fibre_age_yr": 6.0,
                "ont_age_yr": 3.0,
                "expected_rx_power_dbm": -20.5,
                "rx_sensitivity_dbm": -27.0,
                "service_impact_weight": 2.0,
                "customer_priority_weight": 1.5,
                "gt_frailty": 0.8,
                "gt_chronic": True,
            },
        ]
    )

    service_windows = pd.DataFrame(
        {
            "entity_id": ["ONT-00001", "ONT-00002"],
            "install_ts": [timestamps[0], timestamps[0]],
            "decommission_ts": [pd.NaT, pd.NaT],
        }
    )
    tickets = pd.DataFrame(
        [
            {
                "ticket_id": "TKT-OPAQUE1",
                "ont_id": "ONT-00001",
                "reported_ts": timestamps[6],
                "resolved_ts": timestamps[7],
                "reported_symptom": "slow_service",
                "gt_fault_id": "F-00001",
                "gt_fault_type": "connector_contamination",
                "gt_is_nff": False,
                "gt_misattributed": False,
            }
        ]
    )
    engineering = pd.DataFrame(
        [
            {
                "entity_id": "OLT-01-PON-01",
                "ts": timestamps[2],
                "event_type": "planned_maintenance",
                "level_change_db": 0.0,
                "detail": "one hour",
            }
        ]
    )
    return panel, topology, service_windows, tickets, engineering


def evaluation_frames():
    timestamps = pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC")
    registry = pd.DataFrame(
        [
            {
                "gt_fault_id": "F-00001",
                "gt_fault_type": "splitter_degradation",
                "family": "optical_ramp",
                "scope": "l2",
                "target": "L2-001",
                "group_id": "STORM-001",
                "onset_ts": timestamps[4],
                "first_observable_ts": timestamps[5],
                "impact_ts": timestamps[6],
                "repair_ts": timestamps[7],
                "repair_source": "ticket",
                "averted": False,
                "left_censored": False,
            },
            {
                "gt_fault_id": "F-00002",
                "gt_fault_type": "distribution_damage",
                "family": "optical_step",
                "scope": "l1",
                "target": "L1-001",
                "group_id": "STORM-001",
                "onset_ts": timestamps[4],
                "first_observable_ts": timestamps[5],
                "impact_ts": timestamps[6],
                "repair_ts": timestamps[7],
                "repair_source": "ticket",
                "averted": False,
                "left_censored": False,
            },
        ]
    )
    intervals = pd.DataFrame(
        [
            {
                "fault_id": "F-00001",
                "entity_id": "ONT-00001",
                "fault_family": "optical_ramp",
                "channel": "optical",
                "active_start_ts": timestamps[4],
                "active_end_ts": timestamps[7],
                "impact_ts": timestamps[6],
                "contribution_db": 2.0,
            },
            {
                "fault_id": "F-00001",
                "entity_id": "ONT-00002",
                "fault_family": "optical_ramp",
                "channel": "optical",
                "active_start_ts": timestamps[5],
                "active_end_ts": timestamps[7],
                "impact_ts": timestamps[6],
                "contribution_db": 1.4,
            },
            {
                "fault_id": "F-00002",
                "entity_id": "ONT-00002",
                "fault_family": "optical_step",
                "channel": "optical",
                "active_start_ts": timestamps[4],
                "active_end_ts": timestamps[7],
                "impact_ts": timestamps[6],
                "contribution_db": 1.8,
            },
        ]
    )
    groups = pd.DataFrame(
        [
            {
                "group_id": "STORM-001",
                "start_ts": timestamps[3],
                "end_ts": timestamps[7],
                "geo_clusters": "GEO-1",
            }
        ]
    )
    return registry, intervals, groups
