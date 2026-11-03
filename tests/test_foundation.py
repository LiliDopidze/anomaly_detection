"""Measurement semantics, counterfactual controls and stochastic stress tests."""

from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
import yaml
from optical_anomaly.adapter import TelemetryAdapter
from optical_anomaly.assessment import coverage_report, sampling_report
from optical_anomaly.contracts import MetricDefinition
from optical_anomaly.generator import GeneratorConfig, generate, make_topology
from optical_anomaly.diagnostics import healthy_statistics
from optical_anomaly.optics import validate_generated
from optical_anomaly.pipeline import prepare
from optical_anomaly.validation import DataValidator


def test_non_telecom_registry_and_paired_count_exposure():
    definitions = {
        "temperature": MetricDefinition("C", "gauge", "Oven reading"),
        "rejects": MetricDefinition(
            "items", "interval_count", "Rejected items", count_group="production"
        ),
        "produced": MetricDefinition(
            "items", "interval_count", "Produced items", count_group="production"
        ),
    }
    times = pd.date_range("2025-01-01T00:05Z", periods=3, freq="5min")
    native = pd.DataFrame(
        {
            "timestamp": times,
            "entity_id": "oven-1",
            "temperature": [100, 101, 102],
            "rejects": [2, np.nan, 3],
            "produced": [100, 100, 100],
        }
    )
    adapter = TelemetryAdapter(
        metrics={name: name for name in definitions},
        units={name: item.unit for name, item in definitions.items()},
        kinds={name: item.kind for name, item in definitions.items()},
        definitions=definitions,
    )
    telemetry = adapter.transform(native)
    validator = DataValidator("15min", source_interval="5min", definitions=definitions)
    result = validator.transform(telemetry).set_index("metric_name")
    assert result.loc["temperature", "value"] == 101
    assert result.loc[["rejects", "produced"], "value"].isna().all()
    native.loc[1, "rejects"] = 1
    result = validator.transform(adapter.transform(native)).set_index("metric_name")
    assert result.loc["rejects", "value"] / result.loc[
        "produced", "value"
    ] == pytest.approx(6 / 300)
    # Removing the row entirely must have the same exposure consequence as NaN.
    incomplete = telemetry.drop(
        telemetry.index[
            telemetry.metric_name.eq("rejects") & telemetry.timestamp.eq(times[1])
        ]
    )
    result = validator.transform(incomplete).set_index("metric_name")
    assert result.loc[["rejects", "produced"], "value"].isna().all()
    with pytest.raises(ValueError, match="source interval"):
        DataValidator("15min", definitions=definitions).transform(telemetry)


def test_coverage_includes_boundaries_absent_series_and_value_gaps():
    start = pd.Timestamp("2025-01-01", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": [start + pd.Timedelta(minutes=x) for x in [5, 10, 15]],
            "entity_id": "A",
            "metric_name": "temperature",
            "value": [10, np.nan, 12],
        }
    )
    report = coverage_report(
        frame,
        start=start,
        end=start + pd.Timedelta(minutes=25),
        interval="5min",
        entities=["A", "B"],
        metrics=["temperature"],
    )
    a, b = report.iloc[0], report.iloc[1]
    assert (
        a.expected_slots,
        a.absent_rows,
        a.invalid_or_missing_values,
        a.observed_slots,
    ) == (5, 2, 1, 2)
    assert a.observed_coverage == 0.4
    assert b.observed_coverage == 0
    assert b.longest_missing_hours == pytest.approx(25 / 60)
    with pytest.raises(ValueError, match="Duplicate"):
        coverage_report(
            pd.concat([frame, frame]),
            start=start,
            end=start + pd.Timedelta(minutes=25),
            interval="5min",
            entities=["A"],
            metrics=["temperature"],
        )
    irregular = frame.copy()
    irregular.loc[0, "timestamp"] += pd.Timedelta(minutes=1)
    assert sampling_report(irregular, "5min").off_grid_timestamps.iloc[0] == 1
    with pytest.raises(ValueError, match="Off-grid"):
        coverage_report(
            irregular,
            start=start,
            end=start + pd.Timedelta(minutes=25),
            interval="5min",
            entities=["A"],
            metrics=["temperature"],
        )


@pytest.mark.parametrize("seed", [3, 17, 42, 101, 808])
def test_reference_identities_across_seeds(seed):
    config = GeneratorConfig(
        seed=seed, entities=2, days=30, interval_minutes=15, faults_per_entity=0
    )
    native, truth = generate(config)
    stats = healthy_statistics(native, config)
    # Broad, predefined engineering screens: not iid standard errors or CIs.
    assert stats.amplitude_in_band.all()
    assert stats.variance_in_band.all()
    assert stats.correlation_in_band.all()
    assert all(
        validate_generated(native, truth, make_topology(config))["checks"].values()
    )


def test_fault_interventions_leave_normal_prefix_and_recovery_unchanged():
    config = GeneratorConfig(
        entities=3, days=12, interval_minutes=15, missing_probability=0
    )
    faulty, truth = generate(config)
    healthy, _ = generate(replace(config, faults_per_entity=0))
    cutoff = faulty.time.min() + pd.Timedelta(days=config.days * 0.55)
    pd.testing.assert_frame_equal(
        faulty.loc[faulty.time < cutoff], healthy.loc[healthy.time < cutoff]
    )
    for event in truth.itertuples():
        at_end = faulty.device.eq(event.entity_id) & faulty.time.eq(event.end_time)
        # Counter telemetry still covers the preceding interval; compare gauges only.
        pd.testing.assert_series_equal(
            faulty.loc[at_end, "rx_dbm"], healthy.loc[at_end, "rx_dbm"]
        )
        assert event.onset_time < event.recovery_start_time <= event.end_time


def test_shared_faults_have_shared_cause_but_entity_specific_impact():
    config = GeneratorConfig(
        entities=4,
        days=12,
        interval_minutes=15,
        fault_scope="pon_port",
        onts_per_splitter=2,
        splitters_per_port=1,
        missing_probability=0,
    )
    faulty, truth = generate(config)
    healthy, _ = generate(replace(config, faults_per_entity=0))
    assert truth.groupby("cause_id").entity_id.nunique().eq(2).all()
    assert truth.groupby("cause_id").onset_time.nunique().eq(1).all()
    change = (faulty.rx_dbm - healthy.rx_dbm).to_numpy().reshape(4, -1)
    np.testing.assert_allclose(change[0], change[1], atol=0.002)
    np.testing.assert_allclose(change[2], change[3], atol=0.002)
    assert truth.fault_id.is_unique


def test_negative_controls_and_collection_gap_do_not_change_physical_truth():
    config = GeneratorConfig(
        entities=12, days=12, interval_minutes=15, fault_probability=0.5
    )
    native, truth = generate(config)
    assert 0 < truth.entity_id.nunique() < config.entities
    gapped, labels = generate(replace(config, collection_gap_hours=8))
    pd.testing.assert_series_equal(truth.impact_time, labels.impact_time)
    both_observed = native.rx_dbm.notna() & gapped.rx_dbm.notna()
    pd.testing.assert_series_equal(
        native.rx_dbm[both_observed], gapped.rx_dbm[both_observed]
    )
    assert gapped.rx_dbm.isna().sum() >= native.rx_dbm.isna().sum() + 12 * 20


def test_normal_stress_changes_are_not_mislabeled_as_faults():
    config = GeneratorConfig(
        entities=1,
        days=30,
        interval_minutes=15,
        faults_per_entity=0,
        missing_probability=0,
        sensor_noise_db=0,
    )
    reference, _ = generate(config)
    drift, truth = generate(replace(config, drift_db_per_day=0.02))
    elapsed_days = np.arange(len(reference)) * 15 / 1440
    np.testing.assert_allclose(
        reference.rx_dbm - drift.rx_dbm, 0.02 * elapsed_days, atol=0.0011
    )
    assert truth.empty
    volatile, _ = generate(replace(config, noise_end_multiplier=4))
    # Same latent innovations, scaled; seasonal path and measurement coupling persist.
    difference = volatile.rx_dbm - reference.rx_dbm
    assert difference.iloc[-500:].std() > 3 * difference.iloc[:500].std()


def test_prepare_checks_source_defaults_and_runtime_provenance(tmp_path):
    settings = {
        "generator": {"entities": 1, "days": 8},
        "splits": [0.4, 0.55, 0.75, 1],
        "output": str(tmp_path / "run"),
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(settings))
    run = prepare(path)
    assert prepare(path) == run
    manifest = json.loads((run / "generation_manifest.json").read_text())
    assert manifest["generator"]["recovery_hours"] == 2
    manifest["source_sha256"]["generator.py"] = "stale"
    (run / "generation_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Generation code"):
        prepare(path)


def test_bins_use_utc_epoch_alignment_for_non_daily_cadence():
    times = pd.date_range("2025-01-01T00:00Z", periods=30, freq="20min")
    definition = {"count": MetricDefinition("items", "interval_count", "Items")}
    frame = pd.DataFrame(
        {
            "timestamp": times.tz_convert("Asia/Kolkata"),
            "entity_id": "A",
            "metric_name": "count",
            "value": np.arange(len(times)),
        }
    )
    result = DataValidator(
        "100min", source_interval="20min", definitions=definition
    ).transform(frame)
    expected = (
        pd.Series(np.arange(len(times)), index=times.ceil("100min"))
        .groupby(level=0)
        .sum(min_count=5)
    )
    np.testing.assert_array_equal(result.timestamp, expected.index)
    np.testing.assert_allclose(result.value, expected.values, equal_nan=True)


def test_weak_faults_keep_timing_but_reduce_counterfactual_effect():
    config = GeneratorConfig(
        entities=3, days=12, interval_minutes=15, missing_probability=0
    )
    strong, labels = generate(config)
    weak, weak_labels = generate(replace(config, fault_scale=0.1))
    control, _ = generate(replace(config, faults_per_entity=0))
    pd.testing.assert_frame_equal(
        labels[["onset_time", "end_time"]], weak_labels[["onset_time", "end_time"]]
    )
    np.testing.assert_allclose(
        weak.rx_dbm - control.rx_dbm,
        0.1 * (strong.rx_dbm - control.rx_dbm),
        atol=0.0012,
    )
    assert weak_labels.impact_time.notna().sum() <= labels.impact_time.notna().sum()


def test_generator_validation_rejects_partial_fec_tuple(
    generated_data, generator_config
):
    native, truth = generated_data
    broken = native.copy()
    valid_row = broken.downstream_fec_total_codewords.first_valid_index()
    broken.loc[valid_row, "downstream_fec_corrected_codewords"] = np.nan
    with pytest.raises(ValueError, match="paired_count_availability"):
        validate_generated(broken, truth, make_topology(generator_config))


def test_repair_duration_does_not_change_earlier_fault_path():
    config = GeneratorConfig(
        entities=3,
        days=90,
        interval_minutes=15,
        missing_probability=0,
        faults_per_entity=1,
        fault_duration_median_hours=6,
        recovery_hours=1,
    )
    original, labels = generate(config)
    changed, later = generate(replace(config, recovery_hours=4))
    assert not labels.duration_truncated.any()
    assert not later.duration_truncated.any()
    for event in labels.itertuples():
        prefix = original.device.eq(event.entity_id) & original.time.lt(
            event.recovery_start_time
        )
        pd.testing.assert_frame_equal(original.loc[prefix], changed.loc[prefix])
