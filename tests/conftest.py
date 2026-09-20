"""Reusable test data; imports use the installed package, never sys.path edits."""

import numpy as np
import pandas as pd
import pytest
from optical_anomaly.generator import GeneratorConfig, generate, stationary_noise
from optical_anomaly.sources import synthetic_adapter
from optical_anomaly.validation import DataValidator


@pytest.fixture(scope="session")
def generator_config():
    # Eight days preserves the generator's documented minimum; fewer entities
    # and 15-minute cadence keep the reusable fixture small.
    return GeneratorConfig(entities=3, days=8, interval_minutes=15)


@pytest.fixture(scope="session")
def generated_data(generator_config):
    return generate(generator_config)


@pytest.fixture(scope="session")
def stationary_sample():
    # Preserve statistical power for the 4% variance / 0.01 correlation checks.
    return stationary_noise(100_000, 0.2, 0.8, np.random.default_rng(42))


@pytest.fixture
def clean_telemetry():
    # A small feature fixture, independent of generator event placement.
    times = pd.date_range("2025-01-01", periods=2 * 24 * 12, freq="5min", tz="UTC")
    hours = np.arange(len(times)) / 12
    rng = np.random.default_rng(7)
    native = pd.DataFrame(
        {
            "time": times,
            "device": "A",
            "rx_dbm": -22
            + 0.2 * np.sin(2 * np.pi * hours / 24)
            + rng.normal(0, 0.08, len(times)),
        }
    )
    adapter = synthetic_adapter(["rx_dbm"])
    return DataValidator().transform(adapter.transform(native))


@pytest.fixture
def operational_tables():
    # Function scope: tests can alter labels without contaminating another test.
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
    return truth, alerts, telemetry
