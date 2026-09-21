"""Statistical properties, reproducibility and separation of physical truth."""

from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from optical_anomaly.generator import generate


def test_stationary_variance_and_correlation(stationary_sample):
    assert stationary_sample.var() == pytest.approx(0.04, rel=0.04)
    assert np.corrcoef(stationary_sample[:-1], stationary_sample[1:])[
        0, 1
    ] == pytest.approx(0.8, abs=0.01)


def test_reproducible_generation(generator_config, generated_data):
    native, truth = generated_data
    # This second generation is intentional: sharing the same object would not
    # test that independent executions reproduce the same fixture.
    again, labels = generate(generator_config)
    pd.testing.assert_frame_equal(native, again)
    pd.testing.assert_frame_equal(truth, labels)


def test_separate_truth(generated_data):
    native, truth = generated_data
    assert not set(truth.columns).intersection(native.columns)
    assert set(truth.fault_type) == {"random_walk", "exponential", "variance_shift"}
    assert (
        truth.impact_time.dropna() >= truth.loc[truth.impact_time.notna(), "onset_time"]
    ).all()


def test_collection_does_not_change_impact(generator_config, generated_data):
    _, truth = generated_data
    _, missing = generate(replace(generator_config, missing_probability=0.3))
    pd.testing.assert_series_equal(truth.impact_time, missing.impact_time)


@pytest.mark.parametrize(
    "values, intervals, expected",
    [
        ([True, True], 3, None),
        ([False, True, True, True], 3, 3),
        ([True, False, True, True, True], 3, 4),
        ([True], 1, 0),
    ],
)
def test_impact_confirmation_is_not_backdated(values, intervals, expected):
    from optical_anomaly.generator import first_persistent_crossing

    assert first_persistent_crossing(np.array(values), intervals) == expected


def test_variance_shift_has_no_claimed_observable_onset(generated_data):
    _, truth = generated_data
    variance = truth.loc[truth.fault_type.eq("variance_shift")]
    assert len(variance) > 0
    assert variance.observable_onset_time.isna().all()


def test_impact_persistence_on_known_latent_path(monkeypatch):
    import optical_anomaly.generator as module

    config = module.GeneratorConfig(entities=1, days=8)
    times = pd.date_range("2025-01-01", periods=8 * 288, freq="5min", tz="UTC")
    optics = {
        "rx_dbm": np.full(len(times), -26.0),
        "upstream_rx_dbm": np.full(len(times), -20.0),
    }
    monkeypatch.setattr(
        module, "fault_signature", lambda kind, size, dt, rng: np.full(size, -10.0)
    )
    truth = module._inject_fault(
        config,
        0,
        0,
        times,
        optics,
        np.zeros(len(times), dtype=bool),
        np.random.default_rng(7),
    )
    assert truth["impact_time"] == truth["onset_time"] + pd.Timedelta(minutes=10)
