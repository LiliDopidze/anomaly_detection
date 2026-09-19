"""Coverage for the uncertainty helpers used by the new pipeline."""

from telco_anomaly.evaluation import poisson_rate_interval, wilson_interval


def test_poisson_interval_uses_requested_confidence_level():
    _, upper_95 = poisson_rate_interval(20, 1_000, 0.95)
    _, upper_99 = poisson_rate_interval(20, 1_000, 0.99)

    assert upper_99 > upper_95


def test_wilson_interval_uses_requested_confidence_level():
    lower_95, upper_95 = wilson_interval(20, 50, 0.95)
    lower_99, upper_99 = wilson_interval(20, 50, 0.99)

    assert lower_99 < lower_95
    assert upper_99 > upper_95
