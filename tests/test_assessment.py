"""Small examples with hand-checkable temporal and statistical answers."""


import numpy as np
import pandas as pd
import pytest

from ts_anomaly.assessment import (
    aggregate_gauges, bin_coverage, duplicate_report, longest_regular_segment,
    relationship_report, sampling_report, segment_acf, signal_profile,
    split_development, state_runs,
)
from ts_anomaly.contracts import MetricDefinition, validate_schema


DEFINITIONS = {
    "temperature": MetricDefinition("Oven temperature", "degC", "gauge", "measurement"),
    "switch": MetricDefinition("Oven switch", None, "state", "context", (0, 1)),
}


def test_sampling_keeps_acquisition_order_and_reports_gaps_without_rounding():
    t = pd.to_datetime(["2024-01-01 00:00:10", "2024-01-01", "2024-01-01 00:00:10", "2024-01-01 00:00:19", "2024-01-01 00:01:00"], format="mixed")
    data = pd.DataFrame({"timestamp": t, "entity_id": "oven"})
    report, intervals, gaps = sampling_report(data, "15s")
    assert report.duplicate_rows.iloc[0] == 2
    assert report.backward_steps.iloc[0] == 1
    assert report.median_interval_s.iloc[0] == 10
    assert dict(zip(intervals.interval_s, intervals['count'])) == {9.: 1, 10.: 1, 41.: 1}
    assert gaps.elapsed_s.tolist() == [41.]
    assert data.timestamp.iloc[0] == t[0]


def test_sampling_and_profile_keep_entities_separate():
    data = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                         "entity_id": ["a", "b"], "temperature": [10., 100.], "switch": [0, 1]})
    summary, _, gaps = sampling_report(data, "15s")
    assert summary.duplicate_rows.sum() == 0
    assert gaps.empty
    profile = signal_profile(data, DEFINITIONS)
    assert profile.loc[profile.measurement.eq('temperature'), 'median'].tolist() == [10., 100.]


def test_coverage_uses_declared_bounds_and_counts_bins_not_rows():
    times = pd.Series(pd.to_datetime(["2024-01-01 00:01:00", "2024-01-01 00:01:00", "2024-01-01 00:01:10", "2024-01-01 00:03:00"]))
    report = bin_coverage(times, "2024-01-01", "2024-01-01 00:03")
    assert report.observations.tolist() == [0, 2, 0]
    assert report.occupied.mean() == pytest.approx(1/3)


def test_conflicting_duplicates_are_not_averaged():
    data = pd.DataFrame({"timestamp": pd.to_datetime(['2024-01-01']*2),
                         "entity_id": 'oven', "temperature": [2., 8.], "switch": [1, 1]})
    report = duplicate_report(data, list(DEFINITIONS))
    assert report.conflicting_measurements.tolist() == ['temperature']
    with pytest.raises(ValueError, match='unique'):
        aggregate_gauges(data, ['temperature'], '1min', 1)


def test_profile_distinguishes_missing_nonfinite_and_encoding_without_clipping():
    data = pd.DataFrame({"entity_id": "oven", "temperature": [np.nan, np.inf, -100., 500.], "switch": [0., 1., 2., np.nan]})
    result = signal_profile(data, DEFINITIONS).set_index('measurement')
    assert result.loc['temperature', 'missing'] == 1
    assert result.loc['temperature', 'nonfinite'] == 1
    assert result.loc['temperature', 'min'] == -100
    assert result.loc['switch', 'unexpected_encoding'] == 1
    assert data.temperature.iloc[2] == -100


def test_schema_rejects_labels_and_missing_identifiers():
    data = pd.DataFrame({"timestamp": pd.to_datetime(['2024-01-01']), "entity_id": ['oven'], 'temperature': [1.], 'switch': [1]})
    validate_schema(data, DEFINITIONS)
    with pytest.raises(ValueError, match='declared'):
        validate_schema(data.assign(incident=1), DEFINITIONS)
    with pytest.raises(ValueError, match='identifiers'):
        validate_schema(data.assign(entity_id=''), DEFINITIONS)


def test_half_open_splits_and_future_changes_do_not_change_development():
    data = pd.DataFrame({'timestamp': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01']), 'value': [1., 2., 3.]})
    train, development = split_development(data, '2024-01-01', '2024-02-01', '2024-03-01')
    assert train.value.tolist() == [1]
    assert development.value.tolist() == [2]
    data.loc[2, 'value'] = 1e9
    a, b = split_development(data, '2024-01-01', '2024-02-01', '2024-03-01')
    pd.testing.assert_frame_equal(a, train)
    pd.testing.assert_frame_equal(b, development)
    with pytest.raises(ValueError):
        split_development(data, '2024-01-01', '2024-03-01', '2024-02-01')


def test_runs_break_at_gaps_missing_values_and_record_boundaries():
    times = pd.to_datetime(['2024-01-01 00:00:00', '2024-01-01 00:00:10', '2024-01-01 00:00:20', '2024-01-01 00:00:30', '2024-01-01 00:02:00', '2024-01-01 00:02:10', '2024-01-01 00:02:20'])
    runs = state_runs(pd.Series([0, 1, 1, 0, 0, np.nan, 0], index=times), '15s')
    assert runs.samples.tolist() == [1, 2, 1, 1, 1]
    assert runs.observed_span_s.tolist() == [0, 10, 0, 0, 0]
    assert runs.loc[1, 'duration_upper_s'] == 30
    assert not runs.loc[1, ['left_censored', 'right_censored']].any()
    assert runs.loc[3, ['left_censored', 'right_censored']].all()
    assert runs.duration_upper_s.notna().sum() == 1


def test_categorical_runs_and_unobserved_gap_switches():
    times = pd.to_datetime(['2024-01-01', '2024-01-01 00:00:10', '2024-01-01 00:02:00'], format='mixed')
    runs = state_runs(pd.Series(['off', 'on', 'off'], index=times), '15s')
    assert (~runs.left_censored).sum() == 1  # The change across the gap is unknown.
    assert runs.duration_upper_s.isna().all()


def test_aggregation_requires_support_and_does_not_impute_empty_bins():
    data = pd.DataFrame({'timestamp': pd.to_datetime(['2024-01-01', '2024-01-01 00:00:10', '2024-01-01 00:02:00'], format='mixed'), 'entity_id': 'oven', 'x': [1., 3., 8.]})
    values, counts = aggregate_gauges(data, ['x'], '1min', 2)
    assert counts.x.tolist() == [2, 0, 1]
    assert values.x.iloc[0] == 2
    assert values.x.iloc[1:].isna().all()


def test_longest_segment_never_compresses_missing_or_irregular_times():
    times = pd.to_datetime(['2024-01-01 00:00', '2024-01-01 00:01', '2024-01-01 00:03', '2024-01-01 00:04', '2024-01-01 00:05', '2024-01-01 00:06'])
    values = pd.Series([1., 2., 3., np.nan, 5., 6.], index=times)
    selected = longest_regular_segment(values, '1min')
    pd.testing.assert_series_equal(selected, values.iloc[:2])


def test_acf_matches_demeaned_covariance_definition():
    series = pd.Series([1., 4., 2., 7., 3., 8., 5., 9.], index=pd.date_range('2024-01-01', periods=8, freq='1min'))
    result, info = segment_acf(series, '1min', 2)
    centered = series.to_numpy() - series.mean()
    expected = np.dot(centered[1:], centered[:-1]) / np.dot(centered, centered)
    assert result.acf.iloc[1] == pytest.approx(expected)
    assert result.acf.iloc[0] == pytest.approx(1)
    assert info['samples'] == 8


@pytest.mark.parametrize('value', [np.nan, 1.])
def test_undefined_acf_for_missing_or_constant_series(value):
    series = pd.Series(value, index=pd.date_range('2024-01-01', periods=12, freq='1min'))
    result, _ = segment_acf(series, '1min', 3)
    assert result.acf.isna().all()


def test_relationship_counts_use_finite_pairs_and_constant_is_undefined():
    data = pd.DataFrame({'x': [1., 2., np.inf, 3.], 'y': [2., 4., 7., 6.], 'constant': 1})
    result = relationship_report(data, [('x', 'y'), ('x', 'constant')])
    assert result.pairs.tolist() == [3, 3]
    assert result.pearson.iloc[0] == pytest.approx(1)
    assert pd.isna(result.pearson.iloc[1])


def test_state_runs_all_missing_and_single_observation():
    times = pd.date_range('2024-01-01', periods=2, freq='10s')
    assert state_runs(pd.Series([np.nan, np.nan], index=times), '15s').empty
    run = state_runs(pd.Series([1.], index=times[:1]), '15s').iloc[0]
    assert run.observed_span_s == 0
    assert run.left_censored and run.right_censored
    assert pd.isna(run.duration_upper_s)


def test_relationships_reject_pooling_different_entities():
    data = pd.DataFrame({'entity_id': ['a', 'b'], 'x': [1., 2.], 'y': [3., 4.]})
    with pytest.raises(ValueError, match='one entity'):
        relationship_report(data, [('x', 'y')])
