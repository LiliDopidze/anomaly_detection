"""Hand-calculated statistics, information boundaries and temporal support."""

import numpy as np
import pandas as pd
import pytest

from ts_anomaly.baselines import (
    apply_reference, calibrate_reference, compare_predictions, completed_gauges,
    fit_reference, persistence_prediction,
)


START = pd.Timestamp("2020-01-01")


def series(values, seconds=None):
    seconds = np.arange(len(values)) if seconds is None else seconds
    return pd.Series(values, index=START + pd.to_timedelta(seconds, unit="s"), dtype=float)


def recording(values, seconds=None, entity="one"):
    values = series(values, seconds)
    return pd.DataFrame({"timestamp": values.index, "entity_id": entity, "x": values.to_numpy()})


def test_quantiles_residuals_and_frozen_application():
    fitted = fit_reference(series([0, 2, 4, 6, np.nan, np.inf]))
    assert fitted.expected == 3
    assert fitted.lower == pytest.approx(.3)
    assert fitted.upper == pytest.approx(5.7)
    assert fitted.iqr == 3
    before = fitted.copy()
    actual = series([10, 11])
    output = apply_reference(actual, fitted)
    assert output.residual.tolist() == [7, 8]
    pd.testing.assert_series_equal(fitted, before)
    pd.testing.assert_series_equal(output.expected, apply_reference(actual * 100, fitted).expected)


def test_half_open_calibration_and_future_independence():
    values = series([0, 2, 4, 6, 100, 200])
    end = START + pd.Timedelta(seconds=4)
    fitted = calibrate_reference(values, START, end)
    assert fitted.observations == 4 and fitted.expected == 3
    changed = values.copy()
    changed.loc[changed.index >= end] = -9999
    pd.testing.assert_series_equal(fitted, calibrate_reference(changed, START, end))
    predictions = apply_reference(values, fitted, available_from=end)
    assert predictions.expected.iloc[:4].isna().all()
    assert predictions.expected.iloc[4:].eq(3).all()


def test_annotations_are_not_fitting_inputs():
    data = recording([1, 2, 3, 4, 100])
    data['anomaly'] = [0, 0, 0, 1, 1]
    end = START + pd.Timedelta(seconds=4)
    first = calibrate_reference(data.set_index('timestamp').x, START, end)
    data['anomaly'] = 1
    second = calibrate_reference(data.set_index('timestamp').x, START, end)
    pd.testing.assert_series_equal(first, second)
    assert first.observations == 4


def test_occupancy_counts_slots_not_rows_or_span():
    values = series([1, 1, 1, 1, 1], [0, .1, 1, 2, 3])
    fitted = calibrate_reference(values, START, START + pd.Timedelta(seconds=5))
    assert fitted.observations == 5 and fitted.occupied_slots == 4
    assert fitted.occupancy == .8 and fitted.supported
    insufficient = calibrate_reference(values, START, START + pd.Timedelta(seconds=6))
    assert not insufficient.supported
    assert insufficient.expected == 1  # Descriptive audit remains available.
    assert apply_reference(values, insufficient).expected.isna().all()


def test_ties_zero_spread_and_empty_reference():
    fitted = fit_reference(series([4] * 5))
    assert fitted.collapsed_range and fitted.mad == 0
    prediction = apply_reference(series([4, 5]), fitted)
    metrics = compare_predictions(series([4, 5]), {'fixed': prediction}).iloc[0]
    assert metrics.range_coverage == .5 and metrics.range_width == 0
    empty = fit_reference(series([np.nan, np.inf]))
    assert not empty.supported
    assert apply_reference(series([2]), empty).expected.isna().all()


def test_completed_windows_timing_and_partial_edges():
    data = recording(np.arange(12), np.arange(2, 14))
    means, counts, connected = completed_gauges(data, ['x'])
    # [0,5) starts before recording; [10,15) is unfinished at observed end 14.
    assert means.index.tolist() == [START + pd.Timedelta(seconds=10)]
    assert means.x.iloc[0] == 5
    assert counts.x.iloc[0] == 5 and not connected.x.iloc[0]


def test_five_second_persistence_is_causal():
    data = recording(np.arange(20))
    means, _, connected = completed_gauges(data, ['x'])
    prediction = persistence_prediction(means.x, connected.x)
    assert means.x.tolist() == [2, 7, 12, 17]
    assert pd.isna(prediction.expected.iloc[0])
    assert prediction.expected.iloc[1:].tolist() == [2, 7, 12]
    assert prediction.issued_at.iloc[1] == START + pd.Timedelta(seconds=5)
    changed = data.copy()
    changed.loc[changed.timestamp >= START + pd.Timedelta(seconds=10), 'x'] = 999
    future_means, _, future_connected = completed_gauges(changed, ['x'])
    future = persistence_prediction(future_means.x, future_connected.x)
    pd.testing.assert_series_equal(prediction.expected.iloc[:3], future.expected.iloc[:3])


def test_raw_gap_resets_history_even_with_supported_adjacent_bins():
    data = recording([1] * 8, [0, 1, 2, 3, 6, 7, 8, 9])
    means, counts, connected = completed_gauges(data, ['x'])
    assert counts.x.tolist() == [4, 4]
    assert means.x.notna().all()
    assert not connected.x.any()
    predicted = persistence_prediction(means.x, connected.x)
    assert predicted.expected.iloc[1] == 1  # Issued before the unexpected gap.
    assert not predicted.eligible.any() and predicted.residual.isna().all()


def test_missing_bins_reset_history_and_never_fill():
    data = recording(np.arange(20, dtype=float))
    data.loc[5:7, 'x'] = np.nan
    means, _, connected = completed_gauges(data, ['x'])
    assert pd.isna(means.x.iloc[1])
    result = persistence_prediction(means.x, connected.x)
    assert result.residual.iloc[:3].isna().all()
    assert not result.eligible.iloc[:3].any()
    assert result.expected.iloc[3] == means.x.iloc[2]


def test_missing_measurement_resets_even_if_mean_has_four_readings():
    data = recording(np.arange(20, dtype=float))
    data.loc[7, 'x'] = np.nan
    means, _, connected = completed_gauges(data, ['x'])
    assert means.x.notna().all()
    assert not connected.x.iloc[1]
    assert not connected.x.iloc[2]
    assert connected.x.iloc[3]


def test_no_cross_recording_history():
    a = recording(np.arange(10))
    b = recording(np.arange(10), np.arange(10, 20), 'two')
    with pytest.raises(ValueError, match='one entity'):
        completed_gauges(pd.concat([a, b]), ['x'])
    means, _, connected = completed_gauges(b, ['x'])
    assert pd.isna(persistence_prediction(means.x, connected.x).expected.iloc[0])


def test_common_observations_and_unavailable_are_explicit():
    actual = series([1, 3, 9])
    first = apply_reference(actual, fit_reference(series([1])))
    second = apply_reference(actual, fit_reference(series([3])))
    second.iloc[2] = np.nan
    result = compare_predictions(actual, {'a': first, 'b': second}).set_index('method')
    assert result.common.eq(2).all()
    assert result.mae.eq(1).all()
    assert result.loc['a', 'bias'] == 1 and result.loc['b', 'bias'] == -1
    assert result.loc['a', 'own_mae'] == pytest.approx(10 / 3)
    assert result.loc['b', 'availability'] == pytest.approx(2 / 3)
    second[:] = np.nan
    result = compare_predictions(actual, {'a': first, 'b': second})
    assert result.common.eq(0).all() and result.mae.isna().all()
    with pytest.raises(ValueError, match='align'):
        compare_predictions(actual, {'a': first.iloc[::-1]})


def test_duplicate_calibration_timestamps_rejected():
    with pytest.raises(ValueError, match='unique'):
        calibrate_reference(series([1, 2], [0, 0]), START, START + pd.Timedelta(seconds=2))


def test_conditional_line_is_hand_calculable_and_requires_variation():
    from ts_anomaly.baselines import fit_pair_reference
    x = series([1, 2, 3, 4])
    result = fit_pair_reference(x, 2 * x + 1)
    assert result.slope == 2 and result.intercept == 1
    assert result.predictor_min == 1 and result.predictor_max == 4
    with pytest.raises(ValueError, match='varying'):
        fit_pair_reference(x * 0, x)


def test_nonindependent_rows_are_excluded_without_hiding_prediction_availability():
    actual = series([1, 3, 100])
    predicted = apply_reference(actual, fit_reference(series([1])))
    mask = pd.Series([True, True, False], index=actual.index)
    result = compare_predictions(actual, {'fixed': predicted}, mask).iloc[0]
    assert result.available == 3 and result.availability == 1
    assert result.common == 2 and result.excluded_rows == 1
    assert result.mae == result.own_mae == 1
    mask[:] = False
    result = compare_predictions(actual, {'fixed': predicted}, mask).iloc[0]
    assert result.common == 0 and pd.isna(result.mae)


def test_aggregated_calibration_uses_only_whole_completed_fit_windows():
    data = recording(np.arange(20, dtype=float), np.arange(2, 22))
    means, _, _ = completed_gauges(data, ['x'])
    cutoff = START + pd.Timedelta(seconds=13)
    selected = (means.index-pd.Timedelta(seconds=5) >= data.timestamp.min()) & (means.index <= cutoff)
    fitted = fit_reference(means.loc[selected, 'x'])
    assert fitted.observations == 1 and fitted.expected == 5
    data.loc[data.timestamp >= cutoff, 'x'] = 999
    changed, _, _ = completed_gauges(data, ['x'])
    pd.testing.assert_series_equal(fitted, fit_reference(changed.loc[selected, 'x']))


def test_forecast_issuance_does_not_use_future_missingness():
    means = series([2, 7, 12], [5, 10, 15])
    connected = pd.Series([False, True, True], index=means.index)
    before = persistence_prediction(means, connected)
    changed = means.copy()
    changed.iloc[1] = np.nan
    after = persistence_prediction(changed, connected)
    assert before.expected.iloc[1] == after.expected.iloc[1] == 2
    assert before.issued_at.iloc[1] == after.issued_at.iloc[1] == START + pd.Timedelta(seconds=5)
    assert not after.eligible.iloc[1] and pd.isna(after.residual.iloc[1])
    assert pd.isna(after.expected.iloc[2])  # History resets after unavailable target.
    result = compare_predictions(changed, {'persistence': after}).iloc[0]
    assert result.available == 0
