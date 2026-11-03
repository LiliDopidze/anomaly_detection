"""Hand-calculated references and explicit information-availability checks."""

import numpy as np
import pandas as pd
import pytest

from ts_anomaly.baselines import (
    apply_reference, compare_predictions, completed_bins, fit_reference, lag_features,
)


def test_reference_uses_only_half_open_fit_period_and_separates_entities():
    data = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-01"]),
                         "entity_id": ["a", "a", "a", "b"], "x": [0., 10., 999., 100.]})
    ref = fit_reference(data, ["x"], "2024-01-01", "2024-01-03", min_observations=1, min_days=1)
    a = ref.loc[ref.entity_id.eq("a")].iloc[0]
    assert (a.lower, a.expected, a.upper) == (.5, 5., 9.5)
    assert ref.loc[ref.entity_id.eq("b"), "expected"].iloc[0] == 100
    changed = data.copy()
    changed.loc[2, "x"] = -1e12
    pd.testing.assert_frame_equal(ref, fit_reference(changed, ["x"], "2024-01-01", "2024-01-03", min_observations=1, min_days=1))
    predicted = apply_reference(data.iloc[[2]], ref, "x")
    assert predicted.residual.iloc[0] == 994
    with pytest.raises(ValueError, match="follow"):
        apply_reference(data.iloc[[0]], ref, "x")


def test_unsupported_states_sparse_groups_and_collapsed_ranges_are_explicit():
    training = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
                             "entity_id": "a", "command": ["on", "on", "off", "unknown", "unknown"], "x": [2., 2., 10., 20., 30.]})
    ref = fit_reference(training, ["x"], "2024-01-01", "2024-02-01", context="command",
                        allowed_states=["on", "off"], min_observations=2, min_days=2)
    test = pd.DataFrame({"timestamp": pd.date_range("2024-02-01", periods=5, freq="D"),
                         "entity_id": "a", "command": ["on", "off", "unknown", "new", "on"], "x": [4., 1., 1., 1., np.nan]})
    result = apply_reference(test, ref, "x", context="command")
    assert result.expected.iloc[0] == 2 and result.residual.iloc[0] == 2
    assert result.collapsed_range.iloc[0]
    assert result.expected.iloc[1:4].isna().all()
    assert result.status.tolist() == ["available", "unsupported_reference", "unsupported_reference", "unsupported_reference", "missing_measurement"]


def test_reference_requires_repeated_days_not_just_many_correlated_rows():
    data = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=200, freq="s"), "entity_id": "a", "x": 1.})
    ref = fit_reference(data, ["x"], "2024-01-01", "2024-02-01")
    assert not ref.supported.iloc[0]
    assert ref.days.iloc[0] == 1


def test_comparisons_share_targets_but_report_each_methods_availability():
    first = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=3), "entity_id": "a",
                          "observed": [1., 3., 5.], "expected": [2., 2., 2.], "lower": 0., "upper": 4., "state": ["on", "on", "off"]})
    second = first.assign(expected=[0., 4., np.nan], lower=[0., 2., np.nan], upper=[2., 4., np.nan])
    report = compare_predictions({"a": first, "b": second}).set_index("method")
    assert report.common.tolist() == [2, 2]
    assert report.available.tolist() == [3, 2]
    assert report.mae.tolist() == [1., 1.]
    assert report.bias.tolist() == [0., 0.]
    assert report.range_coverage.tolist() == [1., 1.]
    assert report.range_width.tolist() == [4., 2.]
    grouped = compare_predictions({"a": first, "b": second}, ["state"])
    assert grouped.loc[grouped.state.eq("off"), "common"].eq(0).all()
    assert grouped.loc[grouped.state.eq("off"), "mae"].isna().all()
    with pytest.raises(ValueError, match="identical"):
        compare_predictions({"a": first, "b": second.iloc[::-1]})


def test_aggregation_is_available_at_end_and_endpoint_belongs_to_next_bin():
    data = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=7, freq="10s"),
                         "entity_id": "a", "x": [0., 1., 2., 3., 4., 5., 999.]})
    bins = completed_bins(data, ["x"])
    assert bins.index[0] == pd.Timestamp("2024-01-01 00:01")
    assert bins.x.iloc[0] == 2.5
    assert pd.isna(bins.x.iloc[1])
    changed = data.copy()
    changed.loc[6, "x"] = -999
    pd.testing.assert_frame_equal(bins.iloc[:1], completed_bins(changed, ["x"]).iloc[:1])


def test_within_bin_gap_masks_mean_even_with_enough_samples():
    data = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:10", "2024-01-01 00:00:30", "2024-01-01 00:00:40", "2024-01-01 00:00:50"]),
                         "entity_id": "a", "x": 1.})
    result = completed_bins(data, ["x"])
    assert result.source_count.iloc[0] == 5
    assert pd.isna(result.x.iloc[0])


def test_context_must_be_known_at_every_observed_row_when_requested():
    data = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=6, freq="10s"),
                         "entity_id": "a", "x": 1., "context": [1., 1., 1., 1., 1., np.nan]})
    result = completed_bins(data, ["x", "context"], complete_columns=["context"])
    assert result.x.iloc[0] == 1
    assert pd.isna(result.context.iloc[0])


def test_boundary_gap_resets_lags_even_when_both_means_are_supported():
    times = list(pd.date_range("2024-01-01", periods=5, freq="10s"))
    times += list(pd.date_range("2024-01-01 00:01", periods=12, freq="10s"))
    data = pd.DataFrame({"timestamp": times, "entity_id": "a", "x": range(len(times))})
    bins = completed_bins(data, ["x"])
    features = lag_features(bins, {"x": (1,)})
    assert bins.x.notna().all()
    assert pd.isna(features.x_lag1.iloc[1])
    assert features.x_lag1.iloc[2] == bins.x.iloc[1]


def test_missing_values_break_all_lags_without_compressing_time():
    bins = pd.DataFrame({"x": [1., 2., np.nan, 4., 5., 6.], "context": 1., "segment": 0},
                         index=pd.date_range("2024-01-01", periods=6, freq="1min"))
    result = lag_features(bins, {"x": (1, 2), "context": (1,)})
    assert result.x_lag1.iloc[1] == 1
    assert result.x_lag2.iloc[:5].isna().all()
    assert result.x_lag2.iloc[5] == 4
    assert pd.isna(result.context_lag1.iloc[3])
    with pytest.raises(ValueError, match="positive"):
        lag_features(bins, {"x": (0,)})


def test_future_changes_leave_historical_lags_and_fixed_linear_predictions_unchanged():
    bins = pd.DataFrame({"x": np.arange(12.), "context": 1., "segment": 0},
                         index=pd.date_range("2024-01-01", periods=12, freq="1min"))
    original = lag_features(bins, {"x": (1, 3), "context": (1,)})
    changed = bins.copy()
    changed.iloc[8:, changed.columns.get_loc("x")] = 1e6
    changed.iloc[8:, changed.columns.get_loc("context")] = 0
    updated = lag_features(changed, {"x": (1, 3), "context": (1,)})
    columns = ["x_lag1", "x_lag3", "context_lag1"]
    pd.testing.assert_frame_equal(original[columns].iloc[:9], updated[columns].iloc[:9])
    weights = np.array([.8, .2, .5])
    np.testing.assert_allclose(original[columns].iloc[3:9] @ weights, updated[columns].iloc[3:9] @ weights)


def test_references_reject_duplicates_in_fitting_and_application():
    data = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01"] * 2), "entity_id": "a", "x": [1., 2.]})
    with pytest.raises(ValueError, match="Duplicate"):
        fit_reference(data, ["x"], "2024-01-01", "2024-02-01")
    ref = fit_reference(data.iloc[:1], ["x"], "2024-01-01", "2024-02-01", min_observations=1, min_days=1)
    with pytest.raises(ValueError, match="Duplicate"):
        apply_reference(data.assign(timestamp=pd.Timestamp("2024-02-02")), ref, "x")
