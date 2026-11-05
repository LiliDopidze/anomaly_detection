"""Hand-calculated detector scores, decision timing and label-aware evaluation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest

from ts_anomaly.detection import (
    fit_departure_reference, departure_scores, calibrate_threshold, causal_alerts,
)
from ts_anomaly.evaluation import (
    label_windows, point_metrics, observed_episodes, decision_labels, exposure_seconds, episode_metrics,
)

START = pd.Timestamp('2020-01-01')


def series(values, seconds=None):
    seconds = np.arange(len(values)) if seconds is None else seconds
    return pd.Series(values, index=START + pd.to_timedelta(seconds, unit='s'), dtype=float)


def test_median_iqr_and_maximum_departure_are_hand_calculable():
    reference = fit_departure_reference(pd.DataFrame({'x': [0, 2, 4, 6], 'y': [0, 4, 8, 12]}))
    assert reference['median'].tolist() == [3, 6]
    assert reference.iqr.tolist() == [3, 6]
    assert reference.observations.eq(4).all() and reference.supported.all()
    data = pd.DataFrame({'x': [-3, 0, 3, 6, 9, 15], 'y': [6, 6, 18, 6, 6, 6]})
    scores, components = departure_scores(reference, data)
    assert components.x.tolist() == [2, 1, 0, 1, 2, 4]
    assert components.y.tolist() == [0, 0, 2, 0, 0, 0]
    assert scores.tolist() == [2, 1, 2, 1, 2, 4]


def test_departures_keep_growing_beyond_both_fitted_extremes():
    reference = fit_departure_reference(pd.DataFrame({'x': [0, 2, 4, 6]}))
    upper, _ = departure_scores(reference, pd.DataFrame({'x': [9, 15, 33]}))
    lower, _ = departure_scores(reference, pd.DataFrame({'x': [-3, -9, -27]}))
    assert upper.tolist() == lower.tolist() == [2, 4, 10]
    calibration, _ = departure_scores(reference, pd.DataFrame({'x': [3, 6, 9]}))
    threshold = calibrate_threshold(calibration, .99).threshold
    assert threshold == pytest.approx(1.98)
    decisions = causal_alerts(series(upper.to_numpy(), [5, 10, 15]), threshold, 3)
    assert decisions.alarm.tolist() == [0, 0, 1]  # The strict cutoff remains attainable.


def test_zero_iqr_is_unavailable_even_when_the_reference_is_not_constant():
    reference = fit_departure_reference(pd.DataFrame(
        {'constant': [2] * 5, 'tied': [0, 0, 0, 0, 1], 'variable': [0, 1, 2, 3, 4]}))
    assert reference.distinct.tolist() == [1, 2, 5]
    assert reference.supported.tolist() == [False, False, True]
    scores, components = departure_scores(
        reference, pd.DataFrame({'constant': [2, 3], 'tied': [0, 9], 'variable': [2, 4]}))
    assert components[['constant', 'tied']].isna().all().all()
    assert components.variable.tolist() == [0, 1]
    assert scores.isna().all()  # Unsupported features cannot silently disappear from the maximum.
    with pytest.raises(ValueError, match='finite calibration'):
        calibrate_threshold(scores)


def test_departures_do_not_depend_on_measurement_units():
    fit = pd.DataFrame({'x': [0, 2, 4, 6], 'y': [2, 4, 6, 8]})
    values = pd.DataFrame({'x': [-10, 2, 20], 'y': [5, 8, 15]})
    expected = departure_scores(fit_departure_reference(fit), values)
    # Independent affine unit changes preserve dimensionless departure magnitudes.
    scale, offset = pd.Series({'x': .001, 'y': 100}), pd.Series({'x': 10, 'y': -30})
    actual = departure_scores(fit_departure_reference(fit * scale + offset), values * scale + offset)
    pd.testing.assert_series_equal(actual[0], expected[0])
    pd.testing.assert_frame_equal(actual[1], expected[1])


def test_missing_components_do_not_become_normal_scores():
    reference = fit_departure_reference(pd.DataFrame({'x': [0, 1, 2], 'y': [0, 1, 2]}))
    scores, components = departure_scores(reference, pd.DataFrame({'x': [10, np.nan, 0], 'y': [np.inf, 1, 0]}))
    assert scores.iloc[:2].isna().all() and pd.notna(scores.iloc[2])
    assert pd.notna(components.x.iloc[0]) and pd.isna(components.y.iloc[0])
    with pytest.raises(ValueError, match='Feature names'):
        departure_scores(reference, pd.DataFrame({'y': [1], 'x': [1]}))
    with pytest.raises(ValueError, match='nonempty'):
        fit_departure_reference(pd.DataFrame({'x': [np.nan]}))


def test_departure_application_is_frozen_future_and_annotation_independent():
    fitted = fit_departure_reference(pd.DataFrame({'x': [1, 2, 3, 4]}))
    saved = fitted.copy(deep=True)
    first = pd.DataFrame({'x': [0, 2, 100], 'annotation': [0, 0, 1]})
    before = departure_scores(fitted, first[['x']])[0]
    first.loc[2, 'x'] = -1000
    first.annotation = 1
    after = departure_scores(fitted, first[['x']])[0]
    pd.testing.assert_frame_equal(fitted, saved)
    pd.testing.assert_series_equal(before.iloc[:2], after.iloc[:2])


def test_calibration_quantile_and_tied_cutoff_are_not_nominal_false_alarm_guarantees():
    fitted = calibrate_threshold(pd.Series(np.arange(10)), .95)
    assert fitted.threshold == pytest.approx(8.55)
    assert fitted.exceeded_fraction == .1 and not fitted.threshold_at_maximum
    tied = calibrate_threshold(pd.Series([.5] * 20), .99)
    assert tied.threshold == .5 and tied.exceeded_fraction == 0 and tied.threshold_at_maximum
    decisions = causal_alerts(series([.5, .6], [5, 10]), tied.threshold)
    assert decisions.alarm.tolist() == [0, 1]
    with pytest.raises(ValueError, match='finite calibration'):
        calibrate_threshold(pd.Series([np.nan]), .99)


def test_persistence_issues_alert_at_third_endpoint_without_backdating():
    decisions = causal_alerts(series([1, 1, 1, 1, 0, 1], [5, 10, 15, 20, 25, 30]), .5, 3)
    assert decisions.alarm.tolist() == [0, 0, 1, 1, 0, 0]
    assert decisions.index[decisions.alarm_start].tolist() == [START + pd.Timedelta(seconds=15)]


def test_alerts_reset_at_missing_scores_gaps_and_recording_boundaries():
    scores = series([1, 1, np.nan, 1, 1, 1], [5, 10, 15, 20, 25, 30])
    decisions = causal_alerts(scores, .5, 3)
    assert pd.isna(decisions.alarm.iloc[2])
    assert decisions.index[decisions.alarm_start].tolist() == [START + pd.Timedelta(seconds=30)]
    gap = causal_alerts(series([1, 1, 1], [5, 10, 20]), .5, 3)
    assert not gap.alarm_start.any()
    restarted = causal_alerts(series([1, 1], [35, 40]), .5, 3)
    assert restarted.alarm.eq(0).all()


def test_future_scores_cannot_change_earlier_alerts():
    scores = series([0, 1, 1, 1, 0], [5, 10, 15, 20, 25])
    before = causal_alerts(scores, .5, 3)
    changed = scores.copy()
    changed.iloc[3:] = np.nan
    after = causal_alerts(changed, .5, 3)
    pd.testing.assert_frame_equal(before.iloc[:3], after.iloc[:3])


def test_common_availability_applies_before_persistence_not_only_to_metrics():
    scores = series([1, 1, 1, 1, 1], [5, 10, 15, 20, 25])
    common = scores.notna()
    common.iloc[1] = False
    own = causal_alerts(scores, .5, 3)
    paired = causal_alerts(scores.where(common), .5, 3)
    assert own.alarm.iloc[2] == 1 and paired.alarm.iloc[2] == 0
    assert paired.index[paired.alarm_start].tolist() == [START + pd.Timedelta(seconds=25)]


def test_label_bins_are_half_open_and_mixed_bins_remain_unknown():
    labels = series([0] * 5 + [0, 0, 1, 1, 1] + [1] * 5)
    result = label_windows(labels, START + pd.to_timedelta([5, 10, 15, 20], unit='s'))
    assert result.status.tolist() == ['pure', 'mixed', 'pure', 'insufficient_labels']
    assert result.label.iloc[0] == 0 and result.label.iloc[2] == 1
    assert result.label.iloc[[1, 3]].isna().all()
    labels.iloc[5:] = 1
    assert label_windows(labels, result.index).label.iloc[0] == 0


def test_point_metrics_use_common_rows_without_point_adjustment():
    labels = series([0, 0] + [1] * 10)
    scores = series([0, 0, 1] + [0] * 9)
    alarms = scores.copy()
    result = point_metrics(labels, scores, alarms)
    assert result.recall == .1 and result.precision == 1
    scores.iloc[2] = np.nan
    alarms.iloc[2] = np.nan
    result = point_metrics(labels, scores, alarms)
    assert result.positive_bins_monitored == 9 and result.overall_positive_bin_recall == 0
    assert pd.isna(result.precision)
    mask = series([0] * 12).astype(bool)
    result = point_metrics(labels, scores, alarms, mask)
    assert result.common == 0 and pd.isna(result.average_precision)
    assert result.score_available == 11


def test_constant_score_ap_equals_prevalence_not_perfect_detection():
    truth = series([0, 0, 0, 1])
    result = point_metrics(truth, truth * 0, truth * 0)
    assert result.average_precision == .25
    only_positive = series([1, 1])
    assert pd.isna(point_metrics(only_positive, only_positive, only_positive).average_precision)


def test_episode_detection_and_observed_exposure_use_decision_times():
    labels = series([0] * 10 + [1] * 10 + [0] * 11)
    scores = series([0, 1, 1, 0, 1, 0], [5, 10, 15, 20, 25, 30])
    decisions = causal_alerts(scores, .5)
    summary, events = episode_metrics(labels, decisions)
    assert summary.complete_episodes == summary.detected_episodes == 1
    assert summary.median_delay_s == 0 and summary.label_negative_alerts == 1
    assert summary.observed_negative_seconds == 19
    assert summary.monitored_negative_seconds == 14
    assert summary.alert_negative_seconds == 5 and summary.alert_seconds == 15
    assert events.iloc[0].end == START + pd.Timedelta(seconds=20)


def test_standing_alarm_does_not_claim_a_later_event():
    labels = series([0] * 10 + [1] * 10 + [0] * 11)
    decisions = causal_alerts(series([1] * 6, [5, 10, 15, 20, 25, 30]), .5)
    summary, _ = episode_metrics(labels, decisions)
    assert summary.alert_starts == 1 and summary.episode_recall == 0
    assert pd.isna(summary.median_delay_s)
    assert summary.label_negative_alerts == 1


def test_gap_censored_pieces_are_not_counted_as_complete_incidents():
    labels = series([0] * 10 + [1] + [1] * 10 + [0] * 11,
                    np.r_[np.arange(11), np.arange(30, 51)])
    events = observed_episodes(labels)
    assert len(events) == 2 and not events.complete.any()
    assert events.iloc[0].right_censored and events.iloc[1].left_censored
    decisions = causal_alerts(series([0] * 10, np.arange(5, 55, 5)), .5)
    summary, _ = episode_metrics(labels, decisions)
    assert summary.complete_episodes == 0 and summary.censored_positive_pieces == 2
    assert pd.isna(summary.episode_recall)
    assert summary.monitored_seconds == 25  # 5..10 and 30..50; no time credited across the gap.


def test_decision_labels_do_not_fill_gaps_or_transition_intervals():
    labels = series([0, 1, 1, 0], [0, 2, 10, 12])
    result = decision_labels(labels, START + pd.to_timedelta([0, 1, 5, 10, 11, 13], unit='s'))
    assert result.iloc[0] == 0 and result.iloc[3] == 1
    assert result.iloc[[1, 2, 4, 5]].isna().all()


def test_exposure_splits_at_decision_changes_and_does_not_bridge_unavailable_states():
    labels = series([0] * 6, [0, 2, 4, 6, 8, 10])
    alarms = series([1, np.nan], [5, 10])
    exposure = exposure_seconds(labels, alarms)
    assert exposure['observed_negative_seconds'] == 10
    assert exposure['monitored_negative_seconds'] == 5
    assert exposure['alert_negative_seconds'] == 5
    assert exposure_seconds(labels.iloc[:1], alarms)['monitored_seconds'] == 0


def test_timezone_aware_episode_boundaries_are_preserved():
    labels = series([0, 1, 1, 0])
    labels.index = labels.index.tz_localize('Europe/London')
    events = observed_episodes(labels)
    assert events.end.dt.tz == labels.index.tz and events.iloc[0].complete


def test_no_positive_episodes_and_wholly_unavailable_monitoring_are_explicit():
    labels = series([0] * 21)
    decisions = causal_alerts(series([np.nan] * 4, [5, 10, 15, 20]), .5, 3)
    summary, events = episode_metrics(labels, decisions)
    assert events.empty and summary.complete_episodes == 0
    assert pd.isna(summary.episode_recall) and summary.monitored_seconds == 0
    assert summary.observed_negative_seconds == 20
    assert pd.isna(summary.negative_alerts_per_monitored_hour)


def test_isolation_score_is_not_the_contamination_cutoff():
    train = pd.DataFrame({'x': np.arange(20), 'y': np.arange(20) ** 2})
    first = IsolationForest(n_estimators=10, random_state=42, contamination='auto').fit(train)
    second = IsolationForest(n_estimators=10, random_state=42, contamination=.2).fit(train)
    np.testing.assert_allclose(first.score_samples(train), second.score_samples(train))
    assert first.offset_ != second.offset_
