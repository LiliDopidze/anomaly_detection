"""Hand-checkable feature calculations and causal availability boundaries."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ts_anomaly.features import engineer_recording, feature_summary, trailing_statistics

START = pd.Timestamp('2020-01-01')


def recording(values, seconds=None, entity='one'):
    seconds = np.arange(len(values)) if seconds is None else seconds
    return pd.DataFrame({'timestamp': START + pd.to_timedelta(seconds, unit='s'),
                         'entity_id': entity, 'x': np.asarray(values, dtype=float)})


def engineer(data, calibration_seconds=60):
    return engineer_recording(data, ['x'], ['x'], ['x'], ['x'], calibration_seconds)


@pytest.mark.parametrize('constant', [4, .1])
def test_constant_and_known_linear_ramp(constant):
    data = recording(np.ones(120) * constant)
    values, support, refs = engineer(data)
    assert values.loc[START + pd.Timedelta(seconds=65):].eq(0).all().all()
    assert refs.iloc[0].expected == constant and refs.iloc[0].collapsed_range
    data.x = 3 + 2 * np.arange(120)
    values, _, _ = engineer(data)
    assert np.allclose(values.x__slope.dropna(), 120)  # 2 units/s = 120 units/min.
    assert np.allclose(values.x__iqr.dropna(), 29)  # Q75-Q25 of 30 evenly spaced values.


def test_irregular_times_use_elapsed_time_and_occupied_slots():
    seconds = np.delete(np.arange(30), [4, 9, 14, 19, 24])
    series = recording(7 + seconds * 3, seconds).set_index('timestamp').x
    result = trailing_statistics(series, [START + pd.Timedelta(seconds=30)]).iloc[0]
    assert result.slope == pytest.approx(180)
    assert result.occupied_slots == 25 and result.occupancy == pytest.approx(25 / 30)
    assert result.iqr == np.quantile(series, .75) - np.quantile(series, .25)
    # Multiple readings in a nominal second do not manufacture occupancy.
    data = recording([1, 1, 1, 1], [0, .1, 1, 2])
    result = trailing_statistics(data.set_index('timestamp').x,
                                [START + pd.Timedelta(seconds=5)], '5s').iloc[0]
    assert result.observations == 4 and result.occupied_slots == 3
    assert result.reason == 'low_occupancy' and pd.isna(result.iqr)


def test_step_and_completed_availability_times():
    data = recording(np.r_[np.zeros(60), np.full(60, 4)])
    values, support, refs = engineer(data)
    assert values.index[0] == START + pd.Timedelta(seconds=5)
    assert values.loc[:START + pd.Timedelta(seconds=60), 'x__level'].isna().all()
    assert values.loc[START + pd.Timedelta(seconds=65), 'x__level'] == 4
    assert values.loc[START + pd.Timedelta(seconds=65), 'x__history'] == 4
    assert values.loc[START + pd.Timedelta(seconds=70), 'x__history'] == 0
    history = support.loc[support.feature.eq('x__history')].iloc[1]
    assert history.window_start == START + pd.Timedelta(seconds=5)
    assert history.history_start == START and history.previous_bin_available
    assert refs.iloc[0].observations == 12 and refs.iloc[0].expected == 0


def test_level_reference_uses_aggregate_representation_not_native_median():
    data = recording([0, 0, 0, 0, 100] + [0] * 15)
    values, _, refs = engineer(data, 10)
    assert data.x.iloc[:10].median() == 0
    assert refs.iloc[0].expected == 10  # Median of means [20, 0].
    assert refs.iloc[0].lower == 1 and refs.iloc[0].upper == 19
    assert values.loc[START + pd.Timedelta(seconds=15), 'x__level'] == -10


def test_half_open_calibration_future_and_annotation_independence():
    data = recording(np.arange(150))
    before, _, references = engineer(data)
    changed = data.copy()
    cutoff = START + pd.Timedelta(seconds=100)
    changed.loc[changed.timestamp >= cutoff, 'x'] = -9999
    changed['anomaly'] = 1
    after, _, later_refs = engineer(changed)
    pd.testing.assert_frame_equal(before.loc[:cutoff], after.loc[:cutoff])
    pd.testing.assert_frame_equal(references, later_refs)
    changed = data.copy()
    changed.loc[changed.timestamp >= START + pd.Timedelta(seconds=60), 'x'] = 1e6
    _, _, later_refs = engineer(changed)
    pd.testing.assert_frame_equal(references, later_refs)
    annotated = data.assign(anomaly=1, changepoint=1)
    pd.testing.assert_frame_equal(before, engineer(annotated)[0])


def test_non_aligned_calibration_excludes_straddling_bins():
    data = recording(np.arange(120), np.arange(2, 122))
    values, _, refs = engineer(data)
    # Whole fit bins are [5,10), ... [55,60); neither [0,5) nor [60,65).
    assert refs.iloc[0].observations == 11
    assert refs.iloc[0].expected == 30
    assert pd.isna(values.loc[START + pd.Timedelta(seconds=65), 'x__level'])
    assert pd.notna(values.loc[START + pd.Timedelta(seconds=70), 'x__level'])


def test_insufficient_calibration_does_not_remove_unfitted_features():
    seconds = np.r_[np.arange(10), np.arange(40, 120)]
    values, support, refs = engineer(recording(np.ones(len(seconds)), seconds))
    assert not refs.iloc[0].supported
    assert refs.iloc[0].occupancy == .5
    assert values.x__level.isna().all()
    assert values.x__slope.dropna().eq(0).all() and values.x__slope.notna().any()
    failed = support.loc[support.feature.eq('x__level') & support.timestamp.gt(refs.iloc[0].fit_end)]
    assert failed.reason.eq('unsupported_calibration').all()


def test_gap_at_window_edge_and_exact_recovery():
    seconds = np.r_[np.arange(19), np.arange(21, 100)]  # Raw 3-second gap.
    series = recording(np.ones(len(seconds)), seconds).set_index('timestamp').x
    endpoints = START + pd.to_timedelta([50, 51], unit='s')
    result = trailing_statistics(series, endpoints)
    assert result.iloc[0].occupancy > .8  # The left-edge gap still invalidates it.
    assert result.iloc[0].reason == 'collection_gap' and pd.isna(result.iloc[0].slope)
    assert result.iloc[1].reason == 'available' and result.iloc[1].slope == 0


def test_stale_right_edge_uses_no_future_reading():
    data = recording(np.ones(35))
    data.loc[data.timestamp.between(START + pd.Timedelta(seconds=27),
                                    START + pd.Timedelta(seconds=29)), 'x'] = np.nan
    result = trailing_statistics(data.set_index('timestamp').x,
                                 [START + pd.Timedelta(seconds=30)]).iloc[0]
    assert result.occupancy == .9 and result.reason == 'stale_tail'
    data.loc[data.timestamp >= START + pd.Timedelta(seconds=30), 'x'] = np.nan
    after = trailing_statistics(data.set_index('timestamp').x,
                                [START + pd.Timedelta(seconds=30)]).iloc[0]
    pd.testing.assert_series_equal(result, after)


def test_supported_means_can_still_have_boundary_gap_and_reset_history():
    seconds = np.r_[np.arange(64), np.arange(66, 100)]
    values, support, _ = engineer(recording(np.ones(len(seconds)), seconds))
    # Both [60,65) and [65,70) have four readings, but 63 -> 66 exceeds 2 s.
    assert pd.isna(values.loc[START + pd.Timedelta(seconds=70), 'x__level'])
    assert pd.isna(values.loc[START + pd.Timedelta(seconds=75), 'x__history'])
    assert values.loc[START + pd.Timedelta(seconds=80), 'x__history'] == 0
    assert pd.isna(values.loc[START + pd.Timedelta(seconds=95), 'x__slope'])
    assert values.loc[START + pd.Timedelta(seconds=100), 'x__slope'] == 0


def test_nonfinite_missing_windows_and_unavailable_are_not_zero():
    data = recording(np.ones(120))
    data.loc[70:79, 'x'] = np.inf
    values, support, _ = engineer(data)
    assert values.loc[START + pd.Timedelta(seconds=75)].isna().all()
    assert values.loc[START + pd.Timedelta(seconds=85), 'x__level'] == 0
    assert pd.isna(values.loc[START + pd.Timedelta(seconds=85), 'x__history'])
    assert values.loc[START + pd.Timedelta(seconds=90), 'x__history'] == 0
    availability = support.pivot(index='timestamp', columns='feature', values='available')
    pd.testing.assert_frame_equal(availability[values.columns], values.notna(),
                                  check_names=False, check_freq=False)


def test_independent_recordings_and_common_populations():
    first = recording(np.zeros(120))
    second = recording(np.full(120, 10), entity='two')
    values = engineer(second)[0]
    assert pd.isna(values.x__history.iloc[0])
    assert values.x__level.dropna().eq(0).all()
    with pytest.raises(ValueError, match='one entity'):
        engineer(pd.concat([first.iloc[:60], second.iloc[60:]]))
    level_only = values[['x__level']].notna().all(axis=1)
    expanded = values.notna().all(axis=1)
    assert (level_only & expanded).sum() == expanded.sum()
    assert level_only.sum() <= len(values) and len(values) == 24  # No automatic row dropping.


def test_summary_distinguishes_zero_missing_and_constant_spans():
    values = pd.DataFrame({'x': [0, 0, np.nan, 2, 2, 2]},
                         index=START + pd.to_timedelta([5, 10, 15, 20, 25, 35], unit='s'))
    row = feature_summary(values).iloc[0]
    assert row.available == 5 and row.unavailable == 1 and row.zeros == 2
    assert row.distinct == 2 and row.constant_runs == 2
    assert row.longest_constant_span_s == 5  # Missing timestamp 30 splits the last run.


def test_feature_workflow_never_requests_reserved_files(monkeypatch, tmp_path):
    from ts_anomaly import skab
    settings = json.loads((Path(__file__).resolve().parents[1] / 'configs/skab.json').read_text())
    requested = []

    def fake_fetch(cache, commit, item, download):
        assert item['partition'] != 'reserved_test'
        requested.append(item['recording_id'])
        return tmp_path / 'permitted.csv'

    def fake_read(path, identifier):
        return recording(np.ones(80), entity=identifier), None

    monkeypatch.setattr(skab, 'fetch_recording', fake_fetch)
    monkeypatch.setattr(skab, 'read_recording', fake_read)
    data, _ = skab.load_development(tmp_path, settings, download=False)
    for frame in data.values():
        assert engineer(frame)[0].x__level.notna().any()
    assert len(requested) == 27
    reserved = {item['recording_id'] for item in settings['files'] if item['partition'] == 'reserved_test'}
    assert reserved.isdisjoint(requested)
