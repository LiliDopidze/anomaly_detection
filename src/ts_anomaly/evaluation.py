"""Recorded-label evaluation with explicit gaps, timing and unmonitored outcomes."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .assessment import state_runs


def label_windows(labels: pd.Series, endpoints, frequency='5s', min_count=4):
    """Pure observed labels in [t-step,t); mixed/undersupported bins stay unknown."""
    if not labels.dropna().isin([0, 1]).all():
        raise ValueError('Annotations must be binary or missing')
    groups = labels.resample(frequency, closed='left', label='left')
    result = pd.DataFrame({'count': groups.count(), 'minimum': groups.min(), 'maximum': groups.max()})
    result.index += pd.Timedelta(frequency)
    result = result.reindex(endpoints)
    result['status'] = 'insufficient_labels'
    supported = result['count'].ge(min_count)
    result.loc[supported, 'status'] = 'mixed'
    pure = supported & result.minimum.eq(result.maximum)
    result.loc[pure, 'status'] = 'pure'
    result['label'] = result.minimum.where(pure)
    return result


def point_metrics(labels, scores, alarms, comparison_mask=None):
    """Paired metrics and separate availability; no point adjustment or gap filling.

    Overall positive-bin recall counts an unavailable decision as not detected.
    AP requires both classes. Precision without any predicted positives is undefined.
    """
    if not labels.index.equals(scores.index) or not labels.index.equals(alarms.index):
        raise ValueError('Labels, scores and alarms must align')
    mask = pd.Series(True, index=scores.index) if comparison_mask is None else comparison_mask
    if not mask.index.equals(scores.index):
        raise ValueError('Comparison mask must align')
    valid_label = labels.isin([0, 1])
    common = valid_label & np.isfinite(scores) & np.isfinite(alarms) & mask
    truth, predicted = labels.loc[common], alarms.loc[common].eq(1)
    positive, negative = int(truth.eq(1).sum()), int(truth.eq(0).sum())
    tp = int((truth.eq(1) & predicted).sum())
    fp = int((truth.eq(0) & predicted).sum())
    all_positive = labels.eq(1)
    total_positive = int(all_positive.sum())
    ap = average_precision_score(truth, scores.loc[common]) if positive and negative else np.nan
    return pd.Series({'requested': len(scores), 'score_available': int(np.isfinite(scores).sum()),
                      'decision_available': int(np.isfinite(alarms).sum()),
                      'labelled_bins': int(valid_label.sum()), 'common': int(common.sum()),
                      'positive_common': positive, 'negative_common': negative,
                      'average_precision': ap, 'precision': tp / (tp + fp) if tp + fp else np.nan,
                      'recall': tp / positive if positive else np.nan,
                      'negative_bin_alert_fraction': fp / negative if negative else np.nan,
                      'positive_bins_total': total_positive,
                      'positive_bins_monitored': int((all_positive & np.isfinite(alarms)).sum()),
                      'positive_bins_detected': int((all_positive & alarms.eq(1)).sum()),
                      'overall_positive_bin_recall': ((all_positive & alarms.eq(1)).sum() / total_positive
                                                      if total_positive else np.nan)})


def observed_episodes(labels: pd.Series, max_gap='2s'):
    """Positive observed runs, retaining censored pieces rather than inventing incidents."""
    runs = state_runs(labels, max_gap)
    positive = runs.loc[runs.value.eq(1)].copy()
    positive['end'] = pd.Series(pd.NaT, index=positive.index, dtype=labels.index.dtype)
    for i, run in positive.iterrows():
        if not run.right_censored:
            following = labels.index.searchsorted(run['last'], side='right')
            positive.loc[i, 'end'] = labels.index[following]
    positive['complete'] = ~positive.left_censored & ~positive.right_censored
    return positive.reset_index(drop=True)


def decision_labels(labels: pd.Series, timestamps, max_gap='2s'):
    """Retrospective labels at alert times: exact samples or agreeing connected neighbours.

    No assignment across collection gaps, a label transition or outside the source.
    The bracket is used for evaluation only, never to construct a score or alert.
    """
    result = pd.Series(np.nan, index=timestamps)
    for time in timestamps:
        right = labels.index.searchsorted(time, side='left')
        if right < len(labels) and labels.index[right] == time:
            result.loc[time] = labels.iloc[right]
        elif 0 < right < len(labels):
            connected = labels.index[right] - labels.index[right - 1] <= pd.Timedelta(max_gap)
            if connected and labels.iloc[right] == labels.iloc[right - 1]:
                result.loc[time] = labels.iloc[right]
    return result


def exposure_seconds(labels: pd.Series, alarms: pd.Series, frequency='5s', max_gap='2s'):
    """Integrate held alert states over observed native intervals, never over gaps.

    Each decision holds for at most one feature interval. Negative exposure requires
    two connected label-zero endpoints; uncertain transition intervals are excluded.
    A union of timestamps splits intervals exactly at each decision change.
    """
    step = pd.Timedelta(frequency)
    if len(labels) < 2 or alarms.empty:
        return dict.fromkeys(['observed_negative_seconds', 'monitored_negative_seconds',
                              'alert_negative_seconds', 'monitored_seconds', 'alert_seconds'], 0.0)
    edges = labels.index.union(alarms.index).sort_values()
    edges = edges[(edges >= labels.index[0]) & (edges <= labels.index[-1])]
    left, right = edges[:-1], edges[1:]
    native_position = labels.index.searchsorted(left, side='right') - 1
    decision_position = alarms.index.searchsorted(left, side='right') - 1
    inside = (native_position >= 0) & (native_position + 1 < len(labels))
    position = np.clip(native_position, 0, len(labels) - 2)
    raw_gap = labels.index[position + 1] - labels.index[position]
    connected = inside & (raw_gap <= pd.Timedelta(max_gap))
    native_labels = labels.to_numpy(dtype=float)
    negative = connected & (native_labels[position] == 0) & (native_labels[position + 1] == 0)
    alarm_position = np.maximum(decision_position, 0)
    state = alarms.to_numpy()[alarm_position]
    monitored = (decision_position >= 0) & np.isfinite(state) & (right <= alarms.index[alarm_position] + step)
    duration = (right - left).total_seconds().to_numpy()
    return {'observed_negative_seconds': float(duration[negative].sum()),
            'monitored_negative_seconds': float(duration[negative & monitored].sum()),
            'alert_negative_seconds': float(duration[negative & monitored & (state == 1)].sum()),
            'monitored_seconds': float(duration[connected & monitored].sum()),
            'alert_seconds': float(duration[connected & monitored & (state == 1)].sum())}


def episode_metrics(labels, decisions, frequency='5s', max_gap='2s'):
    """New alert starts within complete observed episodes; report censoring separately.

    A standing alert cannot claim a later event. Delays use actual decision endpoints,
    not the beginning of an averaging or persistence window. No end-of-event backfill.
    """
    events = observed_episodes(labels, max_gap)
    onset_times = decisions.index[decisions.alarm_start]
    events['detected'] = False
    events['first_alert'] = pd.Series(pd.NaT, index=events.index, dtype=labels.index.dtype)
    events['delay_s'] = np.nan
    assessment_start = decisions.index[0] - pd.Timedelta(frequency)
    events['in_assessment'] = events.start.ge(assessment_start) & events.start.le(decisions.index[-1])
    for i, event in events.loc[events.complete & events.in_assessment].iterrows():
        hits = onset_times[(onset_times >= event.start) & (onset_times < event.end)]
        if len(hits):
            events.loc[i, ['detected', 'first_alert', 'delay_s']] = [
                True, hits[0], (hits[0] - event.start).total_seconds()]
    eligible = events.loc[events.complete & events.in_assessment]
    matched = int(eligible.detected.sum())
    onset_labels = decision_labels(labels, onset_times, max_gap)
    exposure = exposure_seconds(labels, decisions.alarm, frequency, max_gap)
    negative_hours = exposure['monitored_negative_seconds'] / 3600
    negative_alerts = int(onset_labels.eq(0).sum())
    summary = {'alert_starts': len(onset_times), 'label_negative_alerts': negative_alerts,
               'unknown_label_alert_starts': int(onset_labels.isna().sum()),
               'complete_episodes': len(eligible), 'detected_episodes': matched,
               'censored_positive_pieces': int((~events.complete & events.in_assessment).sum()),
               'episode_recall': matched / len(eligible) if len(eligible) else np.nan,
               'median_delay_s': eligible.loc[eligible.detected, 'delay_s'].median(),
               'negative_alerts_per_monitored_hour': negative_alerts / negative_hours if negative_hours else np.nan,
               **exposure}
    return pd.Series(summary), events
