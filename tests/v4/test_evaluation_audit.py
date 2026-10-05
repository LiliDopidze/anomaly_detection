import numpy as np
import pandas as pd
import pytest
from anomaly_detection.evaluation import coverage, evaluate, grouping_audit
from anomaly_detection.grouping import group
from test_contract import sample
from test_events import events, stream, truth


def test_coverage_missing_rows_uses_independent_schedule():
    _,grid,_=sample(np.ones(12))
    start=grid.event_time.min();end=start+pd.Timedelta(hours=1)
    complete=stream(np.zeros(12))
    absent=complete.drop(index=[3,4,5])
    c=coverage(absent,start,end,grid)
    assert c['schedule_decisions']==12
    assert c['any_coverage']==.75
    assert c['longest_score_gap_minutes']==15
    roster=complete[['route','entity_id','scope','decision_time']]
    assert coverage(complete.iloc[:0],start,end,roster)['any_coverage']==0
    with pytest.raises(ValueError,match='Duplicate'):
        coverage(pd.concat([complete,complete.iloc[:1]]),start,end,grid)


def test_recovery_after_horizon_is_censored_and_alarm_clipped_to_schedule():
    _,grid,_=sample(np.ones(12));start=grid.event_time.min()
    end=start+pd.Timedelta(hours=1)
    e=events([10],[90]);t=truth().iloc[:1]
    a,_,_=evaluate(e,t,grid,start,end)
    e.loc[0,'end_time']=start+pd.Timedelta(hours=100)
    b,_,_=evaluate(e,t,grid,start,end)
    assert a==b or all((pd.isna(a[k]) and pd.isna(b[k])) or a[k]==b[k] for k in a)
    assert a['recovery_sample']==0 and a['recovery_censored_matches']==1
    # No monitoring for the middle half-hour; alarm duration cannot exceed exposure.
    grid=grid.loc[~grid.grid_index.between(3,8)]
    result,_,_=evaluate(events([0]),t,grid,start,end)
    assert result['time_in_alarm_hours']==.5
    assert result['time_in_alarm_fraction']==1


def test_early_cohort_uses_observable_onset_and_boundary_unknown_workload():
    _,grid,_=sample(np.ones(12));start=grid.event_time.min();end=start+pd.Timedelta(hours=1)
    t=truth().iloc[:1].copy()
    t['onset_time']=start-pd.Timedelta(hours=1)
    t['observable_onset_time']=start+pd.Timedelta(minutes=5)
    m,_,_=evaluate(events([10]),t,grid,start,end,lead_minutes=5)
    assert m['physical_faults']==0 and m['faults']==1
    assert m['known_impact_faults']==1 and m['early_tp']==1
    t['observable_onset_time']=pd.NaT
    m,_,_=evaluate(events([10]),t,grid,start,end)
    assert m['unmatched']==1 and m['unverified']==1 and m['fp']==0


def test_group_merge_audit_checks_later_members_not_only_anchor():
    e=events([10,50],[100,100]);t=truth().copy()
    origin=e.confirmation.iloc[0]-pd.Timedelta(minutes=10)
    t.loc[1,'observable_onset_time']=origin+pd.Timedelta(minutes=40)
    t.loc[1,'end_time']=origin+pd.Timedelta(minutes=70)
    _,history=group(e,60,{'x':'symptom'})
    audit,links=grouping_audit(e,history,t,origin,origin+pd.Timedelta(hours=2))
    assert audit['merge_risk_groups']==1 and len(links)==2
    assert audit['fragmented_faults']==0
