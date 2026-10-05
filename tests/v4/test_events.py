import numpy as np
import pandas as pd
import pytest
from anomaly_detection.incidents import Policy,detect,EVENT_COLUMNS
from anomaly_detection.grouping import group
from anomaly_detection.evaluation import match,evaluate,exposure,coverage
from test_contract import sample


def stream(values,stride=5,route='u_level'):
    t=pd.date_range('2025-01-01',periods=len(values),freq=f'{stride}min',tz='UTC')
    return pd.DataFrame(dict(entity_id='A',route=route,scope='x',event_time=t,decision_time=t,grid_index=np.arange(len(t)),score=values))


def test_equality_stride_gap_and_notifications():
    s=stream([3,3,2,4,4,1,0,0,np.nan,4],stride=10)
    p={('u_level','A','x'):Policy(2,1,opening=2,closing=2,stride_minutes=10,gap_minutes=15)}
    e,t,n=detect(s,p)
    assert e.confirmation.iloc[0]==s.decision_time.iloc[1]
    assert e.end_time.iloc[0]==s.decision_time.iloc[7]
    assert len(e)==1 and not t.active.iloc[-1]
    one=stream([3,np.nan,np.nan,3]);p={('u_level','A','x'):Policy(2,1,gap_minutes=5)}
    e,t,n=detect(one,p)
    assert e.reason.iloc[0]=='administrative_gap' and e.end_time.iloc[0]==one.decision_time.iloc[2]
    assert e.confirmation.iloc[0]==one.decision_time.iloc[0]
    assert n.time.isin(one.decision_time).all()
    s=stream([3,2,3]);p={('u_level','A','x'):Policy(2,1,opening=2)}
    assert detect(s,p)[0].empty


def test_route_order_and_prefix_independence():
    a=stream([4,4,0,0,0,4,4,0]);b=stream([0,4,4,0,0,0,4,0],route='u_shift')
    p={(r,'A','x'):Policy(2,1) for r in ['u_level','u_shift']}
    first=detect(pd.concat([a,b]),p);reverse=detect(pd.concat([b,a]),p)
    for x,y in zip(first,reverse):pd.testing.assert_frame_equal(x,y)
    pre=detect(pd.concat([a.iloc[:5],b.iloc[:5]]),p)[2]
    after=first[2].loc[first[2].time<=a.decision_time.iloc[4]].reset_index(drop=True)
    pd.testing.assert_frame_equal(pre,after)


def events(minutes,ends=None):
    origin=pd.Timestamp('2025-01-01',tz='UTC')
    ends=ends or [None]*len(minutes)
    return pd.DataFrame([dict(event_id=f'E{i}',entity_id='A',route='u_level',scope='x',confirmation=origin+pd.Timedelta(minutes=m),end_time=origin+pd.Timedelta(minutes=ends[i]) if ends[i] is not None else pd.NaT,reason='recovery' if ends[i] is not None else 'active') for i,m in enumerate(minutes)],columns=EVENT_COLUMNS)


def test_group_anchor_no_chaining_active_older_group_and_admin():
    e=events([0,25,50,60],[100,70,80,90])
    g,h=group(e,30,{'x':'symptom'})
    assert len(g)==2 and g.confirmation.tolist()==[e.confirmation.iloc[0],e.confirmation.iloc[2]]
    assert h.loc[h.action.eq('open'),'time'].tolist()==g.confirmation.tolist()
    assert g.end_time.tolist()==[e.end_time.iloc[0],e.end_time.iloc[3]]
    e.loc[1,'reason']='administrative_gap'
    assert group(e,30,{'x':'symptom'})[0].reason.iloc[0]=='administrative_or_mixed'
    pd.testing.assert_frame_equal(group(e.iloc[:2],30,{'x':'symptom'})[1].iloc[:2],group(e,30,{'x':'symptom'})[1].iloc[:2])


def truth():
    o=pd.Timestamp('2025-01-01',tz='UTC')
    return pd.DataFrame([dict(fault_id='A',entity_id='A',onset_time=o,observable_onset_time=o,impact_time=o+pd.Timedelta(minutes=20),end_time=o+pd.Timedelta(minutes=30)),
        dict(fault_id='B',entity_id='A',onset_time=o,observable_onset_time=o,impact_time=o+pd.Timedelta(minutes=12),end_time=o+pd.Timedelta(minutes=15))])


def test_maximum_matching_greedy_failure_early_ties_duplicates_halfopen():
    e=events([10,20,30]);f=truth()
    m=match(e,f);assert len(m)==2
    assert dict(zip(m.event_id,m.fault_id))=={'E0':'B','E1':'A'}
    assert len(match(e,f,lead_minutes=2))==1
    assert len(match(events([30]),f))==0
    assert len(match(events([-1]),f))==0
    assert match(events([10,10]),f).event_id.tolist()==['E0','E1']


def test_exposure_undefined_missing_and_duplicates():
    _,grid,reg=sample(np.ones(12))
    o=grid.event_time.min();end=o+pd.Timedelta(hours=1)
    extra=grid.copy();extra.metric_name='another'
    assert exposure(pd.concat([grid,extra]),o,end)==pytest.approx(1/24)
    m,_,_=evaluate(events([10,20,25,30]),truth(),grid,o,end,lead_minutes=2)
    assert m['tp']==2 and m['unmatched']==2 and m['fn']==0
    assert m['nuisance_per_1000_entity_days']==48000
    assert m['early_tp']==1 and m['early_recall']==.5
    empty=events([])
    m,_,_=evaluate(empty,truth().iloc[:0],grid,o,end)
    assert np.isnan(m['precision']) and np.isnan(m['recall'])
    s=stream([np.nan]*12);c=coverage(s,o,end)
    assert c['any_coverage']==0 and c['schedule_decisions']==12
