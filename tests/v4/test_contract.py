import numpy as np
import pandas as pd
import pytest
from anomaly_detection.validation import Metric, schedule, validate, CANONICAL, counter_rate, interval_ratio, aggregate_complete
from anomaly_detection.adapters import non_optical_fixture, optical_loss, fec_fraction
from anomaly_detection.references import References
from anomaly_detection.features import features, joint_vectors, COMPACT, HISTORICAL


def sample(values, direction="two-sided", cadence=5):
    t=pd.date_range("2025-01-01",periods=len(values),freq=f"{cadence}min",tz="UTC")
    registry={"x":Metric("unit",direction=direction,cadence_minutes=cadence,min_observations=3,min_hours=.1)}
    expected=schedule(["A"],registry,t[0],t[-1]+pd.Timedelta(minutes=cadence))
    obs=pd.DataFrame(dict(entity_id="A",metric_name="x",event_time=t,available_time=t,value=values,quality="ok"))[CANONICAL]
    return obs, expected, registry


def test_schedule_gaps_late_duplicates_and_portability():
    obs,expected,reg=sample([1,2,3,4])
    obs.loc[1,"available_time"]+=pd.Timedelta(minutes=5)
    data=validate(obs.drop(index=2),expected,reg)
    assert data.quality.tolist()==["ok","late","missing","ok"]
    assert data.value.isna().tolist()==[False,True,True,False]
    assert data.segment.iloc[-1]!=data.segment.iloc[0]
    with pytest.raises(ValueError,match="duplicate"):
        validate(pd.concat([obs,obs.iloc[:1]]),expected,reg)
    non,reg=non_optical_fixture()
    grid=schedule(["machine-A"],reg,non.event_time.min(),non.event_time.max()+pd.Timedelta(minutes=5))
    data=validate(non,grid,reg)
    ref=References(reg).fit(data,data.event_time.max()+pd.Timedelta(minutes=5))
    assert len(ref.models)==2
    assert np.isfinite(ref.score(data).z).all()


def test_counter_wrap_reset_interval_denominator():
    t=pd.date_range("2025-01-01",periods=5,freq="5min",tz="UTC")
    result=counter_rate([90,95,2,4,1],t,5,modulus=100,wrapped=[0,0,1,0,0],reset=[0,0,0,0,1])
    np.testing.assert_allclose(result,[np.nan,5/300,7/300,2/300,np.nan],equal_nan=True)
    assert np.isnan(counter_rate([2,1],t[:2],5)[1])
    np.testing.assert_allclose(interval_ratio([np.nan,1,1],[10,0,10],t[:3],t[:3],t[1:4],t[1:4]),[np.nan,np.nan,.1],equal_nan=True)
    with pytest.raises(ValueError): interval_ratio([1],[2],t[:1],t[1:2],t[:1],t[:1])
    with pytest.raises(ValueError): fec_fraction([2],[2],[3],t[:1],t[1:2])
    with pytest.raises(ValueError): optical_loss([1],[0],t[:1],t[1:2])
    obs,_,_=sample([1,2,3,4,5,6,7])
    agg=aggregate_complete(obs,5,15,"sum")
    assert agg.iloc[1].value==9
    assert np.isnan(aggregate_complete(obs.drop(index=2),5,15,"sum").iloc[1].value)


def test_residual_centre_floor_future_and_unknown_entity():
    obs,grid,reg=sample([2.,2.,2.,3.,10.,100.])
    data=validate(obs,grid,reg);end=data.event_time.iloc[4]
    ref=References(reg).fit(data,end,allow_seasonal=False)
    model=ref.models['A','x']
    assert model['centre']==0 and model['scale']==.05
    assert ref.score(data).z.iloc[4]==160
    changed=data.copy();changed.loc[4:,"value"]=-1000
    other=References(reg).fit(changed,end,allow_seasonal=False)
    assert other.models['A','x']['scale']==model['scale']
    pd.testing.assert_frame_equal(ref.score(changed).iloc[:4],ref.score(data).iloc[:4])
    unknown=data.copy();unknown.entity_id="new"
    assert ref.score(unknown).z.isna().all()


def test_feature_hand_arithmetic_gap_direction_and_exact_dependency():
    obs,grid,reg=sample([0,1,2,3,4,np.nan,6,7,8,9],cadence=60)
    data=validate(obs,grid,reg);data['z']=data.value
    out=features(data,reg,short=3,long=5,allowance=.5,boundary=2)
    row=out.iloc[4]
    assert row['mean']==3 and row['std']==pytest.approx(np.std([2,3,4]))
    assert row.slope==1 and row.ewma_difference==1
    assert row.long_mean==2 and row.short_minus_long==1
    assert row.persistence==pytest.approx(2/3)
    assert row.cusum==8
    assert out['mean'].iloc[5:8].isna().all()
    assert out.u_shift.iloc[6]==5.5
    assert out.long_count.iloc[8]==3
    v,c=joint_vectors(out,['x']);assert c==['x:'+n for n in COMPACT]
    assert len(joint_vectors(out,['x'],'historical8')[1])==8
    for direction,expected in [('high',[0,0,1.5]),('low',[1.5,1,0]),('two-sided',[1.5,1,1.5])]:
        obs,grid,reg=sample([-2,0,2],direction=direction)
        d=validate(obs,grid,reg);d['z']=d.value
        np.testing.assert_allclose(features(d,reg,short=3,long=4,allowance=.5).u_shift,expected)


def test_prefix_feature_invariance():
    obs,grid,reg=sample(np.sin(np.arange(200)))
    d=validate(obs,grid,reg);d['z']=d.value
    before=features(d,reg,short=3,long=5).iloc[:100]
    d.loc[100:,'z']=1e6
    pd.testing.assert_frame_equal(before,features(d.iloc[:100],reg,short=3,long=5))
    pd.testing.assert_frame_equal(before,features(d,reg,short=3,long=5).iloc[:100])
