from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
import joblib
from optical_anomaly.generator import GeneratorConfig,generate
from optical_anomaly.sources import synthetic_adapter
from anomaly_detection.adapters import gpon
from anomaly_detection.validation import validate
from anomaly_detection.references import References
from anomaly_detection.features import features,joint_vectors
from anomaly_detection.detectors import Forest
from anomaly_detection.distance_profiles import NormalBank
from anomaly_detection.selection import policy_grid,select_routes
from anomaly_detection.evaluation import paired_block_intervals
from test_contract import sample


def test_generator_adapter_exact_parity():
    config=GeneratorConfig(entities=1,days=8,interval_minutes=15)
    native,truth=generate(config);again,labels=generate(config)
    pd.testing.assert_frame_equal(native,again,check_exact=True)
    pd.testing.assert_frame_equal(truth,labels,check_exact=True)
    old=synthetic_adapter().transform(native).rename(columns={'timestamp':'event_time'})
    new=gpon(native)
    pd.testing.assert_frame_equal(old,new[old.columns],check_exact=True)
    assert new.available_time.equals(new.event_time)


def test_future_fit_isolation_missing_joint_channel_and_no_score_mutation():
    obs,grid,reg=sample(np.sin(np.arange(1000)/10))
    data=validate(obs,grid,reg);end=data.event_time.iloc[700]
    mutated=data.copy();mutated.loc[700:,'value']=1e8
    a=References(reg).fit(data,end);b=References(reg).fit(mutated,end)
    assert joblib.hash(a)==joblib.hash(b)
    residual=a.score(data);feature=features(residual,reg,short=3,long=5)
    v,c=joint_vectors(feature,['x']);forest=Forest(c,minimum_hours=1).fit(v,end)
    future=v.copy();future.loc[700:,c]=1e9
    forest2=Forest(c,minimum_hours=1).fit(future,end)
    assert joblib.hash(forest)==joblib.hash(forest2)
    before=joblib.hash(forest);forest.score(v,'x',10);assert before==joblib.hash(forest)
    bank=NormalBank(6,'shape').fit(residual,end);before=joblib.hash(bank)
    bank.score(residual.iloc[:20]);assert before==joblib.hash(bank)
    # Fixed vector has an absent declared channel; valid x residuals still score.
    vectors,columns=joint_vectors(feature,['x','missing'])
    assert vectors[columns].isna().any(axis=1).all()
    assert feature.u_level.notna().all()


def test_selection_final_guard_and_quantile_tie_budget():
    times=pd.date_range('2025-01-01',periods=20,freq='5min',tz='UTC')
    s=pd.DataFrame(dict(route='u_level',entity_id='A',scope='x',event_time=times,decision_time=times,score=np.zeros(20)))
    candidates=policy_grid(s)
    assert len(candidates)==4  # q99=q999 deduplicates; one explicit edge extension.
    with pytest.raises(ValueError,match='final'):
        select_routes(s,pd.DataFrame({'onset_time':[]}),pd.DataFrame(),times[10],times[15],{}, {'u_level':0})


def test_paired_blocks_no_row_bootstrap():
    counts=pd.DataFrame([dict(candidate=c,block=b,tp=1 if c=='a' else 0,faults=2) for c in ['a','b'] for b in ['site1','site2']])
    result=paired_block_intervals(counts,draws=20)
    assert result['paired_difference']['b']==[-.5,-.5]
    assert paired_block_intervals(counts.loc[counts.block.eq('site1')])['status']=='unavailable'
