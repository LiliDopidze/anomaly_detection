import numpy as np
import pandas as pd
import pytest
import joblib
from anomaly_detection.distance_profiles import NormalBank, oracle
from anomaly_detection.detectors import Forest
from anomaly_detection.features import features,joint_vectors
from anomaly_detection.validation import validate
from test_contract import sample


def residual(values):
    obs,grid,reg=sample(values)
    d=validate(obs,grid,reg);d['y']=d.value;d['z']=d.value
    return d,reg


@pytest.mark.parametrize('mode',['shape','level'])
def test_distance_oracle_full_overlap_constants_and_prefix(mode,tmp_path):
    x=np.random.default_rng(4).normal(size=80)
    d,_=residual(x);bank=NormalBank(6,mode).fit(d,d.event_time.iloc[50])
    result=bank.query(('A','x'),x[47:53],d.event_time.iloc[47],d.event_time.iloc[52])
    b=bank.banks['A','x'];windows=np.lib.stride_tricks.sliding_window_view(x[:50],6)
    eligible=b['ends']<d.event_time.iloc[47].value
    expected=oracle(x[47:53],windows,mode=='shape');expected[~eligible]=np.inf
    assert result['score']==pytest.approx(expected.min(),abs=1e-6)
    assert result['overlap_excluded']==3
    first=bank.score(d.iloc[:60]);changed=d.copy();changed.loc[60:,'y']=100;changed.loc[60:,'z']=100
    pd.testing.assert_frame_equal(first,bank.score(changed).iloc[:60].reset_index(drop=True))
    joblib.dump(bank,tmp_path/'bank.joblib');loaded=joblib.load(tmp_path/'bank.joblib')
    pd.testing.assert_frame_equal(first,loaded.score(d.iloc[:60]))
    assert bank.banks['A','x']['hash']==loaded.banks['A','x']['hash']
    const=bank.query(('A','x'),np.ones(6),d.event_time.iloc[60],d.event_time.iloc[65])
    assert (const['reason']=='constant_shape_query')==(mode=='shape')
    affine=oracle(2*x[10:16]+7,np.array([x[10:16]]),shape=True)
    assert affine[0]<1e-12


def test_bank_gap_disjoint_floor_repeated_fault_and_identity():
    x=np.tile([0.,1.,0.,2.,-1.,0.],12);x[20:23]=np.nan
    d,_=residual(x);bank=NormalBank(6,'level').fit(d,d.event_time.iloc[40])
    b=bank.banks['A','x'];assert not b['valid'][15:23].any()
    q=np.arange(6.)+100
    a=bank.query(('A','x'),q,d.event_time.iloc[45],d.event_time.iloc[50])
    c=bank.query(('A','x'),q,d.event_time.iloc[60],d.event_time.iloc[65])
    assert a['score']==c['score'] and a['bank_hash']==c['bank_hash']
    tiny=NormalBank(6,'level').fit(d.iloc[:12],d.event_time.iloc[12])
    assert tiny.query(('A','x'),x[0:6],d.event_time.iloc[0],d.event_time.iloc[5])['reason']=='insufficient_disjoint_bank'
    changed=d.copy();changed.loc[50:,'z']=-100
    again=NormalBank(6,'level').fit(changed,d.event_time.iloc[40])
    assert again.banks['A','x']['hash']==b['hash']


def test_forest_matched_rows_save_load_unknown_and_relationship(tmp_path):
    d,reg=residual(np.sin(np.arange(800)/10)+.1*np.random.default_rng(1).normal(size=800))
    f=features(d,reg,short=3,long=9);end=d.event_time.iloc[650]
    v,c=joint_vectors(f,['x']);w,h=joint_vectors(f,['x'],'historical8')
    a=Forest(c,minimum_hours=1).fit(v,end);b=Forest(h,profile='historical8',minimum_hours=1).fit(w,end)
    assert a.models['A']['rows']==b.models['A']['rows']
    joblib.dump(a,tmp_path/'forest.joblib')
    pd.testing.assert_frame_equal(a.score(v,'x',10),joblib.load(tmp_path/'forest.joblib').score(v,'x',10))
    unknown=v.copy();unknown.entity_id='new';assert a.score(unknown,'x',10).score.isna().all()
    with pytest.raises(ValueError,match='constant'):
        zeros=v.copy();zeros[c]=0;Forest(c,minimum_hours=1).fit(zeros,end)
    # Joint scoring is called directly, with no univariate flags or gates.
    relationship=v.copy();relationship.loc[700:,c[0]]=1.;relationship.loc[700:,c[2]]=-1.
    assert a.score(relationship,'x',10).score.iloc[700:].notna().all()
