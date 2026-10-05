import joblib
import numpy as np
import pandas as pd
from anomaly_detection.fixtures import scenario
from anomaly_detection.pipeline import fit, score, save, load
from anomaly_detection.incidents import detect
from anomaly_detection.selection import policy_grid


def test_complete_pipeline_save_load_order_and_future_mutation(tmp_path):
    data,expected,registry,truth,end,_,_=scenario('relationship_only')
    config=dict(features=dict(short=6,long=18),forest=dict(minimum_hours=24),distance_lengths=[6])
    model=fit(data,registry,config,end)
    before=joblib.hash(model)
    scores,_=score(model,data)
    assert joblib.hash(model)==before
    save(model,tmp_path/'model.joblib')
    restored=load(tmp_path/'model.joblib')
    reversed_scores,_=score(restored,data,routes=list(reversed(scores.route.unique())))
    keys=['route','entity_id','scope','event_time']
    pd.testing.assert_frame_equal(scores.sort_values(keys).reset_index(drop=True),
                                  reversed_scores.sort_values(keys).reset_index(drop=True),check_like=True)
    cut=end+pd.Timedelta(hours=5)
    changed=data.copy();changed.loc[changed.event_time.ge(cut),'value']=1e8
    future,_=score(model,changed)
    pd.testing.assert_frame_equal(scores.loc[scores.event_time.lt(cut)].reset_index(drop=True),
                                  future.loc[future.event_time.lt(cut)].reset_index(drop=True))
    # Each route has independently calibrated policies; notification prefixes agree.
    for route,part in scores.groupby('route'):
        _,policies=policy_grid(part.loc[part.decision_time.lt(end)])[0]
        original=detect(part,policies)[2]
        modified=detect(future.loc[future.route.eq(route)],policies)[2]
        pd.testing.assert_frame_equal(original.loc[original.time.lt(cut)].reset_index(drop=True),
                                      modified.loc[modified.time.lt(cut)].reset_index(drop=True))
