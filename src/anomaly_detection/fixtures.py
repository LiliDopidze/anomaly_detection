"""Isolated diagnostic scenarios, never mixed with unchanged GPON benchmark truth."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from .validation import Metric,schedule,validate,CANONICAL
from .pipeline import fit,score,write_json,provenance
from .selection import policy_grid
from .incidents import detect
from .evaluation import evaluate,coverage


def scenario(name,contamination=0.):
    n=900;t=pd.date_range('2025-01-01',periods=n,freq='5min',tz='UTC')
    rng=np.random.default_rng(74)
    a=np.sin(2*np.pi*np.arange(n)/24)+rng.normal(0,.08,n)
    b=a+rng.normal(0,.025,n)
    intervals=[]
    if name=='spike':a[680]+=6;b[680]+=6;intervals=[(680,681)]
    elif name=='subtle_shift':a[680:752]+=.7;b[680:752]+=.7;intervals=[(680,752)]
    elif name=='step':a[680:752]+=4;b[680:752]+=4;intervals=[(680,752)]
    elif name=='ramp':a[660:804]+=np.linspace(0,4,144);b[660:804]+=np.linspace(0,4,144);intervals=[(661,804)]
    elif name=='relationship_only':
        # Sign reflection preserves symmetric marginals while reversing dependence.
        b[672:768]=-a[672:768];intervals=[(672,768)]
    elif name=='benign_modes':
        mode=(np.arange(n)//72)%2;a+=3*mode;b+=3*mode
    elif name=='missing_channel':b[680:752]=np.nan
    elif name=='repeated':
        for lo,hi in [(660,696),(804,840)]:a[lo:hi]+=4;b[lo:hi]+=4;intervals.append((lo,hi))
    else:raise ValueError(name)
    ids=np.sort(rng.choice(600,int(round(600*contamination)),replace=False))
    a[ids]+=4;b[ids]+=4
    observations=pd.concat([pd.DataFrame(dict(entity_id='machine-A',metric_name=c,event_time=t,available_time=t,value=v,quality=np.where(np.isfinite(v),'ok','missing'))) for c,v in [('a',a),('b',b)]],ignore_index=True)[CANONICAL]
    registry={c:Metric('unit',floor=.05) for c in ['a','b']};expected=schedule(['machine-A'],registry,t[0],t[-1]+pd.Timedelta(minutes=5))
    truth=pd.DataFrame([dict(fault_id=f'{name}-{i}',entity_id='machine-A',fault_type=name,onset_time=t[lo],observable_onset_time=t[lo],impact_time=t[min(hi-1,lo+12)],end_time=t[hi]) for i,(lo,hi) in enumerate(intervals)],columns=['fault_id','entity_id','fault_type','onset_time','observable_onset_time','impact_time','end_time'])
    for c in ['onset_time','observable_onset_time','impact_time','end_time']:truth[c]=pd.to_datetime(truth[c],utc=True)
    return validate(observations,expected,registry),expected,registry,truth,t[600],t[-1]+pd.Timedelta(minutes=5),ids


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    config=dict(features=dict(short=12,long=72,half_life_hours=1.,allowance=.25,boundary=2.),forest=dict(seed=42,stride=3,minimum_rows=100,minimum_hours=48),distance_lengths=[6])
    results=[];selection=[]
    cases=[(n,0.) for n in ['spike','subtle_shift','step','ramp','relationship_only','benign_modes','missing_channel','repeated']]+[('step',c) for c in [.005,.01,.02]]
    write_json(output/'manifest.json',dict(status='started',source=provenance(),cases=cases,config=config,policy='training q99, N_open=1, N_close=3; no fixture tuning',benchmark=False))
    for name,contamination in cases:
        data,expected,registry,truth,train_end,end,ids=scenario(name,contamination)
        model=fit(data,registry,config,train_end);scores,_=score(model,data)
        for route,part in scores.groupby('route'):
            _,policy=policy_grid(part.loc[part.decision_time<train_end],quantiles=(.99,),openings=(1,))[0]
            events,_,_=detect(part,policy)
            metrics,_,_=evaluate(events,truth,expected,train_end,end,lead_minutes=30)
            results.append(dict(scenario=name,contamination=contamination,route=route,**metrics,**coverage(part,train_end,end,expected)))
        selection.append(dict(scenario=name,contamination=contamination,contaminated_row_indices=ids.tolist()))
    table=pd.DataFrame(results);table.to_csv(output/'results.csv',index=False)
    write_json(output/'contamination_rows.json',selection)
    write_json(output/'COMPLETE.json',dict(status='complete',scenarios=len(cases),route_results=len(table),benchmark=False))
    return table


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
