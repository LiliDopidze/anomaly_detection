"""Directed frozen normal-bank profiles using STUMPY MASS (not DAMP).

Reference arrays retain their original grid and NaNs; no concatenation seams.
Exact full-overlap exclusion is applied to the entire profile before argmin.
"""
from dataclasses import dataclass, field
import hashlib
import numpy as np
import pandas as pd
import stumpy
from .detectors import SCORE_KEYS


def disjoint_count(starts,ends,indices):
    # Equal-length sorted intervals: skip directly to the next disjoint start.
    count=0;position=0;eligible_starts=starts[indices]
    while position<len(indices):
        i=indices[position];count+=1
        position=int(np.searchsorted(eligible_starts,ends[i],side='right'))
    return count


def oracle(query, windows, shape=False, floor=1e-8):
    """Small direct RMS oracle only; never the experiment scoring backend."""
    q=np.asarray(query,float);w=np.asarray(windows,float)
    if shape:
        if q.std()<=floor: return np.full(len(w),np.nan)
        valid=w.std(axis=1)>floor
        q=(q-q.mean())/q.std()
        w=np.divide(w-w.mean(axis=1,keepdims=True),w.std(axis=1,keepdims=True),out=np.full_like(w,np.nan),where=valid[:,None])
    return np.sqrt(np.mean((w-q)**2,axis=1))


@dataclass
class NormalBank:
    m: int
    mode: str
    floor: float = 1e-8
    banks: dict = field(default_factory=dict)

    def __post_init__(self):
        if type(self.m) is not int or self.m<3 or self.mode not in {"shape","level"} or self.floor<=0:
            raise ValueError("Declare m>=3, shape/level mode and positive floor")

    @property
    def route(self): return f"mp_{self.mode}_{self.m}"

    def fit(self,residuals,training_end):
        if self.banks: raise ValueError("Bank already frozen")
        self.training_end=pd.Timestamp(training_end)
        train=residuals.loc[(residuals.event_time<self.training_end)&(residuals.decision_time<self.training_end)]
        for key,part in train.groupby(["entity_id","metric_name"],sort=True):
            part=part.sort_values("event_time")
            if len(part)<2*self.m: continue
            if not np.all(np.diff(part.grid_index)==1): raise ValueError("Keep gaps as grid rows; do not concatenate segments")
            values=part["y" if self.mode=="shape" else "z"].to_numpy().copy()
            windows=np.lib.stride_tricks.sliding_window_view(values,self.m)
            valid=np.isfinite(windows).all(axis=1)
            means=np.full(len(windows),np.inf);sds=np.zeros(len(windows))
            means[valid]=windows[valid].mean(axis=1);sds[valid]=windows[valid].std(axis=1,ddof=0)
            segment=part.segment.to_numpy()
            valid &= (segment[:len(windows)]==segment[self.m-1:]) & (segment[self.m-1:]>=0)
            if self.mode=="shape": valid &= sds>self.floor
            time=part.event_time.astype('int64').to_numpy()
            fingerprint=hashlib.sha256(values.tobytes()+time.tobytes()+valid.tobytes()+f'{self.m}|{self.mode}|{self.floor}'.encode()).hexdigest()
            self.banks[key]=dict(values=values,means=means,sds=sds,valid=valid,
                starts=time[:len(windows)],ends=time[self.m-1:],
                rows=part.row_id.tolist(),segment=segment[:len(windows)],hash=fingerprint,
                cadence=int(np.median(np.diff(time))),total_windows=len(windows),eligible_windows=int(valid.sum()))
            self.banks[key]['disjoint_count']=disjoint_count(time[:len(windows)],time[self.m-1:],np.flatnonzero(valid))
        return self

    def query(self,key,query,start,end):
        b=self.banks.get(key)
        empty=dict(score=np.nan,reason="no_normal_bank",bank_hash=None,bank_support=0,disjoint_support=0,
                   overlap_excluded=0,nearest_start=pd.NaT,nearest_end=pd.NaT,nearest_segment=-1)
        if b is None: return empty
        empty['bank_hash']=b['hash']
        q=np.asarray(query,float)
        if len(q)!=self.m or not np.isfinite(q).all(): return {**empty,'reason':'incomplete_window'}
        if self.mode=="shape" and q.std(ddof=0)<=self.floor: return {**empty,'reason':'constant_shape_query'}
        a,e=pd.Timestamp(start).value,pd.Timestamp(end).value
        overlap=(b['starts']<=e)&(b['ends']>=a)
        eligible=b['valid']&~overlap
        indices=np.flatnonzero(eligible)
        # Earliest-finish interval scheduling gives maximum disjoint support.
        count=b['disjoint_count'] if not overlap.any() else disjoint_count(b['starts'],b['ends'],indices)
        info={**empty,'bank_support':len(indices),'disjoint_support':count,'overlap_excluded':int((overlap&b['valid']).sum())}
        if count<2: return {**info,'reason':'insufficient_disjoint_bank'}
        if self.mode=="shape":
            distances=stumpy.mass(q,b['values'],M_T=b['means'],Σ_T=b['sds'],
                T_subseq_isconstant=b['sds']<=self.floor,Q_subseq_isconstant=np.array([False]))
        else:
            distances=stumpy.mass(q,b['values'],normalize=False,T_subseq_isfinite=b['valid'])
        distances=np.asarray(distances,float)/np.sqrt(self.m)
        distances[~eligible]=np.inf
        i=int(np.argmin(distances))
        if not np.isfinite(distances[i]): return {**info,'reason':'nonfinite_distance'}
        return {**info,'score':float(distances[i]),'reason':'ok',
                'nearest_start':pd.Timestamp(b['starts'][i],tz='UTC'),
                'nearest_end':pd.Timestamp(b['ends'][i],tz='UTC'),'nearest_segment':int(b['segment'][i])}

    def score(self,residuals,stride=1):
        records=[]
        for key,part in residuals.groupby(["entity_id","metric_name"],sort=True):
            part=part.sort_values('event_time').reset_index(drop=True)
            values=part['y' if self.mode=='shape' else 'z'].to_numpy()
            times=part.event_time.tolist();segments=part.segment.to_numpy()
            cadence=pd.Timedelta(minutes=5) if len(part)<2 else times[1]-times[0]
            for i,row in enumerate(part.itertuples(index=False)):
                if row.grid_index%stride: continue
                start=times[i]-(self.m-1)*cadence
                query=values[max(0,i-self.m+1):i+1]
                if i<self.m-1 or segments[i]<0 or segments[max(0,i-self.m+1)]!=segments[i]: query=np.full(self.m,np.nan)
                detail=self.query(key,query,start,times[i])
                records.append({**{name:getattr(row,name) for name in SCORE_KEYS},'scope':key[1],
                    'route':self.route,'window_start':start,**detail})
        return pd.DataFrame(records)
