"""Independent local forests; no univariate gating or cross-route score fusion."""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

SCORE_KEYS=["entity_id", "event_time", "decision_time", "grid_index"]


def statistical(frame, route):
    if route not in {"u_level","u_shift"}: raise ValueError("Unknown statistical route")
    out=frame[SCORE_KEYS+["metric_name","window_start","reference_reason"]].copy()
    out=out.rename(columns={"metric_name":"scope","reference_reason":"reason"})
    out["route"],out["score"]=route,frame[route].to_numpy()
    out["window_start"]=out.event_time
    return out


@dataclass
class Forest:
    columns: list
    profile: str = "compact"
    seed: int = 42
    stride: int = 3
    floor: float = 1e-8
    minimum_rows: int = 100
    minimum_hours: float = 48
    models: dict = field(default_factory=dict)
    support: list = field(default_factory=list)

    def fit(self, vectors, training_end):
        if self.models: raise ValueError("Already fitted")
        if self.stride<1 or self.floor<=0 or self.minimum_rows<3: raise ValueError("Invalid forest support")
        self.training_end=pd.Timestamp(training_end)
        train=vectors.loc[(vectors.event_time<self.training_end)&(vectors.decision_time<self.training_end)]
        for entity, part in train.groupby("entity_id",sort=True):
            ok=np.isfinite(part[self.columns]).all(axis=1)&part.grid_index.mod(self.stride).eq(0)
            selected=part.loc[ok].sort_values("event_time")
            if len(selected)>20000:
                selected=selected.iloc[np.sort(np.random.default_rng(self.seed).choice(len(selected),20000,replace=False))]
            span=(selected.event_time.max()-selected.event_time.min()).total_seconds()/3600 if len(selected) else 0
            if len(selected)<self.minimum_rows or span<self.minimum_hours:
                self.support.append(dict(entity_id=entity,status="insufficient_local_support",rows=len(selected),hours=span));continue
            x=selected[self.columns].to_numpy();mean=x.mean(axis=0);sd=x.std(axis=0,ddof=0)
            if np.all(sd<=self.floor): raise ValueError("All-constant joint training")
            scale=np.maximum(sd,self.floor)
            model=IsolationForest(n_estimators=100,max_samples=min(256,len(x)),random_state=self.seed,n_jobs=1,contamination="auto")
            model.fit((x-mean)/scale)
            self.models[entity]=dict(estimator=model,mean=mean,scale=scale,rows=selected.row_id.tolist(),constant_columns=np.flatnonzero(sd<=self.floor).tolist())
            self.support.append(dict(entity_id=entity,status="fitted",rows=len(selected),hours=span))
        return self

    def score(self,vectors,scope,window_minutes):
        if list(vectors[self.columns].columns)!=self.columns: raise ValueError("Feature order mismatch")
        out=vectors[SCORE_KEYS].copy()
        out["route"],out["scope"]=f"if_{self.profile}",scope
        out["window_start"]=out.event_time-pd.Timedelta(minutes=window_minutes)
        out["score"],out["reason"]=np.nan,"no_local_reference"
        for entity,index in vectors.groupby("entity_id").groups.items():
            if entity not in self.models: continue
            model=self.models[entity];x=vectors.loc[index,self.columns].to_numpy()
            ok=np.isfinite(x).all(axis=1)
            out.loc[index,"reason"]="missing_channel_or_window"
            selected=np.asarray(index)[ok]
            if ok.any():
                out.loc[selected,"score"]=-model['estimator'].score_samples((x[ok]-model['mean'])/model['scale'])
                out.loc[selected,"reason"]="ok"
        return out
