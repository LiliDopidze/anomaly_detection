"""Independent strict-threshold state machines and immutable notifications."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

EVENT_COLUMNS=['event_id','entity_id','route','scope','confirmation','end_time','reason']


@dataclass(frozen=True)
class Policy:
    high: float
    low: float
    opening: int = 1
    closing: int = 3
    stride_minutes: float = 5
    gap_minutes: float = 360

    def __post_init__(self):
        if not np.isfinite([self.high,self.low]).all() or self.low>=self.high:
            raise ValueError('Require finite low < high')
        if type(self.opening) is not int or type(self.closing) is not int or min(self.opening,self.closing,self.stride_minutes,self.gap_minutes)<=0:
            raise ValueError('Positive integer confirmation counts and positive times required')


def detect(scores,policies):
    """Replay full chronological history; filtering evaluation periods happens later."""
    records,trace,notifications=[],[],[]
    for key,part in scores.groupby(['route','entity_id','scope'],sort=True):
        route,entity,scope=key
        policy=policies[(route,entity,scope)]
        part=part.sort_values(['decision_time','event_time'])
        if part.decision_time.duplicated().any(): raise ValueError('Duplicate decisions for one route scope')
        high_run=low_run=0;active=None;previous=last_valid=None;number=0
        for row in part.itertuples(index=False):
            t,s=row.decision_time,row.score
            if previous is not None and t-previous!=pd.Timedelta(minutes=policy.stride_minutes): high_run=low_run=0
            if active is not None and last_valid is not None and t-last_valid>pd.Timedelta(minutes=policy.gap_minutes):
                active['end_time'],active['reason']=t,'administrative_gap'
                notifications.append(dict(event_id=active['event_id'],time=t,action='administrative_close'))
                active=None;high_run=low_run=0
            previous=t
            if not np.isfinite(s): high_run=low_run=0
            else:
                last_valid=t
                if active is None:
                    high_run=high_run+1 if s>policy.high else 0
                    if high_run>=policy.opening:
                        number+=1
                        active=dict(event_id=f'{route}|{entity}|{scope}|{number}',entity_id=entity,route=route,scope=scope,
                                    confirmation=t,end_time=pd.NaT,reason='active')
                        records.append(active)
                        notifications.append(dict(event_id=active['event_id'],time=t,action='open'))
                        high_run=low_run=0
                else:
                    low_run=low_run+1 if s<policy.low else 0
                    if low_run>=policy.closing:
                        active['end_time'],active['reason']=t,'recovery'
                        notifications.append(dict(event_id=active['event_id'],time=t,action='recovery'))
                        active=None;high_run=low_run=0
            trace.append(dict(route=route,entity_id=entity,scope=scope,decision_time=t,
                active=active is not None,raw_above=bool(s>policy.high) if np.isfinite(s) else None,
                valid=bool(np.isfinite(s)),opening_run=high_run,closing_run=low_run))
    return pd.DataFrame(records,columns=EVENT_COLUMNS),pd.DataFrame(trace),pd.DataFrame(notifications,columns=['event_id','time','action'])
