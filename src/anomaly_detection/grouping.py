"""Causal within-entity grouping anchored at first confirmation."""
import pandas as pd
from .incidents import EVENT_COLUMNS


def group(events,span_minutes,compatibility):
    if span_minutes<0: raise ValueError('Nonnegative span required')
    if events.event_id.duplicated().any(): raise ValueError('Duplicate event IDs')
    if span_minutes==0:
        history=pd.DataFrame([dict(group_id=r.event_id,event_id=r.event_id,time=r.confirmation,action='open') for r in events.itertuples()],columns=['group_id','event_id','time','action'])
        return events.copy(),history
    actions=[]
    for r in events.itertuples(index=False):
        actions.append((r.confirmation,1,r.event_id,'open',r))
        if pd.notna(r.end_time): actions.append((r.end_time,0,r.event_id,'close',r))
    actions.sort(key=lambda a:a[:3])
    groups=[];membership={};history=[]
    for time,_,eid,action,row in actions:
        if action=='open':
            symptom=compatibility.get(row.scope,row.scope)
            eligible=[g for g in groups if g['entity_id']==row.entity_id and g['symptom']==symptom and g['live'] and time-g['confirmation']<=pd.Timedelta(minutes=span_minutes)]
            chosen=min(eligible,key=lambda g:(g['confirmation'],g['event_id'])) if eligible else None
            first=chosen is None
            if first:
                chosen=dict(event_id=f'G|{eid}',entity_id=row.entity_id,route='grouped',scope=symptom,confirmation=time,end_time=pd.NaT,reason='active',symptom=symptom,live=set(),administrative=False)
                groups.append(chosen)
            chosen['live'].add(eid);membership[eid]=chosen
            history.append(dict(group_id=chosen['event_id'],event_id=eid,time=time,action='open' if first else 'attach'))
        else:
            chosen=membership[eid];chosen['live'].remove(eid)
            chosen['administrative']|=row.reason!='recovery'
            history.append(dict(group_id=chosen['event_id'],event_id=eid,time=time,action='member_'+row.reason))
            if not chosen['live']:
                chosen['end_time']=time
                chosen['reason']='administrative_or_mixed' if chosen['administrative'] else 'recovery'
                history.append(dict(group_id=chosen['event_id'],event_id=eid,time=time,action=chosen['reason']))
    return pd.DataFrame(groups,columns=EVENT_COLUMNS),pd.DataFrame(history,columns=['group_id','event_id','time','action'])
