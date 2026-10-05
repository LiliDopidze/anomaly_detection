"""Schedule denominators, separate ordinary/early matching and block uncertainty."""
import numpy as np
import pandas as pd


def ratio(a,b): return float(a/b) if b else np.nan


def union_seconds(intervals):
    ordered=sorted((a,b) for a,b in intervals if pd.notna(a) and pd.notna(b) and b>a)
    seconds=0.;right=None
    for a,b in ordered:
        left=a if right is None else max(a,right)
        if b>left: seconds+=(b-left).total_seconds()
        right=b if right is None else max(right,b)
    return seconds


def intersect_seconds(active, scheduled):
    """Intersect unions without counting off-schedule time or duplicate channels."""
    def merge(intervals):
        out=[]
        for a,b in sorted((a,b) for a,b in intervals if pd.notna(a) and pd.notna(b) and b>a):
            if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(b,out[-1][1]))
            else:out.append((a,b))
        return out
    a,b=merge(active),merge(scheduled);i=j=0;seconds=0.
    while i<len(a) and j<len(b):
        left,right=max(a[i][0],b[j][0]),min(a[i][1],b[j][1])
        if right>left:seconds+=(right-left).total_seconds()
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return seconds


def exposure(expected,start,end):
    total=0.
    for _,part in expected.groupby('entity_id'):
        total+=union_seconds((max(a,start),min(b,end)) for a,b in zip(part.event_time,part.interval_end))
    return total/86400


def match(events,faults,lead_minutes=None):
    """Maximum cardinality with earlier confirmations then stable IDs preferred.

    Augmenting paths retain every earlier matched alert while adding later ones;
    unlike interval-greedy matching, reassignment resolves overlapping faults.
    """
    ev=events.sort_values(['confirmation','event_id']).reset_index(drop=True)
    ft=faults.sort_values('fault_id').reset_index(drop=True)
    adjacency=[]
    for e in ev.itertuples():
        choices=[]
        for j,f in enumerate(ft.itertuples()):
            ok=e.entity_id==f.entity_id and pd.notna(f.observable_onset_time) and f.observable_onset_time<=e.confirmation<f.end_time
            if lead_minutes is not None:
                ok=ok and pd.notna(f.impact_time) and e.confirmation<=f.impact_time-pd.Timedelta(minutes=lead_minutes)
            if ok: choices.append(j)
        adjacency.append(choices)
    owner={}
    def augment(i,seen):
        for j in adjacency[i]:
            if j in seen: continue
            seen.add(j)
            if j not in owner or augment(owner[j],seen): owner[j]=i;return True
        return False
    for i in range(len(ev)): augment(i,set())
    rows=[]
    for j,i in sorted(owner.items(),key=lambda pair:pair[1]):
        e,f=ev.iloc[i],ft.iloc[j]
        rows.append(dict(event_id=e.event_id,fault_id=f.fault_id,entity_id=e.entity_id,
            confirmation=e.confirmation,delay_minutes=(e.confirmation-f.observable_onset_time).total_seconds()/60,
            lead_minutes=(f.impact_time-e.confirmation).total_seconds()/60 if pd.notna(f.impact_time) else np.nan,
            recovery_delay_minutes=(e.end_time-f.end_time).total_seconds()/60 if pd.notna(e.end_time) else np.nan))
    return pd.DataFrame(rows,columns=['event_id','fault_id','entity_id','confirmation','delay_minutes','lead_minutes','recovery_delay_minutes'])


def evaluate(events,truth,expected,start,end,lead_minutes=30,complete_truth=True):
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    events=events.copy()
    truth=truth.copy()
    for column in ['onset_time','observable_onset_time','impact_time','end_time']:
        truth[column]=pd.to_datetime(truth[column],utc=True)
    for column in ['confirmation','end_time']:
        events[column]=pd.to_datetime(events[column],utc=True)
    if events.event_id.duplicated().any() or truth.fault_id.duplicated().any():
        raise ValueError('Event and fault IDs must be unique')
    # An assessment cannot use recoveries observed after its availability horizon.
    events.loc[events.end_time.ge(end), 'end_time'] = pd.NaT
    new=events.loc[events.confirmation.ge(start)&events.confirmation.lt(end)]
    known=truth.observable_onset_time.notna()
    faults=truth.loc[known&truth.observable_onset_time.ge(start)&truth.observable_onset_time.lt(end)]
    physical=truth.loc[truth.onset_time.ge(start)&truth.onset_time.lt(end)]
    unknown=physical.loc[physical.observable_onset_time.isna()]
    cohort=pd.concat([faults,unknown],ignore_index=True)
    impact=cohort.loc[cohort.impact_time.notna()]
    opportunity=impact.loc[impact.observable_onset_time.notna()&(impact.impact_time-impact.observable_onset_time).ge(pd.Timedelta(minutes=lead_minutes))]
    ordinary=match(new,faults);early=match(new,impact,lead_minutes);early_opportunity=match(new,opportunity,lead_minutes)
    tp=len(ordinary);unmatched=len(new)-tp;fn=len(faults)-tp
    days=exposure(expected,start,end)
    active_seconds=0.
    for entity,part in events.groupby('entity_id'):
        monitoring=expected.loc[expected.entity_id.eq(entity)]
        active_seconds+=intersect_seconds(
            ((max(a,start),min(b if pd.notna(b) else end,end)) for a,b in zip(part.confirmation,part.end_time)),
            zip(monitoring.event_time,monitoring.interval_end))
    unmatched_rows=new.loc[~new.event_id.isin(ordinary.event_id)]
    unknown_intervals=truth.loc[~known&truth.onset_time.lt(end)&truth.end_time.gt(start)]
    unknown_workload=sum(any(e.entity_id==f.entity_id and f.onset_time<=e.confirmation<f.end_time for f in unknown_intervals.itertuples()) for e in unmatched_rows.itertuples())
    boundary=truth.loc[known&truth.observable_onset_time.lt(start)&truth.end_time.gt(start)]
    ongoing=events.loc[events.confirmation.lt(start)&(events.end_time.isna()|events.end_time.gt(start))]
    merge_risk=0;associations={}
    for e in new.itertuples():
        candidates=faults.loc[faults.entity_id.eq(e.entity_id)&faults.observable_onset_time.le(e.confirmation)&faults.end_time.gt(e.confirmation)]
        merge_risk+=len(candidates)>1
        for f in candidates.fault_id: associations[f]=associations.get(f,0)+1
    result=dict(tp=tp,fp=unmatched-unknown_workload if complete_truth else np.nan,fn=fn,
        faults=len(faults),physical_faults=len(physical),unknown_onsets=len(unknown),
        precision=ratio(tp,len(new)),recall=ratio(tp,len(faults)),f1=ratio(2*tp,2*tp+unmatched+fn),
        investigations=len(new),unmatched=unmatched,unverified=unknown_workload if complete_truth else unmatched,
        scheduled_entity_days=days,nuisance_per_1000_entity_days=1000*ratio(unmatched,days),
        investigations_per_1000_entity_days=1000*ratio(len(new),days),
        precision_interpretation='matched_fraction_all_investigations',
        known_impact_faults=len(impact),unknown_impact_faults=int(cohort.impact_time.isna().sum()),
        early_tp=len(early),early_recall=ratio(len(early),len(impact)),
        opportunity_faults=len(opportunity),opportunity_early_tp=len(early_opportunity),opportunity_early_recall=ratio(len(early_opportunity),len(opportunity)),
        median_delay_minutes=float(ordinary.delay_minutes.median()),delay_sample=len(ordinary),
        median_lead_minutes=float(ordinary.lead_minutes.median()),lead_sample=int(ordinary.lead_minutes.notna().sum()),
        median_recovery_delay_minutes=float(ordinary.recovery_delay_minutes.median()),recovery_sample=int(ordinary.recovery_delay_minutes.notna().sum()),
        right_censored_faults=int(faults.end_time.ge(end).sum()),
        recovery_censored_matches=int(ordinary.recovery_delay_minutes.isna().sum()),
        time_in_alarm_hours=active_seconds/3600,time_in_alarm_fraction=ratio(active_seconds,days*86400),
        administrative_closures=int((events.reason.str.contains('administrative')&events.end_time.ge(start)&events.end_time.lt(end)).sum()),
        boundary_faults=len(boundary),ongoing_events=len(ongoing),merge_risk=merge_risk,
        fragmented_faults=sum(n>1 for n in associations.values()),truth_complete=complete_truth)
    return result,ordinary,early


def coverage(scores,start,end,expected=None):
    """Availability on an independently declared schedule, including absent rows.

    ``expected`` may be a canonical channel schedule or an explicit route/scope
    decision schedule. Omitting it is only suitable for complete fixture streams.
    The batch runner requires a common cadence for every joint channel group.
    """
    keys=['route','entity_id','scope','decision_time']
    if scores.duplicated(keys).any():
        raise ValueError('Duplicate score decisions')
    subset=scores.loc[scores.decision_time.ge(start)&scores.decision_time.lt(end)].copy()
    if expected is not None:
        grid=expected.loc[expected.decision_time.ge(start)&expected.decision_time.lt(end)]
        if {'route','scope'}.issubset(grid.columns):
            declared=grid[keys].drop_duplicates()
        else:
            streams=[]
            for route,scope in scores[['route','scope']].drop_duplicates().itertuples(index=False,name=None):
                channels=scope.split('+')
                part=grid.loc[grid.metric_name.isin(channels)]
                count=part.groupby(['entity_id','decision_time']).metric_name.nunique()
                times=count.loc[count.eq(len(channels))].reset_index()[['entity_id','decision_time']]
                streams.append(times.assign(route=route,scope=scope))
            declared=pd.concat(streams,ignore_index=True)[keys] if streams else pd.DataFrame(columns=keys)
        check=subset[keys].merge(declared,on=keys,how='left',indicator=True)
        if check._merge.ne('both').any():
            raise ValueError('Score outside the declared decision schedule')
        subset=declared.merge(subset[keys+['score']],on=keys,how='left',validate='one_to_one')
    subset['valid']=np.isfinite(subset.score)
    # All candidate streams emit a row for each scheduled decision, including abstentions.
    groups=subset.groupby(['entity_id','decision_time']).valid
    any_valid,all_valid=groups.any(),groups.all()
    longest=0.;invalid_decisions=0
    for _,part in subset.groupby(['route','entity_id','scope']):
        part=part.sort_values('decision_time');gap_start=None;times=part.decision_time.tolist()
        stride=(times[1]-times[0]).total_seconds()/60 if len(times)>1 else 0
        for t,v in zip(times,part.valid):
            if not v:
                invalid_decisions+=1
                if gap_start is None:gap_start=t
                longest=max(longest,(t-gap_start).total_seconds()/60+stride)
            else:gap_start=None
    return dict(any_coverage=float(any_valid.mean()),full_coverage=float(all_valid.mean()),
        schedule_decisions=len(any_valid),valid_any_decisions=int(any_valid.sum()),valid_full_decisions=int(all_valid.sum()),
        longest_score_gap_minutes=longest,invalid_stream_decisions=invalid_decisions)


def grouping_audit(events,history,truth,start,end):
    """Inspect every member confirmation, including faults after the group anchor.

    This is diagnostic association, independent of one-to-one detection credit.
    """
    membership=history.loc[history.action.isin(['open','attach'])]
    rows=[]
    for member in membership.itertuples(index=False):
        event=events.loc[events.event_id.eq(member.event_id)].iloc[0]
        if not start<=event.confirmation<end:
            continue
        eligible=truth.loc[truth.entity_id.eq(event.entity_id)&
            truth.observable_onset_time.le(event.confirmation)&truth.end_time.gt(event.confirmation)]
        for fault in eligible.fault_id:
            rows.append(dict(group_id=member.group_id,event_id=member.event_id,
                             fault_id=fault,confirmation=event.confirmation))
    associations=pd.DataFrame(rows,columns=['group_id','event_id','fault_id','confirmation'])
    return dict(merge_risk_groups=int((associations.groupby('group_id').fault_id.nunique()>1).sum()),
                fragmented_faults=int((associations.groupby('fault_id').group_id.nunique()>1).sum())),associations


def timestamp_metrics(scores,truth,policies,trace):
    """Known truth AND valid scores only; raw decisions and active states separated."""
    rows=[]
    for (route,entity,scope),part in scores.groupby(['route','entity_id','scope']):
        truth_e=truth.loc[truth.entity_id.eq(entity)]
        positive=np.zeros(len(part),bool);unknown=np.zeros(len(part),bool)
        for f in truth_e.itertuples():
            if pd.isna(f.observable_onset_time): unknown|=(part.decision_time.ge(f.onset_time)&part.decision_time.lt(f.end_time)).to_numpy()
            else:positive|=(part.decision_time.ge(f.observable_onset_time)&part.decision_time.lt(f.end_time)).to_numpy()
        valid=np.isfinite(part.score).to_numpy();use=valid&~unknown
        predicted=(part.score>policies[(route,entity,scope)].high).to_numpy()
        state=trace.loc[trace.route.eq(route)&trace.entity_id.eq(entity)&trace.scope.eq(scope)].set_index('decision_time').active.reindex(part.decision_time).to_numpy()
        for label,pred in [('raw_threshold',predicted),('active_event',state)]:
            rows.append(dict(route=route,entity_id=entity,scope=scope,kind=label,
                tp=int((use&positive&pred).sum()),fp=int((use&~positive&pred).sum()),fn=int((use&positive&~pred).sum()),tn=int((use&~positive&~pred).sum()),
                excluded_unknown_truth=int(unknown.sum()),excluded_invalid_score=int((~valid).sum()),eligible=int(use.sum())))
    return pd.DataFrame(rows)


def paired_block_intervals(counts,metric='recall',draws=1000,seed=42):
    """Counts by candidate and independent block; same resample for every candidate."""
    blocks=sorted(counts.block.unique())
    if len(blocks)<2:return {'status':'unavailable','reason':'fewer than two independent blocks','blocks':len(blocks)}
    numerator,denominator=('tp','faults') if metric=='recall' else ('unmatched','scheduled_entity_days')
    candidates=sorted(counts.candidate.unique());rng=np.random.default_rng(seed)
    values={c:[] for c in candidates}
    for _ in range(draws):
        sampled=rng.integers(0,len(blocks),len(blocks))
        for c in candidates:
            x=counts.loc[counts.candidate.eq(c)].set_index('block').reindex(blocks).iloc[sampled]
            values[c].append(ratio(x[numerator].sum(),x[denominator].sum()))
    baseline=candidates[0]
    return {'status':'available','blocks':len(blocks),'baseline':baseline,'metric':metric,
        'intervals':{c:np.nanquantile(v,[.025,.975]).tolist() for c,v in values.items()},
        'paired_difference':{c:np.nanquantile(np.array(v)-values[baseline],[.025,.975]).tolist() for c,v in values.items()}}
