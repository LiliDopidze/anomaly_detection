"""A finite, declared policy shortlist; no access to final-test labels."""
from dataclasses import asdict
import numpy as np
import pandas as pd
from .incidents import Policy,detect
from .evaluation import evaluate,coverage
from .grouping import group


def policy_grid(training,quantiles=(.99,.999),openings=(1,3),closing=3,stride=5):
    scope_scores={key:part.score[np.isfinite(part.score)].to_numpy() for key,part in training.groupby(['route','entity_id','scope'])}
    candidates=[];seen=set()
    for position in range(len(quantiles)+1):
        thresholds={}
        for key,x in scope_scores.items():
            if not len(x):
                # Unsupported streams must still abstain, not trigger on placeholders.
                high,low=1.,0.
            else:
                high=float(np.quantile(x,quantiles[position])) if position<len(quantiles) else float(x.max()+max(1e-6,.25*x.std()))
                low=min(float(np.quantile(x,.9)),high-max(1e-6,.1*abs(high)))
            thresholds[key]=(high,low)
        identity=tuple((key,*v) for key,v in sorted(thresholds.items()))
        if identity in seen:continue
        seen.add(identity)
        for opening in openings:
            policies={key:Policy(h,l,opening,closing,stride) for key,(h,l) in thresholds.items()}
            candidates.append((f'q{position}-n{opening}',policies))
    return candidates


def choose(table):
    eligible=table.loc[table.workload_coverage_feasible]
    source=eligible if len(eligible) else table
    best=source.recall.max()
    source=source.loc[source.recall.ge(best-.02)] if pd.notna(best) else source
    delay=source.median_delay_minutes.min()
    source=source.loc[source.median_delay_minutes.le(delay+5)] if pd.notna(delay) else source
    return source.sort_values(['complexity','window_cost','nuisance_per_1000_entity_days','candidate']).iloc[0].candidate


def record(candidate,events,scores,truth,expected,start,end,requirements,complexity=1,window_cost=0):
    metrics,ordinary,early=evaluate(events,truth,expected,start,end,requirements['lead_minutes'])
    availability=coverage(scores,start,end)
    feasible=metrics['nuisance_per_1000_entity_days']<=requirements['workload'] and availability['any_coverage']>=requirements['any_coverage'] and availability['full_coverage']>=requirements['full_coverage']
    accepted=feasible and metrics['recall']>=requirements['recall'] and metrics['early_recall']>=requirements['early_recall'] and metrics['median_delay_minutes']<=requirements['delay_minutes'] and metrics['time_in_alarm_fraction']<=requirements['alarm_fraction']
    return dict(candidate=candidate,**metrics,**availability,workload_coverage_feasible=bool(feasible),accepted=bool(accepted),complexity=complexity,window_cost=window_cost)


def select_routes(scores,truth,expected,train_end,dev_end,requirements,windows):
    if scores.decision_time.ge(dev_end).any() or truth.onset_time.ge(dev_end).any():
        raise ValueError('Selection input includes final period')
    rows=[];policies={};event_tables={};selected={}
    for route,part in scores.groupby('route',sort=False):
        train=part.loc[part.decision_time<train_end]
        route_rows=[]
        for suffix,policy in policy_grid(train):
            cid=f'{route}:{suffix}';events,_,_=detect(part,policy)
            result=record(cid,events,part,truth,expected,train_end,dev_end,requirements,window_cost=windows[route])
            rows.append(result);route_rows.append(result);policies[cid]=policy;event_tables[cid]=events
        selected[route]=choose(pd.DataFrame(route_rows))
    return pd.DataFrame(rows),policies,event_tables,selected


def select_system(scores,truth,expected,train_end,dev_end,requirements,windows,route_table,policies,event_tables,selected,compatibility):
    # Select profile and duration independently within their own route families first.
    forest=choose(route_table.loc[route_table.candidate.isin([v for k,v in selected.items() if k.startswith('if_')])])
    distance=choose(route_table.loc[route_table.candidate.isin([v for k,v in selected.items() if k.startswith('mp_')])])
    level,shift=selected['u_level'],selected['u_shift']
    combos={'level_shift':[level,shift], 'level_forest':[level,forest],
        'level_distance':[level,distance], 'forest_distance':[forest,distance],
        'level_forest_distance':[level,forest,distance]}
    # Standalone selected variants are also eligible for deployment and grouping.
    for name,cid in selected.items():combos[name]=[cid]
    rows=[];systems={};groups={};histories={}
    for name,components in combos.items():
        routes=[c.split(':')[0] for c in components]
        raw=pd.concat([event_tables[c] for c in components],ignore_index=True)
        subset=scores.loc[scores.route.isin(routes)]
        baseline=None
        for span in [0,30,120]:
            cid=f'{name}:g{span}'
            grouped,history=group(raw,span,compatibility)
            result=record(cid,grouped,subset,truth,expected,train_end,dev_end,requirements,len(routes),sum(windows[r] for r in routes))
            result.update(raw_events=int((raw.confirmation.ge(train_end)&raw.confirmation.lt(dev_end)).sum()),group_span_minutes=span)
            if baseline is None:baseline=result.copy()
            result['recall_loss']=baseline['recall']-result['recall']
            result['early_recall_loss']=baseline['early_recall']-result['early_recall']
            result['nuisance_reduction']=1-result['unmatched']/baseline['unmatched'] if baseline['unmatched'] else np.nan
            result['grouping_accepted']=span==0 or (result['recall_loss']<=.02 and result['early_recall_loss']<=.02 and result['nuisance_reduction']>=.5)
            result['workload_coverage_feasible'] &= result['grouping_accepted']
            result['accepted'] &= result['grouping_accepted']
            rows.append(result);systems[cid]=dict(components=components,routes=routes,span_minutes=span)
            groups[cid]=grouped;histories[cid]=history
    table=pd.DataFrame(rows);winner=choose(table)
    return table,systems,groups,histories,winner


def pareto(table):
    keep=[]
    for i,row in table.iterrows():
        dominates=(table.recall>=row.recall)&(table.nuisance_per_1000_entity_days<=row.nuisance_per_1000_entity_days)&(table.full_coverage>=row.full_coverage)
        strict=(table.recall>row.recall)|(table.nuisance_per_1000_entity_days<row.nuisance_per_1000_entity_days)|(table.full_coverage>row.full_coverage)
        if not (dominates&strict).any():keep.append(i)
    return table.loc[keep]
