"""Read-only plots, common-support comparisons and reviewable fault/error tables."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .evaluation import evaluate,coverage,match,timestamp_metrics,paired_block_intervals
from .incidents import detect


def report(output,model,scores,truth,expected,start,end,route_table,system_table,events,grouped,prefix):
    from .pipeline import write_json
    output=Path(output);directory=output/f'{prefix}_diagnostics';directory.mkdir(exist_ok=True)
    faults=truth.loc[truth.observable_onset_time.ge(start)&truth.observable_onset_time.lt(end)].copy()
    detection=faults[['fault_id','entity_id','fault_type','observable_onset_time','impact_time','end_time']].copy()
    selected=model['selected_routes']
    for route,cid in selected.items():
        new=events[cid].loc[events[cid].confirmation.ge(start)&events[cid].confirmation.lt(end)]
        pairs=match(new,faults).set_index('fault_id')
        detection[route]=detection.fault_id.isin(pairs.index)
        detection[f'{route}_delay']=detection.fault_id.map(pairs.delay_minutes)
    detection.to_csv(directory/'complementary_misses.csv',index=False)
    truth.loc[truth.onset_time.ge(start)&truth.onset_time.lt(end)&truth.observable_onset_time.isna()].to_csv(directory/'unknown_onset_faults.csv',index=False)
    truth.loc[truth.observable_onset_time.lt(start)&truth.end_time.gt(start)].to_csv(directory/'boundary_faults.csv',index=False)
    primary=grouped[model['primary']]
    metrics,pairs,early=evaluate(primary,truth,expected,start,end,model['config']['requirements']['lead_minutes'])
    pairs.to_csv(directory/'primary_matches.csv',index=False);early.to_csv(directory/'primary_early_matches.csv',index=False)
    unmatched=primary.loc[primary.confirmation.ge(start)&primary.confirmation.lt(end)&~primary.event_id.isin(pairs.event_id)]
    unmatched.to_csv(directory/'primary_unmatched.csv',index=False)
    primary.loc[primary.confirmation.lt(start)&(primary.end_time.isna()|pd.to_datetime(primary.end_time,utc=True).gt(start))].to_csv(directory/'ongoing_events.csv',index=False)
    # Common time support includes both channels and every fixed standalone comparator.
    chosen_scores=scores.loc[scores.route.isin(selected)].copy()
    chosen_scores['valid']=np.isfinite(chosen_scores.score)
    common=chosen_scores.groupby(['entity_id','decision_time']).valid.all().rename('common').reset_index()
    common.to_parquet(directory/'common_support.parquet',index=False)
    common_scores=chosen_scores.merge(common,on=['entity_id','decision_time'],validate='many_to_one')
    common_scores.loc[~common_scores.common,'score']=np.nan
    comparison=[];timestamp=[];matched_timestamp=[]
    for route,cid in selected.items():
        subset=scores.loc[scores.route.eq(route)]
        _,trace,_=detect(subset,model['policies'][cid])
        period=subset.loc[subset.decision_time.ge(start)&subset.decision_time.lt(end)]
        timestamp.append(timestamp_metrics(period,truth,model['policies'][cid],trace))
        matched=common_scores.loc[common_scores.route.eq(route)]
        e,tr,_=detect(matched,model['policies'][cid])
        m,_,_=evaluate(e,truth,expected,start,end,model['config']['requirements']['lead_minutes'])
        comparison.append(dict(candidate=cid,**m,**coverage(matched,start,end)))
        matched_timestamp.append(timestamp_metrics(matched.loc[matched.decision_time.ge(start)&matched.decision_time.lt(end)],truth,model['policies'][cid],tr))
    pd.DataFrame(comparison).to_csv(directory/'common_support_events.csv',index=False)
    pd.concat(timestamp).to_csv(directory/'timestamp_metrics.csv',index=False)
    pd.concat(matched_timestamp).to_csv(directory/'common_support_timestamp_metrics.csv',index=False)
    # Same monitored channels for marginal routes; the forest uses both jointly.
    # All variants use identical independent truth/opportunity sets.
    topology=pd.read_parquet(output/'topology.parquet');block_rows=[]
    for candidate,table in {**{c:events[c] for c in selected.values()},model['primary']:primary}.items():
        for block,part in topology.groupby('olt_id'):
            ids=set(part.entity_id)
            m,_,_=evaluate(table.loc[table.entity_id.isin(ids)],truth.loc[truth.entity_id.isin(ids)],expected.loc[expected.entity_id.isin(ids)],start,end,model['config']['requirements']['lead_minutes'])
            block_rows.append(dict(candidate=candidate,block=block,**m))
    block_counts=pd.DataFrame(block_rows);block_counts.to_csv(directory/'independent_block_counts.csv',index=False)
    write_json(directory/'uncertainty.json',paired_block_intervals(block_counts))
    # Complete-system operating curves, including failed policies.
    fig,ax=plt.subplots(figsize=(8,5))
    for route,part in route_table.assign(route=route_table.candidate.str.split(':').str[0]).groupby('route'):
        ax.plot(part.nuisance_per_1000_entity_days,part.recall,'o',label=route,alpha=.75)
    ax.axvline(model['config']['requirements']['workload'],color='black',ls='--',lw=1)
    ax.set(xlabel='Unmatched investigations / 1,000 scheduled entity-days',ylabel='Observable fault recall',ylim=(-.03,1.03),title=f'{prefix.title()} standalone route policies')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(directory/'operating_curves.png',dpi=160);plt.close(fig)
    plots=[];cases=[]
    if len(pairs):
        row=pairs.iloc[0];cases.append(('detected',row.entity_id,row.confirmation))
    misses=faults.loc[~faults.fault_id.isin(pairs.fault_id)]
    if len(misses):
        row=misses.iloc[0];cases.append(('missed',row.entity_id,row.observable_onset_time))
    if len(unmatched):
        row=unmatched.iloc[0];cases.append(('unmatched',row.entity_id,row.confirmation))
    else:
        # Inspect a rejected candidate's nuisance rather than manufacture a primary FP.
        for cid,table in events.items():
            new=table.loc[table.confirmation.ge(start)&table.confirmation.lt(end)]
            matched=match(new,faults)
            nuisance=new.loc[~new.event_id.isin(matched.event_id)]
            if len(nuisance):
                row=nuisance.iloc[0];cases.append((f'unmatched_{cid}',row.entity_id,row.confirmation));break
    plotted_routes=['u_level','u_shift','if_compact',next(r for r in selected if r.startswith('mp_level')),next(r for r in selected if r.startswith('mp_shape'))]
    for number,(label,entity,time) in enumerate(cases):
        left=max(start,time-pd.Timedelta(hours=4));right=min(end,time+pd.Timedelta(hours=12))
        fig,axes=plt.subplots(len(plotted_routes),1,figsize=(11,9),sharex=True)
        for ax,route in zip(axes,plotted_routes):
            part=scores.loc[scores.entity_id.eq(entity)&scores.route.eq(route)&scores.decision_time.ge(left)&scores.decision_time.le(right)]
            for scope,p in part.groupby('scope'):
                ax.plot(p.decision_time,p.score,lw=.8,label=scope)
                policy=model['policies'][selected[route]][(route,entity,scope)]
                ax.axhline(policy.high,ls='--',lw=.6,color='black')
            for f in truth.loc[truth.entity_id.eq(entity)].itertuples():
                if pd.notna(f.observable_onset_time) and f.end_time>left and f.observable_onset_time<right:
                    ax.axvspan(max(left,f.observable_onset_time),min(right,f.end_time),color='red',alpha=.09)
            ax.set_ylabel(route,fontsize=8);ax.grid(alpha=.15)
        axes[0].set_title(f'{prefix}: {label} · {entity} · shaded = known observable fault')
        axes[-1].set_xlabel('Decision availability time (UTC)');fig.tight_layout()
        name=f'case_{number}_{label.replace(":","_")}.png';fig.savefig(directory/name,dpi=150);plt.close(fig)
        plots.append(dict(file=name,label=label,entity_id=entity,anchor=time))
    # A highest-distance eligible query and its frozen normal neighbour.
    bank=next(b for b in model['banks'].values() if b.mode=='level')
    candidate=scores.loc[scores.route.eq(bank.route)&scores.decision_time.ge(start)&scores.decision_time.lt(end)&scores.score.notna()]
    if len(candidate):
        row=candidate.loc[candidate.score.idxmax()]
        from .pipeline import canonical
        data,_,_=canonical(output,end);residual=model['references'].score(data)
        channel=residual.loc[residual.entity_id.eq(row.entity_id)&residual.metric_name.eq(row.scope)]
        query=channel.loc[channel.event_time.ge(row.window_start)&channel.event_time.le(row.event_time)].z.to_numpy()
        b=bank.banks[row.entity_id,row.scope];i=np.flatnonzero(b['starts']==row.nearest_start.value)[0]
        reference=b['values'][i:i+bank.m]
        fig,ax=plt.subplots(figsize=(7,4));ax.plot(query,'o-',label='query residual');ax.plot(reference,'o-',label='nearest eligible normal')
        ax.set(title=f'{bank.route} · {row.entity_id} · RMS {row.score:.3f}',xlabel='Position within window',ylabel='Frozen residual units');ax.legend();fig.tight_layout();fig.savefig(directory/'nearest_normal_match.png',dpi=160);plt.close(fig)
    write_json(directory/'plot_manifest.json',dict(cases=plots,primary=model['primary'],interpretation='diagnostic association, not fault probability or cause'))
    write_json(directory/'summary.json',dict(primary=model['primary'],metrics=metrics,unknown_onsets=int(truth.loc[truth.onset_time.ge(start)&truth.onset_time.lt(end)].observable_onset_time.isna().sum()),
        comparison_scope='same two power channels, same fault opportunities; common-support replay provided',
        limitations=['Synthetic evidence only','Variance-fault observable onset unknown','One independent OLT block in bounded pilot','Previously explored seed 42; no independent generalisation claim']))
