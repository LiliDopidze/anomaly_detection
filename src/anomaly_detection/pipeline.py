"""Small reproducible batch workflow with explicit development/final entry points."""
from pathlib import Path
from dataclasses import asdict
import argparse
import ast
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
import traceback
import shutil
import joblib
import numpy as np
import pandas as pd
import yaml
from optical_anomaly.generator import GeneratorConfig,generate,make_topology
from optical_anomaly.optics import validate_generated
from .adapters import gpon,gpon_registry
from .validation import schedule,validate
from .references import References
from .features import features,joint_vectors
from .detectors import Forest,statistical
from .distance_profiles import NormalBank
from .selection import select_routes,select_system,pareto,record
from .incidents import detect
from .grouping import group
from .evaluation import evaluate,coverage,timestamp_metrics,paired_block_intervals


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_json(value):
    if isinstance(value,dict):return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean_json(v) for v in value]
    if isinstance(value,np.ndarray):return clean_json(value.tolist())
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,np.bool_):return bool(value)
    if isinstance(value,(pd.Timestamp,Path)):return str(value)
    return value


def write_json(path,value): Path(path).write_text(json.dumps(clean_json(value),indent=2,allow_nan=False)+'\n')


def provenance():
    root=Path(__file__).resolve().parents[2]
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    dirty=subprocess.check_output(['git','status','--short'],cwd=root,text=True).strip()
    return dict(source_sha=sha,working_tree_status=dirty,python=platform.python_version(),
        dependencies={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','joblib','pyarrow','stumpy']},
        source_hashes={str(p.relative_to(root)):digest(p) for p in sorted((root/'src').rglob('*.py'))})


def bounds(config):
    start=pd.Timestamp('2025-01-01',tz='UTC');duration=pd.Timedelta(days=config['generator']['days'])
    return start,*[start+duration*x for x in config['split_shares']]


def prepare(config_path,output):
    """Generate once, immutable config/input files; resume only identical configuration."""
    config=yaml.safe_load(Path(config_path).read_text());output=Path(output)
    if config['version']!=4 or config['split_shares']!=[.55,.75,1.]:
        raise ValueError('This runner implements the explicitly synthetic 55/20/25 experiment')
    if config['generator']['interval_minutes']!=5:
        raise ValueError('Runner candidate policies require five-minute cadence; reselect for another cadence')
    config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if output.exists():
        saved=json.loads((output/'input_manifest.json').read_text())
        if saved['config_hash']!=config_hash:raise ValueError('Configuration changed; use a new immutable run directory')
        for name,h in saved['files'].items():
            if digest(output/name)!=h:raise ValueError(f'Input changed: {name}')
        return output,config
    output.mkdir(parents=True)
    try:
        native,truth=generate(GeneratorConfig(**config['generator']))
        topology=make_topology(GeneratorConfig(**config['generator']))
        checks=validate_generated(native,truth,topology)
        if not all(checks['checks'].values()):raise ValueError('Generator structural checks failed')
        start,train_end,dev_end,end=bounds(config)
        if truth.onset_time.lt(train_end).any():raise ValueError('Training contains injected faults')
        registry=gpon_registry();expected=schedule(topology.entity_id,registry,start,end)
        (output/'config.yaml').write_text(Path(config_path).read_text())
        for name,frame in [('native',native),('truth',truth),('topology',topology),('schedule',expected)]:
            frame.to_parquet(output/f'{name}.parquet',index=False)
        write_json(output/'generation_checks.json',checks)
        write_json(output/'input_manifest.json',dict(status='prepared',config_hash=config_hash,
            source=provenance(),files={f'{name}.parquet':digest(output/f'{name}.parquet') for name in ['native','truth','topology','schedule']},
            boundaries=dict(start=start,train_end=train_end,development_end=dev_end,end=end),
            registry={name:asdict(spec) for name,spec in registry.items()},
            input_rows=len(native),truth_rows=len(truth),test_status=config['test_status'],
            healthy_training_verified=True,truth_access='structural generation and healthy-training gate only'))
    except Exception:
        write_json(output/'FAILED_prepare.json',dict(error=traceback.format_exc(),status='failed'));raise
    return output,config


def canonical(output,end):
    native=pd.read_parquet(output/'native.parquet',filters=[('time','<',end)])
    expected=pd.read_parquet(output/'schedule.parquet',filters=[('event_time','<',end)])
    registry=gpon_registry()
    observations=gpon(native)
    observations=observations.loc[observations.metric_name.isin(registry)].reset_index(drop=True)
    return validate(observations,expected,registry),expected,registry


def fit(data,registry,config,train_end):
    references=References(registry).fit(data,train_end)
    residuals=references.score(data)
    engineered=features(residuals,registry,**config['features'])
    channels=list(registry)
    forests={}
    for profile in ['compact','historical8']:
        vectors,columns=joint_vectors(engineered,channels,profile)
        forests[profile]=Forest(columns,profile=profile,**config['forest']).fit(vectors,train_end)
    for entity in forests['compact'].models:
        if forests['compact'].models[entity]['rows']!=forests['historical8'].models[entity]['rows']:
            raise ValueError('Feature ablations have different training row identities')
    banks={}
    for m in config['distance_lengths']:
        for mode in ['shape','level']:
            bank=NormalBank(m,mode).fit(residuals,train_end);banks[bank.route]=bank
    return dict(references=references,forests=forests,banks=banks,channels=channels,config=config,train_end=train_end)


def score(model,data,routes=None,cache=None):
    """All fitted objects are frozen. Optional route cache is immutable and manifest-bound."""
    config=model['config'];registry=model['references'].registry
    residuals=model['references'].score(data)
    engineered=features(residuals,registry,**config['features'])
    all_routes=['u_level','u_shift','if_compact','if_historical8',*model['banks']]
    routes=all_routes if routes is None else routes
    if set(routes)-set(all_routes):raise ValueError('Unknown route')
    frames=[];cost=[]
    for route in routes:
        path=Path(cache)/f'scores_{route}.parquet' if cache else None
        if path and path.exists():
            frames.append(pd.read_parquet(path));continue
        started=time.perf_counter()
        if route.startswith('u_'):frame=statistical(engineered,route)
        elif route.startswith('if_'):
            profile=route.removeprefix('if_')
            v,_=joint_vectors(engineered,model['channels'],profile)
            frame=model['forests'][profile].score(v,'+'.join(model['channels']),(config['features']['short']-1)*5)
        else:frame=model['banks'][route].score(residuals)
        frame['model_version']='4.0.0'
        frame['decision_kind']='original_fixed_deadline'
        elapsed=time.perf_counter()-started
        record_cost=dict(route=route,seconds=elapsed,scheduled_scores=len(frame),valid_scores=int(frame.score.notna().sum()),
            frame_bytes=int(frame.memory_usage(deep=True).sum()))
        cost.append(record_cost)
        if path:
            frame.to_parquet(path,index=False)
            write_json(path.with_suffix('.cost.json'),record_cost)
        print(f'{route}: {len(frame)} decisions in {elapsed:.2f}s',flush=True)
        frames.append(frame)
    return pd.concat(frames,ignore_index=True),pd.DataFrame(cost)


def save(model,path): joblib.dump(model,path)
def load(path): return joblib.load(path)  # Only load trusted artifacts.


def reuse_development_cache(previous,config_path,output,source_revision):
    """Copy verified immutable inputs/scores after an evaluation-only correction.

    Never alter the source run. Numerical fitting/scoring sources and function
    bodies must agree; policies and diagnostics are always recomputed. A caller
    supplies the actual committed source snapshot, checked against saved hashes.
    """
    previous,output=Path(previous),Path(output)
    if output.exists():raise FileExistsError(output)
    if not (previous/'DEVELOPMENT_COMPLETE.json').exists():
        raise ValueError('Only a completed development run can supply a cache')
    if (previous/'FINAL_OPENED.json').exists():
        raise ValueError('Cannot use an assessed run to retune the same claim')
    config=yaml.safe_load(Path(config_path).read_text())
    inputs=json.loads((previous/'input_manifest.json').read_text())
    if hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()!=inputs['config_hash']:
        raise ValueError('Different configuration; do not reuse scores')
    frozen=json.loads((previous/'FROZEN.json').read_text())
    fitted=json.loads((previous/'fit_manifest.json').read_text())
    for name,h in {**inputs['files'],**frozen['files'],'fitted.joblib':fitted['fitted_hash']}.items():
        if digest(previous/name)!=h:raise ValueError(f'Prior artifact changed: {name}')
    current=provenance();old=frozen['source']['source_hashes']
    evaluation_only={'src/anomaly_detection/evaluation.py','src/anomaly_detection/selection.py',
                     'src/anomaly_detection/diagnostics.py','src/anomaly_detection/fixtures.py',
                     'src/anomaly_detection/pipeline.py'}
    for name,h in old.items():
        if name not in evaluation_only and current['source_hashes'].get(name)!=h:
            raise ValueError(f'Numerical source changed: {name}')
    root=Path(__file__).resolve().parents[2]
    code=subprocess.check_output(['git','show',f'{source_revision}:src/anomaly_detection/pipeline.py'],cwd=root,text=True)
    if hashlib.sha256(code.encode()).hexdigest()!=old['src/anomaly_detection/pipeline.py']:
        raise ValueError('Revision does not match the recorded runner')
    def bodies(source):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    before,after=bodies(code),bodies(Path(__file__).read_text())
    for name in ['prepare','canonical','fit','score','bounds','save','load']:
        if before[name]!=after[name]:raise ValueError(f'Numerical runner changed: {name}')
    if frozen['source']['dependencies']!=current['dependencies']:
        raise ValueError('Dependency versions changed')
    names=['config.yaml','input_manifest.json','generation_checks.json','native.parquet','truth.parquet',
           'topology.parquet','schedule.parquet','fitted.joblib']
    names += [p.name for p in previous.glob('scores_*') if p.suffix in {'.json','.parquet'}]
    output.mkdir(parents=True)
    for name in names:shutil.copy2(previous/name,output/name)
    write_json(output/'cache_lineage.json',dict(previous=str(previous),source_revision=source_revision,
        previous_frozen_hash=digest(previous/'FROZEN.json'),reason='evaluation correctness audit; no detector changes',
        files={name:digest(output/name) for name in names},original_fit_manifest=fitted,
        numerical_functions_verified=['prepare','canonical','fit','score','bounds','save','load']))
    fitted.update(current)
    write_json(output/'fit_manifest.json',fitted)
    return output


def develop(config_path,output):
    output,config=prepare(config_path,output)
    if (output/'DEVELOPMENT_COMPLETE.json').exists():
        print(f'Reusing frozen development result: {output}',flush=True);return output
    if (output/'FAILED_development.json').exists():
        raise ValueError('Failed run retained; review the failure and use a new run directory')
    try:
        start,train_end,dev_end,end=bounds(config)
        data,expected,registry=canonical(output,dev_end)
        truth=pd.read_parquet(output/'truth.parquet',filters=[('onset_time','<',dev_end)])
        write_json(output/'development_access.json',dict(status='test_closed',max_event_time=data.event_time.max(),max_decision_time=data.decision_time.max(),source=provenance()))
        data.to_parquet(output/'canonical_development.parquet',index=False)
        if (output/'fitted.joblib').exists():
            manifest=json.loads((output/'fit_manifest.json').read_text())
            if manifest['source_hashes']!=provenance()['source_hashes']:raise ValueError('Code changed since cached fit')
            if manifest['fitted_hash']!=digest(output/'fitted.joblib'):raise ValueError('Cached model changed')
            model=load(output/'fitted.joblib')
        else:
            model=fit(data,registry,config,train_end);save(model,output/'fitted.joblib')
            write_json(output/'fit_manifest.json',dict(**provenance(),fitted_hash=digest(output/'fitted.joblib'),
                training_rows={p:{e:m['rows'] for e,m in f.models.items()} for p,f in model['forests'].items()},
                feature_order={p:f.columns for p,f in model['forests'].items()},
                reference_rows={str(k):v['rows'] for k,v in model['references'].models.items()},
                banks={r:{str(k):{name:v[name] for name in ['hash','total_windows','eligible_windows','disjoint_count','rows']} for k,v in b.banks.items()} for r,b in model['banks'].items()}))
        pd.DataFrame(model['references'].diagnostics).to_csv(output/'reference_diagnostics.csv',index=False)
        pd.concat([pd.DataFrame(f.support).assign(profile=p) for p,f in model['forests'].items()]).to_csv(output/'forest_support.csv',index=False)
        scores,cost=score(model,data,cache=output)
        costs=pd.DataFrame([json.loads(p.read_text()) for p in output.glob('scores_*.cost.json')]);costs.to_csv(output/'costs.csv',index=False)
        windows={r:0 if r.startswith('u_') else config['features']['short'] if r.startswith('if_') else int(r.rsplit('_',1)[1]) for r in scores.route.unique()}
        rt,policies,events,selected=select_routes(scores,truth,expected,train_end,dev_end,config['requirements'],windows)
        rt.to_csv(output/'route_trials.csv',index=False)
        # This table is written before combinations are considered.
        rt.loc[rt.candidate.isin(selected.values())].to_csv(output/'separate_routes.csv',index=False)
        compatibility={**{c:'optical_power' for c in model['channels']},'+'.join(model['channels']):'optical_power'}
        st,systems,grouped,histories,winner=select_system(scores,truth,expected,train_end,dev_end,config['requirements'],windows,rt,policies,events,selected,compatibility)
        st.to_csv(output/'system_trials.csv',index=False);pareto(st).to_csv(output/'pareto.csv',index=False)
        model.update(policies=policies,selected_routes=selected,systems=systems,primary=winner,compatibility=compatibility,windows=windows)
        save(model,output/'model.joblib')
        for cid,table in events.items():table.to_parquet(output/f'events_{cid}.parquet',index=False)
        for cid,table in grouped.items():
            table.to_parquet(output/f'groups_{cid}.parquet',index=False)
            histories[cid].to_parquet(output/f'membership_{cid}.parquet',index=False)
        winner_row=st.set_index('candidate').loc[winner]
        write_json(output/'FROZEN.json',dict(status='frozen',source=provenance(),primary=winner,
            deployment_enabled=bool(winner_row.accepted),decision='candidate_for_shadow_pilot' if winner_row.accepted else 'no_feasible_deployment',
            component_selection=selected,systems=systems,
            policies={c:[dict(route=k[0],entity_id=k[1],scope=k[2],**asdict(p)) for k,p in policy.items()] for c,policy in policies.items()},
            files={name:digest(output/name) for name in ['model.joblib','config.yaml','route_trials.csv','system_trials.csv','fit_manifest.json']},
            test_opened=False,test_status=config['test_status']))
        from .diagnostics import report
        report(output,model,scores,truth,expected,train_end,dev_end,rt,st,events,grouped,prefix='development')
        write_json(output/'DEVELOPMENT_COMPLETE.json',dict(status='complete',frozen_hash=digest(output/'FROZEN.json')))
        print(f'Frozen primary {winner}; deployment enabled: {bool(winner_row.accepted)}',flush=True)
    except Exception:
        write_json(output/'FAILED_development.json',dict(status='failed',error=traceback.format_exc()));raise
    return output


def assess(output):
    output=Path(output);frozen=json.loads((output/'FROZEN.json').read_text())
    if not (output/'DEVELOPMENT_COMPLETE.json').exists():raise ValueError('Complete all development diagnostics before assessment')
    if (output/'FINAL_OPENED.json').exists():raise FileExistsError('Final assessment already opened; do not retune/reassess this claim')
    for name,h in frozen['files'].items():
        if digest(output/name)!=h:raise ValueError(f'Frozen artifact changed: {name}')
    if frozen['source']['source_hashes']!=provenance()['source_hashes']:raise ValueError('Scoring source changed since freeze')
    write_json(output/'FINAL_OPENED.json',dict(status='opened',frozen_hash=digest(output/'FROZEN.json'),test_status=frozen['test_status'],opened_at=pd.Timestamp.now(tz='UTC')))
    try:
        model=load(output/'model.joblib');config=model['config'];start,train_end,dev_end,end=bounds(config)
        data,expected,_=canonical(output,end);truth=pd.read_parquet(output/'truth.parquet')
        # Replay full causal history, preserving recursive and event state over boundaries.
        routes=[c.split(':')[0] for c in model['selected_routes'].values()]
        scores,cost=score(model,data,routes=routes)
        scores.loc[scores.decision_time.ge(dev_end)].to_parquet(output/'final_scores.parquet',index=False)
        cost.to_csv(output/'final_costs.csv',index=False)
        events={};route_rows=[]
        for route,cid in model['selected_routes'].items():
            subset=scores.loc[scores.route.eq(route)]
            e,trace,notifications=detect(subset,model['policies'][cid]);events[cid]=e
            e.to_parquet(output/f'final_events_{cid}.parquet',index=False)
            notifications.to_parquet(output/f'final_notifications_{cid}.parquet',index=False)
            trace.to_parquet(output/f'final_state_{cid}.parquet',index=False)
            route_rows.append(record(cid,e,subset,truth,expected,dev_end,end,config['requirements'],window_cost=model['windows'][route]))
        system_rows=[];grouped={}
        for cid,system in model['systems'].items():
            raw=pd.concat([events[c] for c in system['components']],ignore_index=True)
            e,history=group(raw,system['span_minutes'],model['compatibility']);grouped[cid]=e
            subset=scores.loc[scores.route.isin(system['routes'])]
            result=record(cid,e,subset,truth,expected,dev_end,end,config['requirements'],len(system['routes']),sum(model['windows'][r] for r in system['routes']))
            result['raw_events']=int((raw.confirmation.ge(dev_end)&raw.confirmation.lt(end)).sum())
            result['group_span_minutes']=system['span_minutes'];system_rows.append(result)
            e.to_parquet(output/f'final_groups_{cid}.parquet',index=False)
            history.to_parquet(output/f'final_membership_{cid}.parquet',index=False)
        rt,st=pd.DataFrame(route_rows),pd.DataFrame(system_rows)
        rt.to_csv(output/'final_routes.csv',index=False);st.to_csv(output/'final_systems.csv',index=False)
        from .diagnostics import report
        report(output,model,scores,truth,expected,dev_end,end,rt,st,events,grouped,prefix='final')
        primary=st.set_index('candidate').loc[model['primary']].to_dict()
        write_json(output/'FINAL_RESULT.json',dict(status='complete',primary=model['primary'],metrics=primary,
            deployment_enabled=frozen['deployment_enabled'],selection_changed=False,test_status=frozen['test_status']))
        return primary
    except Exception:
        write_json(output/'FAILED_assessment.json',dict(status='failed',error=traceback.format_exc()));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','develop','assess','reuse'])
    parser.add_argument('--config',default='configs/industry_agnostic.yaml')
    parser.add_argument('--output',required=True)
    parser.add_argument('--from-run')
    parser.add_argument('--source-revision')
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.config,args.output)
    elif args.command=='develop':develop(args.config,args.output)
    elif args.command=='assess':assess(args.output)
    else:
        if not args.from_run or not args.source_revision:parser.error('reuse needs --from-run and --source-revision')
        reuse_development_cache(args.from_run,args.config,args.output,args.source_revision)


if __name__=='__main__':main()
