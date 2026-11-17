"""Table4 readouts, table5 random-source LoRA and table6 compute statistics."""
import argparse
import copy
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from . import core
from .locking import ExclusiveLock
from .scheme_common import (PROJECT, bind_contract, bind_execution, read_json,
                            short_lock, status, verify_stage)
from .scheme_report import statistics
from .training import _commit, _save, _utc

READOUTS = ('h_conditional', 'random_conditional', 'linear_aligned')
CONFIG = PROJECT / 'configs/paper_controls_20261117.json'


def default_config():
    cfg = copy.deepcopy(read_json(PROJECT/'configs/tri_three_datasets.json'))
    cfg.update(version='paper-controls-v1-20261117', readouts=list(READOUTS),
               execution=dict(workers=4, threads_per_worker=2),
               random_u=dict(seed_offset=1, normalization='pretrained-train-statistics'),
               expected=dict(table4=90, table5=25, table6=9))
    return cfg


def validate_config(cfg):
    old=read_json(PROJECT/'configs/tri_three_datasets.json')
    if cfg['datasets']!=old['datasets'] or cfg.get('readouts')!=list(READOUTS):
        raise ValueError('Registered dataset, training protocol or readout controls changed')
    if cfg.get('random_u')!=default_config()['random_u']:
        raise ValueError('Random-source pairing/normalization changed')


def expected_counts(cfg):
    return dict(table4=sum(len(d['initial'])*len(d['seeds'])*3 for d in cfg['datasets'].values()),
                table5=sum(len(d['targets'])*len(d['seeds']) for d in cfg['datasets'].values()),
                table6=3*len(cfg['datasets']))


def jobs(cfg,tables,dataset=None,seed=None):
    result=[]
    for table in tables:
        for name in cfg.get('order',cfg['datasets']):
            if dataset and dataset!=name:continue
            seeds=cfg['datasets'][name]['seeds']
            for current in seeds[:1] if table==6 else seeds:
                if seed is None or current==seed:result.append((table,name,current))
    return result


def _leaf(output,table,dataset,seed):
    return Path(output)/f'table{table}'/dataset/str(seed)


def _leaves(folder,table,dc):
    if table==4:return [folder/target/v for target in dc['initial'] for v in READOUTS]
    if table==5:return [folder/target/'random_lora' for target in dc['targets']]
    return [folder/'compute']


def completed_job(folder,table,dc):
    if not (folder/'DONE.json').exists():return False
    for leaf in _leaves(folder,table,dc):verify_stage(leaf,['metrics.json'])
    return True


def _random_source(spec,u,seed,folder):
    from .paper_readouts import make_random_u
    random=make_random_u(spec,u,seed)
    root=folder/'random-source'
    root.mkdir(parents=True,exist_ok=True)
    receipt=dict(paired_seed=seed,initialization_seed=seed+1,
                 normalization_source=core.model_hash(u),model_hash=core.model_hash(random))
    if (root/'COMMITTED.json').exists():
        verify_stage(root,['source.pt','metrics.json'])
        if read_json(root/'metrics.json')!=receipt:raise ValueError('Random source changed')
    else:
        _save(random.state_dict(),root/'source.pt')
        core.dump(root/'metrics.json',receipt);_commit(core,root)
    return random


def run_job(args,cfg,table,dataset,seed,identity):
    from .tri_data import build_spec
    from .scheme_reuse import load_sources
    dc=dict(cfg['datasets'][dataset],execution_identity=identity)
    folder=_leaf(args.output,table,dataset,seed)
    with ExclusiveLock(folder/'worker.lock'):
        if completed_job(folder,table,dc):return
        progress=folder/'status.json'
        status(progress,'running',stage='source-admission',table=table,dataset=dataset,seed=seed)
        with short_lock(args.cache_root/f'.scheme-{dataset}-admission.lock',timeout=86400):
            spec=build_spec(dataset,args.dataset_root,args.cache_root,folder/'data',dc)
        base,caches,envs,u,receipt=load_sources(spec,seed,dc,args.source_root,require_u=True)
        core.dump(folder/'reuse-receipt.json',receipt)
        if table==4:
            from .paper_readouts import fit_control
            _random_source(spec,u,seed,folder)
            for position,target in enumerate(dc['initial']):
                for variant in READOUTS:
                    fit_control(base,u,spec,caches,position,folder/target/variant,dc,seed,variant,progress)
        elif table==5:
            from .stage2_models import build_model
            from .tri_training import train_one
            from .scheme1_runner import _verify_completed
            random=_random_source(spec,u,seed,folder)
            base_hash,u_hash=core.model_hash(base),core.model_hash(u)
            def guard():
                if (core.model_hash(base),core.model_hash(u))!=(base_hash,u_hash):
                    raise ValueError('Original B/U source changed')
                return dict(passed=True,base_hash=base_hash,u_hash=u_hash,
                            original_path_proof='frozen B/U and original readout paths unchanged')
            for position,target in enumerate(dc['targets']):
                actual=seed+position+dc.get('head_seed_offset',3)
                core.seed_all(actual)
                model=build_model(spec,random,actual-3,'tri_lora').to(core.DEVICE)
                destination=folder/target/'random_lora'
                bind_contract(folder/target/'random-lora-contract.json',dict(
                    execution_identity=identity,config=dc,target=target,paired_seed=seed,
                    random_source_hash=core.model_hash(random),base_hash=base_hash,
                    initial_state_hash=core.model_hash(model),data=spec['scheme_data_identity']))
                if (destination/'COMMITTED.json').exists():
                    _verify_completed(model,destination,spec,caches,seed,target,'random_lora')
                train_one(model,spec,caches,envs,target,destination,dc,actual,'random_lora',guard,
                          progress,paired_seed=seed)
                del model
        else:
            from .paper_compute import profile
            rows=spec['sets']['train'];n=min(spec['batch'],len(rows))
            raw={name:torch.as_tensor(np.asarray(rows.x[:n,i]).copy(),device=core.DEVICE)
                 for i,name in enumerate(rows.names)}
            raw['_target']=torch.as_tensor(np.asarray(rows.y[:n,2]).copy(),device=core.DEVICE)
            result=profile(spec,base,u,envs,raw,seed)
            result.update(dataset=dataset,seed=seed,source_receipt=receipt,updated_at=_utc())
            core.dump(folder/'compute/metrics.json',result);_commit(core,folder/'compute')
        core.dump(folder/'DONE.json',dict(table=table,dataset=dataset,seed=seed,
                                       execution_identity=identity,ended_at=_utc()))
        status(progress,'complete',table=table,dataset=dataset,seed=seed)


def reuse_index(args):
    """Small references to existing summaries; no artifact or dataset copies."""
    refs={}
    for name,path in [('tri',args.source_root),('conditional',args.readout_root)]:
        if path is not None:
            p=Path(path)/'summary.json'
            if p.exists():refs[name]=dict(path=str(p.resolve()),sha256=core.file_hash(p))
    p=args.single_task_summary
    if p is not None and p.exists():refs['single_task']=dict(path=str(p.resolve()),sha256=core.file_hash(p))
    return refs


def existing_rows(args,cfg):
    """Populate existing table rows from their own summaries, without retraining."""
    rows={}
    for owner,root in [('tri',args.source_root),('conditional',args.readout_root)]:
        if root is None or not (root/'summary.json').exists():continue
        source=read_json(root/'summary.json');grouped={}
        for key,m in source.get('records',{}).items():
            ds,seed,target,variant=key.split('/')
            if ds not in cfg['datasets']:continue
            dc=cfg['datasets'][ds]
            if int(seed) not in dc['seeds']:raise ValueError('Referenced result seed mismatch')
            if owner=='tri' and target in dc['targets'] and variant in ['mpt','tri_frozen','tri_lora']:
                table=3 if variant=='tri_lora' else 5
            elif owner=='conditional' and target in dc['initial'] and variant=='conditional_readout':table=4
            else:continue
            grouped.setdefault(f'table{table}/{ds}/{target}/{variant}',[]).append(m)
            if owner=='tri' and variant=='mpt':grouped.setdefault(f'table3/{ds}/{target}/mpt',[]).append(m)
            if owner=='tri' and variant=='tri_lora':grouped.setdefault(f'table5/{ds}/{target}/tri_lora',[]).append(m)
            if owner=='conditional':
                baseline=dict(m,test=m['baseline']['test'])
                grouped.setdefault(f'table4/{ds}/{target}/mpt',[]).append(baseline)
        for key,ms in grouped.items():
            rows[key]=dict(source=str(root/'summary.json'),
                           **{metric:statistics([m['test'][metric] for m in ms]) for metric in ['auc','logloss']})
    if args.single_task_summary is not None and args.single_task_summary.exists():
        for row in read_json(args.single_task_summary):
            if row['model']!='single_task':continue
            ds,target=row['dataset'],row['target']
            if target not in cfg['datasets'][ds]['targets']:continue
            rows[f'table3/{ds}/{target}/single_task']=dict(
                source=str(args.single_task_summary),auc=dict(mean=row['auc_mean'],variance=row['auc_variance']),
                aggregation='previously audited five-seed summary; raw files remain at original source')
    return rows


def report(args,cfg):
    records={};stats={};counts=dict(table4=0,table5=0,table6=0)
    for table,dataset,seed in jobs(cfg,[4,5,6]):
        folder=_leaf(args.output,table,dataset,seed)
        for leaf in _leaves(folder,table,cfg['datasets'][dataset]):
            if not (leaf/'COMMITTED.json').exists():continue
            proof=verify_stage(leaf,['metrics.json']);m=read_json(leaf/'metrics.json')
            key=str(leaf.relative_to(args.output)).replace('\\','/')
            if table!=6:
                with np.load(leaf/'predictions.npz',allow_pickle=False) as raw:
                    for split in ['val','test']:
                        actual=core.measures(raw[split+'_y'],raw[split+'_p'])
                        if any(abs(actual[k]-m[split][k])>1e-10 for k in actual):
                            raise ValueError('Raw prediction metric mismatch')
                group=f'table{table}/{dataset}/{m["target"]}/{m["variant"]}'
                stats.setdefault(group,[]).append(m)
                counts[f'table{table}']+=1
            else:counts['table6']+=len(m['variants'])
            records[key]=dict(metrics=m,manifest_sha256=proof['committed_sha256'])
    summary={}
    for group,ms in stats.items():
        summary[group]={metric:statistics([m['test'][metric] for m in ms]) for metric in ['auc','logloss']}
        deltas=[m['test']['auc']-m['baseline']['test']['auc'] for m in ms if 'baseline' in m]
        if deltas:summary[group]['delta_vs_mpt']=statistics(deltas)
    reused_rows=existing_rows(args,cfg)
    result=dict(completed=counts,expected=expected_counts(cfg),summary=summary,records=records,
                existing_rows=reused_rows,
                reused=reuse_index(args),updated_at=_utc())
    core.dump(args.output/'summary.json',result)
    combined={**reused_rows,**summary};lines=['# 表3–6结果','', 'AUC均值（样本方差），完整组为5seed。','']
    for table in [3,4,5]:
        cols=[(ds,t) for ds,dc in cfg['datasets'].items() for t in (dc['initial'] if table==4 else dc['targets'])]
        variants={3:[('single_task','Single Task'),('mpt','MPT'),('tri_lora','条件头＋LoRA')],
                  4:[('mpt','MPT'),('h_conditional','原表示＋条件头'),('random_conditional','随机U＋条件头'),
                     ('linear_aligned','预训练U＋线性头'),('conditional_readout','预训练U＋条件头')],
                  5:[('mpt','MPT'),('tri_frozen','预训练U＋冻结'),('random_lora','随机U＋LoRA'),('tri_lora','预训练U＋LoRA')]}[table]
        lines += [f'## 表{table}','', '| 模型 | '+' | '.join(ds+'/'+t for ds,t in cols)+' |', '|---|'+'---:|'*len(cols)]
        for variant,label in variants:
            cells=[]
            for ds,target in cols:
                item=combined.get(f'table{table}/{ds}/{target}/{variant}')
                if item is None:cells.append('待补');continue
                a=item['auc'];v=a['variance'];n=a.get('n',5)
                value=f'{a["mean"]:.6f}（{v:.3e}）' if v is not None else f'{a["mean"]:.6f}（方差待补）'
                cells.append(value+(f' [{n}/5]' if n!=5 else ''))
            lines.append('| '+' | '.join([label]+cells)+' |')
        lines.append('')
    lines += ['## 表6','', '统计新增任务路径；FLOPs为计入矩阵算子的每批操作数。','',
              '| 数据集 | 模型 | 总参数B＋新目标 | 活跃推理参数 | 可训练参数 | 推理前向FLOPs | 训练前向FLOPs | 反向FLOPs | 训练合计FLOPs |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for item in records.values():
        m=item['metrics']
        for variant,row in m.get('variants',{}).items():
            keys=['deployment_parameters','active_new_task_inference_parameters','trainable_parameters',
                  'inference_forward_flops','training_forward_flops','training_backward_flops','training_total_flops']
            lines.append('| '+' | '.join([m['dataset'],variant]+[str(row[k]) for k in keys])+' |')
    (args.output/'tables.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    return result


def parallel(args,cfg,identity):
    tasks=jobs(cfg,args.tables,args.dataset,args.seed);running=[];failures=[];finished=0
    with ExclusiveLock(args.output/'scheduler.lock'):
        try:
            while tasks or running:
                while tasks and len(running)<args.workers:
                    table,ds,seed=tasks.pop(0);folder=_leaf(args.output,table,ds,seed)
                    if completed_job(folder,table,cfg['datasets'][ds]):finished+=1;continue
                    folder.mkdir(parents=True,exist_ok=True)
                    cmd=[sys.executable,'-m','sska_rec.paper_controls','--config',str(args.config),
                         '--source-root',str(args.source_root),'--dataset-root',str(args.dataset_root),
                         '--cache-root',str(args.cache_root),'--output',str(args.output),
                         '--table',str(table),'--dataset',ds,'--seed',str(seed),
                         '--device',args.device,'--threads',str(args.threads)]
                    log=(folder/'runner.log').open('ab',buffering=0)
                    env=dict(os.environ,OMP_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads),
                             OPENBLAS_NUM_THREADS=str(args.threads),PYTHONUNBUFFERED='1')
                    process=subprocess.Popen(cmd,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    running.append((process,log,table,ds,seed))
                live=[]
                for process,log,table,ds,seed in running:
                    code=process.poll()
                    if code is None:live.append((process,log,table,ds,seed))
                    else:
                        log.close();finished+=1
                        if code:failures.append(dict(table=table,dataset=ds,seed=seed,exit_code=code))
                running=live
                status(args.output/'status.json','running',finished_jobs=finished,pending_jobs=len(tasks),
                       running_jobs=[dict(pid=p.pid,table=t,dataset=d,seed=s) for p,l,t,d,s in running],
                       failures=failures,workers=args.workers,threads_per_worker=args.threads)
                if running:time.sleep(1)
        except BaseException:
            for p,log,*_ in running:
                if p.poll() is None:p.terminate()
                p.wait();log.close()
            raise
        result=report(args,cfg)
        state='failed' if failures else ('complete' if result['completed']==result['expected'] else 'partial')
        status(args.output/'status.json',state,completed=result['completed'],failures=failures)
        if failures:raise RuntimeError(f'{len(failures)} failed jobs; inspect runner.log')
        return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=CONFIG)
    p.add_argument('--source-root',type=Path)
    p.add_argument('--dataset-root',type=Path)
    p.add_argument('--cache-root',type=Path,default=Path('cache'))
    p.add_argument('--readout-root',type=Path)
    p.add_argument('--single-task-summary',type=Path)
    p.add_argument('--output',type=Path,default=Path('outputs/paper-controls'))
    p.add_argument('--table',type=int,choices=[4,5,6])
    p.add_argument('--tables',type=lambda v:[int(x) for x in v.split(',')],default=[4,5,6])
    p.add_argument('--dataset',choices=['census','kuairand','aliccp'])
    p.add_argument('--seed',type=int)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--parallel',action='store_true')
    p.add_argument('--report-only',action='store_true')
    args=p.parse_args(argv)
    for name in ['config','source_root','dataset_root','cache_root','readout_root','single_task_summary','output']:
        value=getattr(args,name)
        if value is not None:setattr(args,name,value.expanduser().resolve())
    cfg=read_json(args.config);validate_config(cfg)
    if args.table is not None:args.tables=[args.table]
    args.output.mkdir(parents=True,exist_ok=True)
    if args.report_only:return report(args,cfg)
    if args.source_root is None or args.dataset_root is None:raise ValueError('Source/dataset roots required')
    if args.workers<1 or args.threads<1 or not set(args.tables)<=set([4,5,6]):raise ValueError('Invalid parallel/table settings')
    identity=bind_execution(args.output,cfg,args.source_root,args.dataset_root)
    if args.parallel:return parallel(args,cfg,identity)
    core.DEVICE=torch.device(args.device);torch.set_num_threads(args.threads)
    tasks=jobs(cfg,[args.table] if args.table else args.tables,args.dataset,args.seed)
    if not tasks:raise ValueError('No registered dataset/seed job selected')
    for table,ds,seed in tasks:
        try:run_job(args,cfg,table,ds,seed,identity)
        except BaseException as exc:
            status(_leaf(args.output,table,ds,seed)/'status.json','failed',error=str(exc));raise


if __name__=='__main__':main()
