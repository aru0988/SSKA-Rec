"""Verify ST raw evidence and join all original/new tasks with five controls."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
from . import core
from .training import _committed,_utc


def stats(values):
    a=np.asarray(values,dtype=float)
    return dict(n=len(a),mean=float(a.mean()),sd=float(a.std(ddof=1)),values=a.tolist())


def report(output,cfg,primary):
    output,primary=Path(output),Path(primary)
    records={};joined={};checks=0
    for dataset,dc in cfg['datasets'].items():
        for seed in dc['seeds']:
            receipt_path=primary/dataset/str(seed)/'reuse-receipt.json'
            receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else None
            for target in dc['st_targets']:
                alignment={}
                folder=output/dataset/str(seed)/target
                if (folder/'COMMITTED.json').exists():
                    _committed(core,folder)
                    m=json.loads((folder/'metrics.json').read_text())
                    expected_seed=seed+dc['st_targets'].index(target)+2
                    if m['paired_seed']!=seed or m['target']!=target or m.get('variant')!='single_task' or m.get('model_seed',m.get('seed'))!=expected_seed:raise ValueError('ST paired identity mismatch')
                    if (folder/'REUSED.json').exists():
                        from .st_reuse import artifact_folder
                        raw_folder=artifact_folder(folder)
                    else:raw_folder=folder
                    with np.load(raw_folder/'predictions.npz',allow_pickle=False) as raw:
                        for split in ['val','test']:
                            y,p,ids=[raw[split+'_'+k] for k in ['y','p','rowid']]
                            actual=core.measures(y,p)
                            if ids.shape!=y.shape or p.shape!=y.shape:raise ValueError('ST raw shape mismatch')
                            if any(abs(actual[k]-m[split][k])>1e-10 for k in actual):raise ValueError('ST raw metric mismatch')
                            alignment[split]=(y.copy(),ids.copy())
                            checks+=1
                    records[(dataset,seed,target)]=m
                    joined[(dataset,seed,target,'single_task')]=m['test']
                for variant in ['mpt','tri_frozen','tri_lora','tri_ssf','tri_full']:
                    if target in dc['initial']:
                        if not receipt:continue
                        linear=receipt['costs']['linear']
                        scores=linear.get('baseline',{}).get('test',{}) if variant=='mpt' else linear.get('test',{})
                        if target in scores:joined[(dataset,seed,target,variant)]=scores[target]
                        if alignment and variant=='mpt':
                            with np.load(receipt['original_predictions_npz'],allow_pickle=False) as raw:
                                for split,(y,ids) in alignment.items():
                                    if not np.array_equal(y,raw[split+'_y'][:,dc['initial'].index(target)]) or not np.array_equal(ids,raw[split+'_rowid']):raise ValueError('ST/primary original-task labels/rowids mismatch')
                    else:
                        stage=primary/dataset/str(seed)/target/variant
                        if (stage/'COMMITTED.json').exists():
                            _committed(core,stage)
                            m=json.loads((stage/'metrics.json').read_text())
                            if (m['paired_seed'],m['target'],m['variant'])!=(seed,target,variant):raise ValueError('Primary paired identity mismatch')
                            if alignment:
                                from .tri_reuse import artifact_folder
                                with np.load(artifact_folder(stage)/'predictions.npz',allow_pickle=False) as raw:
                                    for split,(y,ids) in alignment.items():
                                        if not np.array_equal(y,raw[split+'_y']) or not np.array_equal(ids,raw[split+'_rowid']):raise ValueError('ST/primary new-task labels/rowids mismatch')
                            joined[(dataset,seed,target,variant)]=m['test']
    variants=['single_task','mpt','tri_frozen','tri_lora','tri_ssf','tri_full']
    summaries={};lines=['# 三数据集全部目标：Single Task与五结构对照','']
    for dataset,dc in cfg['datasets'].items():
        summaries[dataset]={}
        for metric in ['auc','logloss']:
            lines += [f'## {dataset} {metric}','', '| 模型 | '+' | '.join(dc['st_targets'])+' |','|---|'+'---:|'*len(dc['st_targets'])]
            for variant in variants:
                cells=[]
                for target in dc['st_targets']:
                    values=[joined[(dataset,s,target,variant)][metric] for s in dc['seeds'] if (dataset,s,target,variant) in joined]
                    if len(values)!=5:cells.append(f'待完成 {len(values)}/5');continue
                    item=stats(values)
                    if variant!='single_task' and all((dataset,s,target,'single_task') in joined for s in dc['seeds']):
                        delta=np.asarray(values)-np.asarray([joined[(dataset,s,target,'single_task')][metric] for s in dc['seeds']])
                        d=stats(delta);margin=float(t.ppf(.975,4)*d['sd']/np.sqrt(5))
                        item['vs_single_task']=dict(d,low=d['mean']-margin,high=d['mean']+margin)
                    summaries[dataset][f'{variant}/{target}/{metric}']=item
                    cells.append(f"{item['mean']:.6f} ± {item['sd']:.6f}")
                lines.append('| '+' | '.join([variant,*cells])+' |')
            lines.append('')
        lines += ['| 模型 | 目标/指标 | 相对ST差值 | 95%t区间 |','|---|---|---:|---|']
        for key,item in summaries[dataset].items():
            d=item.get('vs_single_task')
            if d:lines.append(f"| {key.split('/')[0]} | {'/'.join(key.split('/')[1:])} | {d['mean']:+.6f} | [{d['low']:+.6f}, {d['high']:+.6f}] |")
        lines += ['', '| ST目标 | 完成 | 训练参数 | 平均耗时秒 | 平均峰值MiB | 成本来源 |','|---|---:|---:|---:|---:|---|']
        for target in dc['st_targets']:
            ms=[records[(dataset,s,target)] for s in dc['seeds'] if (dataset,s,target) in records]
            if ms and 'seconds' in ms[0]:
                sources=','.join(sorted({m.get('cost_measurement_source','current-ST') for m in ms}))
                lines.append(f"| {target} | {len(ms)}/5 | {ms[0]['optimizer_parameters']} | {np.mean([m['seconds'] for m in ms]):.3f} | {np.mean([m['peak_cuda_bytes']/1024**2 for m in ms]):.3f} | {sources} |")
        lines.append('')
    core.dump(output/'summary.json',dict(completed=len(records),expected=55,raw_split_checks=checks,summary=summaries,updated_at=_utc(),records={'/'.join(map(str,k)):v for k,v in records.items()}))
    (output/'table.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    return len(records)
