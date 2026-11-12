"""Verify raw records and aggregate the five paired models per dataset."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
from . import core
from .training import _committed, _utc


def stats(values):
    a=np.asarray(values,dtype=float)
    return dict(n=len(a),mean=float(a.mean()),sd=float(a.std(ddof=1)),values=a.tolist())


def report(output,cfg):
    output=Path(output)
    rows={}; raw_checks=0
    for dataset,dc in cfg['datasets'].items():
        for seed in dc['seeds']:
            for target in dc['targets']:
                references={}
                for variant in cfg['variants']:
                    folder=output/dataset/str(seed)/target/variant
                    if not (folder/'COMMITTED.json').exists():continue
                    _committed(core,folder)
                    m=json.loads((folder/'metrics.json').read_text())
                    if (m['paired_seed'],m['target'],m['variant'])!=(seed,target,variant):
                        raise ValueError('Model pairing mismatch')
                    if (folder/'REUSED.json').exists():
                        from .tri_reuse import artifact_folder
                        artifact=artifact_folder(folder)
                    else:artifact=folder
                    with np.load(artifact/'predictions.npz',allow_pickle=False) as raw:
                        for split in ['val','test']:
                            y,p,ids=[raw[split+'_'+k] for k in ['y','p','rowid']]
                            actual=core.measures(y,p)
                            if any(abs(actual[k]-m[split][k])>1e-10 for k in actual):
                                raise ValueError('Raw metric mismatch')
                            if ids.shape!=y.shape or p.shape!=y.shape:raise ValueError('Raw shape mismatch')
                            if split in references:
                                if not np.array_equal(y,references[split][0]) or not np.array_equal(ids,references[split][1]):
                                    raise ValueError('Paired labels/rowids mismatch')
                            else:references[split]=(y.copy(),ids.copy())
                            raw_checks+=1
                    rows[(dataset,seed,target,variant)]=m
    results={}; lines=['# 三数据集：线性U接入与三路融合','']
    for dataset,dc in cfg['datasets'].items():
        results[dataset]={}
        lines += [f'## {dataset}','', '| 模型 | '+' | '.join(f'{target} {metric}' for metric in ['AUC','Logloss'] for target in dc['targets'])+' |', '|---|'+'---:|'*(2*len(dc['targets']))]
        for variant in cfg['variants']:
            cells=[]; results[dataset][variant]={}
            for metric in ['auc','logloss']:
                for target in dc['targets']:
                    values=[rows[(dataset,s,target,variant)]['test'][metric] for s in dc['seeds'] if (dataset,s,target,variant) in rows]
                    if len(values)!=5:cells.append(f'待完成 {len(values)}/5');continue
                    item=stats(values)
                    for reference in ['mpt','tri_full']:
                        if reference==variant:continue
                        if all((dataset,s,target,reference) in rows for s in dc['seeds']):
                            delta=np.asarray(values)-np.asarray([rows[(dataset,s,target,reference)]['test'][metric] for s in dc['seeds']])
                            d=stats(delta);margin=t.ppf(.975,4)*d['sd']/np.sqrt(5)
                            item['vs_'+reference]=dict(d,low=d['mean']-float(margin),high=d['mean']+float(margin))
                    results[dataset][variant][target+'/'+metric]=item
                    cells.append(f"{item['mean']:.6f} ± {item['sd']:.6f}")
            lines.append('| '+' | '.join([variant,*cells])+' |')
        lines += ['', '| 模型 | 参照 | 目标 | 指标 | 配对差值 | 95% t区间 |', '|---|---|---|---|---:|---|']
        for variant,metrics in results[dataset].items():
            for name,item in metrics.items():
                for reference in ['mpt','tri_full']:
                    delta=item.get('vs_'+reference)
                    if delta:lines.append(f"| {variant} | {reference} | {name} | {delta['mean']:+.6f} | [{delta['low']:+.6f}, {delta['high']:+.6f}] |")
        lines += ['', '| 原任务路径 | '+' | '.join(f'{target} {metric}' for metric in ['auc','logloss'] for target in dc['initial'])+' |','|---|'+'---:|'*(len(dc['initial'])*2)]
        original={}
        source_costs=[]
        for seed in dc['seeds']:
            rp=output/dataset/str(seed)/'reuse-receipt.json'
            if not rp.exists():continue
            receipt=json.loads(rp.read_text())
            source_costs.append(dict(seed=seed,reused=receipt['reused'],stages=receipt['stages'],costs=receipt['costs']))
            lm=receipt['costs']['linear']
            for path in ['mpt','linear']:
                scores=lm.get('baseline',{}).get('test') if path=='mpt' else lm.get('test')
                if scores:
                    for target in dc['initial']:
                        for metric in ['auc','logloss']:original.setdefault((path,target,metric),[]).append(scores[target][metric])
        for path in ['mpt','linear']:
            cells=[]
            for metric in ['auc','logloss']:
                for target in dc['initial']:
                    values=original.get((path,target,metric),[])
                    if len(values)==5:
                        r=stats(values);cells.append(f"{r['mean']:.6f} ± {r['sd']:.6f}")
                    else:cells.append(f'待完成 {len(values)}/5')
            lines.append('| '+' | '.join([path,*cells])+' |')
        results[dataset]['original_tasks']={'/'.join(k):stats(v) if len(v)==5 else dict(n=len(v),values=v) for k,v in original.items()}
        results[dataset]['source_costs']=source_costs
        lines += ['', '| 模型 | 目标 | 完成 | U更新参数 | 总训练参数 | 新增路径推理参数（不含B） | 权重MiB | 平均秒 | 平均峰值MiB | 计时条件 |','|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
        for variant in cfg['variants']:
            for target in dc['targets']:
                ms=[rows[(dataset,s,target,variant)] for s in dc['seeds'] if (dataset,s,target,variant) in rows]
                if not ms:continue
                m=ms[0]
                if 'seconds' not in m:continue
                condition=m.get('cost_measurement_source','当前版本；缓存模式见cache-execution.json')
                weight=m.get('weight_file_bytes')
                weight_text=f'{weight/1024**2:.3f}' if weight is not None else '引用旧权重'
                lines.append(f"| {variant} | {target} | {len(ms)}/5 | {m.get('u_update_parameters',0)} | {m['optimizer_parameters']} | {m['inference_parameters']} | {weight_text} | {np.mean([r['seconds'] for r in ms]):.3f} | {np.mean([r['peak_cuda_bytes']/1024**2 for r in ms]):.3f} | {condition} |")
        lines.append('')
    expected=sum(len(d['seeds'])*len(d['targets'])*len(cfg['variants']) for d in cfg['datasets'].values())
    data=dict(completed=len(rows),expected=expected,raw_split_checks=raw_checks,summary=results,updated_at=_utc(),records={'/'.join(map(str,k)):v for k,v in rows.items()})
    core.dump(output/'summary.json',data)
    (output/'table.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    return len(rows)
