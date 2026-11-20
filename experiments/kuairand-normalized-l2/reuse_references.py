"""Audit existing ST artifacts in place and emit an immutable JSON reference."""
import hashlib
import json
from pathlib import Path
import statistics
from datetime import datetime,timezone

import numpy as np
from sklearn.metrics import roc_auc_score,log_loss

HOME=Path.home()
SEEDS=[1685480945,1685463909,1685477428,1685459668,1685496394]
TASKS=['click','like','follow','forward']
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def verify(folder):
    entries=json.loads((folder/'COMMITTED.json').read_text())
    for n,h in entries.items():
        if sha(folder/n)!=h:raise ValueError(f'Hash mismatch {folder/n}')
    return dict(path=str(folder),manifest_sha256=sha(folder/'COMMITTED.json'),files=entries)
def main():
    cache=HOME/'shared/cache/kuairand'
    arrays={s:{k:np.load(cache/f'{s}-{k}.npy',mmap_mode='r') for k in ['y','rowid']} for s in ['val','test']}
    roots=[HOME/'experiments'/n/'outputs/st/kuairand' for n in [
        'single-task-parallel-resume-20261112-v2','single-task-parallel-remaining-20261112']]
    result=dict(version='kuairand-st-reuse-audit-20261120',verified_at=datetime.now(timezone.utc).isoformat(),
                source_files={},artifacts=[],rows={})
    result['source_files'][str(cache/'manifest.json')]=sha(cache/'manifest.json')
    for target in TASKS:
        ms=[]
        for seed in SEEDS:
            p=next((r/str(seed)/target for r in roots if (r/str(seed)/target/'metrics.json').exists()),None)
            if p is None:raise ValueError(f'Missing ST {seed}/{target}')
            wrapper=json.loads((p/'metrics.json').read_text())
            receipts=[verify(p)]
            actual=Path(wrapper['external_artifacts']) if wrapper.get('external_artifacts') else p
            if actual!=p:receipts.append(verify(actual))
            m=json.loads((actual/'metrics.json').read_text())
            if abs(m['embedding_sum_coefficient']-.006/m['embedding_elements'])>1e-15:
                raise ValueError('ST regularization differs')
            if wrapper.get('paired_seed')!=seed or wrapper['target']!=target:
                raise ValueError('ST pairing differs')
            with np.load(actual/'predictions.npz') as raw:
                for split in ['val','test']:
                    if not np.array_equal(raw[split+'_y'],arrays[split]['y'][:,TASKS.index(target)]):
                        raise ValueError('ST labels differ')
                    if not np.array_equal(raw[split+'_rowid'],arrays[split]['rowid']):
                        raise ValueError('ST rows differ')
                    y,pred=raw[split+'_y'],raw[split+'_p']
                    metrics=dict(auc=roc_auc_score(y,pred),logloss=log_loss(y.astype(np.float64),np.clip(pred.astype(np.float64),1e-7,1-1e-7)))
                    for name,value in metrics.items():
                        if abs(value-wrapper[split][name])>1e-9:
                            raise ValueError(f'ST raw metric differs {seed}/{target}/{split}/{name}: {value} vs {wrapper[split][name]}')
            ms.append(wrapper)
            result['artifacts'].append(dict(seed=seed,target=target,receipts=receipts,
                model_hash=wrapper['model_hash'],test=wrapper['test'],val=wrapper['val'],
                started_at=wrapper.get('started_at'),ended_at=wrapper.get('ended_at')))
            result['source_files'][str(actual/'metrics.json')]=sha(actual/'metrics.json')
        row={}
        for name in ['auc','logloss']:
            values=[m['test'][name] for m in ms]
            row[name]=dict(n=5,seeds=SEEDS,values=values,mean=statistics.mean(values),variance=statistics.variance(values))
        result['rows'][f'kuairand/single_task/{target}']=row
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
