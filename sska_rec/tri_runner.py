"""Full-split Census and matched Ali-CCP/KuaiRand five-model queue."""
import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
import numpy as np
import torch
from . import core
from .locking import ExclusiveLock
from .training import _utc


def model_for(spec,u,seed,variant):
    if variant=='tri_ssf':
        from .peft_models import build_model
    else:
        from .stage2_models import build_model
    return build_model(spec,u,seed,variant)


def resident(caches,device):
    """Move one small seed's representations; labels remain CPU for raw writes."""
    nbytes=sum(v.numel()*v.element_size() for cache in caches.values() for k,v in cache.items() if k!='y')
    if device.type!='cuda':return dict(mode='cpu',bytes=nbytes)
    free,total=torch.cuda.mem_get_info(device)
    if nbytes>min(free*.4,total*.35):return dict(mode='mmap_stream',bytes=nbytes)
    for cache in caches.values():
        for key in cache:
            if key!='y':cache[key]=cache[key].to(device)
    return dict(mode='gpu_resident',bytes=nbytes)


def main(argv=None):
    from .tri_data import build_spec
    from .tri_sources import prepare_seed,old_predictions
    from .tri_report import report
    from .tri_training import train_one
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,default=Path('configs/tri_three_datasets.json'))
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,default=Path('cache'))
    p.add_argument('--output',type=Path,default=Path('outputs/tri'))
    p.add_argument('--reuse-map',type=Path)
    p.add_argument('--device',default='cuda')
    p.add_argument('--report-only',action='store_true')
    a=p.parse_args(argv);cfg=json.loads(a.config.read_text())
    a.output.mkdir(parents=True,exist_ok=True)
    if a.report_only:report(a.output,cfg);return
    core.DEVICE=torch.device(a.device)
    if core.DEVICE.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4)
    source={f.name:core.file_hash(f) for f in sorted(Path(__file__).parent.glob('*.py'))}
    reuse=json.loads(a.reuse_map.read_text()) if a.reuse_map else {}
    identity=dict(config=cfg,source=source,dataset_root=str(a.dataset_root.resolve()),reuse_map=reuse,git_commit=os.environ.get('SSKA_COMMIT'))
    status=a.output/'status.json'
    with ExclusiveLock(a.output/'worker.lock'):
        ip=a.output/'identity.json'
        if ip.exists() and json.loads(ip.read_text())!=identity:raise ValueError('Execution identity changed; create new version')
        core.dump(ip,identity)
        core.dump(a.output/'environment.json',dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,platform=platform.platform(),gpu=torch.cuda.get_device_name() if core.DEVICE.type=='cuda' else None,started_at=_utc(),fp32=True))
        try:
            for dataset in cfg.get('order',list(cfg['datasets'])):
                dc=cfg['datasets'][dataset]
                core.dump(status,dict(state='running',pid=os.getpid(),dataset=dataset,stage='data_admission',updated_at=_utc()))
                spec=build_spec(dataset,a.dataset_root,a.cache_root,a.output/dataset/'data',dc)
                for seed in dc['seeds']:
                    folder=a.output/dataset/str(seed)
                    core.dump(status,dict(state='running',pid=os.getpid(),dataset=dataset,seed=seed,stage='prepare_sources',updated_at=_utc()))
                    candidate=reuse.get(dataset,{}).get(str(seed))
                    base,caches,envs,u,heads,receipt=prepare_seed(spec,seed,folder,dc,status,candidate)
                    cm=resident(caches,core.DEVICE)
                    core.dump(folder/'cache-execution.json',cm)
                    core.dump(folder/'reuse-receipt.json',receipt)
                    hashes=[core.model_hash(x) for x in [base,u,heads]]
                    references={s:old_predictions(base,u,heads,caches,spec,s) for s in ['val','test']}
                    with np.load(receipt['original_predictions_npz'],allow_pickle=False) as raw:
                        for split in references:
                            if not np.array_equal(references[split],raw[split+'_p']):raise ValueError('Original linear predictions fail reproduction')
                    def guard():
                        if hashes != [core.model_hash(x) for x in [base,u,heads]]:raise ValueError('Frozen original source changed')
                        for split,expected in references.items():
                            if not np.array_equal(old_predictions(base,u,heads,caches,spec,split),expected):raise ValueError('Original prediction changed')
                        return dict(passed=True,base_hash=hashes[0],u_hash=hashes[1],linear_heads_hash=hashes[2],max_probability_difference=0.)
                    for pos,target in enumerate(dc['targets']):
                        for variant in cfg['variants']:
                            core.dump(status,dict(state='running',pid=os.getpid(),dataset=dataset,seed=seed,target=target,variant=variant,stage='stage2',updated_at=_utc()))
                            actual_seed=seed+pos+dc.get('head_seed_offset',3)
                            core.seed_all(actual_seed)
                            model=model_for(spec,u,actual_seed-3,variant).to(core.DEVICE)
                            contract=dict(trainable={n:p.numel() for n,p in model.named_parameters() if p.requires_grad},frozen={n:p.numel() for n,p in model.named_parameters() if not p.requires_grad},source_hashes=hashes)
                            contract_path=folder/target/(variant+'-parameter-contract.json')
                            if contract_path.exists() and json.loads(contract_path.read_text())!=contract:
                                raise ValueError('Parameter/frozen contract changed')
                            core.dump(contract_path,contract)
                            reused=False
                            if dataset=='kuairand' and variant=='mpt' and candidate:
                                from .tri_reuse import reuse_mpt
                                reused=reuse_mpt(spec,seed,target,actual_seed,model,receipt,folder/target/variant,candidate)
                                guard()
                            if not reused:
                                train_one(model,spec,caches,envs,target,folder/target/variant,dc,actual_seed,variant,guard,status,paired_seed=seed)
                            del model
                    report(a.output,cfg)
                    print(json.dumps(dict(event='SEED_DONE',dataset=dataset,seed=seed,at=_utc())),flush=True)
                    del base,u,heads,caches,references
                    if core.DEVICE.type=='cuda':torch.cuda.empty_cache()
            completed=report(a.output,cfg)
            core.dump(status,dict(state='complete',pid=None,completed=completed,expected=125,finished_at=_utc()))
        except BaseException as exc:
            core.dump(status,dict(state='paused' if isinstance(exc,KeyboardInterrupt) else 'failed',pid=None,error=str(exc),updated_at=_utc()))
            raise


if __name__=='__main__':main()
