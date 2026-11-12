"""Run remaining ST targets beside the main worker; reference completed Census."""
import argparse
import json
import os
import time
from pathlib import Path
from . import core
import torch
from .locking import ExclusiveLock
from .training import _utc,_committed
from .tri_data import build_spec
from .st_runner import compare_data,selected_device,validate_config
from .st_fit import fit_single,build_single
from .st_reuse import reuse_single
from .st_report import report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,required=True)
    p.add_argument('--comparison-root',type=Path,required=True)
    p.add_argument('--completed-st-root',type=Path,required=True)
    p.add_argument('--legacy-kuairand',type=Path,required=True)
    p.add_argument('--resume-st-root',type=Path)
    p.add_argument('--output',type=Path,default=Path('outputs/st'))
    a=p.parse_args();cfg=json.loads(Path('configs/st_three_datasets.json').read_text());validate_config(cfg)
    cfg=dict(cfg,execution_mode='remaining-ST-parallel-to-primary')
    core.DEVICE=selected_device('cuda')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(2)
    out=a.output;out.mkdir(parents=True,exist_ok=True);status=out/'status.json'
    identity=dict(config=cfg,source={f.name:core.file_hash(f) for f in sorted(Path(__file__).parent.glob('*.py'))},git_commit=os.environ.get('SSKA_COMMIT'),completed_st_root=str(a.completed_st_root.resolve()),comparison_root=str(a.comparison_root.resolve()))
    with ExclusiveLock(out/'worker.lock'):
        ip=out/'identity.json'
        if ip.exists() and json.loads(ip.read_text())!=identity:raise ValueError('Parallel execution identity changed')
        core.dump(ip,identity)
        try:
            census=a.completed_st_root/'census'
            for seed in cfg['datasets']['census']['seeds']:
                for target in cfg['datasets']['census']['st_targets']:
                    stage=census/str(seed)/target
                    if not (stage/'COMMITTED.json').exists():raise ValueError('Census ST reuse incomplete')
                    _committed(core,stage)
            link=out/'census'
            if link.is_symlink():
                if link.resolve()!=census.resolve():raise ValueError('Census result source changed')
            elif link.exists():raise ValueError('Census result destination occupied')
            else:link.symlink_to(census.resolve(),target_is_directory=True)
            core.dump(out/'completed-census-receipt.json',dict(source=str(census.resolve()),source_identity_sha256=core.file_hash(a.completed_st_root/'identity.json'),models=20,mode='read-only-directory-reference'))
            for dataset in ['kuairand','aliccp']:
                dc=dict(cfg['datasets'][dataset],cost_measurement_source='current-ST-concurrent-with-primary')
                core.dump(status,dict(state='running',pid=os.getpid(),dataset=dataset,stage='data_admission',completed=20,expected=55,updated_at=_utc()))
                spec=build_spec(dataset,a.dataset_root,a.cache_root,out/dataset/'data',dc)
                primary_data=a.comparison_root/dataset/'data/data-signature.json'
                compare_data(spec['data_signature'],json.loads(primary_data.read_text()))
                core.dump(out/dataset/'comparison-receipt.json',dict(primary_data=str(primary_data.resolve()),primary_data_sha256=core.file_hash(primary_data),verified_at=_utc()))
                for seed in dc['seeds']:
                    for target in dc['st_targets']:
                        while torch.cuda.mem_get_info(core.DEVICE)[0]<2048*1024**2:
                            core.dump(status,dict(state='queued',pid=os.getpid(),stage='waiting_for_gpu_headroom',dataset=dataset,seed=seed,target=target,updated_at=_utc()))
                            time.sleep(15)
                        core.dump(status,dict(state='running',pid=os.getpid(),dataset=dataset,seed=seed,target=target,stage='single_task',updated_at=_utc()))
                        folder=out/dataset/str(seed)/target;reused=False
                        if a.resume_st_root:
                            prior=a.resume_st_root/dataset/str(seed)/target
                            if (prior/'COMMITTED.json').exists() and not folder.exists():
                                previous=json.loads((a.resume_st_root/'identity.json').read_text())
                                for name in ['st_fit.py','backbone.py','core.py','training.py','tri_data.py','data.py','vocabulary.py']:
                                    if previous['source'][name]!=core.file_hash(Path(__file__).with_name(name)):raise ValueError(f'Completed ST source changed: {name}')
                                compare_data(spec['data_signature'],json.loads((a.resume_st_root/dataset/'data/data-signature.json').read_text()))
                                _committed(core,prior)
                                folder.parent.mkdir(parents=True,exist_ok=True)
                                folder.symlink_to(prior.resolve(),target_is_directory=True)
                        if dataset=='kuairand' and target in ['follow','forward']:
                            model=build_single(spec,seed+spec['tasks'].index(target)+2,dc)
                            reused=reuse_single(spec,seed,target,model,folder,a.legacy_kuairand,dc)
                            del model
                        if not reused:fit_single(spec,seed,target,folder,dc,status)
                    n=report(out,cfg,a.comparison_root)
                    print(json.dumps(dict(event='ST_SEED_DONE',dataset=dataset,seed=seed,completed=n,at=_utc())),flush=True)
                    torch.cuda.empty_cache()
            n=report(out,cfg,a.comparison_root)
            if n!=55:raise ValueError(f'ST matrix incomplete {n}/55')
            core.dump(status,dict(state='complete',pid=None,completed=n,expected=55,finished_at=_utc()))
        except BaseException as e:
            core.dump(status,dict(state='paused' if isinstance(e,KeyboardInterrupt) else 'failed',pid=None,error=str(e),updated_at=_utc()))
            raise


if __name__=='__main__':main()
