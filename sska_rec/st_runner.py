"""All-target ST queue, isolated from and serialized after the five-model run."""
import argparse
import json
import os
import time
from contextlib import contextmanager,ExitStack
from pathlib import Path
import torch
from . import core
from .locking import ExclusiveLock
from .training import _utc


def compare_data(actual,reference):
    for key in ['dataset','tasks','initial','features','vocab','input_dim','representation_dim','widths']:
        if actual.get(key)!=reference.get(key):raise ValueError(f'Primary data mismatch: {key}')
    def inputs(s):return [(Path(x['path']).name,x['sha256'],x['bytes']) for x in s['inputs']]
    if inputs(actual)!=inputs(reference):raise ValueError('Primary raw input mismatch')
    if set(actual['splits'])!=set(reference['splits']):raise ValueError('Primary split mismatch')
    for split,entry in actual['splits'].items():
        for key in ['rows','rowid_sha256','labels_sha256','features_sha256','features_dtype','labels_dtype','rowid_dtype','user_groups_sha256']:
            if entry.get(key)!=reference['splits'][split].get(key):raise ValueError(f'Primary data mismatch: {split}/{key}')


def primary_ready(status,worker_alive):
    return status.get('state')=='complete' and status.get('completed')==125 and not status.get('pid') and not worker_alive


def parallel_allowed(cfg,dataset):
    allowed=cfg.get('parallel_datasets',[])
    if set(allowed)-{'census'}:raise ValueError('Only Census ST has parallel resource admission')
    return dataset in allowed


def validate_config(cfg):
    expected={'census':['income','marital','education','sex'],'aliccp':['CTR','CVR','BSI'],'kuairand':['click','like','follow','forward']}
    if set(cfg['datasets'])!=set(expected) or set(cfg['order'])!=set(expected) or len(cfg['order'])!=3:raise ValueError('ST dataset matrix mismatch')
    for dataset,dc in cfg['datasets'].items():
        if dc['st_targets']!=expected[dataset] or len(dc['seeds'])!=5 or len(set(dc['seeds']))!=5:raise ValueError('ST target/seed matrix mismatch')
    if cfg['expected_results']!=55:raise ValueError('ST expected matrix must contain 55 results')


@contextmanager
def primary_lock(primary):
    """Hold the existing main lock without changing a byte of its file."""
    with (primary/'worker.lock').open('rb') as file:
        if os.name=='nt':
            import msvcrt
            file.seek(0);msvcrt.locking(file.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(file.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            if os.name=='nt':
                file.seek(0);msvcrt.locking(file.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(file.fileno(),fcntl.LOCK_UN)


def primary_lock_free(primary):
    try:
        with primary_lock(primary):return True
    except OSError:return False


def discover_primary(experiment):
    """Also detect a main worker when its final status has already cleared pid."""
    if os.name=='nt':return False
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            args=proc.joinpath('cmdline').read_bytes().split(b'\0')
            if b'sska_rec.tri_runner' in args and live_primary(int(proc.name),experiment):return True
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    return False


def live_primary(pid,experiment):
    if not pid:return False
    try:
        os.kill(pid,0)
        if os.name!='nt':
            proc=Path('/proc')/str(pid)
            if proc.joinpath('stat').read_text().split()[2]=='Z':return False
            return proc.joinpath('cwd').resolve()==experiment.resolve()
        return True
    except (ProcessLookupError,FileNotFoundError,PermissionError):return False


def wait_primary(primary,output):
    last_pid=None
    while True:
        path=primary/'status.json'
        state=json.loads(path.read_text()) if path.exists() else {'state':'not_started'}
        if state.get('pid'):last_pid=state['pid']
        alive=live_primary(last_pid,primary.parents[1]) or discover_primary(primary.parents[1])
        if primary_ready(state,alive) and primary_lock_free(primary):return
        core.dump(output/'status.json',dict(state='queued',pid=os.getpid(),stage='waiting_for_primary',completed=0,expected=55,primary_status=state,primary_worker_alive=alive,updated_at=_utc()))
        time.sleep(30)


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,default=Path('configs/st_three_datasets.json'))
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,required=True)
    p.add_argument('--comparison-root',type=Path,required=True)
    p.add_argument('--legacy-kuairand',type=Path)
    p.add_argument('--output',type=Path,default=Path('outputs/st'))
    p.add_argument('--device',default='cuda')
    p.add_argument('--report-only',action='store_true')
    a=p.parse_args(argv);cfg=json.loads(a.config.read_text(encoding='utf8'))
    validate_config(cfg)
    from .st_report import report
    a.output.mkdir(parents=True,exist_ok=True)
    if a.report_only:report(a.output,cfg,a.comparison_root);return
    code={f.name:core.file_hash(f) for f in sorted(Path(__file__).parent.glob('*.py'))}
    identity=dict(config=cfg,source=code,git_commit=os.environ.get('SSKA_COMMIT'),comparison_root=str(a.comparison_root.resolve()),dataset_root=str(a.dataset_root.resolve()),legacy_kuairand=str(a.legacy_kuairand.resolve()) if a.legacy_kuairand else None)
    status=a.output/'status.json'
    with ExclusiveLock(a.output/'worker.lock'),ExitStack() as primary_guard:
        path=a.output/'identity.json'
        if path.exists() and json.loads(path.read_text())!=identity:raise ValueError('ST execution identity changed; select new version')
        core.dump(path,identity)
        core.DEVICE=torch.device(a.device)
        if core.DEVICE.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        torch.set_num_threads(4)
        from .tri_data import build_spec
        from .st_fit import fit_single,build_single
        from .st_reuse import reuse_single
        primary_held=False
        try:
            for dataset in cfg['order']:
                dc=cfg['datasets'][dataset]
                overlap=parallel_allowed(cfg,dataset)
                if not overlap and cfg.get('wait_for_primary',True) and not primary_held:
                    while True:
                        wait_primary(a.comparison_root,a.output)
                        try:
                            primary_guard.enter_context(primary_lock(a.comparison_root));primary_held=True
                            break
                        except OSError:time.sleep(1)
                torch.set_num_threads(2 if overlap else 4)
                core.dump(status,dict(state='running',pid=os.getpid(),dataset=dataset,stage='data_admission',updated_at=_utc()))
                spec=build_spec(dataset,a.dataset_root,a.cache_root,a.output/dataset/'data',dc)
                primary_data=a.comparison_root/dataset/'data/data-signature.json'
                compare_data(spec['data_signature'],json.loads(primary_data.read_text()))
                core.dump(a.output/dataset/'comparison-receipt.json',dict(primary_data=str(primary_data.resolve()),primary_data_sha256=core.file_hash(primary_data),primary_identity_sha256=core.file_hash(a.comparison_root/'identity.json'),verified_at=_utc()))
                for seed in dc['seeds']:
                    for target in dc['st_targets']:
                        if overlap and core.DEVICE.type=='cuda':
                            while torch.cuda.mem_get_info(core.DEVICE)[0]<2048*1024**2:
                                core.dump(status,dict(state='queued',pid=os.getpid(),stage='waiting_for_gpu_headroom',dataset=dataset,updated_at=_utc()))
                                time.sleep(15)
                        core.dump(status,dict(state='running',pid=os.getpid(),dataset=dataset,seed=seed,target=target,stage='single_task',updated_at=_utc()))
                        folder=a.output/dataset/str(seed)/target
                        reused=False
                        if dataset=='kuairand' and target in ['follow','forward'] and a.legacy_kuairand:
                            model=build_single(spec,seed+spec['tasks'].index(target)+2,dc)
                            reused=reuse_single(spec,seed,target,model,folder,a.legacy_kuairand,dc)
                            del model
                        if not reused:fit_single(spec,seed,target,folder,dc,status)
                    count=report(a.output,cfg,a.comparison_root)
                    print(json.dumps(dict(event='ST_SEED_DONE',dataset=dataset,seed=seed,completed=count,at=_utc())),flush=True)
                    if core.DEVICE.type=='cuda':torch.cuda.empty_cache()
            count=report(a.output,cfg,a.comparison_root)
            if count!=55:raise ValueError(f'ST matrix incomplete: {count}/55')
            core.dump(status,dict(state='complete',pid=None,completed=count,expected=55,finished_at=_utc()))
        except BaseException as exc:
            core.dump(status,dict(state='paused' if isinstance(exc,KeyboardInterrupt) else 'failed',pid=None,error=str(exc),updated_at=_utc()))
            raise


if __name__=='__main__':main()
