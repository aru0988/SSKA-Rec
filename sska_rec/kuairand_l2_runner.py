"""Normalized embedding-L2 KuaiRand experiment: dependency-aware Linux fork queue."""
import argparse
import copy
import json
from dataclasses import asdict, dataclass
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

# Never discover/initialize CUDA in the fork parent.
os.environ['SSKA_DEVICE'] = 'cpu'
from . import core
import torch

from .locking import ExclusiveLock
from .scheme_common import bind_contract, digest, read_json, source_hashes
from .training import _utc

PROJECT = Path(__file__).resolve().parents[1]
READOUTS = ('conditional_readout', 'head_only')
STAGE2 = ('mpt', 'tri_frozen', 'random_lora', 'tri_lora')


@dataclass(frozen=True)
class Job:
    stage: str
    seed: int
    target: str = ''
    variant: str = ''

    @property
    def key(self):
        return '/'.join(str(x) for x in (self.stage, self.seed, self.target, self.variant) if x != '')

    @property
    def dependency(self):
        return None if self.stage == 'source' else f'source/{self.seed}'

    @property
    def units(self):
        return 2 if self.stage == 'source' else 1


def jobs(cfg):
    dc = cfg['datasets']['kuairand']
    sources = [Job('source', seed) for seed in dc['seeds']]
    children = [Job(stage, seed, target, variant)
                for seed in dc['seeds']
                for stage, targets, variants in [('readout', dc['initial'], READOUTS),
                                                  ('stage2', dc['targets'], STAGE2)]
                for target in targets for variant in variants]
    return sources + children


def leaf(output, job):
    root = Path(output)/'kuairand'/str(job.seed)
    return root/'source' if job.stage == 'source' else root/('old' if job.stage == 'readout' else 'new')/job.target/job.variant


class JobQueue:
    """Explicit states prevent duplicate workers and release only verified sources."""
    def __init__(self, registered):
        self.registered = list(registered)
        self.pending = {j.key: j for j in self.registered}
        self.running, self.completed, self.failures, self.blocked = {}, set(), {}, {}

    def dependencies(self, job):
        result = [] if job.dependency is None else [job.dependency]
        if job.stage == 'stage2':
            result.extend(j.key for j in self.registered if j.seed == job.seed
                          and j.stage == 'readout' and j.variant == 'conditional_readout')
        return result

    def ready(self):
        return [j for j in self.pending.values() if all(key in self.completed for key in self.dependencies(j))]

    def start(self, job):
        if job not in self.ready():
            raise ValueError(f'Job is already running or dependency is incomplete: {job.key}')
        del self.pending[job.key]
        self.running[job.key] = job

    def take(self):
        ready = self.ready()
        if not ready:
            return None
        self.start(ready[0])
        return ready[0]

    def finish(self, job, exitcode):
        del self.running[job.key]
        if exitcode == 0:
            self.completed.add(job.key)
            return
        self.failures[job.key] = dict(job=asdict(job), exitcode=exitcode)
        for key, dependent in list(self.pending.items()):
            if job.key in self.dependencies(dependent):
                self.blocked[key] = dict(job=asdict(dependent), failed_dependency=job.key)
                del self.pending[key]


def execution_identity(cfg, dataset_root, cache_root, references):
    references = Path(references).resolve()
    return dict(config=cfg, code=source_hashes(), dataset_root=str(Path(dataset_root).resolve()),
                cache_root=str(Path(cache_root).resolve()), references=str(references),
                references_sha256=core.file_hash(references))


def _job_paths(output, job):
    root = Path(output)/'jobs'/job.key
    return root.with_suffix('.log'), root.with_suffix('.status.json'), root.with_suffix('.contract.json')


def _source_ready(root):
    return all((root/path).is_file() for path in ('source-ready.json',
               'source/base/COMMITTED.json', 'source/ssl/COMMITTED.json'))


def verify_readout(base, u, spec, caches, position, folder, cfg, seed, variant):
    """Verify a committed head without rewriting its immutable manifest."""
    from .scheme_common import verify_stage
    from .scheme_fitting import _contract, _verify_state, _verify_raw
    from .scheme_models import ConditionalReadout
    from . import paper_readouts
    verify_stage(folder, ['model.pt', 'metrics.json', 'predictions.npz', 'contract.json'])
    expected = _contract(spec, caches, cfg, seed, 'conditional_readout',
                         dict(base_hash=core.model_hash(base), u_hash=core.model_hash(u)),
                         spec['initial'][position])
    if variant == 'head_only':
        expected.update(version='paper-readouts-v1', variant='h_conditional')
        expected['code']['paper_readouts.py'] = core.file_hash(paper_readouts.__file__)
        expected['settings']['representation'] = 'ConditionalReadout(h,h,p)'
        head = paper_readouts.make_control_head(spec, seed, position, 'h_conditional')
        stored_variant = 'h_conditional'
    else:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed+position+2)
            head = ConditionalReadout(spec['rdim'], rank=4).to(core.DEVICE)
        stored_variant = 'conditional_readout'
    prior = read_json(Path(folder)/'metrics.json')
    if (read_json(Path(folder)/'contract.json') != expected or prior.get('contract') != expected
            or (prior.get('paired_seed'), prior.get('target'), prior.get('variant')) !=
               (seed, spec['initial'][position], stored_variant)):
        raise ValueError('Completed readout protocol identity mismatch')
    _verify_state(folder, head, prior, 'model.pt')
    _verify_raw(folder, spec, caches, position, prior)


def run_job(spec, job, output, dc, status_path):
    """Verify reused leaves through the same source/model APIs used for training."""
    from .kuairand_l2_sources import prepare_source, load_source, freeze_guard
    root = Path(output)/'kuairand'/str(job.seed)
    _, _, contract_path = _job_paths(output, job)
    if job.stage == 'source':
        bind_contract(contract_path, dict(job=asdict(job), data=spec['data_signature'],
                      config=dc, embedding_l2_normalization='parameter_mean', embedding_l2=.006))
        if _source_ready(root):
            base, caches, envs, u = load_source(spec, job.seed, root, dc)
        else:
            base, caches, envs, u = prepare_source(spec, job.seed, root, dc, status_path)
        freeze_guard(base, caches, envs, u, spec)()
        if not _source_ready(root):
            raise ValueError('Source preparation did not commit its complete receipt')
        return
    base, caches, envs, u = load_source(spec, job.seed, root, dc)
    guard = freeze_guard(base, caches, envs, u, spec, readout_root=root/'old')
    guard()
    destination = leaf(output, job)
    contract = dict(job=asdict(job), config=dc,
                    data=spec.get('scheme_data_identity', spec['data_signature']),
                    cache=spec.get('scheme_cache_identity'),
                    base_hash=core.model_hash(base), u_hash=core.model_hash(u),
                    source_ready_sha256=core.file_hash(root/'source-ready.json'))
    if job.stage == 'readout':
        bind_contract(contract_path, contract)
        position = spec['initial'].index(job.target)
        if (destination/'COMMITTED.json').exists():
            verify_readout(base, u, spec, caches, position, destination, dc, job.seed, job.variant)
        elif job.variant == 'conditional_readout':
            from .scheme_fitting import fit_readout
            fit_readout(base, u, spec, caches, position, destination, dc, job.seed, status_path)
        elif job.variant == 'head_only':
            from .paper_readouts import fit_control
            fit_control(base, u, spec, caches, position, destination, dc, job.seed,
                        'h_conditional', status_path)
        else:
            raise ValueError('Unregistered old-task readout')
    elif job.stage == 'stage2':
        from .paper_readouts import make_random_u
        from .stage2_models import build_model
        from .scheme1_runner import _verify_completed
        from .tri_training import train_one
        position = dc['targets'].index(job.target)
        actual_seed = job.seed + position + dc['head_seed_offset']
        core.seed_all(actual_seed)
        source = make_random_u(spec, u, job.seed) if job.variant == 'random_lora' else u
        model_variant = 'tri_lora' if job.variant == 'random_lora' else job.variant
        model = build_model(spec, source, actual_seed-3, model_variant).to(core.DEVICE)
        contract.update(model_seed=actual_seed, initial_state_hash=core.model_hash(model),
                        model_variant=model_variant, random_u_hash=core.model_hash(source)
                        if job.variant == 'random_lora' else None)
        bind_contract(contract_path, contract)
        metadata = dict(base_hash=contract['base_hash'], u_hash=contract['u_hash'],
                        job_contract_sha256=core.file_hash(contract_path), contract=contract)
        if (destination/'COMMITTED.json').exists():
            prior = read_json(destination/'metrics.json')
            if (any(prior.get(key) != value for key, value in metadata.items())
                    or not (destination/'contract.json').is_file()
                    or read_json(destination/'contract.json') != contract):
                raise ValueError('Completed stage source provenance mismatch')
            _verify_completed(model, destination, spec, caches, job.seed, job.target, job.variant)
        train_one(model, spec, caches, envs, job.target, destination, dc, actual_seed,
                  job.variant, guard, status_path, paired_seed=job.seed,
                  completion_metadata=metadata)
    else:
        raise ValueError('Unregistered queue stage')
    guard()


def _worker(spec, job, output, dc, threads, device):
    log_path, status_path, _ = _job_paths(output, job)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', buffering=1) as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        try:
            torch.set_num_threads(threads)
            core.DEVICE = torch.device(device)
            with ExclusiveLock(Path(output)/'locks'/(job.key+'.lock')):
                core.dump(status_path, dict(state='running', pid=os.getpid(), job=asdict(job),
                          threads=threads, device=device, started_at=_utc()))
                # Source/cache metadata mutations stay private to this fork child.
                run_job(dict(spec), job, output, dc, status_path)
                core.dump(status_path, dict(state='complete', pid=None, worker_pid=os.getpid(),
                          job=asdict(job), ended_at=_utc()))
        except BaseException as exc:
            traceback.print_exc()
            core.dump(status_path, dict(state='failed', pid=None, worker_pid=os.getpid(),
                      job=asdict(job), error=str(exc), ended_at=_utc()))
            raise


def _health(process, job, output):
    log_path, status_path, _ = _job_paths(output, job)
    result = dict(pid=process.pid, job=asdict(job), log=str(log_path),
                  log_bytes=log_path.stat().st_size if log_path.exists() else 0)
    if status_path.exists():
        result['progress'] = read_json(status_path)
    try:
        import psutil
    except ImportError:
        psutil = None
    try:
        if psutil is None:
            raise OSError('Use Linux proc resource counters')
        observed = psutil.Process(process.pid)
        result.update(rss_bytes=observed.memory_info().rss, cpu_seconds=sum(observed.cpu_times()[:2]),
                      os_threads=observed.num_threads())
    except ((OSError, psutil.Error) if psutil is not None else (OSError,)):
        try:
            status = Path(f'/proc/{process.pid}/status').read_text()
            for line in status.splitlines():
                if line.startswith('VmRSS:'):
                    result['rss_bytes'] = int(line.split()[1])*1024
                elif line.startswith('Threads:'):
                    result['os_threads'] = int(line.split()[1])
        except OSError:
            pass
    return result


def record_health(output, records, workers, threads):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output/'parallel-profile.json'
    previous = read_json(path) if path.exists() else {}
    sample = dict(scheduler_pid=os.getpid(), workers=records, updated_at=_utc())
    with (output/'health-samples.jsonl').open('a', encoding='utf8') as stream:
        stream.write(json.dumps(sample, ensure_ascii=False)+'\n')
    result = dict(configured_workers=workers, threads_per_worker=threads,
                  cpu_count=os.cpu_count(), scheduler_pid=os.getpid(),
                  observed_max_workers=max(previous.get('observed_max_workers', 0), len(records)),
                  observed_max_worker_rss_bytes=max(previous.get('observed_max_worker_rss_bytes', 0),
                                                    sum(r.get('rss_bytes', 0) for r in records)),
                  updated_at=sample['updated_at'])
    if hasattr(os, 'sched_getaffinity'):
        result['affinity_cpus'] = len(os.sched_getaffinity(0))
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemTotal:'):
                result['memory_bytes'] = int(line.split()[1])*1024
    except OSError:
        pass
    core.dump(path, result)
    return result

def parallel(args, cfg):
    if sys.platform != 'linux':
        raise ValueError('Shared admitted data execution requires Linux fork workers')
    if args.workers < 1 or args.threads < 1:
        raise ValueError('Positive worker/thread counts required')
    from .tri_data import build_spec
    from .kuairand_l2_report import report
    with ExclusiveLock(args.output/'scheduler.lock'):
        identity = execution_identity(cfg, args.dataset_root, args.cache_root, args.references)
        bind_contract(args.output/'identity.json', identity)
        torch.set_num_threads(1)
        dc = copy.deepcopy(cfg['datasets']['kuairand'])
        dc['execution_identity'] = digest(identity)
        core.dump(args.output/'status.json', dict(state='running', pid=os.getpid(),
                  stage='data-admission', updated_at=_utc()))
        spec = build_spec('kuairand', args.dataset_root, args.cache_root, args.output/'data', dc)
        spec.update(embedding_l2_normalization='parameter_mean', embedding_l2=.006)
        bind_contract(args.output/'data-contract.json', dict(data=spec['data_signature'],
                      embedding_l2_normalization='parameter_mean', embedding_l2=.006,
                      execution_identity=dc['execution_identity']))
        queue = JobQueue(jobs(cfg))
        running = {}
        context = mp.get_context('fork')
        base_seen, last_report = set(), 0.
        try:
            while queue.pending or running:
                while len(running) < args.workers:
                    job = queue.take()
                    if job is None:
                        break
                    process = context.Process(target=_worker, args=(spec, job, args.output, dc,
                                              args.threads, args.device))
                    process.start()
                    running[job.key] = (process, job)
                changed = False
                for key, (process, job) in list(running.items()):
                    if not process.is_alive():
                        process.join()
                        queue.finish(job, process.exitcode)
                        del running[key]
                        changed = True
                current_bases = {seed for seed in dc['seeds'] if (args.output/'kuairand'/str(seed)/
                                  'source/base/COMMITTED.json').exists()}
                now = time.monotonic()
                health_records = [_health(p,j,args.output) for p,j in running.values()]
                if changed or current_bases != base_seen or now-last_report >= 30:
                    report(args.output, cfg, args.references)
                    record_health(args.output, health_records, args.workers, args.threads)
                    base_seen, last_report = current_bases, now
                core.dump(args.output/'status.json', dict(state='running', pid=os.getpid(),
                          stage='training', completed_jobs=len(queue.completed), expected_jobs=len(jobs(cfg)),
                          completed_base_sources=sorted(base_seen),
                          running_jobs=health_records,
                          pending_jobs=len(queue.pending), blocked_jobs=queue.blocked, failures=queue.failures,
                          concurrency=dict(workers=args.workers, threads_per_worker=args.threads), updated_at=_utc()))
                if queue.pending and not running and not queue.ready():
                    raise RuntimeError('Queue has unresolved source dependencies')
                if running:
                    time.sleep(2)
        except BaseException as exc:
            for process, _ in running.values():
                if process.is_alive():
                    process.terminate()
                process.join()
            core.dump(args.output/'status.json', dict(state='failed', pid=None, stage='scheduler',
                      error=str(exc), failures=queue.failures, blocked_jobs=queue.blocked, updated_at=_utc()))
            raise
        result = report(args.output, cfg, args.references)
        complete = result['completed_units'] == result['expected_units'] == 70
        core.dump(args.output/'status.json', dict(state='failed' if queue.failures else
                  ('complete' if complete else 'partial'), pid=None, completed_jobs=len(queue.completed),
                  expected_jobs=65, completed_units=result['completed_units'], expected_units=70,
                  failures=queue.failures, blocked_jobs=queue.blocked,
                  concurrency=dict(workers=args.workers, threads_per_worker=args.threads), finished_at=_utc()))
        if queue.failures or not complete:
            raise RuntimeError('KuaiRand queue incomplete; inspect status.json and per-job logs')
        return result


def main(argv=None):
    from .kuairand_l2_sources import default_config, validate_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT/'configs/kuairand_normalized_l2_20261120.json')
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args(argv)
    cfg = read_json(args.config) if args.config.exists() else default_config()
    validate_config(cfg)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        from .kuairand_l2_report import report
        return report(args.output, cfg, args.references)
    if args.dataset_root is None or args.cache_root is None:
        parser.error('dataset-root and cache-root are required for training')
    return parallel(args, cfg)


if __name__ == '__main__':
    main()



