"""Resource-bounded parallel dataset/seed jobs, each protected by its own OS lock."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from . import core
from .locking import ExclusiveLock
from .scheme_common import PROJECT, bind_execution, read_json, selected_jobs, status, validate_config, verify_stage
from .scheme_report import report


def build_jobs(cfg, scheme, output, source_root, dataset_root, cache_root,
               device='cuda:0', dataset=None, seed=None, config=None, threads=2):
    config = config or PROJECT / f'configs/scheme{scheme}_three_datasets.json'
    return [[sys.executable, '-m', f'sska_rec.scheme{scheme}_runner', '--config', str(config),
             '--source-root', str(source_root), '--dataset-root', str(dataset_root),
             '--cache-root', str(cache_root), '--output', str(output), '--dataset', name,
             '--seed', str(current), '--device', device, '--threads', str(threads)]
            for name, current in selected_jobs(cfg, dataset, seed)]


def main(argv=None):
    p = argparse.ArgumentParser(description='Parallel independent scheme experiment jobs')
    p.add_argument('--scheme', type=int, choices=[1, 2], required=True)
    p.add_argument('--config', type=Path)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--cache-root', type=Path, default=Path('cache'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--threads-per-worker', type=int, default=2)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dataset', choices=['census', 'kuairand', 'aliccp'])
    p.add_argument('--seed', type=int)
    args = p.parse_args(argv)
    if args.workers < 1 or args.threads_per_worker < 1:
        raise ValueError('Positive worker and thread limits required')
    args.config = args.config or PROJECT / f'configs/scheme{args.scheme}_three_datasets.json'
    cfg = read_json(args.config)
    validate_config(cfg, args.scheme)
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(cfg, args.scheme, args.output, args.source_root, args.dataset_root,
        args.cache_root, args.device, args.dataset, args.seed, args.config, args.threads_per_worker)
    running, failed, finished = [], [], 0
    with ExclusiveLock(args.output / 'scheduler.lock'):
        identity = bind_execution(args.output,cfg,args.source_root,args.dataset_root)
        previous = report(args.output,cfg,args.scheme)
        pending = []
        for cmd in jobs:
            dataset, seed = cmd[cmd.index('--dataset')+1], cmd[cmd.index('--seed')+1]
            leaf = args.output/dataset/seed
            done = leaf/'DONE.json'
            dc = cfg['datasets'][dataset]
            expected = len(dc['targets'])*len(cfg['variants']) if args.scheme==1 else len(dc['initial'])
            count = sum(k.startswith(f'{dataset}/{seed}/') for k in previous['records'])
            if done.exists() and read_json(done).get('execution_identity') == identity and count == expected:
                receipt = read_json(leaf/'reuse-receipt.json')
                for name,proof in receipt['stages'].items():
                    if name == 'cache':
                        if core.file_hash(Path(proof['path'])/'COMMITTED.json') != proof['manifest_sha256']:
                            raise ValueError('Completed job cache owner identity changed')
                    else:
                        checked=verify_stage(proof['path'])
                        if checked['files'] != proof['files']:
                            raise ValueError('Completed job source identity changed')
                finished += 1
            else:
                pending.append(cmd)
        jobs = pending
        try:
            while jobs or running:
                while jobs and len(running) < args.workers:
                    cmd = jobs.pop(0)
                    dataset, seed = cmd[cmd.index('--dataset')+1], cmd[cmd.index('--seed')+1]
                    leaf = args.output / dataset / seed
                    leaf.mkdir(parents=True, exist_ok=True)
                    # An external worker must not be duplicated. OS locks, not stale PID files, decide.
                    with ExclusiveLock(leaf / 'worker.lock'):
                        pass
                    log = (leaf / 'runner.log').open('ab', buffering=0)
                    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads_per_worker),
                               MKL_NUM_THREADS=str(args.threads_per_worker), PYTHONUNBUFFERED='1')
                    process = subprocess.Popen(cmd, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    running.append((process, log, dataset, seed))
                remaining = []
                for process, log, dataset, seed in running:
                    code = process.poll()
                    if code is None:
                        remaining.append((process, log, dataset, seed))
                    else:
                        log.close()
                        finished += 1
                        if code:
                            failed.append(dict(dataset=dataset, seed=seed, returncode=code))
                running = remaining
                status(args.output / 'status.json', 'running', scheme=args.scheme, finished_jobs=finished,
                       running_jobs=[dict(pid=x.pid, dataset=d, seed=s) for x, _, d, s in running],
                       pending_jobs=len(jobs), failures=failed,
                       concurrency=dict(workers=args.workers, threads_per_worker=args.threads_per_worker))
                if running:
                    time.sleep(1)
        except BaseException:
            for process, log, _, _ in running:
                if process.poll() is None:
                    process.terminate()
                process.wait()
                log.close()
            status(args.output / 'status.json', 'paused', scheme=args.scheme, finished_jobs=finished)
            raise
        summary = report(args.output, cfg, args.scheme)
        status(args.output / 'status.json', 'failed' if failed else
               ('complete' if summary['completed'] == summary['expected'] else 'partial'),
               scheme=args.scheme, completed=summary['completed'], expected=summary['expected'], failures=failed)
        if failed:
            raise RuntimeError(f'{len(failed)} jobs failed; inspect their runner.log before resuming')


if __name__ == '__main__':
    main()
