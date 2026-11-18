"""Admit data once, then run independent table2 models in a bounded Linux fork pool."""
import argparse
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

# core's default GPU discovery must not initialize the CUDA driver in the fork parent.
os.environ['SSKA_DEVICE'] = 'cpu'
from . import core
import numpy as np
import torch

from .baseline_fit import code_hashes, fit_baseline, protocol
from .locking import ExclusiveLock
from .scheme_common import bind_contract, read_json, verify_stage
from .training import _utc

PROJECT = Path(__file__).resolve().parents[1]
VARIANTS = ('sharedbottom', 'mmoe', 'ple', 'adatt')
LABELS = dict(sharedbottom='SharedBottom', mmoe='MMoE', ple='PLE', adatt='AdaTT')
_VERIFIED_STAGES = {}


def default_config():
    previous = read_json(PROJECT/'configs/tri_three_datasets.json')
    return dict(version='table2-initial-baselines-v1-20261118',
                datasets={k: copy.deepcopy(previous['datasets'][k]) for k in ('census', 'aliccp')},
                order=['census', 'aliccp'], variants=list(VARIANTS),
                training=dict(epochs=30, patience=5, embedding_l2=.006, dnn_l2=3e-5),
                execution=dict(workers=4, threads_per_worker=2),
                selection='mean-initial-task-val-auc-earliest-best')


def validate_config(cfg):
    if cfg != default_config():
        raise ValueError('Registered table2 experiment protocol changed')


def jobs(cfg, dataset=None, seed=None, variant=None):
    # Keep each dataset/seed's four architectures adjacent for immediate parallel dispatch.
    return [(d, s, v) for d in cfg['order'] if dataset is None or d == dataset
            for s in cfg['datasets'][d]['seeds'] if seed is None or s == seed
            for v in cfg['variants'] if variant is None or v == variant]


def leaf(output, dataset, seed, variant):
    return Path(output)/dataset/str(seed)/variant


def report(output, cfg):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records, groups = {}, {}
    for dataset, seed, variant in jobs(cfg):
        folder = leaf(output, dataset, seed, variant)
        if not (folder/'COMMITTED.json').is_file():
            continue
        manifest = read_json(folder/'COMMITTED.json')
        if (not {'metrics.json', 'model.pt', 'predictions.npz'}.issubset(manifest)
                or any(not (folder/name).is_file() for name in manifest)
                or core.file_hash(folder/'metrics.json') != manifest['metrics.json']):
            raise ValueError(f'Incomplete/corrupt table2 result: {folder}')
        snapshot = tuple((name, (folder/name).stat().st_size, (folder/name).stat().st_mtime_ns)
                         for name in sorted([*manifest, 'COMMITTED.json']))
        if _VERIFIED_STAGES.get(str(folder.resolve())) != snapshot:
            verify_stage(folder, ['metrics.json', 'model.pt', 'predictions.npz'])
            _VERIFIED_STAGES[str(folder.resolve())] = snapshot
        value = read_json(folder/'metrics.json')
        if (value['dataset'], value['paired_seed'], value['variant']) != (dataset, seed, variant):
            raise ValueError('Table2 result identity mismatch')
        records[f'{dataset}/{seed}/{variant}'] = value
        for task, metrics in value['test'].items():
            groups.setdefault(f'{dataset}/{variant}/{task}', []).append(metrics['auc'])
    summary = {key: dict(n=len(values), mean=float(np.mean(values)),
                        variance=float(np.var(values, ddof=1)) if len(values)>1 else None,
                        values=values) for key, values in groups.items()}
    result = dict(completed_models=len(records), expected_models=len(jobs(cfg)),
                  summary=summary, records=records, updated_at=_utc())
    columns = [(d,t) for d in cfg['order'] for t in cfg['datasets'][d]['initial']]
    lines = ['# 表2 初始任务预测AUC', '', '均值采用已完成配对seed；括号列出实际seed数。', '',
             '| 模型 | '+' | '.join(d+'/'+t for d,t in columns)+' |',
             '|---|'+'---:|'*len(columns)]
    references = read_json(output/'references.json') if (output/'references.json').exists() else {}
    for variant, label in [('single_task','Single Task'), ('sharedbottom','SharedBottom'),
                           ('mmoe','MMoE'), ('ple','PLE'), ('adatt','AdaTT'),
                           ('mpt','MPT-Rec'), ('sska','SSKA-Rec')]:
        cells = []
        for dataset, task in columns:
            item = summary.get(f'{dataset}/{variant}/{task}')
            if item:
                cells.append(f"{item['mean']:.4f}（n={item['n']}）")
            else:
                item = references.get('rows', {}).get(f'{dataset}/{variant}/{task}')
                cells.append(f"{item['mean']:.4f}（n={item['n']}）" if item else '—')
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    result['references'] = references
    core.dump(output/'summary.json', result)
    (output/'tables.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    return result


def admit(args, cfg):
    from .tri_data import build_spec
    specs = {}
    for dataset in cfg['order']:
        if args.dataset and args.dataset != dataset:
            continue
        core.dump(args.output/'status.json', dict(state='running', stage='data-admission', dataset=dataset, updated_at=_utc()))
        spec = build_spec(dataset, args.dataset_root, args.cache_root,
                          args.output/dataset/'data', cfg['datasets'][dataset])
        previous = args.source_root/dataset/'data/data-signature.json'
        if spec['data_signature'] != read_json(previous):
            raise ValueError(f'Current table2 data differs from existing main experiment: {dataset}')
        core.dump(args.output/dataset/'data/reuse-receipt.json',
                  dict(source=str(previous), source_sha256=core.file_hash(previous),
                       current_sha256=core.file_hash(spec['metadata_path']), matched=True))
        specs[dataset] = spec
    return specs


def _worker(spec, seed, variant, output, cfg, threads, device):
    folder = leaf(output, spec['dataset'], seed, variant)
    log_path = folder.parent/(variant+'.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', buffering=1) as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        try:
            torch.set_num_threads(threads)
            core.DEVICE = torch.device(device)
            with ExclusiveLock(Path(output)/'locks'/spec['dataset']/str(seed)/(variant+'.lock')):
                fit_baseline(spec, seed, variant, folder, cfg['training'], folder.parent/(variant+'-status.json'))
        except BaseException:
            traceback.print_exc()
            raise


def parallel(args, cfg):
    if sys.platform != 'linux':
        raise ValueError('Shared admitted data execution requires Linux fork workers')
    if args.workers < 1 or args.threads < 1:
        raise ValueError('Positive worker/thread counts required')
    with ExclusiveLock(args.output/'scheduler.lock'):
        bind_contract(args.output/'identity.json', dict(config=cfg, code=code_hashes(),
                      source_root=str(args.source_root.resolve()), dataset_root=str(args.dataset_root.resolve()),
                      cache_root=str(args.cache_root.resolve())))
        # The parent never initializes CUDA: children initialize independent GPU contexts.
        torch.set_num_threads(1)
        specs = admit(args, cfg)
        pending, reused = [], 0
        for dataset, seed, variant in jobs(cfg, args.dataset, args.seed, args.variant):
            folder = leaf(args.output, dataset, seed, variant)
            if (folder/'COMMITTED.json').exists():
                verify_stage(folder, ['metrics.json','model.pt','predictions.npz'])
                if read_json(folder/'metrics.json')['protocol'] != protocol(specs[dataset], seed, variant, cfg['training']):
                    raise ValueError('Existing model protocol mismatch')
                reused += 1
            else:
                pending.append((dataset, seed, variant))
        running, failures, finished = [], [], reused
        context = mp.get_context('fork')
        try:
            while pending or running:
                while pending and len(running) < args.workers:
                    dataset, seed, variant = pending.pop(0)
                    process = context.Process(target=_worker, args=(specs[dataset], seed, variant,
                              args.output, cfg, args.threads, args.device))
                    process.start()
                    running.append((process, dataset, seed, variant))
                remaining = []
                changed = False
                for process, dataset, seed, variant in running:
                    if process.is_alive():
                        remaining.append((process,dataset,seed,variant))
                    else:
                        process.join()
                        finished += 1
                        changed = True
                        if process.exitcode:
                            failures.append(dict(dataset=dataset, seed=seed, variant=variant, exitcode=process.exitcode))
                running = remaining
                if changed:
                    report(args.output, cfg)
                core.dump(args.output/'status.json', dict(state='running', pid=os.getpid(),
                          stage='training', finished_jobs=finished, reused_jobs=reused,
                          running_jobs=[dict(pid=p.pid,dataset=d,seed=s,variant=v) for p,d,s,v in running],
                          pending_jobs=len(pending), failures=failures,
                          concurrency=dict(workers=args.workers, threads_per_worker=args.threads), updated_at=_utc()))
                if running:
                    time.sleep(2)
        except BaseException:
            for process, *_ in running:
                if process.is_alive():
                    process.terminate()
                process.join()
            core.dump(args.output/'status.json', dict(state='failed', stage='scheduler', updated_at=_utc()))
            raise
        result = report(args.output,cfg)
        core.dump(args.output/'status.json', dict(state='failed' if failures else
                  ('complete' if result['completed_models'] == result['expected_models'] else 'partial'),
                  pid=None, completed_models=result['completed_models'], expected_models=result['expected_models'],
                  failures=failures, finished_at=_utc()))
        if failures:
            raise RuntimeError(f'{len(failures)} table2 workers failed; inspect model logs')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=PROJECT/'configs/table2_baselines_20261118.json')
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', choices=['census','aliccp'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args(argv)
    cfg = read_json(args.config) if args.config.exists() else default_config()
    validate_config(cfg)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        report(args.output,cfg)
        return
    if any(getattr(args,name) is None for name in ('dataset_root','cache_root','source_root')):
        parser.error('dataset-root, cache-root and source-root are required for training')
    parallel(args,cfg)


if __name__ == '__main__':
    main()
