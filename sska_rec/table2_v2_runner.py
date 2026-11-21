"""Run and independently audit the two-task Table 2 formal experiment."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from math import isfinite
import multiprocessing as mp
import os
from pathlib import Path
from statistics import mean, median, stdev, variance
import subprocess
import time

# Fork workers from a parent that has never initialized CUDA.
os.environ['SSKA_DEVICE'] = 'cpu'
from . import core
import numpy as np
import torch

from .baseline_fit import code_hashes, fit_baseline
from .locking import ExclusiveLock
from .scheme_common import digest, read_json, source_hashes, verify_stage
from .scheme_fitting import fit_readout
from .st_fit import fit_single
from .training import _utc


PILOT_DATASETS = ('census', 'aliccp')
JOINT_VARIANTS = ('sharedbottom', 'mmoe', 'ple', 'adatt')
SPLITS = ('val', 'test')
MODEL_ORDER = ('single_task', *JOINT_VARIANTS, 'mpt', 'sska')
JOB_SECONDS = {
    'census': dict(base=800, source=650, readout=180, single_task=450, baseline=550),
    'aliccp': dict(base=2700, source=2250, readout=950, single_task=2800, baseline=2500),
    'kuairand': dict(base=2700, source=1350, readout=360, single_task=480, baseline=950),
}


@dataclass(frozen=True)
class Job:
    dataset: str
    seed: int
    kind: str
    model: str = ''

    @property
    def key(self):
        return '/'.join(str(part) for part in (self.dataset, self.seed, self.kind, self.model) if part != '')

    def dependencies(self):
        prerequisite = {'source': 'base', 'readout': 'source'}.get(self.kind)
        return (Job(self.dataset, self.seed, prerequisite),) if prerequisite else ()

    def folder(self, output):
        return output / 'formal' / self.dataset / str(self.seed)

    def stages(self, output):
        folder = self.folder(output)
        if self.kind == 'source':
            return (folder / 'source' / 'ssl', folder / 'source-evaluation')
        if self.kind == 'base':
            return (folder / 'source' / 'base',)
        parent = 'baselines' if self.kind == 'baseline' else self.kind
        return (folder / parent / self.model,)

    def timing(self, output):
        return output / 'jobs' / f'{self.key}.timing.json'


def make_jobs(cfg):
    jobs = []
    for dataset in cfg['order']:
        targets = cfg['datasets'][dataset]['initial']
        if len(set(targets)) != 2 or len(targets) != 2:
            raise ValueError(f'{dataset} requires exactly two distinct task labels')
        for seed in cfg['formal_seeds']:
            jobs.extend(Job(dataset, seed, kind, model) for kind, models in (
                ('base', ('',)), ('source', ('',)), ('readout', targets),
                ('single_task', targets), ('baseline', JOINT_VARIANTS)) for model in models)
    if len(jobs) != 150 or len(set(jobs)) != len(jobs):
        raise ValueError('Expected 150 distinct formal training jobs')
    return jobs


def verify_gate(pilot_root, config_hash, signatures=None):
    pilot_root = pilot_root.resolve()
    path = pilot_root / 'decision.json'
    if not path.is_file():
        raise ValueError(f'Pilot decision missing: {path}')
    decision = read_json(path)
    if decision.get('state') != 'pass' or decision.get('cfg_sha256') != config_hash:
        raise ValueError('Pilot decision must pass for the current config')
    expected = {(dataset, mode) for dataset in PILOT_DATASETS for mode in ('sum', 'mean')}
    receipts = decision.get('receipts', ())
    if len(receipts) != len(expected) or {(r['dataset'], r['mode']) for r in receipts} != expected:
        raise ValueError('Pilot decision must name all four validation-only stage receipts')
    for receipt in receipts:
        manifest = (pilot_root / receipt['relative_manifest']).resolve()
        expected_path = pilot_root / receipt['dataset'] / receipt['mode'] / 'source' / 'base' / 'COMMITTED.json'
        if (not manifest.is_relative_to(pilot_root) or manifest.name != 'COMMITTED.json'
                or manifest != expected_path or not manifest.is_file()
                or core.file_hash(manifest) != receipt['manifest_sha256']):
            raise ValueError(f'Pilot stage manifest mismatch: {manifest}')
        verify_stage(manifest.parent, required=('base.pt', 'metrics.json', 'protocol.json'))
    if signatures is not None and decision.get('data_signatures') != signatures:
        raise ValueError('Formal dataset signatures differ from pilot admissions')
    return core.file_hash(path)


def bound_receipt(path, contents):
    if not path.exists():
        core.dump(path, contents)
    elif read_json(path) != contents:
        raise ValueError(f'Formal execution identity changed at {path}. Use a new output directory.')


def committed(job, output, audit=False):
    stages = job.stages(output)
    if not all((path / 'COMMITTED.json').is_file() for path in stages):
        return False
    if job.kind == 'source' and not (job.folder(output) / 'source-ready.json').is_file():
        raise ValueError(f'Missing source readiness receipt: {job.key}')
    if audit:
        if job.kind == 'base':
            verify_stage(stages[0], required=('base.pt', 'metrics.json', 'protocol.json'))
        elif job.kind == 'source':
            verify_stage(stages[0], required=('source.pt', 'metrics.json'))
            verify_stage(stages[1], required=('predictions.npz', 'metrics.json'))
        else:
            verify_stage(stages[0], required=('model.pt', 'predictions.npz', 'metrics.json'))
    return True


def execute(job, spec, cfg, output, device, threads, parallelism):
    from .table2_v2_backbone import base_evaluation, fit_backbone, load_source, prepare_source

    core.DEVICE = torch.device(device)
    torch.set_num_threads(threads)
    # Restore the intended device for locally loaded code as well as the shared core object.
    os.environ['SSKA_DEVICE'] = device
    status_path = output / 'jobs' / f'{job.key}.status.json'
    log_path = output / 'jobs' / f'{job.key}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    folder = job.folder(output)
    stage = job.stages(output)[0]
    started_at = _utc()
    started = time.monotonic()
    with log_path.open('a', encoding='utf8', buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        print(f'{started_at} starting {job.key} pid={os.getpid()}', flush=True)
        core.dump(status_path, dict(state='running', pid=os.getpid(), job=job.key, started_at=started_at))
        if job.kind == 'base':
            fit_backbone(spec, job.seed, folder, 'mean', status_path)
        elif job.kind == 'source':
            prepare_source(spec, job.seed, folder, status_path)
            base_evaluation(spec, job.seed, folder)
        elif job.kind == 'readout':
            base, caches, _envs, u = load_source(spec, job.seed, folder)
            position = spec['initial'].index(job.model)
            fit_readout(base, u, spec, caches, position, stage, cfg, job.seed, status_path)
        elif job.kind == 'single_task':
            fit_single(spec, job.seed, job.model, stage, cfg, status_path)
        else:
            fit_baseline(spec, job.seed, job.model, stage, cfg, status_path)
        if not committed(job, output):
            raise RuntimeError(f'Worker returned without a committed result: {job.key}')
        finished_at = _utc()
        core.dump(job.timing(output), dict(job=job.key, dataset=job.dataset, kind=job.kind,
                  started_at=started_at, finished_at=finished_at, seconds=time.monotonic() - started,
                  launch_parallelism=parallelism, threads=threads, device=device))
        core.dump(status_path, dict(state='complete', job=job.key, finished_at=finished_at))
        print(f'{finished_at} completed {job.key}', flush=True)


def memory_usage(device):
    gib = 1024 ** 3
    ram_paths = (Path('/sys/fs/cgroup/memory.current'), Path('/sys/fs/cgroup/memory/memory.usage_in_bytes'))
    ram, working_ram = None, None
    for path in ram_paths:
        if not path.is_file():
            continue
        ram = int(path.read_text().strip())
        stat_path = path.with_name('memory.stat')
        fields = dict(row.split() for row in stat_path.read_text().splitlines()) if stat_path.is_file() else {}
        # Readout and pretraining touch tens of GB of evictable file-backed mmap pages.
        inactive_file = int(fields.get('total_inactive_file', fields.get('inactive_file', 0)))
        working_ram = max(0, ram - inactive_file) / gib
        ram /= gib
        break
    vram, utilization = None, None
    if device.startswith('cuda'):
        command = ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits']
        try:
            lines = subprocess.check_output(command, text=True, timeout=5).splitlines()
            rows = [tuple(float(value) for value in line.split(',')) for line in lines]
            vram = max(memory for memory, _ in rows) / 1024
            utilization = max(gpu for _, gpu in rows)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return dict(ram_gib=ram, working_ram_gib=working_ram, vram_gib=vram,
                gpu_utilization_percent=utilization)


def job_seconds(job, timings):
    return median(timings.get((job.dataset, job.kind), ()) or [JOB_SECONDS[job.dataset][job.kind]])


def ready_jobs(jobs, pending, done, running, timings, aliccp_io_limit):
    source_io = sum(job.dataset == 'aliccp' and job.kind == 'source' for job in running)
    ready = [job for job in jobs if job in pending and all(dep in done for dep in job.dependencies())
             and not (job.dataset == 'aliccp' and job.kind == 'source' and source_io >= aliccp_io_limit)]

    def priority(job):
        chain = {'base': ('source', 'readout'), 'source': ('readout',)}.get(job.kind, ())
        return job_seconds(job, timings) + sum(job_seconds(Job(job.dataset, job.seed, phase), timings)
                                                for phase in chain)

    return sorted(ready, key=priority, reverse=True)


def run_queue(jobs, specs, cfg, settings, output, device):
    ctx = mp.get_context('fork')
    status_path = output / 'status.json'
    start_workers, max_workers = (settings[name] for name in ('start_workers', 'max_workers'))
    threads, io_limit = (settings[name] for name in ('threads_per_worker', 'max_aliccp_io'))
    done = {job for job in jobs if committed(job, output, audit=True)}
    if any(dep not in done for job in done for dep in job.dependencies()):
        raise ValueError('Committed job is missing a committed prerequisite')
    pending = set(jobs) - done
    running, failure = {}, None
    timings, by_parallelism = defaultdict(list), defaultdict(lambda: defaultdict(list))

    def read_timing(job):
        path = job.timing(output)
        if path.is_file():
            record = read_json(path)
            phase = (job.dataset, job.kind)
            timings[phase].append(record['seconds'])
            by_parallelism[record['launch_parallelism']][phase].append(record['seconds'])

    for job in done:
        read_timing(job)
    target_workers = start_workers
    healthy_samples = 0
    pressure_seconds = None
    while pending or running:
        for job, process in list(running.items()):
            if process.is_alive():
                continue
            process.join()
            del running[job]
            if process.exitcode or not committed(job, output):
                failure = failure or f'{job.key} pid={process.pid} exit={process.exitcode}'
            else:
                done.add(job)
                read_timing(job)
                print(f'{_utc()} {len(done)}/{len(jobs)} complete {job.key}', flush=True)
            process.close()

        resources = memory_usage(device)
        pressured = ((resources['working_ram_gib'] or 0) > 48 or (resources['vram_gib'] or 0) > 20)
        observed_default, observed_max = by_parallelism[start_workers], by_parallelism[max_workers]
        # Compare like phases with actual formal timings. A slow active phase is
        # left running while only the admission cap changes.
        common = set(observed_default) & set(observed_max)
        ratios = [median(observed_max[key]) / median(observed_default[key]) for key in common
                  if len(observed_default[key]) >= 2 and len(observed_max[key]) >= 2]
        learned_slowdown = bool(ratios) and median(ratios) > max_workers / start_workers
        if pressured:
            healthy_samples = 0
            pressure_seconds = pressure_seconds or time.monotonic()
            target_workers = min(start_workers, max(1, len(running) - 1))
        else:
            pressure_seconds = None
            healthy_samples += 1
            ready = ready_jobs(jobs, pending, done, running, timings, io_limit)
            enough_work = len(ready) + len(running) >= max_workers
            target_workers = max_workers if healthy_samples > 3 and enough_work and not learned_slowdown else start_workers

        while not failure and not pressured and len(running) < target_workers:
            ready = ready_jobs(jobs, pending, done, running, timings, io_limit)
            if not ready:
                break
            job = ready[0]
            process = ctx.Process(target=execute, args=(job, specs[job.dataset], cfg, output,
                                  device, threads, target_workers), name=job.key)
            process.start()
            running[job] = process
            pending.remove(job)
            print(f'{_utc()} launched {job.key} pid={process.pid}', flush=True)

        status = dict(state='draining_after_failure' if failure else 'running',
                      pid=os.getpid(), updated_at=_utc(), completed=len(done), total=len(jobs),
                      pending=len(pending), active={job.key: process.pid for job, process in running.items()},
                      admission_limit=target_workers, threads_per_worker=threads,
                      resources=resources, failure=failure)
        core.dump(status_path, status)
        with (output / 'execution-samples.jsonl').open('a', encoding='utf8') as samples:
            samples.write(json.dumps(status, separators=(',', ':')) + '\n')
        if not running:
            if failure:
                raise RuntimeError(f'Formal worker failed: {failure}. See the matching job log.')
            if pending and not pressured:
                raise RuntimeError('No eligible formal job despite pending work')
            if pressure_seconds is not None and time.monotonic() - pressure_seconds > 300:
                raise RuntimeError('Resource pressure prevented any formal admission for five minutes')
        if pending or running:
            time.sleep(10)


def stats(values):
    return dict(n=len(values), mean=mean(values), variance=variance(values), sd=stdev(values))


def paired_summary(values, references):
    deltas = [value - reference for value, reference in zip(values, references, strict=True)]
    return dict(**stats(deltas), positive=sum(value > 0 for value in deltas),
                negative=sum(value < 0 for value in deltas), tie=deltas.count(0), per_seed=deltas)


def verify_raw(folder, spec, targets, *, base_path=None):
    result = read_json(folder / 'metrics.json')
    base = np.load(base_path, allow_pickle=False) if base_path else None
    try:
        with np.load(folder / 'predictions.npz', allow_pickle=False) as raw:
            for split in SPLITS:
                rows = spec['sets'][split]
                y, p = raw[f'{split}_y'], raw[f'{split}_p']
                if not np.array_equal(raw[f'{split}_rowid'], rows.row_ids) or p.shape != y.shape:
                    raise ValueError(f'Raw row IDs or prediction shape mismatch: {folder}/{split}')
                positions = [spec['tasks'].index(target) for target in targets]
                labels = rows.y[:, positions[0]] if len(positions) == 1 else rows.y
                if not np.array_equal(y, labels) or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
                    raise ValueError(f'Raw labels or probabilities mismatch: {folder}/{split}')
                if base is not None:
                    base_p = raw[f'{split}_base_p']
                    if (not np.array_equal(base_p, base[f'{split}_p'][:, positions[0]])
                            or (result['selected_epoch'] == 0 and not np.array_equal(base_p, p))):
                        raise ValueError(f'Readout baseline prediction mismatch: {folder}/{split}')
                for i, target in enumerate(targets):
                    recorded = result[split] if len(positions) == 1 else result[split][target]
                    actual = core.measures(y, p) if len(positions) == 1 else core.measures(y[:, i], p[:, i])
                    if any(not isfinite(recorded[key]) or abs(recorded[key] - value) > 1e-10
                           for key, value in actual.items()):
                        raise ValueError(f'Raw metric mismatch: {folder}/{split}/{target}')
    finally:
        if base is not None:
            base.close()
    return result


def report(output, cfg, jobs, specs):
    done = {job for job in jobs if committed(job, output, audit=True)}
    progress = dict(completed=len(done), planned=len(jobs),
                    datasets={name: sum(job.dataset == name for job in done) for name in cfg['order']})
    if len(done) != len(jobs):
        return progress

    measured, zero, receipt = {}, defaultdict(int), {}
    for job in jobs:
        if job.kind == 'base':
            continue
        spec, paths = specs[job.dataset], job.stages(output)
        variant = {'source': 'mpt', 'readout': 'sska', 'single_task': 'single_task'}.get(job.kind, job.model)
        targets = [job.model] if job.kind in ('readout', 'single_task') else list(spec['initial'])
        reference = job.folder(output) / 'source-evaluation' / 'predictions.npz' if job.kind == 'readout' else None
        result = verify_raw(paths[-1], spec, targets, base_path=reference)
        if result['paired_seed'] != job.seed:
            raise ValueError(f'Recorded paired seed does not match {job.key}')
        if job.kind in ('source', 'readout'):
            source = read_json(job.folder(output) / 'source-ready.json')
            base_path = job.folder(output) / 'source' / 'base'
            ssl_path = job.folder(output) / 'source' / 'ssl'
            if (source['seed'] != job.seed or source['data_sha256'] != digest(spec['data_signature'])
                    or source['base_hash'] != read_json(base_path / 'metrics.json')['model_hash']
                    or source['u_hash'] != read_json(ssl_path / 'metrics.json')['model_hash']
                    or source['base_protocol_sha256'] != core.file_hash(base_path / 'protocol.json')
                    or source['ssl_manifest_sha256'] != core.file_hash(ssl_path / 'COMMITTED.json')):
                raise ValueError(f'SSKA source receipt does not match committed stages: {job.key}')
            if job.kind == 'source' and result['model_hash'] != source['base_hash']:
                raise ValueError(f'MPT prediction does not match the trained backbone: {job.key}')
            if job.kind == 'readout' and (result['base_hash'], result['u_hash']) != (source['base_hash'], source['u_hash']):
                raise ValueError(f'Readout backbone/U source binding mismatch: {job.key}')
        if job.kind == 'readout':
            zero[job.dataset, job.model] += result['selected_epoch'] == 0
        for target in targets:
            measured[job.dataset, target, variant, job.seed] = {
                split: result[split] if len(targets) == 1 else result[split][target] for split in SPLITS}
        receipt[job.key] = [str(path.relative_to(output)) for path in paths]

    compact, full = {}, {}
    for dataset in cfg['order']:
        spec, seeds = specs[dataset], cfg['formal_seeds']
        elements = sum(spec['vocab'].values()) * spec['embed']
        counts = {split: len(rows) for split, rows in spec['sets'].items()}
        entry = dict(rows=counts, input_dim=spec['xdim'], embedding_elements=elements,
                     embedding_l2_per_element=cfg['training']['embedding_l2'] / elements, tasks={})
        compact[dataset] = entry
        full[dataset] = {}
        for task in spec['initial']:
            entries = {model: [measured[dataset, task, model, seed] for seed in seeds] for model in MODEL_ORDER}
            auc = {model: [row['test']['auc'] for row in rows] for model, rows in entries.items()}
            entry['tasks'][task] = {
                model: dict(auc=stats(auc[model]), logloss=stats([row['test']['logloss'] for row in rows]),
                            gain_vs_single_task=mean(auc[model]) - mean(auc['single_task']))
                for model, rows in entries.items()}
            full[dataset][task] = dict(metrics=entries, sska_epoch_zero=zero[dataset, task],
                paired_vs_st={model: paired_summary(auc[model], auc['single_task']) for model in MODEL_ORDER},
                paired_sska_vs_mpt=paired_summary(auc['sska'], auc['mpt']))

    compact_record = dict(state='complete', **progress, generated_at=_utc(), source=str(output), data=compact)
    detailed_record = dict(**compact_record, seeds=cfg['formal_seeds'], details=full, receipts=receipt)
    core.dump(output / 'metrics-mean-variance.json', compact_record)
    core.dump(output / 'summary.json', detailed_record)
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'dataset-root', 'output', 'pilot-root', 'source-root'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, default=Path('cache'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    cfg = read_json(args.config)
    jobs = make_jobs(cfg)
    output = args.output.resolve()
    with ExclusiveLock(output / '.formal.lock'):
        from .table2_v2_data import build_table2_spec

        config_hash = digest(cfg)
        verify_gate(args.pilot_root, config_hash)
        torch.set_num_threads(cfg['execution']['threads_per_worker'])
        specs = {name: build_table2_spec(name, args.dataset_root, args.cache_root,
                                        output / 'data' / name, cfg['datasets'][name], source_root=args.source_root)
                 for name in cfg['order']}
        signatures = {name: digest(spec['data_signature']) for name, spec in specs.items()}
        pilot_hash = verify_gate(args.pilot_root, config_hash, {name: signatures[name] for name in PILOT_DATASETS})
        admission = read_json(args.pilot_root / 'admission.json')
        if admission['cfg_sha256'] != config_hash or admission['data_signatures'] != signatures:
            raise ValueError('Formal dataset signatures differ from the full pilot admission')
        identity = dict(config_sha256=config_hash, config_file_sha256=core.file_hash(args.config),
                        pilot_decision_sha256=pilot_hash, data_signatures=signatures,
                        source_hashes=source_hashes(), vendor_and_baseline_hashes=code_hashes())
        bound_receipt(output / 'identity.json', identity)
        try:
            if not args.report_only:
                run_queue(jobs, specs, cfg['training'] | {'execution_identity': digest(identity)},
                          cfg['execution'], output, args.device)
                core.dump(output / 'status.json', dict(state='auditing', pid=os.getpid(), updated_at=_utc(),
                          completed=len(jobs), total=len(jobs)))
            progress = report(output, cfg, jobs, specs)
        except Exception as error:
            core.dump(output / 'failure.json', dict(error=str(error), at=_utc()))
            raise
        print(progress, flush=True)
        if progress['completed'] == progress['planned']:
            core.dump(output / 'status.json', dict(state='complete', updated_at=_utc(), **progress))


if __name__ == '__main__':
    main()
