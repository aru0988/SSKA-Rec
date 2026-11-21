"""Run four validation-only L2 checks before the formal Table 2 queue."""
import argparse
import multiprocessing as mp
import os
from pathlib import Path

os.environ.setdefault('SSKA_DEVICE', 'cpu')

import torch

from . import core
from .locking import ExclusiveLock
from .scheme_common import bind_contract, digest, read_json, source_hashes, status, verify_stage
from .table2_v2_backbone import VERSION, fit_backbone
from .table2_v2_data import build_table2_spec


def _train(spec, seed, mode, folder, threads):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'runner.log').open('a', buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        core.DEVICE = torch.device('cuda:0')
        torch.set_num_threads(threads)
        fit_backbone(spec, seed, folder, mode, folder/'history-live.json')


def decide(output, cfg, signatures):
    receipts = []
    pairs = {}
    for dataset in cfg['pilot']['datasets']:
        results = {}
        for mode in ('sum', 'mean'):
            base = output/dataset/mode/'source/base'
            verified = verify_stage(base, ['base.pt', 'metrics.json', 'protocol.json'])
            receipts.append(dict(dataset=dataset, mode=mode,
                                 relative_manifest=(base/'COMMITTED.json').relative_to(output).as_posix(),
                                 manifest_sha256=verified['committed_sha256']))
            results[mode] = read_json(base/'metrics.json')
        if results['sum']['initial_state_hash'] != results['mean']['initial_state_hash']:
            raise ValueError(f'MPT initialization differs across L2 arms for {dataset}')
        tasks = cfg['datasets'][dataset]['initial']
        delta = {t: results['mean']['val'][t]['auc'] - results['sum']['val'][t]['auc'] for t in tasks}
        mean_delta = sum(delta.values()) / len(tasks)
        review = mean_delta <= -cfg['pilot']['mean_margin'] or min(delta.values()) <= -cfg['pilot']['task_margin']
        pairs[dataset] = dict(val={mode: row['val'] for mode, row in results.items()},
                              selected_epochs={mode: row['selected_epoch'] for mode, row in results.items()},
                              mean_delta=mean_delta, task_delta=delta, review=review)
    return dict(state='review' if any(pair['review'] for pair in pairs.values()) else 'pass',
                cfg_sha256=digest(cfg), data_signatures={d: signatures[d] for d in pairs},
                pairs=pairs, receipts=receipts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'dataset-root', 'cache-root', 'output', 'source-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--admit-only', action='store_true')
    args = parser.parse_args()
    cfg = read_json(args.config)
    expected = {'census': (127, ['income', 'marital']), 'aliccp': (80, ['CTR', 'CVR']),
                'kuairand': (17, ['click', 'like'])}
    if (cfg['version'] != VERSION or set(cfg['order']) != set(expected)
            or cfg['pilot_seed'] != 1501773270 or cfg['pilot']['datasets'] != ['census', 'aliccp']
            or cfg['pilot']['workers'] != 4 or cfg['pilot']['mean_margin'] != .001
            or cfg['pilot']['task_margin'] != .002):
        raise ValueError('Unregistered Table 2 pilot')
    torch.set_num_threads(1)
    output = args.output.resolve()
    with ExclusiveLock(output/'.pilot.lock'):
        status(output/'status.json', 'running', phase='data_admission')
        try:
            specs = {}
            for name in cfg['order']:
                spec = build_table2_spec(name, args.dataset_root, args.cache_root,
                                         output/'data'/name, cfg['datasets'][name], args.source_root)
                dim, tasks = expected[name]
                if spec['xdim'] != dim or list(spec['tasks']) != tasks or list(spec['initial']) != tasks:
                    raise ValueError(f'Invalid two-task input protocol for {name}')
                if any(rows.y.shape != (len(rows), 2) for rows in spec['sets'].values()):
                    raise ValueError(f'Invalid two-task label matrix for {name}')
                specs[name] = spec
            signatures = {name: digest(spec['data_signature']) for name, spec in specs.items()}
            bind_contract(output/'admission.json', dict(cfg_sha256=digest(cfg), data_signatures=signatures,
                          dimensions={name: spec['xdim'] for name, spec in specs.items()},
                          tasks={name: list(spec['tasks']) for name, spec in specs.items()},
                          sizes={name: {k: len(rows) for k, rows in spec['sets'].items()}
                                 for name, spec in specs.items()}))
            if args.admit_only:
                status(output/'status.json', 'admitted', phase='data_admission')
                return
            bind_contract(output/'pilot-contract.json', dict(cfg_sha256=digest(cfg), data=signatures,
                                                            source=source_hashes()))
            ctx = mp.get_context('fork')
            processes = {}
            try:
                for name in cfg['pilot']['datasets']:
                    for mode in ('sum', 'mean'):
                        child = ctx.Process(target=_train, name=f'{name}-{mode}',
                                            args=(specs[name], cfg['pilot_seed'], mode, output/name/mode,
                                                  cfg['pilot']['threads_per_worker']))
                        child.start()
                        processes[child.name] = child
                status(output/'status.json', 'running', phase='pilot', workers={k: p.pid for k,p in processes.items()})
                for process in processes.values():
                    process.join()
                errors = {name: p.exitcode for name, p in processes.items() if p.exitcode}
                if errors:
                    raise RuntimeError(f'MPT pilot workers failed: {errors}')
            finally:
                for process in processes.values():
                    if process.is_alive():
                        process.terminate()
                    process.join()
            decision = decide(output, cfg, signatures)
            core.dump(output/'decision.json', decision)
            status(output/'status.json', decision['state'], phase='pilot', pairs=decision['pairs'])
        except BaseException as exc:
            status(output/'status.json', 'failed', error=str(exc))
            raise


if __name__ == '__main__':
    main()
