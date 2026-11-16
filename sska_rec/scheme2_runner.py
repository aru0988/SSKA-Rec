"""Scheme2: original-task frozen-U readout only; no new-task stage is scheduled."""
from pathlib import Path

from . import core
from .locking import ExclusiveLock
from .scheme_common import bind_execution, parser, read_json, selected_jobs, short_lock, status, validate_config
from .scheme_report import report


def run_seed(spec, cfg, seed, folder, source_root, status_path=None):
    from .scheme_reuse import load_sources
    from .scheme_fitting import fit_readout
    folder = Path(folder)
    status_path = Path(status_path or folder / 'status.json')
    base, caches, envs, u, receipt = load_sources(spec, seed, cfg, source_root, require_u=True)
    core.dump(folder / 'reuse-receipt.json', receipt)
    results = {}
    for pos, target in enumerate(spec['initial']):
        status(status_path, 'running', scheme=2, dataset=spec['dataset'], seed=seed,
               target=target, stage='old-task-readout')
        results[target] = fit_readout(base, u, spec, caches, pos, folder / 'old' / target,
                                      cfg, seed, status_path)
    status(status_path, 'complete', scheme=2, dataset=spec['dataset'], seed=seed,
           completed=len(results), expected=len(spec['initial']))
    core.dump(folder / 'DONE.json', dict(scheme=2, completed=list(results),
                                       execution_identity=cfg.get('execution_identity')))
    return results


def main(argv=None):
    import torch
    args = parser(2).parse_args(argv)
    cfg = read_json(args.config)
    validate_config(cfg, 2)
    if args.report_only:
        return report(args.output, cfg, 2)
    if args.source_root is None or args.dataset_root is None:
        raise ValueError('--source-root and --dataset-root are required for training')
    if args.threads < 1:
        raise ValueError('Positive CPU thread count required')
    core.DEVICE = torch.device(args.device)
    if core.DEVICE.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(args.threads)
    identity = bind_execution(args.output, cfg, args.source_root, args.dataset_root)
    from .tri_data import build_spec
    for dataset, seed in selected_jobs(cfg, args.dataset, args.seed):
        folder = args.output / dataset / str(seed)
        with ExclusiveLock(folder / 'worker.lock'):
            try:
                status(folder / 'status.json', 'running', scheme=2, dataset=dataset, seed=seed, stage='data-admission')
                dc = dict(cfg['datasets'][dataset], execution_identity=identity)
                with short_lock(args.cache_root / f'.scheme-{dataset}-admission.lock', timeout=86400):
                    spec = build_spec(dataset, args.dataset_root, args.cache_root, folder / 'data', dc)
                run_seed(spec, dc, seed, folder, args.source_root)
            except BaseException as exc:
                status(folder / 'status.json', 'paused' if isinstance(exc, KeyboardInterrupt) else 'failed',
                       scheme=2, dataset=dataset, seed=seed, error=str(exc))
                raise
    return report(args.output, cfg, 2)


if __name__ == '__main__':
    main()
