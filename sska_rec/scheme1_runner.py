"""Scheme1: new complementary U sources and task-conditioned three-way routing."""
from pathlib import Path

import numpy as np
import torch

from . import core
from .locking import ExclusiveLock
from .scheme_common import (bind_contract, bind_execution, parser, read_json,
                            selected_jobs, short_lock, source_hashes, status, validate_config, verify_stage)
from .scheme_report import report


def _verify_completed(model, folder, spec, caches, seed, target, variant):
    verify_stage(folder, ['model.pt', 'metrics.json', 'predictions.npz'])
    m = read_json(Path(folder) / 'metrics.json')
    if (m['paired_seed'], m['target'], m['variant']) != (seed, target, variant):
        raise ValueError('Completed stage pairing differs')
    model.load_state_dict(torch.load(Path(folder) / 'model.pt', map_location=core.DEVICE, weights_only=True), strict=True)
    if core.model_hash(model) != m['model_hash']:
        raise ValueError('Completed model state hash differs')
    index = spec['tasks'].index(target)
    with np.load(Path(folder) / 'predictions.npz', allow_pickle=False) as raw:
        for split in ['val', 'test']:
            y, p, ids = [raw[f'{split}_{k}'] for k in ['y', 'p', 'rowid']]
            if not np.array_equal(y, caches[split]['y'][:, index].numpy()) or not np.array_equal(ids, spec['sets'][split].row_ids):
                raise ValueError('Completed raw labels/rowids differ')
            measures = core.measures(y, p)
            if any(abs(measures[k] - m[split][k]) > 1e-10 for k in measures):
                raise ValueError('Completed raw metric differs')


def run_seed(spec, cfg, seed, folder, source_root, variants=None, pretrain_only=False, status_path=None):
    from .scheme_reuse import load_sources
    from .scheme_fitting import fit_decoder, fit_complementary_u
    from .scheme_models import SCHEME1_VARIANTS, build_scheme1_model
    from .scheme_controls import reuse_control
    from .tri_training import train_one
    folder = Path(folder)
    status_path = Path(status_path or folder / 'status.json')
    variants = list(variants or SCHEME1_VARIANTS)
    base, caches, envs, _, receipt = load_sources(spec, seed, cfg, source_root, require_u=False)
    core.dump(folder / 'reuse-receipt.json', receipt)
    decoder, decoder_metrics = fit_decoder(spec, caches, folder / 'source/gs_decoder', cfg, seed, status_path)
    u, u_metrics = fit_complementary_u(spec, caches, decoder, (decoder.mean, decoder.scale),
                                      folder / 'source/complementary_u', cfg, seed, status_path)
    u.eval().requires_grad_(False)
    core.dump(folder / 'complementary-source.json', dict(base_source=receipt,
        decoder_model_hash=decoder_metrics['model_hash'], u_model_hash=u_metrics['model_hash']))
    base_hash, u_hash = core.model_hash(base), core.model_hash(u)

    def guard():
        if core.model_hash(base) != base_hash or core.model_hash(u) != u_hash:
            raise ValueError('Frozen original B/U source changed')
        if any(p.requires_grad for module in [base, u] for p in module.parameters()):
            raise ValueError('Original B/U source is trainable')
        return dict(passed=True, base_hash=base_hash, u_hash=u_hash,
                    original_path_proof='unchanged frozen MPT weights and eval graph; no U readout')

    guard()
    if pretrain_only:
        status(status_path, 'pretrain_complete', scheme=1, seed=seed, dataset=spec['dataset'])
        return {}
    results = {}
    relevant = {k:v for k,v in source_hashes().items() if k in
                ['scheme_models.py', 'tri_training.py', 'stage2_models.py', 'peft_models.py', 'backbone.py', 'core.py']}
    for pos, target in enumerate(spec['tasks'][2:]):
        for variant in variants:
            if variant not in SCHEME1_VARIANTS:
                raise ValueError('Unknown scheme1 variant')
            actual_seed = seed + pos + cfg.get('head_seed_offset', 3)
            core.seed_all(actual_seed)
            model = build_scheme1_model(spec, u, actual_seed - 3, variant).to(core.DEVICE)
            destination = folder / target / variant
            contract = dict(scheme=1, seed=seed, model_seed=actual_seed, target=target, variant=variant,
                config=cfg, base_hash=base_hash, u_hash=u_hash if variant != 'mpt' else None,
                data=spec.get('scheme_data_identity', spec.get('data_signature')),
                code=relevant, initial_state_hash=core.model_hash(model))
            bind_contract(folder / target / f'{variant}-contract.json', contract)
            status(status_path, 'running', scheme=1, dataset=spec['dataset'], seed=seed,
                   target=target, variant=variant, stage='new-task-adaptation')
            reused = variant == 'mpt' and reuse_control(model, spec, caches, envs, seed, target, cfg,
                destination, receipt['source_folder'], base_hash, guard)
            if not reused:
                if (destination / 'COMMITTED.json').exists():
                    _verify_completed(model, destination, spec, caches, seed, target, variant)
                results[target, variant] = train_one(model, spec, caches, envs, target, destination, cfg,
                    actual_seed, variant, guard, status_path, paired_seed=seed)
            else:
                results[target, variant] = read_json(destination / 'metrics.json')
            guard()
            del model
    status(status_path, 'complete', scheme=1, dataset=spec['dataset'], seed=seed,
           completed=len(results), expected=len(spec['tasks'][2:]) * len(variants))
    core.dump(folder / 'DONE.json', dict(scheme=1, completed=len(results),
                                       execution_identity=cfg.get('execution_identity')))
    return results


def main(argv=None):
    args = parser(1).parse_args(argv)
    cfg = read_json(args.config)
    validate_config(cfg, 1)
    if args.report_only:
        return report(args.output, cfg, 1)
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
                status(folder / 'status.json', 'running', scheme=1, dataset=dataset, seed=seed, stage='data-admission')
                dc = dict(cfg['datasets'][dataset], execution_identity=identity,
                          redundancy_weight=cfg['redundancy_weight'], variance_weight=cfg['variance_weight'])
                with short_lock(args.cache_root / f'.scheme-{dataset}-admission.lock', timeout=86400):
                    spec = build_spec(dataset, args.dataset_root, args.cache_root, folder / 'data', dc)
                run_seed(spec, dc, seed, folder, args.source_root, cfg['variants'], args.pretrain_only)
            except BaseException as exc:
                status(folder / 'status.json', 'paused' if isinstance(exc, KeyboardInterrupt) else 'failed',
                       scheme=1, dataset=dataset, seed=seed, error=str(exc))
                raise
    return report(args.output, cfg, 1)


if __name__ == '__main__':
    main()
