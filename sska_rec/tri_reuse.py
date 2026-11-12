"""Reference independently trained historical KuaiRand MPT artifacts in place."""

import json
from pathlib import Path

import numpy as np
import torch

from . import core
from .training import _commit, _rowids


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _verify(folder, required):
    """Check committed bytes without repairing or writing the owning folder."""
    folder = Path(folder).resolve()
    manifest = folder / 'COMMITTED.json'
    if not manifest.is_file():
        raise ValueError(f'Missing committed reuse artifacts: {folder}')
    entries = _read(manifest)
    if not set(required).issubset(entries):
        raise ValueError(f'Incomplete committed reuse artifacts: {folder}')
    for name, digest in entries.items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder) or not path.is_file() or core.file_hash(path) != digest:
            raise ValueError(f'Reused artifact integrity failure: {path}')
    return dict(path=str(folder), committed_sha256=core.file_hash(manifest), files=entries)


def artifact_folder(folder):
    """Resolve raw predictions/weights, rechecking local and external commits."""
    folder = Path(folder).resolve()
    reused = folder / 'REUSED.json'
    if not reused.exists():
        return folder
    _verify(folder, ['metrics.json', 'REUSED.json'])
    receipt = _read(reused)
    external = _verify(receipt['artifacts']['path'], ['model.pt', 'predictions.npz', 'metrics.json'])
    if external != receipt['artifacts']:
        raise ValueError('External reused artifact integrity identity changed')
    base = _verify(receipt['base']['path'], ['base.pt', 'metrics.json'])
    if base != receipt['base']:
        raise ValueError('External reused base integrity identity changed')
    return Path(external['path'])


def reuse_mpt(spec, seed, target, init_seed, model, source_receipt,
              destination, legacy_root):
    """Verify an old independent MPT and commit two small local reference files.

    ``init_seed`` is the actual NewTask RNG seed, including the dataset offset.
    The caller supplies an already audited source receipt and a fresh model.
    A present candidate that fails any check raises instead of retraining.
    """
    if spec['dataset'] != 'kuairand' or legacy_root is None:
        return False
    if target not in ('follow', 'forward'):
        raise ValueError(f'Unknown historical KuaiRand target: {target}')
    pos = ('follow', 'forward').index(target)
    expected_seed = seed + pos + 4
    if init_seed != expected_seed:
        raise ValueError('Historical MPT initialization seed mismatch')
    if spec['batch'] != 512 or spec['lr'] != .001:
        raise ValueError('Historical MPT batch/learning-rate protocol mismatch')
    if getattr(model, 'variant', None) != 'mpt':
        raise ValueError('Historical MPT reuse requires an mpt model')

    destination = Path(destination).resolve()
    if (destination / 'COMMITTED.json').exists() and not (destination / 'REUSED.json').exists():
        return False
    candidate = Path(legacy_root).resolve()
    if not (candidate / 'source/base').is_dir():
        candidate = candidate / str(seed)
    base_folder = candidate / 'source/base'
    external = candidate / 'no_u' / f'stage{pos + 1}'
    base_commit = _verify(base_folder, ['base.pt', 'metrics.json'])
    artifacts = _verify(external, ['model.pt', 'predictions.npz', 'metrics.json'])
    base_metrics = _read(base_folder / 'metrics.json')
    if source_receipt['model_hashes']['base'] != base_metrics['model_hash']:
        raise ValueError('Historical MPT frozen base hash mismatch')
    old = _read(external / 'metrics.json')
    if old['seed'] != expected_seed:
        raise ValueError('Recorded historical MPT seed mismatch')
    if old['epochs'] > 30 or old['selected_epoch'] > old['epochs']:
        raise ValueError('Historical MPT epoch budget mismatch')
    if old['stop_reason'] not in ('patience', 'budget') or (old['stop_reason'] == 'budget' and old['epochs'] != 30):
        raise ValueError('Historical MPT stop reason/epoch budget mismatch')

    reference = dict(protocol='audited-kuairand-independent-mpt-v1', seed=seed,
                     target=target, init_seed=init_seed, artifacts=artifacts,
                     base=base_commit, frozen_base_hash=base_metrics['model_hash'])
    if (destination / 'REUSED.json').exists():
        if (destination / 'COMMITTED.json').exists():
            artifact_folder(destination)
        if _read(destination / 'REUSED.json') != reference:
            raise ValueError('Existing historical MPT reuse identity changed')

    model.load_state_dict(torch.load(external / 'model.pt', map_location=core.DEVICE, weights_only=True), strict=True)
    model.eval()
    if core.model_hash(model) != old['model_hash']:
        raise ValueError('Historical MPT model hash mismatch')
    base_state = torch.load(base_folder / 'base.pt', map_location=core.DEVICE, weights_only=True)
    envs = [value.detach() for value in base_state['env_embedding_network.weight']]
    cache_folder = Path(source_receipt['stages']['cache']['path'])
    target_idx = spec['tasks'].index(target)
    with np.load(external / 'predictions.npz', allow_pickle=False) as raw:
        for split in ['val', 'test']:
            rows = spec['sets'][split]
            y, p, ids = [raw[split + '_' + key] for key in ('y', 'p', 'rowid')]
            if not np.array_equal(y, rows.y[:, target_idx]) or not np.array_equal(ids, _rowids(rows)):
                raise ValueError(f'Historical MPT label/row alignment mismatch: {split}')
            cache = {name: torch.from_numpy(np.load(cache_folder / f'{split}-{name}.npy', mmap_mode='c'))
                     for name in ('x', 'g', 's0', 's1')}
            cache['y'] = torch.from_numpy(np.asarray(rows.y))
            actual = core.probabilities(model, cache, envs, spec)
            if p.dtype != np.float32 or not np.array_equal(actual, p):
                raise ValueError(f'Historical MPT prediction mismatch: {split}')
            measures = core.measures(y, p)
            if any(abs(measures[key] - old[split][key]) > 1e-10 for key in measures):
                raise ValueError(f'Historical MPT raw metric mismatch: {split}')

    if (destination / 'COMMITTED.json').exists():
        return True
    destination.mkdir(parents=True, exist_ok=True)
    if set(p.name for p in destination.iterdir()) - {'metrics.json', 'REUSED.json'}:
        raise ValueError('Historical MPT reuse destination contains unrelated artifacts')
    metrics = dict(old, variant='mpt', target=target, paired_seed=seed,
                   seed=init_seed, reused=True, ready_for_commit=True,
                   learning_rate=.001, patience=5, validation_fallback=False,
                   exclusive_parameters=model.exclusive_parameters,
                   inference_parameters=model.inference_parameters,
                   u_update_parameters=0, cost_measurement_source='historical-continual-20261030',
                   cost_measurement_scope='Original historical training time and peak memory; no current cache residency normalization',
                   history_unavailable_fields=['train_bce', 'train_regularization', 'data_wait_seconds', 'samples_per_second'],
                   frozen_original_paths=dict(base_hash=base_metrics['model_hash'], source_model_hashes=source_receipt['model_hashes']),
                   external_artifacts=str(external))
    core.dump(destination / 'metrics.json', metrics)
    core.dump(destination / 'REUSED.json', reference)
    _commit(core, destination)
    return True
