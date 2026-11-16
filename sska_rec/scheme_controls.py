"""Reuse historical MPT controls only after independent dependency/raw checks."""
from pathlib import Path

import numpy as np
import torch

from . import core
from .scheme_common import read_json, verify_stage
from .scheme_report import resolve_artifacts
from .training import _commit


def _execution(source_folder):
    for root in [source_folder, *source_folder.parents]:
        path = root / 'identity.json'
        if path.is_file():
            value = read_json(path)
            if 'source' in value and 'config' in value:
                return value
    raise ValueError('MPT control has no recorded execution identity')


def reuse_control(model, spec, caches, envs, seed, target, cfg, destination,
                  source_folder, base_hash, guard):
    destination, source_folder = Path(destination), Path(source_folder)
    old = source_folder / target / 'mpt'
    if not (old / 'COMMITTED.json').exists():
        if old.exists() and any(old.iterdir()):
            raise ValueError('Explicit MPT control is incomplete; inspect its owner')
        return False
    owner = _execution(source_folder)
    for name in ['backbone.py', 'stage2_models.py', 'tri_training.py']:
        if owner['source'].get(name) != core.file_hash(Path(__file__).parent / name):
            raise ValueError(f'MPT control owning source differs: {name}')
    oldcfg = owner['config']['datasets'][spec['dataset']]
    for name in ['epochs', 'patience', 'learning_rate', 'batch_size']:
        if oldcfg.get(name) != cfg.get(name):
            raise ValueError(f'MPT control training protocol differs: {name}')
    saved = read_json(old / 'metrics.json')
    pos = spec['tasks'].index(target)
    init_seed = seed + spec['tasks'][2:].index(target) + cfg.get('head_seed_offset', 3)
    if (saved.get('paired_seed'), saved.get('target'), saved.get('variant'), saved.get('seed')) != (seed, target, 'mpt', init_seed):
        raise ValueError('MPT control pairing/initialization mismatch')
    if saved.get('frozen_original_paths', {}).get('base_hash') != base_hash:
        raise ValueError('MPT control frozen base differs')
    if saved.get('validation_fallback', False):
        raise ValueError('MPT control checkpoint selection differs')
    if not 1 <= saved.get('selected_epoch', 0) <= saved.get('epochs', 0) <= cfg['epochs']:
        raise ValueError('MPT control epoch protocol differs')
    if saved.get('stop_reason') not in ['budget', 'patience'] or (saved['stop_reason'] == 'budget' and saved['epochs'] != cfg['epochs']):
        raise ValueError('MPT control stopping protocol differs')
    # Historical local references may themselves point at older committed files.
    from .tri_reuse import artifact_folder
    artifact = artifact_folder(old) if (old / 'REUSED.json').exists() else old
    proof = verify_stage(artifact, ['model.pt', 'metrics.json', 'predictions.npz'])
    verify_stage(old, ['metrics.json'])
    model.load_state_dict(torch.load(artifact / 'model.pt', map_location=core.DEVICE, weights_only=True), strict=True)
    model.eval()
    if core.model_hash(model) != saved['model_hash']:
        raise ValueError('MPT control loaded model hash differs')
    with np.load(artifact / 'predictions.npz', allow_pickle=False) as raw:
        for split in ['val', 'test']:
            y, p, ids = [raw[f'{split}_{k}'] for k in ['y', 'p', 'rowid']]
            if not np.array_equal(y, caches[split]['y'][:, pos].numpy()) or not np.array_equal(ids, spec['sets'][split].row_ids):
                raise ValueError('MPT control raw labels/rowids differ')
            scores = core.measures(y, p)
            if any(abs(scores[k] - saved[split][k]) > 1e-10 for k in scores):
                raise ValueError('MPT control raw metric differs')
            if not np.array_equal(p, core.probabilities(model, caches[split], envs, spec)):
                raise ValueError('MPT control predictions do not reproduce on current execution')
    frozen = guard()
    reference = dict(protocol='scheme1-audited-mpt-control-v1', artifacts=proof,
                     origin_stage=str(old.resolve()), base_hash=base_hash)
    if (destination / 'COMMITTED.json').exists():
        resolve_artifacts(destination)
        if read_json(destination / 'REUSED.json') != reference:
            raise ValueError('Existing MPT control reuse identity changed')
        return True
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('MPT reference destination is not empty')
    destination.mkdir(parents=True, exist_ok=True)
    core.dump(destination / 'metrics.json', dict(saved, reused=True,
        cost_measurement_source=saved.get('cost_measurement_source', 'historical-three-dataset-MPT'),
        frozen_original_paths=frozen))
    core.dump(destination / 'REUSED.json', reference)
    _commit(core, destination)
    return True
