"""Read-only admission of the corrected, normalized-L2 KuaiRand ST runs."""

from pathlib import Path

import numpy as np
import torch

from . import core
from .training import _commit, _rowids
from .tri_reuse import _read, _verify
from .tri_sources import _array_hash, _object_hash, _semantic_nodes

EXECUTION_SHA256 = '0820cfcd2d8fe4bf6bc92981d0f180a07a1068d2c258a369bc22ba3f8711a292'
_MODEL_SOURCE = 'MPT-Rec-exp-universal-representation/multitaskrec/model.py'


def artifact_folder(folder):
    """Resolve a receipt while rechecking both local and external committed bytes."""
    folder = Path(folder).resolve()
    if not (folder / 'REUSED.json').exists():
        return folder
    _verify(folder, ['metrics.json', 'REUSED.json'])
    receipt = _read(folder / 'REUSED.json')
    if receipt.get('protocol') != 'audited-kuairand-single-normalized-l2-v2' or receipt.get('base') is not None:
        raise ValueError('Historical ST receipt identity mismatch')
    external = _verify(receipt['artifacts']['path'], ['model.pt', 'predictions.npz', 'metrics.json'])
    if external != receipt['artifacts']:
        raise ValueError('External reused ST artifact integrity identity changed')
    return Path(external['path'])


def _source_identity(spec, seed, legacy, cfg):
    output = legacy / 'outputs'
    identity = _read(output / 'identity.json')
    source = Path(cfg.get('legacy_source_root', legacy.parent / 'continual-20261030')).resolve()
    if source.name == 'paper_continual_20261030':
        source = source.parent
    project = source / 'paper_continual_20261030'
    lock = _read(project / 'source-lock.json')
    config = _read(project / 'config.json')
    own_lock = _read(legacy / 'source-lock.json')
    if own_lock.get('train.py') != EXECUTION_SHA256 or core.file_hash(legacy / 'train.py') != EXECUTION_SHA256:
        raise ValueError('Historical ST execution source hash mismatch')
    for name, digest in lock['files'].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or not path.is_file() or core.file_hash(path) != digest:
            raise ValueError(f'Historical ST source lock integrity mismatch: {name}')
    if _MODEL_SOURCE not in lock['files'] or 'paper_continual_20261030/config.json' not in lock['files']:
        raise ValueError('Historical ST source model/config not covered by lock')
    previous = _semantic_nodes(source / _MODEL_SOURCE)
    current = _semantic_nodes(Path(__file__).with_name('backbone.py'))
    classes = ['SingleTask', 'EmbeddingNetwork', 'MLP']
    if any(name not in previous or previous[name] != current.get(name) for name in classes):
        raise ValueError('Historical ST model source semantics differ')
    manifest_path = Path(spec['preprocessing_manifest']).resolve()
    manifest = _read(manifest_path)
    expected = dict(version='kuairand-st-normalized-l2-v2', source=_object_hash(lock),
                    execution=EXECUTION_SHA256, data=manifest['preproc_hash'], seeds=config['seeds'],
                    embedding_l2='0.006/(2*N_E)*sum(E^2)', epochs=30, patience=5,
                    lr=.001, batch=512, shuffle=False)
    if identity != expected or seed not in config['seeds']:
        raise ValueError('Historical ST source/data/seed identity mismatch')
    oldcfg = config['kuairand']
    expected_cfg = dict(epochs=30, patience=5, learning_rate=.001, batch_size=512,
                        embedding_size=spec['embed'], hidden=spec['hidden'], tower=spec['tower'], rdim=spec['rdim'])
    if any(oldcfg.get(key) != value for key, value in expected_cfg.items()):
        raise ValueError('Historical ST training/architecture protocol mismatch')
    if 'paper_continual_20261030/data.py' in lock['files'] and manifest.get('source_sha256') != lock['files']['paper_continual_20261030/data.py']:
        raise ValueError('Historical ST preprocessing source differs')
    input_hashes = {Path(value['path']).name:value['sha256'] for value in manifest.get('inputs', [])}
    if 'expected_inputs_sha256' in oldcfg and input_hashes != oldcfg['expected_inputs_sha256']:
        raise ValueError('Historical ST raw inputs differ')
    current_inputs = spec.get('data_signature', {}).get('inputs')
    if current_inputs is not None and {Path(v['path']).name:v['sha256'] for v in current_inputs} != input_hashes:
        raise ValueError('Historical ST current raw input identity differs')
    arrays = {}
    for split in ['train', 'val', 'test']:
        rows = spec['sets'][split]
        info = manifest['splits'][split]
        if info['rows'] != len(rows) or list(rows.names) != manifest['features']['names']:
            raise ValueError(f'Historical ST actual rows/schema differ: {split}')
        arrays[split] = {}
        for kind, actual in [('x',rows.x), ('y',rows.y), ('rowid',_rowids(rows))]:
            path = (manifest_path.parent / info[kind]).resolve()
            if not path.is_relative_to(manifest_path.parent) or core.file_hash(path) != info['sha256'][kind]:
                raise ValueError(f'Historical ST preprocessing array integrity mismatch: {split}/{kind}')
            original = np.load(path, mmap_mode='r', allow_pickle=False)
            if original.dtype != actual.dtype or original.shape != actual.shape or _array_hash(original) != _array_hash(actual):
                raise ValueError(f'Historical ST actual rows differ: {split}/{kind}')
            arrays[split][kind] = info['sha256'][kind]
    return dict(identity=identity, execution_sha256=EXECUTION_SHA256,
                identity_sha256=core.file_hash(output / 'identity.json'),
                source_lock_sha256=core.file_hash(project / 'source-lock.json'),
                source_root=str(source), semantic_classes=classes,
                model_source_sha256=lock['files'][_MODEL_SOURCE],
                current_backbone_sha256=core.file_hash(Path(__file__).with_name('backbone.py')),
                preprocessing_manifest_sha256=core.file_hash(manifest_path), preprocessing_arrays=arrays)


def _validate_history(old):
    history = old['history']
    if not history or len(history) != old['epochs'] or not 1 <= old['epochs'] <= 30:
        raise ValueError('Historical ST epoch history mismatch')
    best, stale, selected = -float('inf'), 0, None
    for epoch, row in enumerate(history, 1):
        if row['epoch'] != epoch or not np.isfinite(row['val_auc']):
            raise ValueError('Historical ST validation history mismatch')
        if row['val_auc'] > best:
            best, stale, selected = row['val_auc'], 0, epoch
        else:
            stale += 1
        if stale >= 5 and epoch != len(history):
            raise ValueError('Historical ST history continued after early stopping')
    expected_stop = 'patience' if stale >= 5 else 'budget'
    if old['selected_epoch'] != selected or old['stop_reason'] != expected_stop or (expected_stop == 'budget' and len(history) != 30):
        raise ValueError('Historical ST early stopping history mismatch')
    row = history[selected - 1]
    if any(abs(row['val_' + key] - old['val'][key]) > 1e-10 for key in ('auc','logloss')):
        raise ValueError('Historical ST selected validation metric mismatch')


def reuse_single(spec, paired_seed, target, model, destination, legacy_root, cfg):
    """Reference a verified v2 ST candidate in place; a bad candidate raises."""
    if spec['dataset'] != 'kuairand' or legacy_root is None:
        return False
    if target not in ('follow', 'forward') or spec['tasks'] != ['click','like','follow','forward']:
        raise ValueError('Historical ST target/task protocol mismatch')
    expected_protocol = dict(epochs=30, patience=5, embedding_l2=.006, dnn_l2=3e-5,
                             learning_rate=.001, batch_size=512, shuffle=False, precision='float32')
    if spec['batch'] != 512 or spec['lr'] != .001 or any(cfg.get(k, v) != v for k,v in expected_protocol.items()):
        raise ValueError('Historical ST requested training protocol mismatch')
    elements = sum(spec['vocab'].values()) * spec['embed']
    if model.reg_embedding != .006 / elements or model.reg_dnn != 3e-5:
        raise ValueError('Historical ST requested regularization mismatch')
    legacy = Path(legacy_root).resolve()
    if legacy.name == 'outputs':
        legacy = legacy.parent
    external = legacy / 'outputs' / str(paired_seed) / ('is_' + target)
    artifacts = _verify(external, ['model.pt', 'predictions.npz', 'metrics.json'])
    provenance = _source_identity(spec, paired_seed, legacy, cfg)
    old = _read(external / 'metrics.json')
    seed = paired_seed + spec['tasks'].index(target) + 2
    if old['seed'] != seed or old['target'] != 'is_' + target:
        raise ValueError('Historical ST model seed/target mismatch')
    if old['embedding_elements'] != elements or old['embedding_sum_coefficient'] != .006 / elements or old['dnn_coefficient'] != 3e-5:
        raise ValueError('Historical ST stored regularization mismatch')
    _validate_history(old)
    # v2 did not store a state hash; the committed .pt bytes and strict loading
    # bind the existing state, and the current canonical hash is recorded below.
    state = torch.load(external / 'model.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    model_hash = core.model_hash(model)
    if old.get('model_hash', model_hash) != model_hash:
        raise ValueError('Historical ST model hash mismatch')
    parameters = sum(p.numel() for p in model.parameters())
    if old['optimizer_parameters'] != parameters:
        raise ValueError('Historical ST optimizer parameter mismatch')
    idx = spec['tasks'].index(target)
    with np.load(external / 'predictions.npz', allow_pickle=False) as raw:
        for split in ['val','test']:
            y, p, rowid = [raw[split + '_' + key] for key in ('y','p','rowid')]
            rows = spec['sets'][split]
            if not np.array_equal(y, rows.y[:,idx]) or not np.array_equal(rowid, _rowids(rows)):
                raise ValueError(f'Historical ST raw label/row alignment mismatch: {split}')
            if p.shape != y.shape or p.dtype != np.float32 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
                raise ValueError(f'Historical ST raw prediction schema mismatch: {split}')
            actual = core.measures(y,p)
            if any(abs(actual[k] - old[split][k]) > 1e-10 for k in actual):
                raise ValueError(f'Historical ST raw metric mismatch: {split}')
    reference = dict(protocol='audited-kuairand-single-normalized-l2-v2', paired_seed=paired_seed,
                     seed=seed, target=target, artifacts=artifacts, base=None,
                     model_hash=model_hash, provenance=provenance,
                     prediction_verification='committed raw hashes; complete label/row alignment; raw metrics; strict state loading and AST-equivalent forward')
    destination = Path(destination).resolve()
    if (destination / 'COMMITTED.json').exists():
        if not (destination / 'REUSED.json').exists():
            return False
        artifact_folder(destination)
        if _read(destination / 'REUSED.json') != reference:
            raise ValueError('Existing historical ST reuse identity changed')
        return True
    if destination.exists() and set(p.name for p in destination.iterdir()) - {'metrics.json','REUSED.json'}:
        raise ValueError('Historical ST destination contains unrelated artifacts')
    if (destination / 'REUSED.json').exists() and _read(destination / 'REUSED.json') != reference:
        raise ValueError('Pending historical ST reuse identity changed')
    destination.mkdir(parents=True, exist_ok=True)
    history = [dict(row, auc=row['val_auc'], logloss=row['val_logloss'],
                    train_regularization=row.get('embedding_l2',0) + row.get('dnn_l2',0)) for row in old['history']]
    metrics = dict(old, history=history, variant='single_task', target=target, paired_seed=paired_seed, seed=seed, model_seed=seed,
                   model_hash=model_hash, reused=True, ready_for_commit=True,
                   total_parameters=parameters, exclusive_parameters=parameters, inference_parameters=parameters,
                   u_update_parameters=0, frozen_original_paths={}, validation_fallback=False,
                   learning_rate=.001, patience=5, batch_size=512, embedding_l2=.006, dnn_l2=3e-5,
                   embedding_elements=elements, weight_file_bytes=(external/'model.pt').stat().st_size,
                   trainable_tensor_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                   cost_measurement_source='historical-kuairand-st-20261031-v2',
                   cost_measurement_scope='Original training, including an additional full training-split evaluation each epoch',
                   history_unavailable_fields=['train_seconds','samples_per_second','batch_fetch_and_forward_seconds'],
                   external_artifacts=str(external))
    core.dump(destination / 'metrics.json', metrics)
    core.dump(destination / 'REUSED.json', reference)
    _commit(core, destination)
    return True
