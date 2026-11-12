"""Full-split B/U sources and two frozen, dimension-generic linear access heads."""

import copy
import ast
import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import core, training
from .training import _prepare, _feature_cache, _committed, _commit, _save, _rowids, _utc


class LinearAccessHead(nn.Module):
    def __init__(self, rdim):
        super().__init__()
        self.delta = nn.Linear(rdim, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, batch, tower=None):
        logits = torch.logit(batch['p'].clamp(1e-7, 1 - 1e-7))
        shifted = (logits + self.delta(batch['u']).reshape(-1)).sigmoid()
        return (batch['p'] + (shifted - logits.sigmoid())).clamp(0, 1)


def _freeze(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _array_hash(array):
    array = np.asarray(array)
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(str(array.shape).encode())
    for offset in range(0, len(array), 65536):
        h.update(np.ascontiguousarray(array[offset:offset + 65536]).tobytes())
    return h.hexdigest()


def loader_settings(cuda):
    return dict(num_workers=2 if cuda else 0, pin_memory=bool(cuda), shuffle=False, precision='float32')


def pipeline_loader(factory, settings):
    def load(*args, **kwargs):
        # Only transport/prefetch options change; callers keep their batch order.
        kwargs.setdefault('num_workers', settings['num_workers'])
        kwargs.setdefault('pin_memory', settings['pin_memory'])
        return factory(*args, **kwargs)
    return load


@contextmanager
def runtime_data_loaders():
    settings = loader_settings(core.DEVICE.type == 'cuda')
    original = training.DataLoader
    try:
        if core.DEVICE.type == 'cuda':
            training.DataLoader = pipeline_loader(original, settings)
        yield settings
    finally:
        training.DataLoader = original


def _split_identity(spec, split, rows):
    signature = spec.get('data_signature', {})
    admitted = signature.get('splits', {}).get(split)
    if admitted is None:
        return dict(n=len(rows), names=list(rows.names), rowid_sha256=_array_hash(_rowids(rows)), features_sha256=_array_hash(rows.x), old_labels_sha256=_array_hash(rows.y[:, :2]))
    schema = dict(features_dtype=str(rows.x.dtype), rowid_dtype=str(_rowids(rows).dtype), labels_dtype=str(rows.y.dtype))
    if admitted.get('rows') != len(rows) or signature.get('features') != list(rows.names) or signature.get('tasks') != spec['tasks'] or any(admitted.get(k) != v for k,v in schema.items()) or rows.x.shape != (len(rows),len(rows.names)) or rows.y.shape != (len(rows),len(spec['tasks'])):
        raise ValueError(f'Data signature schema mismatch: {split}')
    old_labels = {name:admitted.get('labels',{}).get(name,{}).get('sha256') for name in spec['initial']}
    if not admitted.get('features_sha256') or not admitted.get('rowid_sha256') or not all(old_labels.values()):
        raise ValueError(f'Data signature fingerprints missing: {split}')
    return dict(n=len(rows), names=list(rows.names), **schema, rowid_sha256=admitted['rowid_sha256'], features_sha256=admitted['features_sha256'], old_labels_sha256=_object_hash(old_labels), old_labels_order=list(spec['initial']), fingerprint_source='dataset-signature-admission')


def source_identity(spec, seed, cfg):
    """Bind checkpoint interpretation, actual rows, and source-training protocol."""
    code = Path(__file__).parent
    return dict(
        protocol='three-dataset-fullsplit-linear-sources-v1',
        dataset=spec['dataset'], seed=seed,
        data_signature=spec.get('data_signature', {}),
        architecture={k: spec.get(k) for k in ['initial', 'vocab', 'embed', 'xdim', 'rdim', 'hidden', 'tower', 'widths', 'env_coe', 'reg_embedding', 'reg_dnn']},
        training=dict(batch=spec['batch'], lr=spec['lr'], base_epochs=cfg.get('base_epochs', 30), ssl_epochs=cfg.get('ssl_epochs', cfg.get('base_epochs', 30)), patience=5),
        runtime_pipeline=loader_settings(core.DEVICE.type == 'cuda'),
        source_hashes={name: core.file_hash(code / name) for name in ['core.py', 'backbone.py', 'base_training.py', 'expert.py', 'training.py', 'vocabulary.py']},
        splits={s:_split_identity(spec,s,rows) for s,rows in spec['sets'].items()},
    )


def _status(path, seed, stage, **extra):
    if path is not None:
        core.dump(Path(path), dict(state='running', pid=os.getpid(), seed=seed, stage=stage, updated_at=_utc(), **extra))


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _semantic_nodes(path):
    class Normalize(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body.pop(0)
            return node
        def visit_ClassDef(self, node):
            self.generic_visit(node)
            node.body = [n for n in node.body if getattr(n, 'name', None) != '__deepcopy__']
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body.pop(0)
            return node
    tree = Normalize().visit(ast.parse(Path(path).read_text(encoding='utf8')))
    return {node.name:ast.dump(node, include_attributes=False) for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def _historical_sources(spec, seed, candidate, folder, cfg):
    """Verify the original KuaiRand protocol, then read B/U without copying."""
    if spec['dataset'] != 'kuairand':
        raise ValueError('Historical adapter supports the verified KuaiRand protocol')
    legacy = candidate.parents[2]
    # Cloud bundles retain the original project subtree below their bundle root.
    nested = legacy / 'paper_continual_20261030'
    project, workspace = (nested, legacy) if (nested / 'source-lock.json').is_file() else (legacy, legacy.parent)
    lock_path, config_path = project / 'source-lock.json', project / 'config.json'
    lock, config, identity = _read(lock_path), _read(config_path), _read(candidate / 'identity.json')
    expected_identity = dict(seed=seed, source=_object_hash(lock), config=_object_hash(config), data=_read(spec['preprocessing_manifest'])['preproc_hash'])
    if identity != expected_identity or seed not in config['seeds']:
        raise ValueError('Historical source identity/config/data mismatch')
    for name, digest in lock['files'].items():
        path = (workspace / name).resolve()
        if not path.is_relative_to(workspace.resolve()) or not path.is_file() or core.file_hash(path) != digest:
            raise ValueError(f'Historical source lock mismatch: {name}')
    local = Path(__file__).parent
    equivalents = [
        ('paper_complete_20261026/core.py', 'core.py', ['build_base','make_u','batches','seed_all']),
        ('paper_continual_20261030/training.py', 'training.py', ['_prepare','_feature_cache']),
        ('MPT-Rec-exp-universal-representation/multitaskrec/model.py', 'backbone.py', ['MLP','EmbeddingNetwork','MPTRec','NewTask']),
        ('MPT-Rec-exp-universal-representation/universal_experiment/model.py', 'expert.py', ['UniversalExpert']),
        ('MPT-Rec-exp-universal-representation/universal_experiment/paired_run.py', 'base_training.py', ['PairedTrainManager']),
    ]
    semantics = {}
    for old_name, current_name, nodes in equivalents:
        if old_name not in lock['files']:
            raise ValueError(f'Historical source not covered by lock: {old_name}')
        old, current = _semantic_nodes(workspace / old_name), _semantic_nodes(local / current_name)
        if any(name not in old or old[name] != current.get(name) for name in nodes):
            raise ValueError(f'Historical source semantics differ: {old_name}')
        semantics[current_name] = dict(original_sha256=lock['files'][old_name], current_sha256=core.file_hash(local / current_name), nodes=nodes)
    oldcfg = config['kuairand']
    check = dict(base_epochs=cfg.get('base_epochs',30), ssl_epochs=cfg.get('ssl_epochs',30), patience=5, learning_rate=spec['lr'], batch_size=spec['batch'], embedding_size=spec['embed'], rdim=spec['rdim'], hidden=spec['hidden'], tower=spec['tower'], env_coe=spec['env_coe'])
    if any(oldcfg.get(k) != value for k,value in check.items()):
        raise ValueError('Historical source hyperparameters differ')
    manifest_path = Path(spec['preprocessing_manifest'])
    manifest = _read(manifest_path)
    locked_data = lock['files'].get('paper_continual_20261030/data.py')
    if locked_data is not None and manifest.get('source_sha256') != locked_data:
        raise ValueError('Historical preprocessing source differs')
    manifest_names = manifest['features']['names']
    if any(list(rows.names) != manifest_names for rows in spec['sets'].values()):
        raise ValueError('Historical feature order differs')
    input_hashes = {Path(v['path']).name:v['sha256'] for v in manifest.get('inputs',[])}
    if 'expected_inputs_sha256' in oldcfg and input_hashes != oldcfg['expected_inputs_sha256']:
        raise ValueError('Historical raw inputs differ')
    current_inputs = spec.get('data_signature',{}).get('inputs')
    if current_inputs is not None and {Path(v['path']).name:v['sha256'] for v in current_inputs} != input_hashes:
        raise ValueError('Historical current raw input identity differs')
    verified_arrays = {}
    for split, rows in spec['sets'].items():
        info = manifest['splits'][split]
        if info['rows'] != len(rows):
            raise ValueError(f'Historical split size differs: {split}')
        verified_arrays[split] = {}
        for kind, actual in [('x',rows.x),('y',rows.y),('rowid',_rowids(rows))]:
            path = manifest_path.parent / info[kind]
            digest = core.file_hash(path)
            if digest != info['sha256'][kind]:
                raise ValueError(f'Historical preprocessing array hash mismatch: {split}/{kind}')
            original = np.load(path, mmap_mode='r', allow_pickle=False)
            if original.dtype != actual.dtype or original.shape != actual.shape or _array_hash(original) != _array_hash(actual):
                raise ValueError(f'Historical actual rows differ: {split}/{kind}')
            verified_arrays[split][kind] = digest
            del original
    for name, required in [('source/base',['base.pt','metrics.json']),('source/ssl',['source.pt','metrics.json'])]:
        _verify_stage(candidate / name, required)
    base, u = core.build_base(spec), core.make_u(spec, seed)
    base.load_state_dict(torch.load(candidate / 'source/base/base.pt',map_location=core.DEVICE,weights_only=True))
    u.load_state_dict(torch.load(candidate / 'source/ssl/source.pt',map_location=core.DEVICE,weights_only=True))
    for model, stage in [(base,'source/base'),(u,'source/ssl')]:
        if core.model_hash(model) != _read(candidate / stage / 'metrics.json')['model_hash']:
            raise ValueError(f'Historical loaded model hash mismatch: {stage}')
        _freeze(model)
    cache_folder = folder
    if (candidate / 'feature-cache' / 'COMMITTED.json').is_file():
        _verify_stage(candidate / 'feature-cache', ['identity.json', *[f'{s}-{k}.npy' for s in spec['sets'] for k in ['x','g','s0','s1','old']]])
        cache_folder = candidate
    caches, envs = _feature_cache(core, base, spec, cache_folder)
    proof = dict(source_lock_sha256=core.file_hash(lock_path), identity=identity, semantic_equivalence=semantics, manifest_sha256=core.file_hash(manifest_path), preprocessing_arrays=verified_arrays, cache_root=str((cache_folder / 'feature-cache').resolve()))
    return base, caches, envs, u, proof


def _verify_stage(folder, required):
    """Read-only integrity checking: never archive or repair an external source."""
    folder = Path(folder).resolve()
    manifest = folder / 'COMMITTED.json'
    if not manifest.is_file():
        raise ValueError(f'Missing committed source: {folder}')
    files = _read(manifest)
    if not set(required).issubset(files):
        raise ValueError(f'Incomplete committed source: {folder}')
    for name, digest in files.items():
        file = (folder / name).resolve()
        if not file.is_relative_to(folder) or not file.is_file() or core.file_hash(file) != digest:
            raise ValueError(f'Committed source integrity failure: {file}')
    return dict(path=str(folder), manifest_sha256=core.file_hash(manifest), files=files)


def old_predictions(base, u, heads, caches, spec, split):
    """Stream U0 once per batch; no additional full-size U or task-feature cache."""
    output = []
    with torch.no_grad():
        for batch in core.batches(caches[split], spec['batch']):
            latent = u(batch['x'])
            output.append(torch.stack([heads[i](dict(u=latent, p=batch['old'][:, i]), base.tower_networks[i]) for i in range(2)], 1).cpu())
    return torch.cat(output).numpy()


def _score(spec, y, predictions):
    return {name: core.measures(y[:, i], predictions[:, i]) for i, name in enumerate(spec['initial'])}


def _train_linear(base, caches, u, spec, seed, folder, cfg, status_path):
    heads = nn.ModuleList([LinearAccessHead(spec['rdim']) for _ in range(2)]).to(core.DEVICE)
    dependencies = dict(base=core.model_hash(base), u=core.model_hash(u))
    budget = cfg.get('access_epochs', cfg.get('linear_epochs', 30))
    if not 1 <= budget <= 30:
        raise ValueError('Linear access requires 1 to 30 epochs')
    protocol = dict(epochs=budget, patience=5, batch=spec['batch'], learning_rate=spec['lr'], rdim=spec['rdim'])
    if _committed(core, folder):
        metrics = _read(folder / 'metrics.json')
        if metrics['dependencies'] != dependencies or metrics['protocol'] != protocol:
            raise ValueError('Linear source identity/protocol changed')
        heads.load_state_dict(torch.load(folder / 'model.pt', map_location=core.DEVICE, weights_only=True))
        if core.model_hash(heads) != metrics['model_hash']:
            raise ValueError('Linear source model hash mismatch')
        heads = _freeze(heads)
        with np.load(folder / 'predictions.npz', allow_pickle=False) as saved:
            for split in ['val', 'test']:
                if not np.array_equal(saved[split + '_y'], spec['sets'][split].y[:, :2]) or not np.array_equal(saved[split + '_rowid'], _rowids(spec['sets'][split])):
                    raise ValueError(f'Linear raw label/row identity mismatch: {split}')
                if not np.array_equal(saved[split + '_p'], old_predictions(base, u, heads, caches, spec, split)):
                    raise ValueError(f'Linear raw prediction mismatch: {split}')
        metrics.setdefault('baseline', {split:_score(spec, spec['sets'][split].y[:, :2], caches[split]['old'].numpy()) for split in ['val','test']})
        return heads, metrics
    core.seed_all(seed + 20)
    for split in ['val']:
        if not np.array_equal(old_predictions(base, u, heads, caches, spec, split), caches[split]['old'].numpy()):
            raise ValueError('Zero linear access failed exact baseline preservation')
    clock, started = time.perf_counter(), _utc()
    if core.DEVICE.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(core.DEVICE)
    optimizer = torch.optim.Adam(heads.parameters(), lr=spec['lr'])
    history, best, stale, state = [], -float('inf'), 0, None
    for epoch in range(1, budget + 1):
        heads.train()
        total, count = 0., 0
        for batch in core.batches(caches['train'], spec['batch']):
            with torch.no_grad():
                latent = u(batch['x'])
            predictions = torch.stack([heads[i](dict(u=latent, p=batch['old'][:, i])) for i in range(2)], 1)
            loss = F.binary_cross_entropy(predictions, batch['y'][:, :2])
            if not torch.isfinite(loss):
                raise ValueError('Non-finite linear access loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(predictions)
            count += len(predictions)
        heads.eval()
        scores = _score(spec, spec['sets']['val'].y[:, :2], old_predictions(base, u, heads, caches, spec, 'val'))
        mean_auc = float(np.mean([v['auc'] for v in scores.values()]))
        row = dict(epoch=epoch, train_bce=total / count, train_loss=total / count, train_regularization=0., val=scores, val_mean_auc=mean_auc, at=_utc())
        history.append(row)
        core.dump(folder / 'history-live.json', history)
        _status(status_path, seed, 'linear-access', last_epoch=row)
        print(json.dumps(dict(seed=seed, stage='linear-access', **row)), flush=True)
        if mean_auc > best:
            best, stale, state = mean_auc, 0, copy.deepcopy(heads.state_dict())
        else:
            stale += 1
        if stale >= 5:
            break
    heads.load_state_dict(state)
    _freeze(heads)
    raw, scores = {}, {}
    for split in ['val', 'test']:
        y, predictions = spec['sets'][split].y[:, :2], old_predictions(base, u, heads, caches, spec, split)
        raw.update({split + '_y': y, split + '_p': predictions, split + '_base_p': caches[split]['old'].numpy(), split + '_rowid': _rowids(spec['sets'][split])})
        scores[split] = _score(spec, y, predictions)
    if dependencies != dict(base=core.model_hash(base), u=core.model_hash(u)):
        raise ValueError('Frozen source changed during linear access')
    _save(state, folder / 'model.pt')
    np.savez_compressed(folder / 'predictions.npz', **raw)
    baseline = {split:_score(spec, spec['sets'][split].y[:, :2], caches[split]['old'].numpy()) for split in ['val','test']}
    metrics = dict(seed=seed, protocol=protocol, dependencies=dependencies, frozen_base_hash=dependencies['base'], frozen_u_hash=dependencies['u'], model_hash=core.model_hash(heads), history=history, epochs=len(history), selected_epoch=max(history, key=lambda r:r['val_mean_auc'])['epoch'], stop_reason='patience' if stale >= 5 else 'budget', seconds=time.perf_counter() - clock, started_at=started, ended_at=_utc(), optimizer_parameters=sum(p.numel() for p in heads.parameters()), peak_cuda_bytes=torch.cuda.max_memory_allocated(core.DEVICE) if core.DEVICE.type == 'cuda' else 0, baseline=baseline, **scores)
    core.dump(folder / 'metrics.json', metrics)
    _commit(core, folder)
    return heads, metrics


def _load_external(spec, seed, candidate, expected):
    """Accept only an exact full identity; old layouts need an explicit adapter."""
    identity = candidate / 'tri-source-identity.json'
    if not identity.is_file() or _read(identity) != expected:
        raise ValueError('External source identity absent or incompatible')
    for name, required in [('source/base', ['base.pt', 'metrics.json']), ('source/ssl', ['source.pt', 'metrics.json']), ('feature-cache', ['identity.json']), ('linear', ['model.pt', 'metrics.json', 'predictions.npz'])]:
        _verify_stage(candidate / name, required)
    # Every stage has been checked before _prepare can see this external folder.
    return candidate


def prepare_seed(spec, seed, folder, cfg, status_path, reuse_root=None):
    if spec['dataset'] == 'census' and tuple(len(spec['sets'][s]) for s in ['train','val','test']) != (199523,49881,49881):
        raise ValueError('Sources require the complete Census split (199523, 49881, 49881)')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    identity = source_identity(spec, seed, cfg)
    identity_path = folder / 'tri-source-identity.json'
    if identity_path.exists() and _read(identity_path) != identity:
        raise ValueError('Source identity changed')
    budget = cfg.get('base_epochs', 30)
    if budget != cfg.get('ssl_epochs', budget) or budget < 1:
        raise ValueError('B and SSL require the same positive epoch budget')
    _status(status_path, seed, 'preparing-sources')
    source_folder, reused, rejected, historical = folder, False, None, None
    prepared = None
    if reuse_root is not None:
        root = Path(reuse_root).resolve()
        candidate = root if (root / 'tri-source-identity.json').is_file() or (root / 'identity.json').is_file() else root / str(seed)
        # Census always gets a newly trained full-split B/U in this experiment.
        if spec['dataset'] == 'census':
            rejected = 'Census protocol requires new full-split B/U sources'
        else:
            try:
                if (candidate / 'tri-source-identity.json').is_file():
                    source_folder = _load_external(spec, seed, candidate, identity)
                else:
                    *parts, historical = _historical_sources(spec, seed, candidate, folder, cfg)
                    prepared = tuple(parts)
                    source_folder = candidate
                reused = True
            except (ValueError, OSError, KeyError) as exc:
                if spec['dataset'] == 'kuairand':
                    raise ValueError(f'Explicit KuaiRand reuse failed verification: {exc}') from exc
                rejected = str(exc)
        if rejected:
            print(f'Source reuse rejected seed={seed}: {rejected}; preparing local sources', flush=True)
            _status(status_path, seed, 'preparing-sources', reuse_rejected=rejected)
    core.dump(identity_path, identity)
    if prepared is not None:
        base, caches, envs, u = prepared
    elif reused:
        base, caches, envs, u = _prepare(core, spec, seed, source_folder, budget)
    else:
        with runtime_data_loaders():
            base, caches, envs, u = _prepare(core, spec, seed, source_folder, budget)
    if core.model_hash(base) != _read(source_folder / 'source/base/metrics.json')['model_hash'] or core.model_hash(u) != _read(source_folder / 'source/ssl/metrics.json')['model_hash']:
        raise ValueError('Source checkpoint model hash mismatch')
    base, u = _freeze(base), _freeze(u)
    linear_folder = folder / 'linear' if historical is not None else source_folder / 'linear'
    heads, linear = _train_linear(base, caches, u, spec, seed, linear_folder, cfg, status_path)
    roots = dict(base=source_folder / 'source/base', u=source_folder / 'source/ssl', linear=linear_folder, cache=Path(historical['cache_root']) if historical is not None else source_folder / 'feature-cache')
    stages = {name: _verify_stage(path, required) for name,path,required in [('base',roots['base'],['base.pt','metrics.json']), ('u',roots['u'],['source.pt','metrics.json']), ('linear',roots['linear'],['model.pt','metrics.json','predictions.npz']), ('cache',roots['cache'],['identity.json'])]}
    receipt = dict(seed=seed, reused=reused, reuse_rejected=rejected, source_root=str(source_folder.resolve()), identity_sha256=core.file_hash(identity_path), data_signature=identity['data_signature'], source_hashes=identity['source_hashes'], model_hashes=dict(base=core.model_hash(base), u=core.model_hash(u), linear=core.model_hash(heads)), stages=stages, costs={name:_read(path / ('cost.json' if name == 'cache' else 'metrics.json')) for name,path in roots.items()}, original_predictions_npz=str((linear_folder / 'predictions.npz').resolve()))
    if historical is not None:
        receipt['historical_verification'] = historical
    receipt['costs']['linear'] = linear
    receipt['runtime_pipeline'] = identity['runtime_pipeline']
    core.dump(folder / 'source-receipt.json', receipt)
    return base, caches, envs, u, heads, receipt


def make_guard(base, u, heads, caches, spec):
    """Capture both frozen module states and bitwise original-task probabilities."""
    modules = dict(base=base, u=u, linear=heads)
    hashes = {name:core.model_hash(model) for name,model in modules.items()}
    baseline = {split:old_predictions(base, u, heads, caches, spec, split) for split in ['val','test']}
    def guard():
        for name, model in modules.items():
            if core.model_hash(model) != hashes[name] or any(p.requires_grad for p in model.parameters()):
                raise ValueError(f'Frozen source changed: {name}')
        for split, expected in baseline.items():
            if not np.array_equal(old_predictions(base, u, heads, caches, spec, split), expected):
                raise ValueError(f'Frozen original predictions changed: {split}')
        return dict(historical_max_probability_difference=0.0, model_hashes=hashes)
    return guard
