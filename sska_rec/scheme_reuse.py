"""Read-only admission of committed B/cache and optional original SSL U sources."""

import json
import re
from pathlib import Path

from . import core, tri_data, tri_sources
import numpy as np
import torch
from .training import _rowids


_OWNER_FILES = ('core.py', 'backbone.py', 'base_training.py', 'expert.py',
                'training.py', 'vocabulary.py')
_CACHE_WIDTHS = ('x', 'g', 's0', 's1', 'old')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical_data(value):
    """Remove storage locations and timestamps, preserving every content field."""
    if isinstance(value, dict):
        locations = {'path', 'legacy_source', 'current_source', 'source_lock'}
        return {k: canonical_data(v) for k, v in value.items()
                if k not in locations and k not in ('mtime', 'mtime_ns')}
    if isinstance(value, list):
        return [canonical_data(v) for v in value]
    return value


def resolve_source_folder(spec, seed, source_root):
    """Resolve only the three documented roots; never discover a different seed."""
    if source_root is None or not str(source_root).strip():
        raise ValueError('An explicit source_root is required')
    root = Path(source_root).resolve()
    for folder in (root, root / str(seed), root / spec['dataset'] / str(seed)):
        if (folder / 'tri-source-identity.json').is_file():
            return folder
    raise ValueError(f'Missing canonical source identity below source_root: {root}')


def _actual_data(spec):
    signature = spec.get('data_signature')
    if not isinstance(signature, dict) or not signature.get('splits'):
        raise ValueError('Source admission requires full data_signature fingerprints')
    if set(spec['sets']) != {'train', 'val', 'test'}:
        raise ValueError('Source admission requires all three splits')
    if spec['dataset'] == 'census' and tuple(len(spec['sets'][s]) for s in tri_data.SPLITS) != tri_data.EXPECTED_ROWS['census']:
        raise ValueError('Source admission requires the complete Census split')
    expected = dict(dataset=spec['dataset'], tasks=list(spec['tasks']),
                    initial=list(spec['initial']), vocab=spec['vocab'],
                    input_dim=spec['xdim'], representation_dim=spec['rdim'],
                    widths=spec['widths'])
    if any(signature.get(k) != v for k, v in expected.items()):
        raise ValueError('Data signature dataset/targets/vocabulary/schema mismatch')
    tri_data._rowid_admission(spec['sets'])
    checked = {}
    for split, rows in spec['sets'].items():
        admitted = signature['splits'].get(split, {})
        if (list(rows.names) != signature.get('features') or
                rows.x.shape != (len(rows), len(rows.names)) or
                rows.y.shape != (len(rows), len(spec['tasks']))):
            raise ValueError(f'Actual data schema mismatch: {split}')
        values = dict(rows=len(rows), rowid_dtype=str(_rowids(rows).dtype),
                      features_dtype=str(rows.x.dtype), labels_dtype=str(rows.y.dtype),
                      rowid_sha256=tri_data._array_hash(_rowids(rows)),
                      features_sha256=tri_data._array_hash(rows.x),
                      labels_sha256=tri_data._array_hash(rows.y))
        if hasattr(rows, 'user_groups'):
            values['user_groups_sha256'] = tri_data._array_hash(rows.user_groups)
        if any(admitted.get(k) != v for k, v in values.items()):
            raise ValueError(f'Actual data fingerprint mismatch: {split}')
        for i, target in enumerate(spec['tasks']):
            labels = rows.y[:, i]
            if not np.isin(labels, [0, 1]).all():
                raise ValueError(f'Actual nonbinary labels: {split}/{target}')
            fingerprint = dict(sha256=tri_data._array_hash(labels), positive=int(labels.sum()),
                               negative=len(rows) - int(labels.sum()))
            if admitted.get('labels', {}).get(target) != fingerprint:
                raise ValueError(f'Actual target label fingerprint mismatch: {split}/{target}')
        checked[split] = values
    return checked


def _identity(spec, seed, cfg, folder):
    requested = dict(patience=5, learning_rate=spec['lr'], batch_size=spec['batch'],
                     precision='float32', shuffle=False)
    if any(cfg.get(k,v) != v for k,v in requested.items()):
        raise ValueError('Requested source training protocol mismatch')
    actual = _read(folder / 'tri-source-identity.json')
    expected = tri_sources.source_identity(spec, seed, cfg)
    # Runtime transport differs between CPU admission and CUDA training; precision
    # remains part of the owner protocol.
    for key in ('protocol', 'dataset', 'seed', 'architecture', 'training', 'splits'):
        if actual.get(key) != expected[key]:
            raise ValueError(f'Source identity mismatch: {key}')
    if canonical_data(actual.get('data_signature')) != canonical_data(expected['data_signature']):
        raise ValueError('Source data_signature identity mismatch')
    if actual.get('runtime_pipeline', {}).get('precision') != 'float32':
        raise ValueError('Source precision protocol mismatch')
    owner = actual.get('source_hashes', {})
    current = {name: core.file_hash(Path(core.__file__).with_name(name)) for name in _OWNER_FILES}
    if any(owner.get(name) != digest for name, digest in current.items()):
        raise ValueError('Source owning code SHA256 identity mismatch')
    return actual


def _stage(folder, required, recorded=None):
    folder = Path(folder).resolve()
    manifest = folder / 'COMMITTED.json'
    if not manifest.is_file():
        raise ValueError(f'Missing committed source stage: {folder}')
    files = _read(manifest)
    if not isinstance(files, dict) or not set(required).issubset(files):
        raise ValueError(f'Incomplete committed source stage: {folder}')
    for name, digest in files.items():
        path = (folder / name).resolve()
        if (not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest) or
                not path.is_relative_to(folder) or path == manifest or
                not path.is_file() or core.file_hash(path) != digest):
            raise ValueError(f'Committed source integrity failure: {path}')
    result = dict(path=str(folder), manifest_sha256=core.file_hash(manifest), files=files)
    if recorded is not None:
        digest = recorded.get('manifest_sha256', recorded.get('committed_sha256'))
        if recorded.get('files') != files or digest != result['manifest_sha256']:
            raise ValueError(f'Recorded source stage hashes changed: {folder}')
    return result


def _historical(spec, seed, cfg, receipt, paths):
    """Recheck the existing KuaiRand semantic proof and its actual old owners."""
    proof = receipt.get('historical_verification')
    if spec['dataset'] != 'kuairand' or not isinstance(proof, dict):
        raise ValueError('External legacy source stages require historical semantic proof')
    candidate = Path(receipt['source_root']).resolve()
    if paths['base'] != candidate / 'source/base' or ('u' in paths and paths['u'] != candidate / 'source/ssl'):
        raise ValueError('Historical source stage owner path mismatch')
    legacy = candidate.parents[2]
    nested = legacy / 'paper_continual_20261030'
    project, workspace = (nested, legacy) if (nested / 'source-lock.json').is_file() else (legacy, legacy.parent)
    lock_path = project / 'source-lock.json'
    lock, config, identity = _read(lock_path), _read(project/'config.json'), _read(candidate/'identity.json')
    if (core.file_hash(lock_path) != proof.get('source_lock_sha256') or
            identity != proof.get('identity') or identity.get('seed') != seed or
            identity.get('source') != tri_sources._object_hash(lock) or
            identity.get('config') != tri_sources._object_hash(config) or seed not in config['seeds']):
        raise ValueError('Historical source lock/config/seed proof mismatch')
    for name, digest in lock['files'].items():
        path = (workspace / name).resolve()
        if not path.is_relative_to(workspace.resolve()) or not path.is_file() or core.file_hash(path) != digest:
            raise ValueError(f'Historical source locked bytes differ: {name}')
    equivalents = {
        'core.py': ('paper_complete_20261026/core.py', ['build_base','make_u','batches','seed_all']),
        'training.py': ('paper_continual_20261030/training.py', ['_prepare','_feature_cache']),
        'backbone.py': ('MPT-Rec-exp-universal-representation/multitaskrec/model.py', ['MLP','EmbeddingNetwork','MPTRec','NewTask']),
        'expert.py': ('MPT-Rec-exp-universal-representation/universal_experiment/model.py', ['UniversalExpert']),
        'base_training.py': ('MPT-Rec-exp-universal-representation/universal_experiment/paired_run.py', ['PairedTrainManager']),
    }
    for name, (old_name, nodes) in equivalents.items():
        entry = proof.get('semantic_equivalence', {}).get(name, {})
        local = Path(core.__file__).with_name(name)
        if (entry.get('original_sha256') != lock['files'].get(old_name) or
                entry.get('current_sha256') != core.file_hash(local) or entry.get('nodes') != nodes):
            raise ValueError(f'Historical semantic proof identity mismatch: {name}')
        old, current = tri_sources._semantic_nodes(workspace/old_name), tri_sources._semantic_nodes(local)
        if any(n not in old or old[n] != current.get(n) for n in nodes):
            raise ValueError(f'Historical semantic source mismatch: {name}')
    oldcfg = config['kuairand']
    settings = dict(base_epochs=cfg.get('base_epochs',30), ssl_epochs=cfg.get('ssl_epochs',cfg.get('base_epochs',30)),
                    patience=5, learning_rate=spec['lr'], batch_size=spec['batch'], embedding_size=spec['embed'],
                    rdim=spec['rdim'], hidden=spec['hidden'], tower=spec['tower'], env_coe=spec['env_coe'])
    if any(oldcfg.get(k) != v for k,v in settings.items()):
        raise ValueError('Historical source training protocol mismatch')
    manifest = _read(spec['preprocessing_manifest'])
    if identity.get('data') != manifest.get('preproc_hash'):
        raise ValueError('Historical preprocessing identity mismatch')
    for split, rows in spec['sets'].items():
        for kind, actual in [('x',rows.x), ('y',rows.y), ('rowid',_rowids(rows))]:
            info = manifest['splits'][split]
            root = Path(spec['preprocessing_manifest']).resolve().parent
            path = (root/info[kind]).resolve()
            digest = info['sha256'][kind]
            if (not path.is_relative_to(root) or not path.is_file() or core.file_hash(path) != digest or
                    proof.get('preprocessing_arrays', {}).get(split, {}).get(kind) != digest):
                raise ValueError('Historical preprocessing array hash proof mismatch')
            original = np.load(path, mmap_mode='r', allow_pickle=False)
            if original.dtype != actual.dtype or original.shape != actual.shape or tri_data._array_hash(original) != tri_data._array_hash(actual):
                raise ValueError('Historical actual preprocessing rows differ')
    if Path(proof['cache_root']).resolve() != paths['cache']:
        raise ValueError('Historical feature cache path proof mismatch')
    return proof


def _load_sources(spec, seed, cfg, source_root, require_u):
    folder = resolve_source_folder(spec, seed, source_root)
    checked_data = _actual_data(spec)
    identity = _identity(spec, seed, cfg, folder)
    receipt_path = folder/'reuse-receipt.json'
    if not receipt_path.is_file():
        receipt_path = folder/'source-receipt.json'
    original = _read(receipt_path) if receipt_path.is_file() else None
    if original is not None:
        if (original.get('seed') != seed or original.get('identity_sha256') != core.file_hash(folder/'tri-source-identity.json') or
                canonical_data(original.get('data_signature')) != canonical_data(identity['data_signature']) or
                any(original.get('source_hashes', {}).get(n) != identity['source_hashes'][n] for n in _OWNER_FILES)):
            raise ValueError('Original source receipt owner/data/code identity mismatch')
    required = {'base':['base.pt','metrics.json'], 'cache':['identity.json', *[f'{s}-{k}.npy' for s in spec['sets'] for k in _CACHE_WIDTHS]]}
    defaults = dict(base=folder/'source/base', cache=folder/'feature-cache')
    if require_u:
        required['u'] = ['source.pt','metrics.json']
        defaults['u'] = folder/'source/ssl'
    stages, paths = {}, {}
    for name, files in required.items():
        recorded = original.get('stages', {}).get(name) if original else None
        if original and recorded is None:
            raise ValueError(f'Missing source receipt stage: {name}')
        raw_path = Path(recorded['path']) if recorded else defaults[name]
        path = raw_path if raw_path.is_absolute() else folder/raw_path
        stages[name] = _stage(path, files, recorded)
        paths[name] = Path(stages[name]['path'])
    historical = None
    if paths['base'] != defaults['base'] or (require_u and paths['u'] != defaults['u']):
        source_owner = paths['base'].parent.parent
        if (source_owner/'tri-source-identity.json').is_file():
            _identity(spec,seed,cfg,source_owner)
            if require_u and paths['u'] != source_owner/'source/ssl':
                raise ValueError('B and U source owners differ')
        else:
            historical = _historical(spec,seed,cfg,original or {},paths)
    base = core.build_base(spec)
    base.load_state_dict(torch.load(paths['base']/'base.pt', map_location=core.DEVICE, weights_only=True))
    tri_sources._freeze(base)
    models = dict(base=core.model_hash(base))
    if models['base'] != _read(paths['base']/'metrics.json').get('model_hash'):
        raise ValueError('Loaded base model hash mismatch')
    cache_identity = _read(paths['cache']/'identity.json')
    expected_cache = dict(base_hash=models['base'], splits={s:dict(rows=len(r),rowid_sha256=tri_data._array_hash(_rowids(r))) for s,r in spec['sets'].items()})
    if cache_identity != expected_cache:
        raise ValueError('Feature cache base/row identity mismatch')
    caches = {}
    widths = dict(x=spec['xdim'], g=spec['rdim'], s0=spec['rdim'], s1=spec['rdim'], old=2)
    for split, rows in spec['sets'].items():
        caches[split] = {}
        for key, width in widths.items():
            array = np.load(paths['cache']/f'{split}-{key}.npy',mmap_mode='c',allow_pickle=False)
            if array.dtype != np.dtype('float32') or array.shape != (len(rows),width):
                raise ValueError(f'Feature cache schema mismatch: {split}/{key}')
            caches[split][key] = torch.from_numpy(array)
        # Labels always come from this admitted dataset, including new targets.
        labels = rows.y
        if isinstance(labels,np.memmap) and not labels.flags.writeable:
            labels = np.load(labels.filename,mmap_mode='c',allow_pickle=False)
            if labels.shape != rows.y.shape or labels.dtype != rows.y.dtype:
                raise ValueError(f'Label mmap schema mismatch: {split}')
        caches[split]['y'] = torch.from_numpy(labels)
    with torch.no_grad():
        envs = [base.env_embedding_network(torch.tensor(k,device=core.DEVICE)).detach() for k in range(2)]
    u = None
    if require_u:
        u = core.make_u(spec,seed)
        u.load_state_dict(torch.load(paths['u']/'source.pt',map_location=core.DEVICE,weights_only=True))
        tri_sources._freeze(u)
        models['u'] = core.model_hash(u)
        if models['u'] != _read(paths['u']/'metrics.json').get('model_hash'):
            raise ValueError('Loaded U model hash mismatch')
    if original and any(original.get('model_hashes', {}).get(k) != v for k,v in models.items()):
        raise ValueError('Original receipt model hash mismatch')
    receipt = dict(protocol='schemes-readonly-source-admission-v1', dataset=spec['dataset'], seed=seed,
                   source_folder=str(folder), source_root=str(folder), require_u=bool(require_u),
                   identity_sha256=core.file_hash(folder/'tri-source-identity.json'),
                   owner_identity=identity, data_signature=spec['data_signature'],
                   checked_actual_data=checked_data, source_hashes=identity['source_hashes'],
                   model_hashes=models, stages=stages)
    if original:
        receipt['original_receipt'] = dict(path=str(receipt_path),sha256=core.file_hash(receipt_path))
    if historical:
        receipt['historical_verification'] = historical
    data_digest = tri_sources._object_hash(canonical_data(spec['data_signature']))
    spec['scheme_data_identity'] = data_digest
    spec['scheme_cache_identity'] = dict(protocol='schemes-audited-cache-v1',
        dataset=spec['dataset'],seed=seed,base_hash=models['base'],
        cache_manifest_sha256=stages['cache']['manifest_sha256'],
        cache_files=dict(stages['cache']['files']),data_signature_sha256=data_digest)
    return base,caches,envs,u,receipt


def load_sources(spec, seed, cfg, source_root, require_u=True):
    """Audit sources without writing, training, repairing, copying, or old heads."""
    try:
        return _load_sources(spec,seed,cfg,source_root,require_u)
    except (OSError, KeyError, TypeError, RuntimeError, IndexError) as exc:
        raise ValueError(f'Explicit source audit failed: {exc}') from exc
