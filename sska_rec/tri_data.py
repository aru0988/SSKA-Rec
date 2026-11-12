"""Full-split adapters and immutable provenance for the three-dataset protocol.

Raw datasets are opened in place. Ali-CCP and KuaiRand reuse encoded caches;
Census keeps the original complete split and records its deterministic indices.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from . import core
from .data import prepare_kuairand

SPLITS = ('train', 'val', 'test')
EXPECTED_ROWS = {'census': (199523, 49881, 49881), 'aliccp': (10000000, 1000000, 10000000)}
_HASHES = {}


def input_identity(paths):
    """Hash each stable raw input once per process and return its identity."""
    result = []
    for value in paths:
        path = Path(value).resolve()
        before = path.stat()
        stamp = (str(path), before.st_size, before.st_mtime_ns)
        digest = _HASHES.get(stamp)
        if digest is None:
            digest = core.file_hash(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f'Input changed during hashing: {path}')
            _HASHES[stamp] = digest
        result.append(dict(path=str(path), sha256=digest, bytes=before.st_size,
                           mtime_ns=before.st_mtime_ns))
    return result


def _array_hash(value):
    digest = hashlib.sha256()
    array = np.asarray(value)
    for start in range(0, len(array), 100000):
        digest.update(np.ascontiguousarray(array[start:start + 100000]).tobytes())
    return digest.hexdigest()


def _rowid_admission(sets):
    total = sum(len(r) for r in sets.values())
    # Original source rowids are a permutation of [0,total); a bitmap is bounded
    # by the source row count, rather than a Python set of tens of millions IDs.
    seen = np.zeros(total, dtype=np.bool_)
    for split in SPLITS:
        rows = sets[split]
        ids = np.asarray(rows.row_ids)
        if ids.shape != (len(rows),) or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError(f'Invalid rowid array: {split}')
        for start in range(0, len(ids), 100000):
            chunk = ids[start:start + 100000]
            if (chunk < 0).any() or (chunk >= total).any():
                raise ValueError(f'rowid outside complete source range: {split}')
            if len(np.unique(chunk)) != len(chunk) or seen[chunk].any():
                raise ValueError(f'Duplicate or overlapping rowid: {split}')
            seen[chunk] = True
    if not seen.all():
        raise ValueError('Incomplete source rowids')


def dataset_signature(spec, inputs):
    """Validate exclusion/disjointness and hash rows, labels and model features."""
    _rowid_admission(spec['sets'])
    blocked = {
        'census': {'income', 'marital', 'label_income', 'label_marital', 'education', 'sex'},
        'aliccp': {'101', '301', 'CTR', 'CVR', 'BSI'},
        'kuairand': {'click', 'like', 'follow', 'forward', 'is_click', 'is_like', 'is_follow', 'is_forward'},
    }[spec['dataset']]
    names = list(spec['sets']['train'].names)
    if set(names) & blocked:
        raise ValueError(f'Target/user-group field in model features: {sorted(set(names) & blocked)}')
    if spec['dataset'] == 'kuairand' and set(names) != {'user_id', 'video_id', 'tab', 'hour', 'duration_ms'}:
        raise ValueError('KuaiRand feature allowlist mismatch')
    info = {}
    for split in SPLITS:
        rows = spec['sets'][split]
        if list(rows.names) != names or rows.x.shape != (len(rows), len(names)):
            raise ValueError(f'Feature schema mismatch: {split}')
        if rows.y.shape != (len(rows), len(spec['tasks'])):
            raise ValueError(f'Label schema mismatch: {split}')
        counts = {}
        for i, target in enumerate(spec['tasks']):
            values = np.asarray(rows.y[:, i])
            if not np.isin(values, [0, 1]).all():
                raise ValueError(f'Nonbinary labels: {split}/{target}')
            positive = int(values.sum())
            if positive == 0 or positive == len(values):
                raise ValueError(f'Single-class labels: {split}/{target}')
            counts[target] = dict(positive=positive, negative=len(values) - positive,
                                  sha256=_array_hash(values))
        item = dict(rows=len(rows), rowid_sha256=_array_hash(rows.row_ids),
                    labels_sha256=_array_hash(rows.y), features_sha256=_array_hash(rows.x),
                    labels=counts, rowid_dtype=str(rows.row_ids.dtype),
                    features_dtype=str(rows.x.dtype), labels_dtype=str(rows.y.dtype))
        if spec['dataset'] == 'aliccp' and not hasattr(rows, 'user_groups'):
            raise ValueError(f'Missing user-group metadata: {split}')
        if hasattr(rows, 'user_groups'):
            if len(rows.user_groups) != len(rows):
                raise ValueError(f'User-group row mismatch: {split}')
            item['user_groups_sha256'] = _array_hash(rows.user_groups)
        info[split] = item
    return dict(version='linear-trifusion-full-splits-v1', dataset=spec['dataset'],
                inputs=inputs, tasks=list(spec['tasks']), initial=list(spec['initial']),
                features=names, vocab=spec['vocab'], input_dim=spec['xdim'],
                representation_dim=spec['rdim'], widths=spec['widths'], splits=info)


def _check_inputs(inputs, cfg):
    expected = cfg.get('expected_inputs_sha256')
    if expected is not None:
        actual = {Path(item['path']).name: item['sha256'] for item in inputs}
        if actual != expected:
            raise ValueError('Raw input SHA256 identity mismatch')


def _publish(metadata_dir, signature, indices=None):
    root = Path(metadata_dir).resolve()
    path = root / 'data-signature.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != signature:
        raise ValueError(f'Data signature changed: {path}')
    root.mkdir(parents=True, exist_ok=True)
    if indices is not None:
        index_path = root / 'split-indices.npz'
        if index_path.exists():
            with np.load(index_path) as previous:
                if set(previous.files) != set(indices) or any(not np.array_equal(previous[k], v) for k, v in indices.items()):
                    raise ValueError(f'Split indices changed: {index_path}')
        else:
            temporary = index_path.with_suffix('.npz.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **indices)
            temporary.replace(index_path)
    core.dump(path, signature)
    return path


def _kuairand(dataset_root, cache_root, cfg):
    keys = ('log_files', 'categorical_features', 'numeric_features', 'split_fractions',
            'min_positive_per_split', 'chunksize')
    manifest_path = prepare_kuairand(dataset_root / 'KuaiRand-Pure' / 'data',
                                    cache_root / 'kuairand',
                                    {k: cfg[k] for k in keys if k in cfg})
    manifest_path = Path(manifest_path)
    m = json.loads(manifest_path.read_text(encoding='utf-8'))
    if m['version'] != 'kuairand-pure-exposures-v2-epoch-days':
        raise ValueError('KuaiRand preprocessing version mismatch')
    if m['features']['fit_split'] != 'train':
        raise ValueError('KuaiRand schema must be fit on train')
    if m['labels'] != ['click', 'like', 'follow', 'forward']:
        raise ValueError('KuaiRand label order mismatch')
    _check_inputs(m['inputs'], cfg)
    names = m['features']['names']
    vocab = {k: v['cardinality'] for k, v in m['features']['categorical'].items()}
    sets = {}
    for split in SPLITS:
        info = m['splits'][split]
        arrays = {kind: np.load(manifest_path.parent / info[kind], mmap_mode='r') for kind in ('x', 'y', 'rowid')}
        rows = core.Rows(arrays['x'], arrays['y'], names, [0, 1])
        rows.row_ids = arrays['rowid']
        sets[split] = rows
    e = cfg.get('embedding_size', 4)
    numeric = [n for n in names if n not in vocab]
    spec = dict(sets=sets, tasks=['click', 'like', 'follow', 'forward'], initial=['click', 'like'],
                vocab=vocab, xdim=len(numeric) + len(vocab) * e, rdim=128,
                batch=512, lr=.001, env_coe=.1, hidden=[256, 128], tower=[64, 32],
                embed=e, widths=[1] * len(numeric) + [e] * len(vocab), dataset='kuairand',
                data_files=[item['path'] for item in m['inputs']],
                preprocessing_manifest=str(manifest_path.resolve()))
    return spec, m


def build_spec(dataset, dataset_root, cache_root, metadata_dir, cfg):
    """Build complete training datasets and persist identity without raw copies.

    ``cfg`` is the selected entry of ``tri_three_datasets.json['datasets']``.
    Returns the existing core spec interface plus ``data_signature`` and
    ``metadata_path``. Task names are income/marital/education/sex,
    CTR/CVR/BSI, or click/like/follow/forward.
    """
    if dataset not in ('census', 'aliccp', 'kuairand'):
        raise ValueError(f'Unknown dataset: {dataset}')
    for key in ('base_epochs', 'ssl_epochs', 'access_epochs', 'epochs'):
        if cfg.get(key, 30) != 30:
            raise ValueError(f'{key} must equal 30')
    if cfg.get('patience', 5) != 5:
        raise ValueError('patience must equal 5')
    root, cache_root = Path(dataset_root).resolve(), Path(cache_root).resolve()
    indices = None
    preprocessing = None
    if dataset == 'kuairand':
        spec, preprocessing = _kuairand(root, cache_root, cfg)
        inputs = preprocessing['inputs']
    else:
        files = ([root / 'Census-income' / f'{s}.gz' for s in ('train', 'test')]
                 if dataset == 'census' else [root / 'AliCCP' / f'ctr_cvr.{s}' for s in ('train', 'dev', 'test')])
        inputs = input_identity(files)
        _check_inputs(inputs, cfg)
        previous_data, previous_cache = core.DATA, core.CACHE
        try:
            core.DATA, core.CACHE = root, cache_root / 'aliccp'
            core.CACHE.mkdir(parents=True, exist_ok=True)
            spec = (core.census(True, ['income', 'marital']) if dataset == 'census'
                    else core.aliccp(['CTR', 'CVR'], data_hashes={item['path']: item['sha256'] for item in inputs}))
        finally:
            core.DATA, core.CACHE = previous_data, previous_cache
        actual = tuple(len(spec['sets'][s]) for s in SPLITS)
        if actual != EXPECTED_ROWS[dataset]:
            raise ValueError(f'{dataset} requires complete split rows {EXPECTED_ROWS[dataset]}, got {actual}')
        if dataset == 'census':
            if spec['xdim'] != 119:
                raise ValueError('Census input dimension must equal 119')
            vi, ti = core.make_split(actual[1] + actual[2], 20260929)
            ids = dict(train=np.arange(actual[0], dtype=np.int64), val=actual[0] + vi, test=actual[0] + ti)
            indices = dict(train_indices=np.arange(actual[0], dtype=np.int64), val_indices=vi, test_indices=ti)
            for split in SPLITS:
                spec['sets'][split].row_ids = ids[split]
                indices[split + '_rowid'] = ids[split]
        else:
            offset = 0
            for split in SPLITS:
                rows = spec['sets'][split]
                rows.row_ids = np.arange(offset, offset + len(rows), dtype=np.int64)
                offset += len(rows)
        spec['data_files'] = [item['path'] for item in inputs]
    expected_lr, expected_batch, expected_rdim = {'census': (.001, 256, 128), 'aliccp': (.0001, 2000, 64), 'kuairand': (.001, 512, 128)}[dataset]
    if cfg.get('learning_rate', expected_lr) != expected_lr or cfg.get('batch_size', expected_batch) != expected_batch:
        raise ValueError(f'{dataset} learning rate/batch protocol mismatch')
    if spec['rdim'] != expected_rdim:
        raise ValueError(f'{dataset} representation dimension mismatch')
    spec.update(epochs1=30, epochs2=30, ssl_epochs=30, access_epochs=30, patience=5,
                lr=expected_lr, batch=expected_batch, reg_dnn=3e-5)
    signature = dataset_signature(spec, inputs)
    if dataset == 'census':
        signature.update(split_seed=20260929, split_policy='complete train; sorted 50/50 indices of original test',
                         schema_source='existing encoded Census vocabulary and numeric fields; no joint split fitting')
    elif dataset == 'aliccp':
        signature.update(split_policy='original encoded CSV order; first 10000000/1000000/10000000 rows',
                         excluded_features=['101', '301'], bsi_definition='301 != 2',
                         user_group_field='101', evaluation_samples='all protocol rows for CTR and existing CVR semantics',
                         schema_source='existing encoded Ali-CCP vocabulary; no refitting')
    else:
        signature['preprocessing'] = {k: preprocessing[k] for k in ('version', 'preproc_hash', 'features', 'splits')}
    path = _publish(metadata_dir, signature, indices)
    spec.update(data_signature=signature, metadata_path=str(path))
    return spec
