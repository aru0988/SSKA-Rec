"""Prepare the two initial targets and bind their data to the Table 2 protocol."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import core, tri_data
from .locking import ExclusiveLock
from .vocabulary import CensusIncome_Vocabulary_Size

PROTOCOL = 'table2-two-task-v2-20261121'
CENSUS_CACHE_VERSION = 'census-mpt-two-task-127-v1'
CENSUS_SPLIT_SEED = 20260929
CENSUS_ROWS = (199523, 49881, 49881)
CENSUS_COLUMNS = (
    'age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education',
    'wage_per_hour', 'hs_college', 'marital_stat', 'major_ind_code',
    'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
    'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses',
    'stock_dividends', 'tax_filer_stat', 'region_prev_res', 'state_prev_res',
    'det_hh_fam_stat', 'det_hh_summ', 'instance_weight', 'mig_chg_msa',
    'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt', 'num_emp',
    'fam_under_18', 'country_father', 'country_mother', 'country_self',
    'citizenship', 'own_or_self', 'vet_question', 'vet_benefits',
    'weeks_worked', 'year', 'income_50k',
)
CENSUS_TARGETS = ('income_50k', 'marital_stat')
CENSUS_FEATURES = [c for c in CENSUS_COLUMNS if c not in CENSUS_TARGETS]
CENSUS_CATEGORICAL = [c for c in CENSUS_FEATURES if c in CensusIncome_Vocabulary_Size]
CENSUS_NUMERIC = [c for c in CENSUS_FEATURES if c not in CensusIncome_Vocabulary_Size]
TASKS = {'census': ['income', 'marital'], 'aliccp': ['CTR', 'CVR'], 'kuairand': ['click', 'like']}
INPUT_DIMS = {'census': 127, 'aliccp': 80, 'kuairand': 17}


def _read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _raw_census(path, encoded_path):
    # ``NA`` is a literal value of hisp_origin in the original data.
    frame = pd.read_csv(path, header=None, names=CENSUS_COLUMNS, skipinitialspace=True,
                        keep_default_na=False, na_values=[''])
    if not isinstance(frame.index, pd.RangeIndex) or frame.isna().any().any():
        raise ValueError(f'Missing or malformed Census field: {path}')
    income = frame['income_50k'].str.strip()
    marital = frame['marital_stat'].str.strip()
    if set(income) != {'- 50000.', '50000+.'}:
        raise ValueError(f'Unexpected Census income encoding: {path}')
    labels = np.column_stack((income.eq('50000+.'), marital.eq('Never married'))).astype(np.float32)
    encoded = pd.read_csv(encoded_path, usecols=['label_income', 'label_marital'])
    if not np.array_equal(labels, encoded[['label_income', 'label_marital']].to_numpy()):
        raise ValueError(f'Census labels differ row by row from {encoded_path.name}: {path}')
    return frame, labels


def _encode_census(train, heldout):
    features = [np.empty((len(frame), len(CENSUS_FEATURES)), dtype=np.float32)
                for frame in (train, heldout)]
    categories, bounds = {}, {}
    for i, name in enumerate(CENSUS_FEATURES):
        if name in CENSUS_CATEGORICAL:
            levels = sorted(set(train[name]))
            categories[name] = levels
            ids = {value: index for index, value in enumerate(levels, start=1)}
            for frame, x in zip((train, heldout), features):
                x[:, i] = frame[name].map(ids).fillna(0).to_numpy(dtype=np.float32)
        else:
            low, high = float(train[name].min()), float(train[name].max())
            bounds[name] = [low, high]
            for frame, x in zip((train, heldout), features):
                values = frame[name].to_numpy(dtype=np.float64)
                x[:, i] = (values - low) / (high - low or 1.0)
    return features, categories, bounds


def _census_spec(sets, categories, inputs):
    vocab = {name: len(categories[name]) + 1 for name in CENSUS_CATEGORICAL}
    return dict(sets=sets, tasks=TASKS['census'].copy(), initial=TASKS['census'].copy(),
                vocab=vocab, xdim=INPUT_DIMS['census'], rdim=128, batch=256, lr=.001,
                env_coe=.1, hidden=[256, 128], tower=[64, 32], embed=4,
                widths=[1] * len(CENSUS_NUMERIC) + [4] * len(CENSUS_CATEGORICAL),
                epochs1=30, epochs2=30, ssl_epochs=30, access_epochs=30, patience=5,
                reg_dnn=3e-5, dataset='census', data_files=[item['path'] for item in inputs])


def _save_array(path, value):
    temporary = path.with_suffix('.npy.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, value, allow_pickle=False)
    temporary.replace(path)


def _census(dataset_root, cache_root, metadata_dir, cfg):
    if cfg.get('split_seed', CENSUS_SPLIT_SEED) != CENSUS_SPLIT_SEED:
        raise ValueError('Census must use its original holdout split seed')
    folder = dataset_root / 'Census-income'
    raw = folder / 'raw' if (folder / 'raw').is_dir() else folder
    raw_paths = [raw / f'census-income.{suffix}' for suffix in ('data', 'test')]
    encoded_paths = [folder / f'{split}.gz' for split in ('train', 'test')]
    inputs = tri_data.input_identity(raw_paths + encoded_paths)
    expected = cfg.get('expected_inputs_sha256', {})
    if expected:
        matched = [item for item in inputs if Path(item['path']).name in expected]
        tri_data._check_inputs(matched, cfg)
    if 'expected_raw_inputs_sha256' in cfg:
        tri_data._check_inputs(inputs[:2], {'expected_inputs_sha256': cfg['expected_raw_inputs_sha256']})
    cache = cache_root / CENSUS_CACHE_VERSION
    manifest_path = cache / 'manifest.json'
    train_ids = np.arange(CENSUS_ROWS[0], dtype=np.int64)
    val_idx, test_idx = core.make_split(sum(CENSUS_ROWS[1:]), CENSUS_SPLIT_SEED)
    split_indices = dict(zip(tri_data.SPLITS, (train_ids, val_idx, test_idx)))
    indices = {split + '_indices': idx for split, idx in split_indices.items()}
    indices.update(train_rowid=train_ids, val_rowid=len(train_ids) + val_idx,
                   test_rowid=len(train_ids) + test_idx)

    with ExclusiveLock(cache / 'prepare.lock'):
        if manifest_path.is_file():
            manifest = _read_json(manifest_path)
            if manifest['version'] != CENSUS_CACHE_VERSION or manifest['signature']['inputs'] != inputs:
                raise ValueError(f'Census two-task cache identity differs: {manifest_path}')
        else:
            (train, train_y), (heldout, heldout_y) = [
                _raw_census(path, reference) for path, reference in zip(raw_paths, encoded_paths)
            ]
            if (len(train), len(heldout)) != (CENSUS_ROWS[0], sum(CENSUS_ROWS[1:])):
                raise ValueError('Census raw row counts do not match the full partition')
            (train_x, heldout_x), categories, bounds = _encode_census(train, heldout)
            for split, x, y in (('train', train_x, train_y),
                                ('val', heldout_x[val_idx], heldout_y[val_idx]),
                                ('test', heldout_x[test_idx], heldout_y[test_idx])):
                _save_array(cache / f'{split}-x.npy', x)
                _save_array(cache / f'{split}-y.npy', y)
            manifest = dict(version=CENSUS_CACHE_VERSION, categories=categories, numeric_min_max=bounds)

        sets = {}
        for split in tri_data.SPLITS:
            x = np.load(cache / f'{split}-x.npy', allow_pickle=False)
            y = np.load(cache / f'{split}-y.npy', allow_pickle=False)
            rows = core.Rows(x, y, CENSUS_FEATURES.copy(), [0, 1])
            rows.row_ids = indices[split + '_rowid']
            sets[split] = rows
        spec = _census_spec(sets, manifest['categories'], inputs)
        signature = tri_data.dataset_signature(spec, inputs, blocked_features=set(CENSUS_TARGETS))
        signature.update(version=PROTOCOL, schema_source=CENSUS_CACHE_VERSION,
                         split_seed=CENSUS_SPLIT_SEED, excluded_features=list(CENSUS_TARGETS),
                         preprocessing=dict(fit_split='train', categorical_oov=0,
                                            categorical_levels=manifest['categories'],
                                            numeric_min_max=manifest['numeric_min_max']))
        if 'signature' in manifest and manifest['signature'] != signature:
            raise ValueError(f'Census cached arrays differ from their recorded signature: {manifest_path}')
        if 'signature' not in manifest:
            core.dump(manifest_path, dict(manifest, signature=signature))
    path = tri_data._publish(metadata_dir, signature, indices)
    return dict(spec, data_signature=signature, metadata_path=str(path),
                preprocessing_manifest=str(manifest_path))


def _project_source(dataset, dataset_root, cache_root, metadata_dir, cfg, source_root):
    old = tri_data.build_spec(dataset, dataset_root, cache_root, metadata_dir / 'source-admission', cfg)
    original = old['data_signature']
    if source_root is not None:
        path = Path(source_root).expanduser().resolve() / dataset / 'data' / 'data-signature.json'
        if original != _read_json(path):
            raise ValueError(f'Historical data signature differs: {path}')
        provenance = dict(path=str(path), sha256=core.file_hash(path))
    else:
        digest = hashlib.sha256(json.dumps(original, sort_keys=True).encode('utf-8')).hexdigest()
        provenance = dict(admitted_source_signature_sha256=digest)
    tasks = TASKS[dataset]
    if old['tasks'][:2] != tasks or old['initial'] != tasks:
        raise ValueError(f'Unexpected two-task source label order: {dataset}')

    sets, signatures = {}, {}
    for split, old_rows in old['sets'].items():
        rows = core.Rows(old_rows.x, old_rows.y[:, :2], old_rows.names, [0, 1])
        rows.row_ids = old_rows.row_ids
        if hasattr(old_rows, 'user_groups'):
            rows.user_groups = old_rows.user_groups
        sets[split] = rows
        source_signature = original['splits'][split]
        signatures[split] = dict(source_signature, labels={t: source_signature['labels'][t] for t in tasks},
                                 labels_sha256=tri_data._array_hash(rows.y))
    signature = dict(original, version=PROTOCOL, tasks=tasks, initial=tasks,
                     splits=signatures, source_provenance=provenance)
    signature.pop('bsi_definition', None)
    path = tri_data._publish(metadata_dir, signature)
    return dict(old, sets=sets, tasks=tasks.copy(), initial=tasks.copy(),
                data_signature=signature, metadata_path=str(path))


def build_table2_spec(dataset, dataset_root, cache_root, metadata_dir, cfg, source_root=None):
    """Return the core spec and immutable signature for one Table 2 dataset."""
    tasks, width = TASKS[dataset], INPUT_DIMS[dataset]
    if cfg.get('initial', tasks) != tasks or cfg.get('input_dim', width) != width:
        raise ValueError(f'{dataset} Table 2 target/input configuration differs')
    root = Path(dataset_root).expanduser().resolve()
    cache = Path(cache_root).expanduser().resolve()
    metadata = Path(metadata_dir).expanduser().resolve()
    if dataset == 'census':
        spec = _census(root, cache, metadata, cfg)
    else:
        spec = _project_source(dataset, root, cache, metadata, cfg, source_root)
    if spec['xdim'] != width or spec['tasks'] != tasks:
        raise ValueError(f'{dataset} does not match the two-task data protocol')
    return dict(spec, table2_protocol=PROTOCOL)
