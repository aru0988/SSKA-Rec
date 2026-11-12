import hashlib
import json
from pathlib import Path

import numpy as np
import unittest
import tempfile
from unittest.mock import patch

from sska_rec import core


def _cfg():
    return dict(base_epochs=30, ssl_epochs=30, access_epochs=30, epochs=30,
                patience=5, learning_rate=.001, batch_size=256)


def _rows(n, names=('feature',), tasks=4):
    x = np.zeros((n, len(names)), dtype=np.float32)
    y = np.broadcast_to((np.arange(n) % 2)[:, None], (n, tasks)).astype(np.float32)
    return core.Rows(x, y, list(names), [0, 1])


def _census_spec(sizes=(199523, 49881, 49881)):
    return dict(sets={s: _rows(n) for s, n in zip(('train', 'val', 'test'), sizes)},
                tasks=['income', 'marital', 'education', 'sex'], initial=['income', 'marital'],
                vocab={}, xdim=119, rdim=128, batch=256, lr=.001, env_coe=.1,
                hidden=[256, 128], tower=[64, 32], embed=4, widths=[119], dataset='census')


def _raw_census(tmp_path):
    root = tmp_path / 'dataset'
    folder = root / 'Census-income'
    folder.mkdir(parents=True)
    for name in ('train.gz', 'test.gz'):
        (folder / name).write_bytes(name.encode())
    return root


def test_full_census_keeps_original_split_and_persists_identity(tmp_path, monkeypatch):
    from sska_rec import tri_data
    root = _raw_census(tmp_path)
    fake = _census_spec()
    calls = []
    monkeypatch.setattr(core, 'census', lambda continuous, initial: calls.append((continuous, initial)) or fake)
    spec = tri_data.build_spec('census', root, tmp_path / 'cache', tmp_path / 'meta', _cfg())
    assert calls == [(True, ['income', 'marital'])]
    vi, ti = core.make_split(99762, 20260929)
    assert np.array_equal(spec['sets']['val'].row_ids, 199523 + vi)
    assert np.array_equal(spec['sets']['test'].row_ids, 199523 + ti)
    assert spec['epochs1'] == spec['epochs2'] == spec['ssl_epochs'] == spec['access_epochs'] == 30
    signature = json.loads(Path(spec['metadata_path']).read_text())
    assert signature == spec['data_signature']
    assert signature['splits']['train']['rows'] == 199523
    assert signature['splits']['val']['labels']['sex']['positive'] == 24940
    assert signature['split_seed'] == 20260929
    assert len(signature['inputs']) == 2
    with np.load(tmp_path / 'meta' / 'split-indices.npz') as saved:
        assert np.array_equal(saved['val_indices'], vi)
        assert np.array_equal(saved['test_rowid'], 199523 + ti)


def test_census_rejects_old_subset_before_writing_metadata(tmp_path, monkeypatch):
    from sska_rec import tri_data
    monkeypatch.setattr(core, 'census', lambda *a: _census_spec((20000, 5000, 10000)))
    with unittest.TestCase().assertRaisesRegex(ValueError, '199523'):
        tri_data.build_spec('census', _raw_census(tmp_path), tmp_path / 'cache', tmp_path / 'meta', _cfg())
    assert not (tmp_path / 'meta' / 'data-signature.json').exists()


def test_input_digest_is_reused_and_changed_input_is_rehashed(tmp_path, monkeypatch):
    from sska_rec import tri_data
    p = tmp_path / 'raw'
    p.write_bytes(b'first')
    count = []
    original = core.file_hash
    monkeypatch.setattr(core, 'file_hash', lambda path: count.append(path) or original(path))
    first = tri_data.input_identity([p])
    assert tri_data.input_identity([p]) == first
    p.write_bytes(b'second version')
    assert tri_data.input_identity([p])[0]['sha256'] != first[0]['sha256']
    assert len(count) == 2


def test_signature_rejects_overlapping_rowids_and_target_features(tmp_path):
    from sska_rec import tri_data
    spec = _census_spec((4, 4, 4))
    for rows in spec['sets'].values():
        rows.row_ids = np.arange(4, dtype=np.int64)
    with unittest.TestCase().assertRaisesRegex(ValueError, 'rowid'):
        tri_data.dataset_signature(spec, [])
    for i, rows in enumerate(spec['sets'].values()):
        rows.row_ids = np.arange(4 * i, 4 * (i + 1), dtype=np.int64)
    spec['sets']['train'].names = ['sex']
    with unittest.TestCase().assertRaisesRegex(ValueError, 'feature'):
        tri_data.dataset_signature(spec, [])


def test_kuairand_reuses_epoch_day_manifest_and_train_schema(tmp_path, monkeypatch):
    from sska_rec import tri_data
    cache = tmp_path / 'cache' / 'kuairand'
    cache.mkdir(parents=True)
    features = dict(names=['user_id', 'video_id', 'tab', 'hour', 'duration_ms'],
                    categorical={k: {'cardinality': 5} for k in ['user_id', 'video_id', 'tab', 'hour']},
                    numeric={'duration_ms': {'mean': 2., 'std': 1.}}, fit_split='train')
    splits = {}
    for i, split in enumerate(('train', 'val', 'test')):
        arrays = {'x': np.zeros((4, 5), dtype=np.float32),
                  'y': np.tile([[0, 1, 0, 1], [1, 0, 1, 0]], (2, 1)).astype(np.float32),
                  'rowid': np.arange(i * 4, (i + 1) * 4, dtype=np.int64)}
        splits[split] = dict(rows=4, dates=[f'2022-04-{i + 1:02}'])
        for kind, value in arrays.items():
            path = cache / f'{split}-{kind}.npy'
            np.save(path, value)
            splits[split][kind] = path.name
        splits[split]['sha256'] = {k: core.file_hash(cache / splits[split][k]) for k in arrays}
    manifest = dict(version='kuairand-pure-exposures-v2-epoch-days', preproc_hash='test-preproc',
                    features=features, splits=splits, inputs=[], labels=['click', 'like', 'follow', 'forward'])
    path = cache / 'manifest.json'
    path.write_text(json.dumps(manifest))
    called = []
    monkeypatch.setattr(tri_data, 'prepare_kuairand', lambda *a: called.append(a) or path)
    cfg = dict(_cfg(), batch_size=512, embedding_size=4, rdim=128,
               env_coe=.1, hidden=[256, 128], tower=[64, 32])
    spec = tri_data.build_spec('kuairand', tmp_path / 'dataset', tmp_path / 'cache', tmp_path / 'meta', cfg)
    assert spec['tasks'] == ['click', 'like', 'follow', 'forward']
    assert spec['initial'] == ['click', 'like']
    assert spec['xdim'] == 17 and spec['widths'] == [1, 4, 4, 4, 4]
    assert spec['data_signature']['preprocessing']['features']['fit_split'] == 'train'
    assert called[0][0] == (tmp_path / 'dataset' / 'KuaiRand-Pure' / 'data').resolve()
    for rows in spec['sets'].values():
        for value in (rows.x, rows.y, rows.row_ids):
            value._mmap.close()


def test_config_has_exact_five_seed_125_result_protocol():
    cfg = json.loads((Path(__file__).parents[1] / 'configs' / 'tri_three_datasets.json').read_text())
    assert cfg['variants'] == ['mpt', 'tri_frozen', 'tri_lora', 'tri_ssf', 'tri_full']
    assert cfg['order'] == ['census', 'kuairand', 'aliccp']
    assert sum(len(d['seeds']) * len(d['targets']) * len(cfg['variants']) for d in cfg['datasets'].values()) == cfg['expected_stage2_results'] == 125
    for dataset, d in cfg['datasets'].items():
        assert len(set(d['seeds'])) == 5
        assert all(d[k] == 30 for k in ('base_epochs', 'ssl_epochs', 'access_epochs', 'epochs'))
        assert d['patience'] == 5


def test_aliccp_passes_raw_sha_to_cache_and_preserves_grouping(tmp_path, monkeypatch):
    from sska_rec import tri_data
    root = tmp_path / 'dataset'
    (root / 'AliCCP').mkdir(parents=True)
    for split in ('train', 'dev', 'test'):
        (root / 'AliCCP' / ('ctr_cvr.' + split)).write_bytes(split.encode())
    spec0 = dict(_census_spec((4, 4, 4)), dataset='aliccp', tasks=['CTR', 'CVR', 'BSI'],
                 initial=['CTR', 'CVR'], rdim=64, xdim=80)
    for rows in spec0['sets'].values():
        rows.y = rows.y[:, :3]
        rows.user_groups = np.array([7, 7, 9, 9], dtype=np.int32)
    calls = []
    def aliccp(initial, data_hashes):
        calls.append((initial, data_hashes))
        return spec0
    monkeypatch.setattr(core, 'aliccp', aliccp)
    monkeypatch.setattr(tri_data, 'EXPECTED_ROWS', dict(tri_data.EXPECTED_ROWS, aliccp=(4, 4, 4)))
    cfg = dict(_cfg(), learning_rate=.0001, batch_size=2000)
    spec = tri_data.build_spec('aliccp', root, tmp_path / 'cache', tmp_path / 'meta', cfg)
    assert calls[0][0] == ['CTR', 'CVR']
    assert len(calls[0][1]) == 3
    assert spec['data_signature']['excluded_features'] == ['101', '301']
    assert spec['data_signature']['splits']['train']['user_groups_sha256'] == hashlib.sha256(spec0['sets']['train'].user_groups.tobytes()).hexdigest()
    assert np.array_equal(spec['sets']['test'].row_ids, [8, 9, 10, 11])


def test_aliccp_signature_requires_user_groups():
    from sska_rec import tri_data
    spec = dict(_census_spec((4, 4, 4)), dataset='aliccp')
    for i, rows in enumerate(spec['sets'].values()):
        rows.row_ids = np.arange(i * 4, i * 4 + 4)
    with unittest.TestCase().assertRaisesRegex(ValueError, 'user.group'):
        tri_data.dataset_signature(spec, [])


class _MonkeyPatch:
    def __init__(self):
        self.patches = []

    def setattr(self, obj, name, value):
        item = patch.object(obj, name, value)
        item.start()
        self.patches.append(item)

    def close(self):
        for item in reversed(self.patches):
            item.stop()


class TriDataTests(unittest.TestCase):
    pass


def _adapt(function):
    def run(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[1]) as folder:
            monkeypatch = _MonkeyPatch()
            try:
                args = {'tmp_path': Path(folder), 'monkeypatch': monkeypatch}
                import inspect
                function(**{k: args[k] for k in inspect.signature(function).parameters})
            finally:
                monkeypatch.close()
    return run


for _name, _function in list(globals().items()):
    if _name.startswith('test_'):
        setattr(TriDataTests, _name, _adapt(_function))


if __name__ == '__main__':
    unittest.main()
