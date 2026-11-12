import importlib.util
import gc
import shutil
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from sska_rec import core


class TriSourceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        previous = core.DEVICE
        core.DEVICE = torch.device('cpu')
        self.addCleanup(setattr, core, 'DEVICE', previous)

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('sska_rec.tri_sources'))
        from sska_rec import tri_sources
        return tri_sources

    def spec(self):
        rng = np.random.default_rng(19)
        x = np.column_stack([rng.normal(size=24), rng.integers(0, 3, size=24)])
        y = np.column_stack([(np.arange(24) // k) % 2 for k in [1, 2, 3, 4]]).astype('float32')
        spec = dict(sets={s: core.Rows(x, y, ['dense', 'cat'], [0, 1]) for s in ['train', 'val', 'test']}, tasks=['click', 'like', 'follow', 'forward'], initial=['click', 'like'], vocab={'cat': 3}, embed=2, xdim=3, rdim=4, batch=8, lr=.001, hidden=[8, 4], tower=[4], widths=[1, 2], env_coe=.1, dataset='kuairand', data_signature={'raw_sha256': {'sample':'abc'}, 'full_split': True})
        for rows in spec['sets'].values():
            rows.row_ids = np.arange(24) + 100
        return spec

    def test_linear_width_zero_residual_and_gradient(self):
        api = self.api()
        for width in [4, 64, 128]:
            head = api.LinearAccessHead(width)
            p = torch.tensor([0., 1., .1234567, .5])
            u = torch.randn(4, width)
            output = head({'p': p, 'u': u}, None)
            self.assertTrue(torch.equal(output, p))
            self.assertEqual(sum(v.numel() for v in head.parameters()), width + 1)
            output[2:].sum().backward()
            self.assertGreater(head.delta.weight.grad.abs().sum().item(), 0)

    def test_source_identity_uses_admitted_fingerprints_without_array_rescan(self):
        api, spec = self.api(), self.spec()
        signature = spec['data_signature']
        signature.update(features=['dense','cat'], tasks=spec['tasks'], splits={})
        for split, rows in spec['sets'].items():
            signature['splits'][split] = dict(rows=len(rows), features_sha256='f'*64, rowid_sha256='r'*64, features_dtype=str(rows.x.dtype), rowid_dtype=str(rows.row_ids.dtype), labels_dtype=str(rows.y.dtype), labels={name:dict(sha256=str(i)*64) for i,name in enumerate(spec['tasks'])})
        with patch.object(api, '_array_hash', side_effect=AssertionError('must use admission fingerprints')):
            identity = api.source_identity(spec, 17, dict(base_epochs=1))
        self.assertEqual(identity['splits']['train']['features_sha256'], 'f'*64)
        signature['splits']['train']['rows'] += 1
        with self.assertRaisesRegex(ValueError, 'signature schema'):
            api.source_identity(spec, 17, dict(base_epochs=1))

    def test_runtime_loader_wrapper_preserves_batches_and_restores_binding(self):
        api = self.api()
        from sska_rec import training
        original = training.DataLoader
        rows = self.spec()['sets']['train']
        expected = list(original(rows, batch_size=8, shuffle=False))
        recorded = []
        def recording_loader(*args, **kwargs):
            recorded.append(dict(kwargs))
            kwargs.update(num_workers=0, pin_memory=False)
            return original(*args, **kwargs)
        settings = api.loader_settings(True)
        wrapped = api.pipeline_loader(recording_loader, settings)
        actual = list(wrapped(rows, batch_size=8, shuffle=False))
        self.assertEqual(recorded[0]['num_workers'], 2)
        self.assertTrue(recorded[0]['pin_memory'])
        for old, new in zip(expected, actual):
            for left,right in zip(old[:3], new[:3]):
                self.assertTrue(torch.equal(left,right))
            for key in old[3]:
                self.assertTrue(torch.equal(old[3][key],new[3][key]))
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            with api.runtime_data_loaders() as active:
                self.assertEqual(active, api.loader_settings(False))
                self.assertIs(training.DataLoader, original)
                raise RuntimeError('interrupted')
        self.assertIs(training.DataLoader, original)

    def test_prepare_resume_and_guard_detect_source_mutation(self):
        api = self.api()
        spec = self.spec()
        cfg = dict(base_epochs=1, ssl_epochs=1, access_epochs=2, patience=5)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            folder = Path(temp) / '17'
            status = Path(temp) / 'status.json'
            base, caches, envs, u, heads, receipt = api.prepare_seed(spec, 17, folder, cfg, status)
            self.assertTrue(all(not p.requires_grad for model in [base, u, heads] for p in model.parameters()))
            self.assertEqual(caches['train']['y'].device.type, 'cpu')
            self.assertEqual(sum(p.numel() for p in heads.parameters()), 10)
            self.assertNotIn('h0', caches['train'])
            self.assertNotIn('u', caches['train'])
            self.assertEqual(len(envs), 2)
            self.assertTrue((folder / 'linear' / 'COMMITTED.json').exists())
            self.assertIn('baseline', receipt['costs']['linear'])
            with np.load(folder / 'linear' / 'predictions.npz') as raw:
                np.testing.assert_array_equal(raw['val_y'], spec['sets']['val'].y[:, :2])
                np.testing.assert_array_equal(raw['val_p'], api.old_predictions(base, u, heads, caches, spec, 'val'))
                np.testing.assert_array_equal(raw['test_rowid'], spec['sets']['test'].row_ids)
            guard = api.make_guard(base, u, heads, caches, spec)
            self.assertEqual(guard()['historical_max_probability_difference'], 0.0)
            with patch.object(core.PairedTrainManager, 'train_two_task', side_effect=AssertionError('must resume')):
                resumed = api.prepare_seed(spec, 17, folder, cfg, status)
            self.assertEqual(receipt['model_hashes'], resumed[-1]['model_hashes'])
            with torch.no_grad():
                next(u.parameters()).add_(1.)
            with self.assertRaisesRegex(ValueError, 'Frozen'):
                guard()
            del caches, guard, resumed
            gc.collect()

    def test_explicit_kuairand_verification_failure_never_retrains(self):
        api, spec = self.api(), self.spec()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            with patch.object(core.PairedTrainManager, 'train_two_task', side_effect=AssertionError('must not train')):
                with self.assertRaisesRegex(ValueError, 'Explicit KuaiRand reuse failed'):
                    api.prepare_seed(spec, 17, root / 'new', dict(base_epochs=1), root / 'status.json', reuse_root=root / 'absent')

    def test_census_subset_cannot_be_source(self):
        api, spec = self.api(), self.spec()
        spec['dataset'] = 'census'
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, 'complete Census'):
                api.prepare_seed(spec, 17, root / 'new', dict(base_epochs=1), root / 'status.json')

    def test_reject_changed_identity_and_committed_corruption(self):
        api = self.api()
        spec, cfg = self.spec(), dict(base_epochs=1, ssl_epochs=1, access_epochs=1)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            folder, status = Path(temp) / '17', Path(temp) / 'status.json'
            api.prepare_seed(spec, 17, folder, cfg, status)
            spec['sets']['train'].row_ids = np.arange(24) + 200
            with self.assertRaisesRegex(ValueError, 'identity'):
                api.prepare_seed(spec, 17, folder, cfg, status)
            spec['sets']['train'].row_ids = np.arange(24) + 100
            (folder / 'linear' / 'model.pt').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'integrity'):
                api.prepare_seed(spec, 17, folder, cfg, status)

    def test_exact_external_sources_reused_without_training_or_copying(self):
        api = self.api()
        spec, cfg = self.spec(), dict(base_epochs=1, ssl_epochs=1, access_epochs=1)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            old, new = Path(temp) / 'old' / '17', Path(temp) / 'new' / '17'
            initial = api.prepare_seed(spec, 17, old, cfg, Path(temp) / 'status.json')
            with patch.object(core.PairedTrainManager, 'train_two_task', side_effect=AssertionError('must reuse')):
                reused = api.prepare_seed(spec, 17, new, cfg, Path(temp) / 'status.json', reuse_root=old.parent)
            self.assertEqual(initial[-1]['model_hashes'], reused[-1]['model_hashes'])
            self.assertFalse((new / 'source' / 'base' / 'base.pt').exists())
            self.assertTrue(reused[-1]['reused'])
            del initial, reused
            gc.collect()

    def test_historical_kuairand_reuses_B_U_and_trains_only_new_linear(self):
        api, spec = self.api(), self.spec()
        cfg = dict(base_epochs=1, ssl_epochs=1, access_epochs=1)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            legacy = root / 'paper_continual_20261030'
            old = legacy / 'outputs' / 'kuairand' / '17'
            first = api.prepare_seed(spec, 17, old, cfg, root / 'status.json')
            del first
            gc.collect()
            (old / 'tri-source-identity.json').unlink()
            config = dict(seeds=[17], kuairand=dict(base_epochs=1, ssl_epochs=1, patience=5, learning_rate=spec['lr'], batch_size=spec['batch'], embedding_size=spec['embed'], rdim=spec['rdim'], hidden=spec['hidden'], tower=spec['tower'], env_coe=spec['env_coe']))
            core.dump(legacy / 'config.json', config)
            manifest = dict(preproc_hash='preproc-test', source_sha256='unused', features=dict(names=['dense','cat']), inputs=[], splits={})
            for split, rows in spec['sets'].items():
                files = {}
                for kind, array in [('x', rows.x), ('y', rows.y), ('rowid', rows.row_ids)]:
                    p = root / 'data' / f'{split}-{kind}.npy'
                    p.parent.mkdir(exist_ok=True)
                    np.save(p, array)
                    files[kind] = p.name
                manifest['splits'][split] = dict(rows=len(rows), **files, sha256={k:core.file_hash(root / 'data' / name) for k,name in files.items()})
            core.dump(root / 'data' / 'manifest.json', manifest)
            spec['preprocessing_manifest'] = str(root / 'data' / 'manifest.json')
            module = Path(core.__file__).parent
            layout = {'core.py':'paper_complete_20261026/core.py', 'training.py':'paper_continual_20261030/training.py', 'backbone.py':'MPT-Rec-exp-universal-representation/multitaskrec/model.py', 'expert.py':'MPT-Rec-exp-universal-representation/universal_experiment/model.py', 'base_training.py':'MPT-Rec-exp-universal-representation/universal_experiment/paired_run.py'}
            locked = {'paper_continual_20261030/config.json':core.file_hash(legacy / 'config.json')}
            for current, name in layout.items():
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(module / current, p)
                locked[name] = core.file_hash(p)
            lock = dict(files=locked)
            core.dump(legacy / 'source-lock.json', lock)
            core.dump(old / 'identity.json', dict(seed=17, source=api._object_hash(lock), config=api._object_hash(config), data='preproc-test'))
            new = root / 'new' / '17'
            with patch.object(core.PairedTrainManager, 'train_two_task', side_effect=AssertionError('B must reuse')), patch.object(core.UniversalExpert, 'losses', side_effect=AssertionError('SSL must reuse')):
                second = api.prepare_seed(spec, 17, new, cfg, root / 'status.json', reuse_root=old.parent)
            self.assertTrue(second[-1]['reused'])
            self.assertTrue((new / 'linear' / 'COMMITTED.json').exists())
            self.assertFalse((new / 'source' / 'base' / 'base.pt').exists())
            self.assertIn('historical_verification', second[-1])
            del second
            gc.collect()


if __name__ == '__main__':
    unittest.main()
