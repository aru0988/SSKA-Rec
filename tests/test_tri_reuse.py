import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from sska_rec import core
import numpy as np
import torch

from sska_rec.stage2_models import build_model
from sska_rec.training import _commit


class TriReuseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        previous = core.DEVICE
        core.DEVICE = torch.device('cpu')
        self.addCleanup(setattr, core, 'DEVICE', previous)
        self.seed = 101
        self.legacy = self.root / str(self.seed)
        self.destination = self.root / 'new' / 'follow' / 'mpt'
        self.spec = dict(dataset='kuairand', tasks=['click', 'like', 'follow', 'forward'],
                         initial=['click', 'like'], xdim=3, rdim=4, tower=[4],
                         batch=512, lr=.001, sets={})
        torch.manual_seed(9)
        self.caches = {}
        cache_folder = self.root / 'current-cache'
        cache_folder.mkdir()
        for split in ['val', 'test']:
            y = np.column_stack([np.arange(8) % 2] * 4).astype('float32')
            rows = core.Rows(np.zeros((8, 1)), y, ['dense'], [0, 1])
            rows.row_ids = np.arange(8, dtype='int64') + (8 if split == 'test' else 0)
            self.spec['sets'][split] = rows
            self.caches[split] = {}
            for name, width in [('x', 3), ('g', 4), ('s0', 4), ('s1', 4)]:
                value = torch.randn(8, width)
                np.save(cache_folder / (split + '-' + name + '.npy'), value.numpy())
                self.caches[split][name] = value
            self.caches[split]['y'] = torch.from_numpy(y)
        self.base_folder = self.legacy / 'source' / 'base'
        self.base_folder.mkdir(parents=True)
        self.base_state = {'env_embedding_network.weight': torch.randn(2, 4)}
        torch.save(self.base_state, self.base_folder / 'base.pt')
        core.dump(self.base_folder / 'metrics.json', dict(model_hash='verified-base'))
        _commit(core, self.base_folder)
        self.receipt = dict(model_hashes=dict(base='verified-base'),
                            stages=dict(cache=dict(path=str(cache_folder))))
        self.old = self.legacy / 'no_u' / 'stage1'
        self.old.mkdir(parents=True)
        self.model = build_model(self.spec, None, self.seed + 1, 'mpt')
        self.model.eval()
        envs = list(self.base_state['env_embedding_network.weight'])
        raw = {}
        for split in ['val', 'test']:
            raw[split + '_y'] = self.spec['sets'][split].y[:, 2]
            raw[split + '_rowid'] = self.spec['sets'][split].row_ids
            raw[split + '_p'] = core.probabilities(self.model, self.caches[split], envs, self.spec)
        self.raw = raw
        torch.save(self.model.state_dict(), self.old / 'model.pt')
        np.savez_compressed(self.old / 'predictions.npz', **raw)
        self.metrics = dict(seed=self.seed + 4, model_hash=core.model_hash(self.model),
                            epochs=6, selected_epoch=1, stop_reason='patience', seconds=1.,
                            history=[dict(epoch=1, auc=.6)], optimizer_parameters=self.model.trainable_parameters,
                            total_parameters=self.model.trainable_parameters, peak_cuda_bytes=123,
                            val=core.measures(raw['val_y'], raw['val_p']),
                            test=core.measures(raw['test_y'], raw['test_p']))
        core.dump(self.old / 'metrics.json', self.metrics)
        _commit(core, self.old)

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('sska_rec.tri_reuse'))
        from sska_rec import tri_reuse
        return tri_reuse

    def reuse(self, **overrides):
        args = dict(spec=self.spec, seed=self.seed, target='follow', init_seed=self.seed + 4,
                    model=self.model, source_receipt=self.receipt,
                    destination=self.destination, legacy_root=self.legacy)
        args.update(overrides)
        return self.api().reuse_mpt(**args)

    def test_reuses_external_artifacts_and_revalidates_restart(self):
        self.assertTrue(self.reuse())
        self.assertEqual(self.api().artifact_folder(self.destination), self.old.resolve())
        self.assertEqual({p.name for p in self.destination.iterdir()},
                         {'metrics.json', 'REUSED.json', 'COMMITTED.json'})
        metrics = json.loads((self.destination / 'metrics.json').read_text())
        self.assertEqual((metrics['variant'], metrics['target'], metrics['paired_seed']),
                         ('mpt', 'follow', self.seed))
        self.assertEqual(metrics['cost_measurement_source'], 'historical-continual-20261030')
        self.assertTrue(self.reuse())

    def test_rejects_wrong_initial_seed_or_base(self):
        with self.assertRaisesRegex(ValueError, 'seed'):
            self.reuse(init_seed=self.seed + 3)
        self.receipt['model_hashes']['base'] = 'different-base'
        with self.assertRaisesRegex(ValueError, 'base'):
            self.reuse()

    def test_rejects_rowid_or_prediction_mismatch(self):
        self.spec['sets']['val'].row_ids = np.arange(8) + 77
        with self.assertRaisesRegex(ValueError, 'row'):
            self.reuse()
        self.spec['sets']['val'].row_ids = self.raw['val_rowid']
        self.raw['val_p'] = 1 - self.raw['val_p']
        np.savez_compressed(self.old / 'predictions.npz', **self.raw)
        _commit(core, self.old)
        with self.assertRaisesRegex(ValueError, 'prediction'):
            self.reuse()

    def test_committed_external_change_is_detected_after_reuse(self):
        self.reuse()
        with (self.old / 'model.pt').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            self.api().artifact_folder(self.destination)

    def test_budget_stop_reuses_only_complete_thirty_epoch_budget(self):
        self.metrics.update(stop_reason='budget', epochs=20)
        core.dump(self.old / 'metrics.json', self.metrics)
        _commit(core, self.old)
        with self.assertRaisesRegex(ValueError, 'budget'):
            self.reuse()
        self.metrics['epochs'] = 30
        core.dump(self.old / 'metrics.json', self.metrics)
        _commit(core, self.old)
        self.assertTrue(self.reuse())

    def test_other_dataset_has_no_reuse_candidate(self):
        self.spec['dataset'] = 'census'
        self.assertFalse(self.reuse())


if __name__ == '__main__':
    unittest.main()
