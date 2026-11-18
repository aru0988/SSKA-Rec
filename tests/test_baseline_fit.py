import importlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sska_rec import core
import numpy as np
import torch


class BaselineFitTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('sska_rec.baseline_fit'),
                             'Joint baseline training module is missing')
        self.fit = importlib.import_module('sska_rec.baseline_fit')
        torch.set_num_threads(1)
        old = core.DEVICE
        core.DEVICE = torch.device('cpu')
        self.addCleanup(setattr, core, 'DEVICE', old)
        rng = np.random.default_rng(19)
        x = np.column_stack([rng.normal(size=17), np.arange(17) % 3])
        y = np.column_stack([(np.arange(17) // k) % 2 for k in (1, 2, 3, 4)]).astype('float32')
        self.spec = dict(dataset='census', tasks=['income', 'marital', 'education', 'sex'],
                         initial=['income', 'marital'], vocab={'cat': 3}, embed=2,
                         xdim=3, hidden=[8, 4], tower=[4], batch=8, lr=.001, sets={})
        for i, split in enumerate(('train', 'val', 'test')):
            rows = core.Rows(x.copy(), y.copy(), ['dense', 'cat'], [0, 1])
            rows.row_ids = np.arange(17) + i * 100
            self.spec['sets'][split] = rows

    def test_common_checkpoint_uses_mean_and_earliest_tie(self):
        history = [dict(epoch=1, val_mean_auc=.7), dict(epoch=2, val_mean_auc=.8),
                   dict(epoch=3, val_mean_auc=.8)]
        self.assertEqual(self.fit.selected_epoch(history), 2)

    def test_fit_raw_alignment_selection_and_verified_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'model'
            result = self.fit.fit_baseline(self.spec, 71, 'sharedbottom', folder, {}, Path(tmp)/'status.json')
            self.assertEqual(result['tasks'], self.spec['initial'])
            self.assertEqual(result['selected_epoch'], self.fit.selected_epoch(result['history']))
            self.assertEqual(result['protocol']['selection'], 'mean-initial-task-val-auc-earliest-best')
            for row in result['history']:
                self.assertAlmostEqual(row['val_mean_auc'], np.mean([v['auc'] for v in row['val'].values()]))
                self.assertEqual(row['rows_seen'], 17)
            with np.load(folder/'predictions.npz') as raw:
                for split in ('val', 'test'):
                    np.testing.assert_array_equal(raw[split+'_y'], self.spec['sets'][split].y[:, :2])
                    np.testing.assert_array_equal(raw[split+'_rowid'], self.spec['sets'][split].row_ids)
                    self.assertEqual(raw[split+'_p'].shape, (17, 2))
            with patch.object(torch.optim, 'Adam', side_effect=AssertionError('completed models must not retrain')):
                again = self.fit.fit_baseline(self.spec, 71, 'sharedbottom', folder, {}, Path(tmp)/'status.json')
                self.assertEqual(result, again)
            (folder/'model.pt').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                self.fit.fit_baseline(self.spec, 71, 'sharedbottom', folder, {}, Path(tmp)/'status.json')

    def test_heldout_target_labels_do_not_affect_learning(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.fit.fit_baseline(self.spec, 71, 'sharedbottom', Path(tmp)/'a', {}, Path(tmp)/'s.json')
            for rows in self.spec['sets'].values():
                rows.y[:, 2:] = 1 - rows.y[:, 2:]
            self.fit.fit_baseline(self.spec, 71, 'sharedbottom', Path(tmp)/'b', {}, Path(tmp)/'s.json')
            with np.load(Path(tmp)/'a/predictions.npz') as a, np.load(Path(tmp)/'b/predictions.npz') as b:
                np.testing.assert_array_equal(a['test_p'], b['test_p'])


if __name__ == '__main__':
    unittest.main()
