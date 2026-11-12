import importlib
import importlib.util
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sska_rec import core
import numpy as np
import torch


class SingleTaskFitTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('sska_rec.st_fit'),
                             'SingleTask training module must exist')
        self.fit = importlib.import_module('sska_rec.st_fit')
        torch.set_num_threads(1)
        self.old_device = core.DEVICE
        core.DEVICE = torch.device('cpu')
        self.addCleanup(setattr, core, 'DEVICE', self.old_device)
        rng = np.random.default_rng(41)
        x = np.column_stack([rng.normal(size=19), np.arange(19) % 3])
        y = np.column_stack([(np.arange(19) // k) % 2 for k in (1, 2, 3, 4)]).astype('float32')
        self.spec = dict(dataset='kuairand', tasks=['click', 'like', 'follow', 'forward'],
                         vocab={'cat': 3}, embed=2, xdim=3, hidden=[8, 4], tower=[4],
                         batch=8, lr=.001, sets={})
        for split in ('train', 'val', 'test'):
            rows = core.Rows(x, y, ['dense', 'cat'], [0, 1])
            rows.row_ids = np.arange(19) + 100
            self.spec['sets'][split] = rows
        self.cfg = dict(epochs=30, patience=5, embedding_l2=.006, dnn_l2=3e-5)

    def test_embedding_normalizer_and_all_network_updates(self):
        model = self.fit.build_single(self.spec, 54, self.cfg)
        self.assertEqual(model.reg_embedding, .006 / 6)
        emb = model.embedding_network.get_l2_reg() * .006 / 6
        expected = model.embedding_network.embedding_dict['cat'].weight.square().sum() * .006 / 12
        torch.testing.assert_close(emb, expected)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        x = {k: torch.tensor(self.spec['sets']['train'].x[:8, i])
             for i, k in enumerate(['dense', 'cat'])}
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        loss = torch.nn.functional.binary_cross_entropy(model(x)[0].reshape(-1),
                    torch.tensor(self.spec['sets']['train'].y[:8, 2])) + model.get_l2_reg()
        loss.backward()
        optimizer.step()
        for prefix in ('embedding_network.', 'base_network.', 'tower_network.'):
            self.assertTrue(any(not torch.equal(before[n], p) for n, p in model.named_parameters()
                                if n.startswith(prefix)), prefix)
        again = self.fit.build_single(self.spec, 54, self.cfg)
        for name, parameter in again.named_parameters():
            torch.testing.assert_close(parameter, before[name])

    def test_full_raw_fresh_resume_and_pending_recovery(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            folder = Path(tmp) / 'follow'
            status = Path(tmp) / 'status.json'
            result = self.fit.fit_single(self.spec, 50, 'follow', folder, self.cfg, status)
            self.assertEqual(result['model_seed'], 54)
            self.assertEqual(result['seed'], 54)
            self.assertEqual(result['paired_seed'], 50)
            self.assertEqual(result['total_parameters'], result['optimizer_parameters'])
            self.assertEqual(result['embedding_elements'], 6)
            self.assertTrue(result['started_at'].endswith('+00:00'))
            self.assertTrue((folder / 'COMMITTED.json').is_file())
            expected_epoch = max(result['history'], key=lambda r: r['val_auc'])['epoch']
            self.assertEqual(result['selected_epoch'], expected_epoch)
            for row in result['history']:
                self.assertAlmostEqual(row['train_loss'], row['train_bce'] + row['embedding_l2'] + row['dnn_l2'])
                self.assertIn('train_auc', row)
                self.assertIn('train_eval_logloss', row)
                self.assertEqual(row['rows_seen'], 19)
            with np.load(folder / 'predictions.npz') as raw:
                for split in ('val', 'test'):
                    np.testing.assert_array_equal(raw[split + '_y'], self.spec['sets'][split].y[:, 2])
                    np.testing.assert_array_equal(raw[split + '_rowid'], np.arange(19) + 100)
                    self.assertEqual(len(raw[split + '_p']), 19)
            with patch.object(torch.optim, 'Adam', side_effect=AssertionError('must not retrain')):
                self.assertEqual(result, self.fit.fit_single(self.spec, 50, 'follow', folder, self.cfg, status))
                (folder / 'COMMITTED.json').unlink()
                self.assertEqual(result, self.fit.fit_single(self.spec, 50, 'follow', folder, self.cfg, status))
            self.assertTrue((folder / 'COMMITTED.json').exists())
            # Even a valid commit cannot be reused for another target/model seed.
            with self.assertRaises(ValueError):
                self.fit.fit_single(self.spec, 50, 'forward', folder, self.cfg, status)
            (folder / 'predictions.npz').unlink()
            with self.assertRaises(ValueError):
                self.fit.fit_single(self.spec, 50, 'follow', folder, self.cfg, status)

    def test_protocol_and_nonfinite_rejected(self):
        for key, wrong in [('epochs', 29), ('patience', 4), ('embedding_l2', .006 / 2), ('dnn_l2', 0)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.fit.build_single(self.spec, 54, dict(self.cfg, **{key: wrong}))
        ali = dict(self.spec, dataset='aliccp')
        model = self.fit.build_single(ali, 54, self.cfg)
        self.assertEqual([layer.p for layer in model.base_network.modules()
                          if isinstance(layer, torch.nn.Dropout)], [.1, .3])
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            self.spec['sets']['train'].x[0, 0] = float('nan')
            with self.assertRaisesRegex(ValueError, 'finite'):
                self.fit.fit_single(self.spec, 50, 'follow', Path(tmp) / 'follow', self.cfg, Path(tmp) / 'status.json')

    def test_legacy_optimizer_sequence_and_weighted_tail_batch_match(self):
        # Independent corrected-Kuai loop: an uneven tail detects unweighted means.
        from sska_rec.training import _Target
        from torch.utils.data import DataLoader
        model = self.fit.build_single(self.spec, 55, self.cfg)
        loaders = {s: DataLoader(_Target(rows, 3), batch_size=8, shuffle=False)
                   for s, rows in self.spec['sets'].items()}
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        best, stale, selected = -float('inf'), 0, None
        first_losses = None
        for epoch in range(1, 31):
            model.train()
            totals = np.zeros(3)
            for x, y in loaders['train']:
                prediction = model(x)[0].reshape(-1)
                bce = torch.nn.functional.binary_cross_entropy(prediction, y.float())
                emb = (.006 / 6) * model.embedding_network.get_l2_reg()
                dnn = 3e-5 * (model.base_network.get_l2_reg() + model.tower_network.get_l2_reg())
                optimizer.zero_grad()
                (bce + emb + dnn).backward()
                optimizer.step()
                totals += np.array([float(bce.detach()), float(emb.detach()), float(dnn.detach())]) * len(y)
            if first_losses is None:
                first_losses = totals / 19
            model.eval()
            with torch.no_grad():
                # Training evaluation precedes val, as in the reference runner.
                for split in ('train', 'val'):
                    ps, ys = [], []
                    for x, y in loaders[split]:
                        ps.append(model(x)[0].reshape(-1))
                        ys.append(y)
                    score = core.measures(torch.cat(ys).numpy(), torch.cat(ps).numpy())
            if score['auc'] > best:
                best, stale, selected = score['auc'], 0, copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= 5:
                break
        model.load_state_dict(selected)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            folder = Path(tmp) / 'forward'
            result = self.fit.fit_single(self.spec, 50, 'forward', folder, self.cfg, Path(tmp) / 'status.json')
            self.assertEqual(result['model_hash'], core.model_hash(model))
            row = result['history'][0]
            np.testing.assert_allclose([row['train_bce'], row['embedding_l2'], row['dnn_l2']],
                                       first_losses, rtol=0, atol=1e-12)
            # Re-hash a corrupt pending output: semantic alignment must still fail.
            (folder / 'COMMITTED.json').unlink()
            with np.load(folder / 'predictions.npz') as raw:
                payload = {k: raw[k] for k in raw.files}
            payload['test_rowid'] = payload['test_rowid'] + 1
            np.savez_compressed(folder / 'predictions.npz', **payload)
            with self.assertRaisesRegex(ValueError, 'alignment'):
                self.fit.fit_single(self.spec, 50, 'forward', folder, self.cfg, Path(tmp) / 'status.json')


if __name__ == '__main__':
    unittest.main()
