"""Small real-tensor contracts for complementary pretraining and readout stages."""
import copy
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sska_rec import core
import numpy as np
import torch
from torch import nn


class SchemeFittingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        generator = torch.Generator().manual_seed(17)
        self.caches, sets = {}, {}
        for i, split in enumerate(('train', 'val', 'test')):
            x = torch.randn(12, 3, generator=generator) + (20 if split != 'train' else 0)
            y = torch.tensor([[j % 2, (j // 2) % 2] for j in range(12)]).float()
            self.caches[split] = dict(x=x, g=x[:, :2].clone(), s0=x[:, :2].clone() / 2,
                                     s1=x[:, :2].clone() / 3, y=y, old=.1 + .8*y)
            class Rows(SimpleNamespace):
                def __len__(self):
                    return len(self.y)
            sets[split] = Rows(y=y.numpy(), x=x.numpy(), row_ids=np.arange(i*12, (i+1)*12))
        self.spec = dict(dataset='toy', sets=sets, tasks=['a', 'b'], initial=['a', 'b'],
                         widths=[1, 2], xdim=3, rdim=2, lr=.005, batch=4)
        self.cfg = dict(decoder_epochs=2, ssl_epochs=2, epochs=2, patience=2)

    def fitting(self):
        self.assertIsNotNone(importlib.util.find_spec('sska_rec.scheme_fitting'), 'Resumable scheme stage fitting must exist')
        return importlib.import_module('sska_rec.scheme_fitting')

    def test_decoder_train_only_normalizer_and_verified_resume(self):
        fit = self.fitting().fit_decoder
        before = copy.deepcopy(self.caches)
        decoder, result = fit(self.spec, self.caches, self.root/'decoder', self.cfg, 19)
        expected = self.caches['train']['x'].double()
        torch.testing.assert_close(decoder.mean.cpu().double(), expected.mean(0), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(decoder.scale.cpu().double(), expected.std(0, unbiased=False).clamp_min(.1), rtol=1e-6, atol=1e-6)
        self.assertTrue(np.isfinite([r['val_loss'] for r in result['history']]).all())
        saved = (self.root/'decoder'/'model.pt').stat().st_mtime_ns
        resumed, second = fit(self.spec, self.caches, self.root/'decoder', self.cfg, 19)
        self.assertEqual(result, second)
        self.assertEqual(saved, (self.root/'decoder'/'model.pt').stat().st_mtime_ns)
        self.assertEqual(core.model_hash(decoder), core.model_hash(resumed))
        for split in before:
            for key in before[split]:
                self.assertTrue(torch.equal(before[split][key], self.caches[split][key]))
        with self.assertRaisesRegex(ValueError, 'contract|identity|protocol'):
            fit(self.spec, self.caches, self.root/'decoder', dict(self.cfg, patience=3), 19)
        self.assertTrue((self.root/'decoder'/'COMMITTED.json').is_file())

    def test_fresh_complementary_u_and_frozen_decoder_resume(self):
        fitting = self.fitting()
        decoder, _ = fitting.fit_decoder(self.spec, self.caches, self.root/'decoder', self.cfg, 19)
        before = core.model_hash(decoder)
        u, result = fitting.fit_complementary_u(self.spec, self.caches, decoder,
            (decoder.mean, decoder.scale), self.root/'u', self.cfg, 19)
        self.assertEqual(before, core.model_hash(decoder))
        self.assertTrue(np.isfinite([r['val_loss'] for r in result['history']]).all())
        resumed, second = fitting.fit_complementary_u(self.spec, self.caches, decoder,
            (decoder.mean, decoder.scale), self.root/'u', self.cfg, 19)
        self.assertEqual(core.model_hash(u), core.model_hash(resumed))
        self.assertEqual(result, second)
        self.assertTrue((self.root/'u'/'source.pt').is_file())
        with torch.no_grad():
            next(decoder.parameters()).add_(1)
        with self.assertRaisesRegex(ValueError, 'contract|identity|protocol'):
            fitting.fit_complementary_u(self.spec, self.caches, decoder,
                (decoder.mean, decoder.scale), self.root/'u', self.cfg, 19)

    def test_readout_epoch0_frozen_sources_raw_rows_and_recovery(self):
        fitting = self.fitting()
        from sska_rec.expert import UniversalExpert
        from sska_rec.backbone import MPTRec
        base = MPTRec(2, {}, 1, 3, [2], [2], device=core.DEVICE).to(core.DEVICE)
        u = UniversalExpert([1, 2], 2, seed=20).to(core.DEVICE)
        original = (core.model_hash(base), core.model_hash(u))
        folder = self.root/'readout'
        result = fitting.fit_readout(base, u, self.spec, self.caches, 0, folder, self.cfg, 19)
        self.assertEqual(result['selected_epoch'], 0)  # original AUC already 1
        self.assertEqual(original, (core.model_hash(base), core.model_hash(u)))
        with np.load(folder/'predictions.npz') as raw:
            for split in ('val', 'test'):
                np.testing.assert_array_equal(raw[split+'_p'], raw[split+'_base_p'])
                np.testing.assert_array_equal(raw[split+'_rowid'], self.spec['sets'][split].row_ids)
        (folder/'COMMITTED.json').unlink()
        self.assertEqual(result, fitting.fit_readout(base, u, self.spec, self.caches, 0, folder, self.cfg, 19))
        self.assertTrue((folder/'COMMITTED.json').is_file())
        with (folder/'model.pt').open('ab') as stream:
            stream.write(b'corruption')
        with self.assertRaisesRegex(ValueError, 'integrity|hash'):
            fitting.fit_readout(base, u, self.spec, self.caches, 0, folder, self.cfg, 19)

    def test_changed_rows_and_duplicate_split_ids_are_rejected(self):
        fitting = self.fitting()
        fitting.fit_decoder(self.spec, self.caches, self.root/'decoder', self.cfg, 19)
        self.spec['sets']['val'].row_ids[0] = 0
        with self.assertRaisesRegex(ValueError, 'row|identity|contract'):
            fitting.fit_decoder(self.spec, self.caches, self.root/'decoder', self.cfg, 19)

    def test_committed_missing_metrics_is_integrity_error(self):
        fitting = self.fitting()
        folder = self.root/'decoder'
        fitting.fit_decoder(self.spec, self.caches, folder, self.cfg, 19)
        (folder/'metrics.json').unlink()
        with self.assertRaisesRegex(ValueError, 'integrity|metrics'):
            fitting.fit_decoder(self.spec, self.caches, folder, self.cfg, 19)

    def test_pending_readout_rejects_raw_corruption_before_commit(self):
        fitting = self.fitting()
        from sska_rec.expert import UniversalExpert
        from sska_rec.backbone import MPTRec
        base = MPTRec(2, {}, 1, 3, [2], [2], device=core.DEVICE).to(core.DEVICE)
        u = UniversalExpert([1, 2], 2, seed=20).to(core.DEVICE)
        folder = self.root/'head'
        fitting.fit_readout(base, u, self.spec, self.caches, 1, folder, self.cfg, 19)
        (folder/'COMMITTED.json').unlink()
        with np.load(folder/'predictions.npz') as raw:
            arrays = {key: raw[key] for key in raw.files}
        arrays['val_rowid'][0] += 100
        np.savez_compressed(folder/'predictions.npz', **arrays)
        with self.assertRaisesRegex(ValueError, 'integrity|hash|rowid'):
            fitting.fit_readout(base, u, self.spec, self.caches, 1, folder, self.cfg, 19)
        self.assertFalse((folder/'COMMITTED.json').exists())

    def test_incomplete_attempt_preserved_and_mismatch_never_mutates(self):
        fitting = self.fitting()
        folder = self.root/'decoder'
        fitting.fit_decoder(self.spec, self.caches, folder, self.cfg, 19)
        (folder/'COMMITTED.json').unlink()
        (folder/'metrics.json').unlink()
        (folder/'partial.bin').write_bytes(b'checkpoint to preserve')
        snapshot = {p.name:p.read_bytes() for p in folder.iterdir()}
        with self.assertRaisesRegex(ValueError, 'contract|identity'):
            fitting.fit_decoder(self.spec, self.caches, folder, dict(self.cfg, decoder_epochs=3), 19)
        self.assertEqual(snapshot, {p.name:p.read_bytes() for p in folder.iterdir()})
        fitting.fit_decoder(self.spec, self.caches, folder, self.cfg, 19)
        archives = list(self.root.glob('decoder.interrupted-*'))
        self.assertEqual(len(archives), 1)
        self.assertEqual((archives[0]/'partial.bin').read_bytes(), b'checkpoint to preserve')

    def test_wrong_u_normalizer_rejects_before_creating_leaf(self):
        fitting = self.fitting()
        decoder, _ = fitting.fit_decoder(self.spec, self.caches, self.root/'decoder', self.cfg, 19)
        folder = self.root/'u'
        with self.assertRaisesRegex(ValueError, 'normalizer'):
            fitting.fit_complementary_u(self.spec, self.caches, decoder,
                (decoder.mean + 1, decoder.scale), folder, self.cfg, 19)
        self.assertFalse(folder.exists())

    def test_pending_decoder_validation_loss_is_recomputed(self):
        fitting = self.fitting()
        folder = self.root/'decoder'
        fitting.fit_decoder(self.spec, self.caches, folder, self.cfg, 19)
        (folder/'COMMITTED.json').unlink()
        result = json.loads((folder/'metrics.json').read_text())
        result['val_loss'] += 3
        (folder/'metrics.json').write_text(json.dumps(result), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'loss|metric'):
            fitting.fit_decoder(self.spec, self.caches, folder, self.cfg, 19)
        self.assertFalse((folder/'COMMITTED.json').exists())

    def test_pending_u_validation_loss_is_recomputed(self):
        fitting = self.fitting()
        decoder, _ = fitting.fit_decoder(self.spec, self.caches, self.root/'decoder', self.cfg, 19)
        folder = self.root/'u'
        fitting.fit_complementary_u(self.spec, self.caches, decoder,
            (decoder.mean, decoder.scale), folder, self.cfg, 19)
        (folder/'COMMITTED.json').unlink()
        result = json.loads((folder/'metrics.json').read_text())
        result['val_loss'] += 3
        (folder/'metrics.json').write_text(json.dumps(result), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'loss|metric'):
            fitting.fit_complementary_u(self.spec, self.caches, decoder,
                (decoder.mean, decoder.scale), folder, self.cfg, 19)
        self.assertFalse((folder/'COMMITTED.json').exists())


if __name__ == '__main__':
    unittest.main()
