import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from sska_rec import core
from sska_rec.scheme_models import ConditionalReadout


class PaperReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        previous = core.DEVICE
        core.DEVICE = torch.device('cpu')
        self.addCleanup(setattr, core, 'DEVICE', previous)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.spec = dict(dataset='tiny', tasks=['a', 'b', 'c'], initial=['a', 'b'],
                         widths=[2, 2], xdim=4, rdim=4, lr=.01, batch=8, sets={})
        self.base = nn.Module()
        self.base.env_embedding_network = nn.Embedding(2, 4)
        self.base.gate_networks = nn.ModuleList([
            nn.Sequential(nn.Linear(4, 2), nn.Softmax(-1)) for _ in range(2)])
        with torch.no_grad():
            self.base.env_embedding_network.weight.fill_(1)
            for gate in self.base.gate_networks:
                gate[0].weight.zero_(); gate[0].bias.zero_()
        self.u = core.make_u(self.spec, 17)
        with torch.no_grad():
            self.u.mean.copy_(torch.tensor([.2, .3, .4, .5]))
            self.u.scale.copy_(torch.tensor([1., 2., 3., 4.]))
            self.u.encoder[0].weight.add_(.1)
        self.caches = {}
        for i, split in enumerate(('train', 'val', 'test')):
            y = np.tile(np.array([[0, 0, 1], [1, 1, 0]], dtype='float32'), (8, 1))
            sign = torch.from_numpy(y[:, 0] * 2 - 1)
            h = sign[:, None] * torch.tensor([[1., -1., .5, -.5]])
            rows = core.Rows(h.numpy(), y, ['a','b','c','d'], [])
            rows.row_ids = np.arange(16) + i*16
            self.spec['sets'][split] = rows
            self.caches[split] = dict(x=h, g=h, s0=h, s1=h, y=torch.from_numpy(y),
                                      old=torch.full((16, 2), .5))
        self.cfg = dict(epochs=30, patience=5)

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('sska_rec.paper_readouts'),
                             'Table 4 readout controls are missing')
        from sska_rec import paper_readouts
        return paper_readouts

    def test_zero_linear_is_exact_and_matches_probability_difference(self):
        api = self.api()
        head = api.LinearAlignedReadout(4)
        h, u = torch.randn(6, 4), torch.randn(6, 4)
        p = torch.tensor([0., 1., 1e-8, .3, .8, 1-1e-7])
        self.assertTrue(torch.equal(head(h,u,p), p))
        with torch.no_grad():
            head.linear.weight.fill_(.2); head.linear.bias.fill_(.1)
        logits = torch.logit(p.clamp(1e-7, 1-1e-7))
        difference = (torch.sigmoid(logits + head.linear(u).reshape(-1))
                      - torch.sigmoid(logits))
        expected = (p + difference).clamp(0, 1)
        torch.testing.assert_close(head(h,u,p), expected, rtol=0, atol=0)
        self.assertEqual(sum(v.numel() for v in head.parameters()), 5)

    def test_h_control_has_identical_capacity_and_uses_h_twice(self):
        api = self.api()
        head = api.make_control_head(self.spec, 17, 0, 'h_conditional')
        reference = ConditionalReadout(4)
        self.assertIs(type(head), type(reference))
        self.assertEqual(sum(v.numel() for v in head.parameters()),
                         sum(v.numel() for v in reference.parameters()))
        with patch.object(self.u, 'forward', side_effect=AssertionError('h control must not read U')):
            result = api.fit_control(self.base,self.u,self.spec,self.caches,0,
                                     self.root/'h',self.cfg,17,'h_conditional')
        self.assertEqual(result['optimizer_parameters'], 53)
        self.assertGreater(result['selected_epoch'], 0)
        self.assertEqual(result['val']['auc'], 1.)

    def test_random_source_copies_only_normalizer_and_preserves_rng(self):
        api = self.api()
        before = torch.random.get_rng_state().clone()
        random = api.make_random_u(self.spec, self.u, 17)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        expected = core.make_u(self.spec, 17)
        second = api.make_random_u(self.spec,self.u,17)
        self.assertEqual(core.model_hash(random),core.model_hash(second))
        self.assertTrue(torch.equal(random.mean, self.u.mean))
        self.assertTrue(torch.equal(random.scale, self.u.scale))
        self.assertNotEqual(random.mean.data_ptr(), self.u.mean.data_ptr())
        for key, value in random.encoder.state_dict().items():
            self.assertTrue(torch.equal(value,expected.encoder.state_dict()[key]))
        self.assertNotEqual(core.model_hash(random),core.model_hash(self.u))
        self.assertTrue(all(not p.requires_grad for p in random.parameters()))
        self.assertFalse(random.training)
        self.assertNotEqual(core.model_hash(random),core.model_hash(api.make_random_u(self.spec,self.u,18)))

    def test_all_controls_train_freeze_and_resume_independent_heads(self):
        api = self.api()
        before = (core.model_hash(self.base), core.model_hash(self.u))
        for variant in ('h_conditional','random_conditional','linear_aligned'):
            for position in (0,1):
                with self.subTest(variant=variant,position=position):
                    folder = self.root/variant/str(position)
                    result = api.fit_control(self.base,self.u,self.spec,self.caches,position,
                                             folder,self.cfg,17,variant)
                    self.assertGreater(result['selected_epoch'],0)
                    self.assertEqual(result['val']['auc'],1.)
                    self.assertEqual(result['target'],self.spec['initial'][position])
                    self.assertTrue((folder/'COMMITTED.json').exists())
                    self.assertIn('paper_readouts.py', result['contract']['code'])
                    self.assertEqual(result['contract']['variant'],variant)
                    if variant == 'random_conditional':
                        self.assertEqual(result['random_u_hash'],core.model_hash(api.make_random_u(self.spec,self.u,17)))
                        self.assertEqual(result['random_u_seed'],18)
                    with np.load(folder/'predictions.npz') as raw:
                        np.testing.assert_array_equal(raw['val_rowid'],self.spec['sets']['val'].row_ids)
                        np.testing.assert_array_equal(raw['val_base_p'],self.caches['val']['old'][:,position])
                    stamp = (folder/'model.pt').stat().st_mtime_ns
                    resumed = api.fit_control(self.base,self.u,self.spec,self.caches,position,
                                              folder,self.cfg,17,variant)
                    self.assertEqual(result,resumed)
                    self.assertEqual(stamp,(folder/'model.pt').stat().st_mtime_ns)
            a = json.loads((self.root/variant/'0/metrics.json').read_text())
            b = json.loads((self.root/variant/'1/metrics.json').read_text())
            self.assertNotEqual(a['model_seed'],b['model_seed'])
        self.assertEqual(before,(core.model_hash(self.base),core.model_hash(self.u)))
        self.assertTrue(all(not p.requires_grad for m in (self.base,self.u) for p in m.parameters()))
        self.assertTrue(all(not m.training for m in (self.base,self.u)))

    def test_perfect_baseline_ties_keep_epoch_zero_and_patience(self):
        api = self.api()
        for cache in self.caches.values():
            cache['old'] = .2 + .6*cache['y'][:,:2]
        for variant in ('h_conditional','random_conditional','linear_aligned'):
            folder = self.root/variant
            result = api.fit_control(self.base,self.u,self.spec,self.caches,0,
                                     folder,self.cfg,17,variant)
            self.assertEqual(result['selected_epoch'],0)
            self.assertEqual(result['epochs'],5)
            self.assertEqual(result['stop_reason'],'patience')
            self.assertEqual([h['epoch'] for h in result['history']],list(range(6)))
            with np.load(folder/'predictions.npz') as raw:
                np.testing.assert_array_equal(raw['val_p'],raw['val_base_p'])
                np.testing.assert_array_equal(raw['test_p'],raw['test_base_p'])

    def test_committed_contract_rejects_variant_and_random_source_change(self):
        api = self.api()
        folder = self.root/'random'
        api.fit_control(self.base,self.u,self.spec,self.caches,0,folder,self.cfg,17,'random_conditional')
        with self.assertRaisesRegex(ValueError,'identity'):
            api.fit_control(self.base,self.u,self.spec,self.caches,0,folder,self.cfg,17,'h_conditional')
        with torch.no_grad(): self.u.mean.add_(1)
        with self.assertRaisesRegex(ValueError,'identity'):
            api.fit_control(self.base,self.u,self.spec,self.caches,0,folder,self.cfg,17,'random_conditional')


if __name__ == '__main__':
    unittest.main()
