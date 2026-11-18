"""Table 2 model behavior, source equivalence, and regularization contracts."""

import importlib
import importlib.util
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from sska_rec import core


class BaselineModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('sska_rec.baseline_models'),
                             'Table 2 baseline model implementation is missing')
        return importlib.import_module('sska_rec.baseline_models')

    def spec(self):
        return dict(dataset='census', vocab={'category': 5, 'other': 3},
                    embed=2, xdim=5, hidden=[8, 4], tower=[3, 2])

    def features(self, size):
        return dict(numeric=torch.linspace(-1., 1., size, device=core.DEVICE),
                    category=torch.arange(size, device=core.DEVICE) % 5,
                    other=torch.arange(size, device=core.DEVICE) % 3)

    def test_forward_backward_and_singleton_for_all_models(self):
        mod = self.module()
        for name in ('sharedbottom', 'mmoe', 'ple', 'adatt'):
            for size in (1, 7):
                with self.subTest(model=name, size=size):
                    model = mod.build_baseline(self.spec(), 29, name)
                    prediction = model(self.features(size))
                    self.assertEqual(len(prediction), 2)
                    target = torch.arange(size, device=core.DEVICE).float() % 2
                    loss = model.get_l2_reg()
                    for p in prediction:
                        self.assertEqual(p.shape, (size,))
                        self.assertEqual(p.dtype, torch.float32)
                        self.assertTrue(torch.isfinite(p).all())
                        self.assertTrue(((p >= 0) & (p <= 1)).all())
                        loss = loss + F.binary_cross_entropy(p, target)
                    loss.backward()
                    for key, p in model.embedding_network.named_parameters():
                        self.assertTrue(p.requires_grad, key)
                        self.assertIsNotNone(p.grad, key)
                        self.assertTrue(torch.isfinite(p.grad).all(), key)

    def test_same_seed_reproduces_parameters_and_changed_seed_changes_them(self):
        mod = self.module()
        for name in ('sharedbottom', 'mmoe', 'ple', 'adatt'):
            first = mod.build_baseline(self.spec(), 17, name)
            second = mod.build_baseline(self.spec(), 17, name)
            third = mod.build_baseline(self.spec(), 18, name)
            self.assertEqual(core.model_hash(first), core.model_hash(second))
            self.assertNotEqual(core.model_hash(first), core.model_hash(third))

    def test_extracted_models_match_unadapted_source_for_nonsingleton_batches(self):
        mod = self.module()
        # The untouched source subset is bundled alongside the batch-safe subset.
        source = importlib.import_module('sska_rec.vendor.mpt_baselines_original')
        safe = importlib.import_module('sska_rec.vendor.mpt_baselines')
        common = dict(num_tasks=2, feature_vocabulary={'category': 5, 'other': 3},
                      embedding_size=2, input_size=5, tower_dnn_hidden_units=[3, 2],
                      reg_embedding=.003, reg_dnn=3e-5)
        for cls, extras in (
            ('SharedBottom', dict(shared_dnn_hidden_units=[8, 4])),
            ('MMOE', dict(num_experts=3, expert_dnn_hidden_units=[8, 4])),
            ('PLE', dict(expert_dnn_hidden_units=[4], num_levels=2)),
        ):
            original = getattr(source, cls)(**common, **extras).to(core.DEVICE)
            adapted = getattr(safe, cls)(**common, **extras).to(core.DEVICE)
            adapted.load_state_dict(original.state_dict())
            for a, b in zip(original(self.features(7)), adapted(self.features(7))):
                self.assertTrue(torch.equal(a, b), cls)
            self.assertTrue(torch.equal(original.get_l2_reg(), adapted.get_l2_reg()))

    def test_ple_progresses_through_dataset_widths_and_layer_dropout(self):
        mod = self.module()
        spec = self.spec()
        spec['dataset'] = 'aliccp'
        model = mod.build_baseline(spec, 17, 'ple')
        for level, (input_width, output_width, dropout) in enumerate(((5, 8, .1), (8, 4, .3))):
            cgc = model.cgc_networks[level]
            self.assertEqual(len(cgc.specific_expert_networks), 2)
            self.assertEqual(len(cgc.shared_expert_networks), 1)
            for expert in (*cgc.specific_expert_networks, *cgc.shared_expert_networks):
                linear = next(x for x in expert.modules() if isinstance(x, nn.Linear))
                self.assertEqual((linear.in_features, linear.out_features), (input_width, output_width))
                self.assertEqual(next(x.p for x in expert.modules() if isinstance(x, nn.Dropout)), dropout)

    def test_adatt_matches_pinned_official_sp_module_and_has_local_residual(self):
        mod = self.module()
        from sska_rec.vendor.adatt_mtl_lib import AdaTTSp
        model = mod.build_baseline(self.spec(), 23, 'adatt')
        official = AdaTTSp(input_dim=5, expert_out_dims=[[8], [4]], num_tasks=2, num_task_experts=1, self_exp_res_connect=True).to(core.DEVICE)
        official.load_state_dict(model.extraction.state_dict())
        embedded = model.embedding_network(self.features(7)).unsqueeze(1).expand(-1, 2, -1)
        expected = official(embedded)
        actual = model.extraction(embedded)
        if isinstance(expected, list):
            for a, b in zip(actual, expected):
                self.assertTrue(torch.equal(a, b))
        else:
            self.assertTrue(torch.equal(actual, expected))
        for task, prediction in enumerate(model(self.features(7))):
            self.assertTrue(torch.equal(
                prediction, model.tower_networks[task](expected[:, task, :]).squeeze(-1)))
        # Independently reconstruct all-task gating plus the unit native residual.
        manual = embedded
        for experts, gates in zip(official.experts, official.gate_weights):
            expert_outputs = torch.stack([expert(manual[:, task, :])
                                          for task, expert in enumerate(experts)], dim=1)
            weights = torch.stack([gate(manual[:, task, :])
                                   for task, gate in enumerate(gates)], dim=1)
            manual = torch.bmm(weights, expert_outputs) + expert_outputs
        self.assertTrue(torch.equal(manual, expected))
        self.assertFalse(any(isinstance(x, nn.Dropout) for x in model.extraction.modules()))
        self.assertTrue(model.extraction.self_exp_res_connect)

    def test_regularization_excludes_gates_and_biases_and_normalizes_embedding(self):
        mod = self.module()
        for name in ('sharedbottom', 'mmoe', 'ple', 'adatt'):
            model = mod.build_baseline(self.spec(), 1, name)
            embedding = sum(p.square().sum() / 2 for p in model.embedding_network.parameters())
            if name == 'sharedbottom':
                modules = [model.shared_bottom, *model.tower_networks]
            elif name == 'mmoe':
                modules = [*model.expert_networks, *model.tower_networks]
            elif name == 'ple':
                modules = [expert for cgc in model.cgc_networks for expert in
                           (*cgc.specific_expert_networks, *cgc.shared_expert_networks)]
                modules += list(model.tower_networks)
            else:
                modules = [*model.extraction.experts, *model.tower_networks]
            dnn = sum(layer.weight.square().sum() / 2 for module in modules
                      for layer in module.modules() if isinstance(layer, nn.Linear))
            expected = embedding * (.006 / 16) + dnn * 3e-5
            self.assertTrue(torch.allclose(expected, model.get_l2_reg()), name)

    def test_rejects_unregistered_regularization_aliases(self):
        mod = self.module()
        for key in ('reg_embedding', 'reg_dnn'):
            with self.assertRaisesRegex(ValueError, key):
                mod.build_baseline(self.spec(), 1, 'sharedbottom', {key: 0})


if __name__ == '__main__':
    unittest.main()
