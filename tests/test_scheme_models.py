"""Behavioral contracts for the two independent scheme model families."""

import copy
import importlib
import importlib.util
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from sska_rec.backbone import MPTRec
from sska_rec.expert import UniversalExpert
from sska_rec.stage2_models import build_model as historical_model


class SchemeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("sska_rec.scheme_models"),
                             "Scheme model implementation is missing")
        return importlib.import_module("sska_rec.scheme_models")

    def fixture(self):
        spec = dict(xdim=7, rdim=8, widths=[1, 2, 4], tower=[6, 4])
        source = UniversalExpert(spec['widths'], rep_dim=8, seed=31)
        generator = torch.Generator().manual_seed(23)
        batch = {k: torch.randn(9, width, generator=generator)
                 for k, width in [('x', 7), ('g', 8), ('s0', 8), ('s1', 8)]}
        envs = [torch.randn(8, generator=generator) for _ in range(2)]
        return spec, source, batch, envs

    def test_decoder_normalizes_like_u_and_maps_all_gs_blocks(self):
        mod = self.module()
        decoder = mod.GSDecoder(8, 7)
        source = UniversalExpert([1, 2, 4], rep_dim=8)
        mean, std = torch.arange(7).float(), torch.arange(7).float() / 10
        decoder.set_normalizer(mean, std)
        source.set_normalizer(mean, std)
        x = torch.tensor([[100., -100., 2., 5., 6., 7., 8.]])
        self.assertTrue(torch.equal(decoder.normalize(x), source.normalize(x)))
        self.assertEqual(sum(p.numel() for p in decoder.parameters()), 175)
        self.assertEqual(decoder(torch.randn(1, 24)).shape, (1, 7))
        self.assertIn('mean', decoder.state_dict())
        self.assertIn('scale', decoder.state_dict())

    def test_complementary_target_equal_fields_and_frozen_dependencies(self):
        mod = self.module()
        spec, _, batch, _ = self.fixture()
        expert = mod.ComplementaryExpert(spec['widths'], rep_dim=8, seed=5)
        old = UniversalExpert(spec['widths'], rep_dim=8, seed=5)
        self.assertEqual(set(expert.state_dict()), set(old.state_dict()))
        for key, value in old.state_dict().items():
            self.assertTrue(torch.equal(value, expert.state_dict()[key]), key)
        decoder = mod.GSDecoder(8, 7)
        for value in batch.values():
            value.requires_grad_()
        mask = torch.tensor([[True, False, True]] * len(batch['x']))
        g, specifics = batch['g'], [batch['s0'], batch['s1']]
        target = expert.normalize(batch['x']) - decoder(torch.cat([g, *specifics], 1)).detach()
        predicted = expert.decoder(expert.encoder(expert.masked_input(batch['x'], mask)))
        errors = torch.stack([v.mean(1) for v in (predicted-target).square().split(spec['widths'], 1)], 1)
        expected = (errors * mask).sum() / mask.sum()
        losses = expert.losses(batch['x'], g, specifics, decoder, mask=mask,
                               redundancy_weight=.03, variance_weight=.04)
        self.assertTrue(torch.allclose(expected, losses['reconstruction']))
        self.assertTrue(torch.allclose(losses['total'], expected + .03*losses['decorrelation'] + .04*losses['variance']))
        clean = expert(batch['x'])
        standardize = lambda v: (v-v.mean(0))/v.std(0, unbiased=False).clamp_min(.01)
        corr = torch.stack([(standardize(clean).T @ standardize(v.detach()) / len(clean)).square().mean()
                            for v in [g, *specifics]]).mean()
        self.assertTrue(torch.allclose(corr, losses['decorrelation']))
        losses['total'].backward()
        self.assertTrue(all(p.grad is None for p in decoder.parameters()))
        self.assertTrue(all(value.grad is None for value in batch.values()))
        self.assertTrue(all(p.grad is not None for p in expert.parameters()))

    def test_conditional_readout_counts_zero_endpoints_and_only_head_gradients(self):
        mod = self.module()
        for rdim, count in [(128, 1541), (64, 773)]:
            head = mod.ConditionalReadout(rdim)
            self.assertEqual(sum(p.numel() for p in head.parameters()), count)
            p = torch.tensor([0., 1., 1e-8, .3, .99999999], requires_grad=True)
            h = torch.randn(5, rdim, requires_grad=True)
            u = torch.randn(5, rdim, requires_grad=True)
            self.assertTrue(torch.equal(head.delta(h, u), torch.zeros(5)))
            self.assertTrue(torch.equal(head(h, u, p), p))
            head(h, u, p).sum().backward()
            self.assertIsNone(h.grad)
            self.assertIsNone(u.grad)
            self.assertIsNone(p.grad)
            self.assertTrue(all(parameter.grad is not None for parameter in head.parameters()))
            with torch.no_grad():
                head.out.weight.fill_(.25)
                head.out.bias.fill_(.1)
            expected_delta = head.out(torch.sigmoid(head.A(torch.cat([F.layer_norm(h, (rdim,)), F.layer_norm(u, (rdim,))], 1))) * head.C(F.layer_norm(u, (rdim,)))).reshape(-1)
            self.assertTrue(torch.allclose(head.delta(h, u), expected_delta))

    def test_old_task_hidden_is_exact_original_fusion_including_one_row(self):
        mod = self.module()
        _, _, batch, _ = self.fixture()
        base = MPTRec(2, {}, 4, 7, [12, 8], [6, 4])
        for n in [9, 1]:
            rows = {k: v[:n] for k, v in batch.items()}
            for position in [0, 1]:
                env = base.env_embedding_network.weight[position]
                experts = torch.stack([rows[f's{position}'] * env, rows['g']], 2)
                expected = (experts @ base.gate_networks[position](rows['x']).unsqueeze(2)).squeeze(-1)
                actual = mod.old_task_hidden(base, rows, position)
                self.assertEqual(actual.shape, (n, 8))
                self.assertTrue(torch.equal(actual, expected))
                self.assertFalse(actual.requires_grad)

    def test_scheme1_exact_lora_counts_for_all_three_dataset_architectures(self):
        mod = self.module()
        cases = [(119, 128, [1]*11+[4]*27, [64, 32], 30901),
                 (17, 128, [1]+[4]*4, [64, 32], 23833),
                 (80, 64, [5]*16, [32, 32], 11233)]
        for xdim, rdim, widths, tower, count in cases:
            spec = dict(xdim=xdim, rdim=rdim, widths=widths, tower=tower)
            model = mod.build_scheme1_model(spec, UniversalExpert(widths, rep_dim=rdim), 17, 'tri_lora')
            self.assertEqual(model.trainable_parameters, count)

    def test_scheme1_initialization_scopes_private_source_and_rng(self):
        mod = self.module()
        spec, source, batch, envs = self.fixture()
        source_state = copy.deepcopy(source.state_dict())
        flags = [p.requires_grad for p in source.parameters()]
        rng = torch.random.get_rng_state().clone()
        variants = ['mpt', 'tri_frozen', 'tri_lora', 'tri_ssf', 'tri_full']
        models = [mod.build_scheme1_model(spec, source, 17, v) for v in variants]
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        old = historical_model(spec, source, 17, 'mpt')
        for key, value in old.state_dict().items():
            self.assertTrue(torch.equal(value, models[0].state_dict()[key]), key)
        initial_gate = models[1].core.gate_network[0].weight.detach().clone()
        initial_prediction = models[1](batch, envs).detach().clone()
        for model in models[1:]:
            for key, value in old.core.state_dict().items():
                if not key.startswith('gate_network.'):
                    self.assertTrue(torch.equal(value, model.core.state_dict()[key]), key)
            gate = model.core.gate_network[0]
            self.assertEqual(gate.weight.shape, (3, 24))
            self.assertIsNone(gate.bias)
            self.assertTrue(torch.equal(gate.weight, initial_gate))
            self.assertIsNot(source, model.u)
            model.u.decoder.register_forward_pre_hook(lambda *_: self.fail('prediction called decoder'))
            self.assertTrue(torch.equal(initial_prediction, model(batch, envs)))
            model.train()
            self.assertFalse(model.u.decoder.training)
            if model.variant == 'tri_frozen':
                self.assertFalse(model.u.training)
            model(batch, envs).sum().backward()
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    self.assertIsNone(parameter.grad, name)
                else:
                    self.assertIsNotNone(parameter.grad, name)
            for name, parameter in model.u.named_parameters():
                expected = (not name.startswith('decoder.')) and (
                    model.variant == 'tri_full' or
                    model.variant == 'tri_lora' and name.endswith(('.A', '.B')) or
                    model.variant == 'tri_ssf' and name.endswith(('.scale', '.shift')))
                self.assertEqual(parameter.requires_grad, expected, (model.variant, name))
            torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1).step()
        for key, value in source_state.items():
            self.assertTrue(torch.equal(value, source.state_dict()[key]), key)
        self.assertEqual(flags, [p.requires_grad for p in source.parameters()])
        for model in models:
            self.assertEqual(model({k: v[:1] for k, v in batch.items()}, envs).shape, (1,))
        with self.assertRaisesRegex(ValueError, 'variant'):
            mod.build_scheme1_model(spec, source, 17, 'residual')

    def test_scheme1_routes_normalized_gsu_and_fuses_original_values(self):
        mod = self.module()
        spec, source, batch, envs = self.fixture()
        model = mod.build_scheme1_model(spec, source, 17, 'tri_frozen')
        core = model.core
        weights = (core.projection_network(batch['x']) @ torch.stack(envs, 1) / core.temperature).softmax(-1)
        specific = (torch.stack([batch['s0'], batch['s1']], 2) @ weights.unsqueeze(2)).squeeze(-1)
        values = [batch['g'], specific*core.env_embedding_network(core.new_env_idx).squeeze(0), model.u(batch['x'])]
        router_input = torch.cat([F.layer_norm(value, (8,)) for value in values], 1)
        fused = (torch.stack(values, 2) @ core.gate_network(router_input).unsqueeze(2)).squeeze(-1)
        self.assertTrue(torch.equal(model(batch, envs), core.tower_network(fused).reshape(-1)))


if __name__ == '__main__':
    unittest.main()
