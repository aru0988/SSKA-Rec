"""Stage2 prediction, initialization, and allowed-update contracts."""

import copy
import importlib
import math
import unittest

import torch
from torch import nn

from sska_rec.backbone import NewTask
from sska_rec.expert import UniversalExpert


class Stage2ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.spec = dict(xdim=119, rdim=128, tower=[64, 32], widths=[1] * 11 + [4] * 27)
        self.source = UniversalExpert(self.spec["widths"], seed=29)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(23)
            self.batch = dict(
                x=torch.randn(7, 119),
                g=torch.randn(7, 128),
                s0=torch.randn(7, 128),
                s1=torch.randn(7, 128),
            )
            self.envs = [torch.randn(128), torch.randn(128)]

    def build(self, variant, source=None):
        module = importlib.import_module("sska_rec.stage2_models")
        return module.build_model(
            self.spec, self.source if source is None else source, 17, variant
        )

    def test_trainable_counts_and_source_copy(self):
        source_state = copy.deepcopy(self.source.state_dict())
        source_flags = [p.requires_grad for p in self.source.parameters()]
        expected = dict(
            mpt=26799,
            residual=116528,
            tri_frozen=26918,
            tri_lora=30106,
            tri_full=100262,
        )
        for variant, count in expected.items():
            with self.subTest(variant=variant):
                model = self.build(variant)
                self.assertEqual(model.trainable_parameters, count)
                self.assertEqual(
                    sum(p.numel() for p in model.parameters() if p.requires_grad), count
                )
                if variant == "mpt":
                    self.assertFalse(
                        any(
                            name.startswith("u.")
                            for name, _ in model.named_parameters()
                        )
                    )
                    continue
                self.assertIsNot(model.u, self.source)
                self.assertTrue(
                    all(not p.requires_grad for p in model.u.decoder.parameters())
                )
                for key in ["mean", "scale"]:
                    self.assertNotEqual(
                        getattr(model.u, key).data_ptr(),
                        getattr(self.source, key).data_ptr(),
                    )
                with torch.no_grad():
                    next(model.u.parameters()).add_(1)
        for key, tensor in source_state.items():
            self.assertTrue(torch.equal(self.source.state_dict()[key], tensor), key)
        self.assertEqual(
            source_flags, [p.requires_grad for p in self.source.parameters()]
        )

    def test_shared_head_initialization_and_seed(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20)
            original = NewTask(119, 128, [64, 32], 3e-5)
        reference = self.build("mpt")
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, reference.core.state_dict()[key]), key)
        tri_gate = None
        for variant in ["residual", "tri_frozen", "tri_lora", "tri_full"]:
            model = self.build(variant)
            for key, value in reference.core.state_dict().items():
                if variant.startswith("tri_") and key.startswith("gate_network."):
                    continue
                self.assertTrue(
                    torch.equal(value, model.core.state_dict()[key]), (variant, key)
                )
            self.assertTrue(torch.equal(model.get_l2_reg(), reference.get_l2_reg()))
            if variant.startswith("tri_"):
                gate = model.core.gate_network[0]
                self.assertEqual(gate.weight.shape, (3, 119))
                self.assertIsNone(gate.bias)
                if tri_gate is None:
                    tri_gate = gate.weight.detach().clone()
                self.assertTrue(torch.equal(gate.weight, tri_gate))

    def test_tri_initial_predictions_and_explicit_expert_order(self):
        models = [
            self.build(variant) for variant in ["tri_frozen", "tri_lora", "tri_full"]
        ]
        outputs = [model(self.batch, self.envs) for model in models]
        for output in outputs[1:]:
            self.assertTrue(torch.equal(output, outputs[0]))
        model = models[0]
        h = model.core
        weights = (
            h.projection_network(self.batch["x"]) @ torch.stack(self.envs, 1) / 150
        ).softmax(-1)
        specific = (
            torch.stack([self.batch["s0"], self.batch["s1"]], 2) @ weights.unsqueeze(2)
        ).squeeze(-1)
        experts = torch.stack(
            [
                specific * h.env_embedding_network(h.new_env_idx).squeeze(0),
                self.batch["g"],
                model.u(self.batch["x"]),
            ],
            2,
        )
        expected = h.tower_network(
            (experts @ h.gate_network(self.batch["x"]).unsqueeze(2)).squeeze(-1)
        ).reshape(-1)
        self.assertTrue(torch.equal(expected, outputs[0]))

    def test_lora_zero_delta_and_trainable_factors_only(self):
        model = self.build("tri_lora")
        self.assertTrue(
            torch.equal(model.u(self.batch["x"]), self.source(self.batch["x"]))
        )
        for index in [0, 2]:
            layer = model.u.encoder[index]
            self.assertEqual(layer.rank, 4)
            self.assertEqual(layer.scaling, 1.0)
            self.assertGreater(layer.A.abs().sum().item(), 0)
            self.assertEqual(layer.B.abs().sum().item(), 0)
            self.assertTrue(layer.A.requires_grad and layer.B.requires_grad)
            self.assertTrue(all(not p.requires_grad for p in layer.base.parameters()))

    def test_backward_updates_only_allowed_parameters_and_never_calls_decoder(self):
        for variant in ["mpt", "residual", "tri_frozen", "tri_lora", "tri_full"]:
            with self.subTest(variant=variant):
                model = self.build(variant)
                model.train()
                if variant != "mpt":
                    model.u.decoder.register_forward_pre_hook(
                        lambda *_: self.fail("decoder used for prediction")
                    )
                if variant == "tri_frozen":
                    self.assertFalse(model.u.training)
                self.assertEqual(model(self.batch, self.envs).shape, (7,))
                model(self.batch, self.envs).sum().backward()
                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        self.assertIsNone(p.grad, name)
                    else:
                        self.assertIsNotNone(p.grad, name)
                if variant in ["residual", "tri_full"]:
                    self.assertGreater(
                        sum(
                            p.grad.abs().sum().item()
                            for p in model.u.encoder.parameters()
                        ),
                        0,
                    )
                if variant == "tri_lora":
                    self.assertGreater(
                        sum(
                            model.u.encoder[i].B.grad.abs().sum().item() for i in [0, 2]
                        ),
                        0,
                    )
                before = {
                    name: p.detach().clone()
                    for name, p in model.named_parameters()
                    if not p.requires_grad
                }
                torch.optim.SGD(
                    [p for p in model.parameters() if p.requires_grad], lr=0.1
                ).step()
                for name, p in model.named_parameters():
                    if name in before:
                        self.assertTrue(torch.equal(before[name], p), name)

    def test_residual_matches_historical_forward_and_initialization(self):
        model = self.build("residual")
        self.assertTrue(torch.equal(model.projection.weight, torch.eye(128)))
        self.assertIsNone(model.projection.bias)
        self.assertAlmostEqual(model.logit.item(), math.log(0.1 / 0.9), places=6)
        self.assertAlmostEqual(model.logit.sigmoid().item(), 0.1, places=7)
        # Independent transcription of historical ResidualHead.forward, including
        # detach_extra=False (universal_experiment/model.py:102).
        h = model.core
        env = h.env_embedding_network(h.new_env_idx).squeeze(0)
        weights = (
            h.projection_network(self.batch["x"])
            @ torch.stack(self.envs, 1)
            / h.temperature
        ).softmax(-1)
        specific = (
            torch.stack([self.batch["s0"], self.batch["s1"]], 2) @ weights.unsqueeze(2)
        ).squeeze(-1)
        fused = (
            torch.stack([specific * env, self.batch["g"]], 2)
            @ h.gate_network(self.batch["x"]).unsqueeze(2)
        ).squeeze(-1)
        extra = model.u(self.batch["x"])
        extra.retain_grad()
        expected = h.tower_network(
            fused + model.logit.sigmoid() * model.projection(extra)
        ).reshape(-1)
        self.assertTrue(torch.equal(model(self.batch, self.envs), expected))
        expected.sum().backward()
        self.assertGreater(extra.grad.abs().sum().item(), 0)

    def test_mpt_matches_original_and_all_variants_accept_single_row(self):
        model = self.build("mpt")
        expected = model.core(
            self.batch["x"],
            self.batch["g"],
            [self.batch["s0"], self.batch["s1"]],
            self.envs,
        )
        self.assertTrue(torch.equal(expected, model(self.batch, self.envs)))
        single = {key: value[:1] for key, value in self.batch.items()}
        for variant in ["mpt", "residual", "tri_frozen", "tri_lora", "tri_full"]:
            self.assertEqual(self.build(variant)(single, self.envs).shape, (1,))

    def test_build_preserves_rng_and_rejects_unknown_variant(self):
        before = torch.random.get_rng_state().clone()
        for variant in ["mpt", "residual", "tri_frozen", "tri_lora", "tri_full"]:
            self.build(variant)
            self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        with self.assertRaisesRegex(ValueError, "variant"):
            self.build("unknown")


if __name__ == "__main__":
    unittest.main()
