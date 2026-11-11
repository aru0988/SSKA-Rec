"""Matched PEFT initialization, updates, and cached-adapter contracts."""

import copy
import importlib
import unittest

import torch
from torch import nn

from sska_rec.expert import UniversalExpert
from sska_rec.stage2_models import build_model as build_reference


class PEFTModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.spec = dict(xdim=119, rdim=128, tower=[64, 32], widths=[1] * 11 + [4] * 27)
        self.source = UniversalExpert(self.spec["widths"], seed=29)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(23)
            self.batch = dict(
                x=torch.randn(7, 119),
                g=torch.randn(7, 128),
                s0=torch.randn(7, 128),
                s1=torch.randn(7, 128),
            )
            self.envs = [torch.randn(128), torch.randn(128)]

    def build(self, variant, source=None):
        module = importlib.import_module("sska_rec.peft_models")
        return module.build_model(
            self.spec, self.source if source is None else source, 17, variant
        )

    def test_counts_and_bitwise_matched_initialization(self):
        reference = build_reference(self.spec, self.source, 17, "tri_frozen")
        expected = dict(
            tri_bitfit=(27302, 100262),
            tri_ssf=(27686, 101030),
            tri_adapter=(29102, 102446),
        )
        for variant, (trainable, inference) in expected.items():
            with self.subTest(variant=variant):
                model = self.build(variant)
                self.assertEqual(model.trainable_parameters, trainable)
                self.assertEqual(model.exclusive_parameters, trainable)
                self.assertEqual(model.inference_parameters, inference)
                self.assertTrue(
                    torch.equal(
                        model(self.batch, self.envs), reference(self.batch, self.envs)
                    )
                )
                for key, value in reference.core.state_dict().items():
                    self.assertTrue(
                        torch.equal(value, model.core.state_dict()[key]), key
                    )

    def test_bitfit_trains_only_existing_encoder_biases(self):
        model = self.build("tri_bitfit")
        self.assertEqual(
            sum(p.numel() for p in model.u.parameters() if p.requires_grad), 384
        )
        self.assertEqual(
            [name for name, p in model.u.named_parameters() if p.requires_grad],
            ["encoder.0.bias", "encoder.2.bias"],
        )
        for key, value in self.source.state_dict().items():
            self.assertTrue(torch.equal(value, model.u.state_dict()[key]), key)

    def test_ssf_scales_affine_output_before_existing_relu(self):
        model = self.build("tri_ssf")
        for index in [0, 2]:
            layer = model.u.encoder[index]
            self.assertTrue(torch.equal(layer.scale, torch.ones_like(layer.scale)))
            self.assertTrue(torch.equal(layer.shift, torch.zeros_like(layer.shift)))
            self.assertTrue(all(not p.requires_grad for p in layer.base.parameters()))
            self.assertTrue(layer.scale.requires_grad and layer.shift.requires_grad)
            with torch.no_grad():
                layer.scale.fill_(0.5)
                layer.shift.fill_(-0.25)
        x = torch.cat([model.u.normalize(self.batch["x"]), torch.zeros(7, 38)], 1)
        first = model.u.encoder[0]
        second = model.u.encoder[2]
        expected = second.base(torch.relu(first.base(x) * 0.5 - 0.25)) * 0.5 - 0.25
        self.assertTrue(torch.equal(model.u(self.batch["x"]), expected))
        self.assertIsInstance(model.u.encoder[1], nn.ReLU)

    def test_adapter_identity_bottleneck_and_cached_forward(self):
        model = self.build("tri_adapter")
        self.assertEqual(model.adapter.down.weight.shape, (8, 128))
        self.assertEqual(model.adapter.up.weight.shape, (128, 8))
        self.assertGreater(model.adapter.down.weight.abs().sum().item(), 0)
        self.assertEqual(model.adapter.up.weight.abs().sum().item(), 0)
        self.assertEqual(model.adapter.up.bias.abs().sum().item(), 0)
        self.assertTrue(all(not p.requires_grad for p in model.u.parameters()))
        cached = dict(self.batch, u0=model.u(self.batch["x"]).detach())
        self.assertTrue(
            torch.equal(model(self.batch, self.envs), model(cached, self.envs))
        )
        with torch.no_grad():
            model.adapter.up.weight.fill_(0.02)
            model.adapter.up.bias.fill_(0.01)
        expected = model(self.batch, self.envs)
        handle = model.u.register_forward_pre_hook(
            lambda *_: self.fail("cached adapter prediction recomputed U")
        )
        self.assertTrue(torch.equal(expected, model(cached, self.envs)))
        handle.remove()
        u0 = cached["u0"]
        self.assertTrue(
            torch.equal(
                model.adapter(u0),
                u0 + model.adapter.up(torch.relu(model.adapter.down(u0))),
            )
        )

    def test_only_allowed_parameters_receive_gradients_and_step(self):
        for variant in ["tri_bitfit", "tri_ssf", "tri_adapter"]:
            with self.subTest(variant=variant):
                model = self.build(variant).train()
                model.u.decoder.register_forward_pre_hook(
                    lambda *_: self.fail("decoder used in prediction")
                )
                self.assertFalse(model.u.decoder.training)
                before = {
                    name: p.detach().clone() for name, p in model.named_parameters()
                }
                model(self.batch, self.envs).sum().backward()
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, name)
                    else:
                        self.assertIsNone(parameter.grad, name)
                if variant == "tri_bitfit":
                    adaptation = [
                        p for p in model.u.encoder.parameters() if p.requires_grad
                    ]
                elif variant == "tri_ssf":
                    adaptation = [
                        p for p in model.u.encoder.parameters() if p.requires_grad
                    ]
                else:
                    adaptation = list(model.adapter.parameters())
                self.assertGreater(
                    sum(p.grad.abs().sum().item() for p in adaptation), 0
                )
                torch.optim.SGD(
                    [p for p in model.parameters() if p.requires_grad], lr=0.1
                ).step()
                updated_adaptation = False
                for name, parameter in model.named_parameters():
                    changed = not torch.equal(parameter, before[name])
                    if not parameter.requires_grad:
                        self.assertFalse(changed, name)
                    if (
                        name.startswith("u.") or name.startswith("adapter.")
                    ) and changed:
                        updated_adaptation = True
                self.assertTrue(updated_adaptation)

    def test_source_and_global_rng_are_isolated(self):
        source_state = copy.deepcopy(self.source.state_dict())
        flags = [p.requires_grad for p in self.source.parameters()]
        rng = torch.random.get_rng_state().clone()
        for variant in ["tri_bitfit", "tri_ssf", "tri_adapter"]:
            model = self.build(variant)
            duplicate = self.build(variant)
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            self.assertIsNot(model.u, self.source)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, duplicate.state_dict()[key]), key)
            for key in ["mean", "scale"]:
                self.assertNotEqual(
                    getattr(model.u, key).data_ptr(),
                    getattr(self.source, key).data_ptr(),
                )
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(1)
                model.u.mean.add_(1)
        for key, value in source_state.items():
            self.assertTrue(torch.equal(value, self.source.state_dict()[key]), key)
        self.assertEqual(flags, [p.requires_grad for p in self.source.parameters()])

    def test_float64_source_and_single_row(self):
        for variant in ["tri_bitfit", "tri_ssf", "tri_adapter"]:
            model = self.build(variant, source=copy.deepcopy(self.source).double())
            self.assertTrue(all(p.dtype == torch.float64 for p in model.parameters()))
            single = {key: value[:1].double() for key, value in self.batch.items()}
            self.assertEqual(
                model(single, [env.double() for env in self.envs]).shape, (1,)
            )

    def test_rejects_missing_source_and_unknown_variant(self):
        module = importlib.import_module("sska_rec.peft_models")
        with self.assertRaisesRegex(ValueError, "variant"):
            self.build("unknown")
        with self.assertRaisesRegex(ValueError, "source"):
            module.build_model(self.spec, None, 17, "tri_adapter")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_source_device_initial_identity_and_cache(self):
        source = copy.deepcopy(self.source).cuda()
        batch = {key: value.cuda() for key, value in self.batch.items()}
        envs = [env.cuda() for env in self.envs]
        reference = build_reference(self.spec, source, 17, "tri_frozen")
        cpu_rng = torch.random.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone()
        for variant in ["tri_bitfit", "tri_ssf", "tri_adapter"]:
            model = self.build(variant, source=source)
            self.assertTrue(
                all(p.device == source.mean.device for p in model.parameters())
            )
            self.assertTrue(torch.equal(cpu_rng, torch.random.get_rng_state()))
            self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state()))
            self.assertTrue(torch.equal(model(batch, envs), reference(batch, envs)))
            if variant == "tri_adapter":
                cached = dict(batch, u0=model.u(batch["x"]).detach())
                self.assertTrue(torch.equal(model(batch, envs), model(cached, envs)))


if __name__ == "__main__":
    unittest.main()
