import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from sska_rec.u_access import (
    AccessHead,
    stratified_indices,
    paired_interval,
    run_access,
)
from sska_rec import core


class UAccessTests(unittest.TestCase):
    def test_real_access_training_freezes_sources_and_saves_raw(self):
        torch.set_num_threads(1)
        base = nn_base = torch.nn.Module()
        base.tower_networks = torch.nn.ModuleList(
            [
                torch.nn.Sequential(torch.nn.Linear(128, 1), torch.nn.Sigmoid())
                for _ in range(2)
            ]
        )
        u = torch.nn.Linear(128, 128)
        for p in [*base.parameters(), *u.parameters()]:
            p.requires_grad_(False)
        cache = {}
        for split in ["train", "val", "test"]:
            h0 = torch.randn(16, 128)
            h1 = torch.randn(16, 128)
            cache[split] = dict(
                g=torch.randn(16, 128),
                s0=torch.randn(16, 128),
                s1=torch.randn(16, 128),
                u=torch.randn(16, 128),
                h0=h0,
                h1=h1,
                old=torch.stack(
                    [
                        base.tower_networks[0](h0).flatten(),
                        base.tower_networks[1](h1).flatten(),
                    ],
                    1,
                ),
                y=torch.tensor(
                    np.column_stack([np.arange(16) % 2, (np.arange(16) // 2) % 2]),
                    dtype=torch.float32,
                ),
            )
        spec = {
            "batch": 8,
            "sets": {s: type("Rows", (), {"row_ids": np.arange(16)})() for s in cache},
        }
        cfg = {"learning_rate": 0.001, "access_epochs": 1, "patience": 5}
        old_device = core.DEVICE
        core.DEVICE = torch.device("cpu")
        self.addCleanup(setattr, core, "DEVICE", old_device)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            for variant in ["mpt", "linear", "gated_linear", "original", "bottleneck"]:
                result = run_access(
                    base,
                    u,
                    cache,
                    spec,
                    3,
                    variant,
                    Path(tmp) / variant,
                    cfg,
                    Path(tmp) / "status.json",
                )
                self.assertFalse(result["validation_fallback"])
                with np.load(Path(tmp) / variant / "predictions.npz") as raw:
                    self.assertEqual(raw["test_p"].shape, (16, 2))
        self.assertTrue(
            all(p.grad is None for p in [*base.parameters(), *u.parameters()])
        )

    def test_parameter_counts_zero_initialization_and_gradient(self):
        torch.set_num_threads(1)
        features = dict(
            g=torch.randn(9, 128),
            s0=torch.randn(9, 128),
            s1=torch.randn(9, 128),
            u=torch.randn(9, 128),
            h=torch.randn(9, 128),
            p=torch.rand(9) * 0.8 + 0.1,
        )
        for variant, count in [
            ("linear", 129),
            ("gated_linear", 258),
            ("original", 32898),
            ("bottleneck", 2441),
        ]:
            tower = torch.nn.Sequential(torch.nn.Linear(128, 1), torch.nn.Sigmoid())
            for p in tower.parameters():
                p.requires_grad_(False)
            model = AccessHead(variant)
            self.assertEqual(sum(p.numel() for p in model.parameters()), count)
            out = model(features, tower)
            self.assertTrue(torch.equal(out, features["p"]))
            out.sum().backward()
            self.assertGreater(
                sum(
                    float(p.grad.abs().sum())
                    for p in model.parameters()
                    if p.grad is not None
                ),
                0.0,
            )
            self.assertTrue(all(p.grad is None for p in tower.parameters()))

    def test_fixed_joint_stratification(self):
        y = np.column_stack([np.arange(1000) % 2, (np.arange(1000) // 2) % 2])
        a = stratified_indices(y, 100, 20261111)
        self.assertTrue(np.array_equal(a, stratified_indices(y, 100, 20261111)))
        self.assertEqual(len(set(a)), 100)
        self.assertEqual(
            np.bincount((y[a, 0] * 2 + y[a, 1]).astype(int)).tolist(), [25] * 4
        )

    def test_interval_uses_paired_differences(self):
        result = paired_interval([0.1, 0.2, 0.3, 0.4, 0.5], [0, 0.1, 0.2, 0.3, 0.4])
        self.assertAlmostEqual(result["mean"], 0.1)
        self.assertAlmostEqual(result["low"], 0.1)
        self.assertAlmostEqual(result["high"], 0.1)
