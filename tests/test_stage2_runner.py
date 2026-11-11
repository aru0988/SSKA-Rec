import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from sska_rec import core
from sska_rec.stage2_runner import train_one


class TinyHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(3, 1)
        self.exclusive_parameters = 4
        self.inference_parameters = 4

    def forward(self, batch, envs):
        return self.layer(batch["x"]).sigmoid().reshape(-1)

    def get_l2_reg(self):
        return self.layer.weight.square().sum() * 3e-5


class RunnerTests(unittest.TestCase):
    def test_fit_raw_alignment_frozen_checks_and_committed_reuse(self):
        torch.set_num_threads(1)
        old = core.DEVICE
        core.DEVICE = torch.device("cpu")
        self.addCleanup(setattr, core, "DEVICE", old)
        cache = {
            s: {
                "x": torch.randn(32, 3),
                "y": torch.tensor(
                    np.column_stack([np.arange(32) % 2] * 4), dtype=torch.float32
                ),
            }
            for s in ["train", "val", "test"]
        }
        spec = {
            "batch": 8,
            "lr": 0.001,
            "tasks": ["income", "marital", "education", "sex"],
            "sets": {
                s: type("Rows", (), {"row_ids": np.arange(32) + 100})() for s in cache
            },
        }
        calls = []

        def guard():
            calls.append(1)
            return {"passed": True}

        cfg = {"epochs": 2, "patience": 5, "learning_rate": 0.001}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            folder = Path(tmp) / "model"
            status = Path(tmp) / "status.json"
            model = TinyHead()
            m = train_one(
                model,
                spec,
                cache,
                [],
                "education",
                folder,
                cfg,
                51,
                "mpt",
                guard,
                status,
            )
            self.assertGreaterEqual(len(calls), 2)
            self.assertEqual(m["optimizer_parameters"], 4)
            self.assertEqual(m["epochs"], 2)
            with np.load(folder / "predictions.npz") as raw:
                self.assertTrue(np.array_equal(raw["test_rowid"], np.arange(32) + 100))
                self.assertTrue(
                    np.array_equal(raw["test_y"], cache["test"]["y"][:, 2].numpy())
                )
            self.assertEqual(
                m,
                train_one(
                    TinyHead(),
                    spec,
                    cache,
                    [],
                    "education",
                    folder,
                    cfg,
                    51,
                    "mpt",
                    guard,
                    status,
                ),
            )
            pending = Path(tmp) / "pending"
            with patch(
                "sska_rec.stage2_runner._commit",
                side_effect=RuntimeError("interrupted at final commit"),
            ):
                with self.assertRaises(RuntimeError):
                    train_one(
                        TinyHead(),
                        spec,
                        cache,
                        [],
                        "education",
                        pending,
                        cfg,
                        51,
                        "mpt",
                        guard,
                        status,
                    )
            self.assertFalse((pending / "COMMITTED.json").exists())
            recovered = TinyHead()
            with patch.object(
                recovered,
                "forward",
                side_effect=AssertionError("retrained completed weights"),
            ):
                result = train_one(
                    recovered,
                    spec,
                    cache,
                    [],
                    "education",
                    pending,
                    cfg,
                    51,
                    "mpt",
                    guard,
                    status,
                )
            self.assertEqual(result["epochs"], 2)
            self.assertTrue((pending / "COMMITTED.json").exists())
            (folder / "predictions.npz").write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                train_one(
                    TinyHead(),
                    spec,
                    cache,
                    [],
                    "education",
                    folder,
                    cfg,
                    51,
                    "mpt",
                    guard,
                    status,
                )
