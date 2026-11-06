import importlib.util
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

from sska_rec import core
from sska_rec.training import train_single, train_seed as train_kuairand_seed, _prepare


class TrainingTests(unittest.TestCase):
    def test_single_task_only_does_not_prepare_sources(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            with patch(
                "sska_rec.training._prepare",
                side_effect=AssertionError("Unneeded source training"),
            ):
                result = train_kuairand_seed(
                    core, self.spec, 54, Path(tmp), 1, models=["single_task"]
                )
            self.assertEqual(set(result), {"single_task"})

    def test_core_ablations_preserve_old_predictions(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            result = train_kuairand_seed(
                core, self.spec, 55, Path(tmp), 1, models=["random_u", "fixed_fusion"]
            )
            for model in result:
                self.assertEqual(
                    result[model][-1]["historical_max_probability_difference"], 0.0
                )

    def setUp(self):
        torch.set_num_threads(1)
        old_device = core.DEVICE
        self.addCleanup(setattr, core, "DEVICE", old_device)
        core.DEVICE = torch.device("cpu")
        rng = np.random.default_rng(9)
        x = np.column_stack([rng.normal(size=32), rng.integers(0, 3, size=32)])
        y = np.column_stack([(np.arange(32) // k) % 2 for k in [1, 2, 3, 4]]).astype(
            "float32"
        )
        self.spec = dict(
            sets={
                s: core.Rows(x, y, ["dense", "cat"], [0, 1])
                for s in ["train", "val", "test"]
            },
            tasks=["click", "like", "follow", "forward"],
            initial=["click", "like"],
            vocab={"cat": 3},
            embed=2,
            xdim=3,
            rdim=4,
            batch=8,
            lr=0.001,
            hidden=[8, 4],
            tower=[4],
            widths=[1, 2],
            env_coe=0.1,
            dataset="kuairand",
            epochs1=1,
            epochs2=1,
        )
        for rows in self.spec["sets"].values():
            rows.row_ids = np.arange(32) + 100

    def test_single_uses_requested_fourth_label_and_atomic_resume(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            folder = Path(tmp)
            result = train_single(core, self.spec, 51, "forward", folder, 1)
            self.assertEqual(result["seed"], 51)
            with np.load(folder / "predictions.npz") as raw:
                np.testing.assert_array_equal(
                    raw["test_y"], self.spec["sets"]["test"].y[:, 3]
                )
                np.testing.assert_array_equal(raw["test_rowid"], np.arange(32) + 100)
            self.assertTrue(result["started_at"].endswith("+00:00"))
            self.assertEqual(
                result, train_single(core, self.spec, 51, "forward", folder, 1)
            )
            (folder / "predictions.npz").unlink()
            with self.assertRaises(ValueError):
                train_single(core, self.spec, 51, "forward", folder, 1)

    def test_continual_private_copies_and_reusable_results(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            folder = Path(tmp)
            result = train_kuairand_seed(core, self.spec, 52, folder, 1)
            self.assertEqual(set(result), {"single_task", "no_u", "sska"})
            self.assertEqual(len(result["sska"]), 2)
            self.assertEqual(result["single_task"][0]["seed"], 52 + 1 + 3)
            self.assertEqual(
                result["single_task"][0]["seed"], result["sska"][0]["new"]["seed"]
            )
            self.assertEqual(
                result["sska"][0]["historical_max_probability_difference"], 0.0
            )
            self.assertEqual(
                result["sska"][1]["historical_max_probability_difference"], 0.0
            )
            self.assertEqual(
                result, train_kuairand_seed(core, self.spec, 52, folder, 1)
            )

    def test_base_reused_after_interrupted_ssl_and_memmap_cache_reused(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            folder = Path(tmp)
            with patch.object(
                core, "make_u", side_effect=RuntimeError("interrupted SSL")
            ):
                with self.assertRaises(RuntimeError):
                    _prepare(core, self.spec, 53, folder, 1)
            base_path = folder / "source" / "base" / "base.pt"
            digest = core.file_hash(base_path)
            (folder / "source" / "ssl" / "partial.tmp").write_text(
                "interrupted", encoding="utf-8"
            )
            with patch.object(
                core.PairedTrainManager,
                "train_two_task",
                side_effect=AssertionError("B retrained"),
            ):
                _, cache, _, _ = _prepare(core, self.spec, 53, folder, 1)
            self.assertEqual(core.file_hash(base_path), digest)
            self.assertEqual(
                len(list((folder / "source").glob("ssl.interrupted-*"))), 1
            )
            path = folder / "feature-cache" / "train-x.npy"
            stamp = path.stat().st_mtime_ns
            with patch.object(
                core.MPTRec, "get_infos", side_effect=AssertionError("cache recomputed")
            ):
                _prepare(core, self.spec, 53, folder, 1)
            self.assertEqual(path.stat().st_mtime_ns, stamp)
            del cache


if __name__ == "__main__":
    unittest.main()
