import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from sska_rec import core
from sska_rec.u_access import AccessHead


class FakeBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.env_embedding_network = nn.Embedding(2, 128)
        self.tower_networks = nn.ModuleList([nn.Identity(), nn.Identity()])


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "outputs" / "u-access"
        self.source.mkdir(parents=True)
        self.dataset = self.root / "dataset"
        (self.dataset / "Census-income").mkdir(parents=True)
        self.raw = {}
        for name in ["train.gz", "test.gz"]:
            p = self.dataset / "Census-income" / name
            p.write_bytes(name.encode())
            self.raw[name] = core.file_hash(p)
        self.sizes = dict(train=4, val=4, test=4)
        self.spec = dict(
            tasks=["income", "marital", "education", "sex"],
            initial=["income", "marital"],
            xdim=119,
            rdim=128,
            batch=256,
            sets={},
        )
        vi, ti = core.make_split(16, 20260929)
        ids = dict(train=np.arange(8), val=8 + vi, test=8 + ti)
        indices, splits = {}, {}
        for split in self.sizes:
            y = np.column_stack(
                [
                    np.arange(8) % 2,
                    np.arange(8) // 2 % 2,
                    np.arange(8) // 4 % 2,
                    1 - np.arange(8) % 2,
                ]
            ).astype("float32")
            self.spec["sets"][split] = core.Rows(
                np.arange(8)[:, None], y, ["age"], [0, 1]
            )
            ix = np.array([0, 2, 5, 7], dtype="int64")
            indices[split + "_indices"] = ix
            indices[split + "_rowid"] = ids[split][ix]
            splits[split] = dict(
                n=4,
                rowid_sha256=hashlib.sha256(ids[split][ix].tobytes()).hexdigest(),
                joint_label_counts=np.bincount(
                    (y[ix, 0] * 2 + y[ix, 1]).astype(int), minlength=4
                ).tolist(),
            )
        subset = self.source / "subset"
        subset.mkdir()
        np.savez(subset / "indices.npz", **indices)
        self.data = dict(
            raw_sha256=self.raw,
            split_seed=20260929,
            subset_seed=20261111,
            splits=splits,
            features=["age"],
            input_dim=119,
        )
        self.protocol = dict(
            protocol="census-original-task-u-access-v1-20261111",
            seeds=[17],
            sizes=self.sizes,
            split_seed=20260929,
            subset_seed=20261111,
            batch_size=256,
            learning_rate=0.001,
            base_epochs=30,
            access_epochs=30,
            input_dim=119,
            validation_fallback=False,
        )
        old_sources = self.root / "sska_rec"
        old_sources.mkdir()
        (old_sources / "core.py").write_text("old core")
        self.identity = dict(
            protocol=self.protocol,
            data=self.data,
            source={"core.py": core.file_hash(old_sources / "core.py")},
        )
        self.write_json(self.source / "identity.json", self.identity)
        self.write_json(subset / "subset.json", self.data)
        self.base, self.u = FakeBase(), nn.Linear(119, 128)
        self.heads = nn.ModuleList([AccessHead("linear"), AccessHead("linear")])
        with torch.no_grad():
            self.heads[0].delta.weight.fill_(0.01)
            self.heads[1].delta.bias.fill_(0.2)
        seed = self.source / "17"
        for stage, model, filename in [
            ("source/base", self.base, "base.pt"),
            ("source/ssl", self.u, "source.pt"),
            ("linear", self.heads, "model.pt"),
        ]:
            folder = seed / stage
            folder.mkdir(parents=True)
            torch.save(model.state_dict(), folder / filename)
            metrics = (
                dict(model_hash=core.model_hash(model))
                if stage != "linear"
                else dict(
                    frozen_base_hash=core.model_hash(self.base),
                    frozen_u_hash=core.model_hash(self.u),
                )
            )
            self.write_json(folder / "metrics.json", metrics)
        cache = seed / "feature-cache"
        cache.mkdir()
        for split in self.sizes:
            for name, width in dict(x=119, g=128, s0=128, s1=128, old=2).items():
                np.save(
                    cache / f"{split}-{name}.npy",
                    np.full((4, width), 0.3, dtype="float32"),
                )
        self.write_json(
            cache / "identity.json",
            dict(
                base_hash=core.model_hash(self.base),
                splits={
                    s: dict(rows=4, rowid_sha256=splits[s]["rowid_sha256"])
                    for s in self.sizes
                },
            ),
        )
        raw = {}
        for split in ["val", "test"]:
            with torch.no_grad():
                b = dict(u=self.u(torch.full((4, 119), 0.3)), p=torch.full((4,), 0.3))
                raw[split + "_p"] = torch.stack(
                    [h(b, None) for h in self.heads], 1
                ).numpy()
            raw[split + "_y"] = self.spec["sets"][split].y[
                indices[split + "_indices"], :2
            ]
            raw[split + "_rowid"] = indices[split + "_rowid"]
        np.savez(seed / "linear" / "predictions.npz", **raw)
        for name in ["source/base", "source/ssl", "linear", "feature-cache"]:
            self.commit(seed / name)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def write_json(path, data):
        path.write_text(json.dumps(data), encoding="utf8")

    def commit(self, folder):
        self.write_json(
            folder / "COMMITTED.json",
            {
                p.name: core.file_hash(p)
                for p in folder.iterdir()
                if p.name != "COMMITTED.json"
            },
        )

    def load(self):
        from sska_rec import stage2_reuse as reuse

        with patch.object(reuse, "EXPECTED_SIZES", self.sizes), patch.object(
            reuse, "RAW_SHA256", self.raw
        ), patch.object(reuse, "_verify_loaded_sources"), patch.object(
            core, "census", side_effect=lambda *a: copy.deepcopy(self.spec)
        ), patch.object(
            core, "build_base", side_effect=lambda spec: FakeBase()
        ), patch.object(
            core, "make_u", side_effect=lambda spec, seed: nn.Linear(119, 128)
        ):
            return reuse.load_reused_seed(self.source, self.dataset, 17, "cpu")

    def test_preserves_four_labels_frozen_paths_and_predictions(self):
        from sska_rec import stage2_reuse as reuse

        before = {
            p: (p.stat().st_mtime_ns, core.file_hash(p))
            for p in self.source.rglob("*")
            if p.is_file()
        }
        spec, base, u, heads, caches, envs, receipt = self.load()
        self.assertEqual(spec["tasks"], ["income", "marital", "education", "sex"])
        for split in self.sizes:
            np.testing.assert_array_equal(
                caches[split]["y"], self.spec["sets"][split].y[[0, 2, 5, 7]]
            )
            self.assertEqual(caches[split]["y"].shape, (4, 4))
            np.testing.assert_array_equal(spec["sets"][split].x[:, 0], [0, 2, 5, 7])
        for model in [base, u, heads]:
            self.assertFalse(model.training)
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertEqual(len(envs), 2)
        with np.load(self.source / "17/linear/predictions.npz") as raw:
            np.testing.assert_array_equal(
                reuse.original_predictions(base, u, heads, caches, spec, "test"),
                raw["test_p"],
            )
        self.assertEqual(
            Path(receipt["original_predictions_npz"]),
            self.source / "17/linear/predictions.npz",
        )
        self.assertEqual(
            before,
            {
                p: (p.stat().st_mtime_ns, core.file_hash(p))
                for p in self.source.rglob("*")
                if p.is_file()
            },
        )

    def test_rejects_old_source_hash_change(self):
        (self.root / "sska_rec/core.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "source.*hash|Source.*hash"):
            self.load()

    def test_rejects_rowid_mismatch(self):
        p = self.source / "subset/indices.npz"
        with np.load(p) as raw:
            arrays = dict(raw)
        arrays["val_rowid"] = arrays["val_rowid"] + 1
        np.savez(p, **arrays)
        with self.assertRaisesRegex(ValueError, "row.?id"):
            self.load()

    def test_rejects_committed_cache_corruption(self):
        p = self.source / "17/feature-cache/train-g.npy"
        p.write_bytes(p.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.load()


if __name__ == "__main__":
    unittest.main()
