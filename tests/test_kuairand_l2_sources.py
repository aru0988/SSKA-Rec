import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from sska_rec import core


class KuaiNormalizedSourcesTests(unittest.TestCase):
    def setUp(self):
        self.previous = core.DEVICE
        core.DEVICE = torch.device("cpu")
        torch.set_num_threads(1)
        self.addCleanup(setattr, core, "DEVICE", self.previous)

    def spec(self):
        x = np.column_stack([np.linspace(-1, 1, 24), np.arange(24) % 3]).astype("float32")
        y = np.column_stack([(np.arange(24)//k)%2 for k in (1,2,3,4)]).astype("float32")
        sets = {}
        for pos, split in enumerate(("train","val","test")):
            sets[split] = core.Rows(x.copy(), y.copy(), ["dense","cat"], [0,1])
            sets[split].row_ids = np.arange(24)+pos*100
        return dict(dataset="kuairand", vocab={"cat":3}, embed=2, xdim=3, rdim=4,
                    hidden=[8,4], tower=[4], widths=[1,2], batch=8, lr=.001,
                    env_coe=.1, sets=sets, initial=["click","like"],
                    tasks=["click","like","follow","forward"],
                    data_signature={"full_split":True,"signature":"tiny"},
                    embedding_l2_normalization="parameter_mean", embedding_l2=.006)

    def test_normalized_base_changes_only_embedding_penalty(self):
        spec = self.spec()
        core.seed_all(13); new = core.build_base(spec)
        legacy = dict(spec); legacy.pop("embedding_l2_normalization")
        core.seed_all(13); old = core.build_base(legacy)
        self.assertAlmostEqual(new.reg_embedding, .006/6)
        self.assertEqual(old.reg_embedding, .006)
        self.assertEqual(core.model_hash(new), core.model_hash(old))
        self.assertEqual(new.reg_dnn, old.reg_dnn)
        for model in (new, old):
            model.zero_grad()
            (model.reg_embedding * model.embedding_network.get_l2_reg()).backward()
        a = next(new.embedding_network.parameters()).grad
        b = next(old.embedding_network.parameters()).grad
        self.assertTrue(torch.allclose(a*6,b))

    def test_unknown_normalization_is_rejected(self):
        spec = self.spec(); spec["embedding_l2_normalization"] = "unexpected"
        with self.assertRaises(ValueError):
            core.build_base(spec)

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("sska_rec.kuairand_l2_sources"))
        from sska_rec import kuairand_l2_sources
        return kuairand_l2_sources

    def test_registered_config_keeps_other_protocol_values(self):
        api = self.api(); cfg = api.default_config(); api.validate_config(cfg)
        self.assertEqual(list(cfg["datasets"]), ["kuairand"])
        dc = cfg["datasets"]["kuairand"]
        self.assertEqual(dc["batch_size"],512)
        self.assertEqual(dc["learning_rate"],.001)
        self.assertEqual(dc["embedding_l2_normalization"],"parameter_mean")
        self.assertEqual(len(dc["seeds"]),5)

    def test_source_handoff_loads_without_training_and_records_evaluation(self):
        api = self.api(); spec = self.spec()
        cfg = dict(base_epochs=1,ssl_epochs=1,patience=5,epochs=30,learning_rate=.001,
                   embedding_l2_normalization="parameter_mean",embedding_l2=.006)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base,caches,envs,u = api.prepare_source(spec,19,root,cfg,root/"progress.json")
            self.assertFalse((root/"linear").exists())
            receipt = json.loads((root/"source-ready.json").read_text())
            self.assertIn("regularization",receipt)
            ev = json.loads((root/"source-evaluation/metrics.json").read_text())
            self.assertEqual(set(ev["test"]),{"click","like"})
            loaded = api.load_source(spec,19,root,cfg)
            self.assertEqual(core.model_hash(base),core.model_hash(loaded[0]))
            self.assertEqual(core.model_hash(u),core.model_hash(loaded[3]))
            guard = api.freeze_guard(*loaded, spec)
            self.assertTrue(guard()["passed"])
            with torch.no_grad():
                next(loaded[0].parameters()).add_(.01)
            with self.assertRaises(ValueError):
                guard()
            wrong = dict(cfg,embedding_l2_normalization="sum")
            with self.assertRaises(ValueError):
                api.load_source(spec,19,root,wrong)
            del guard, loaded, base, caches, envs, u
            import gc
            gc.collect()
