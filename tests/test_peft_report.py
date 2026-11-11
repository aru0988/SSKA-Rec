import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from sska_rec import core
from sska_rec.training import _commit
from sska_rec.peft_report import report


class ReportTests(unittest.TestCase):
    def test_full_paired_report_and_seed_identity(self):
        cfg = json.loads(
            (Path(__file__).parents[1] / "configs/u_peft_census.json").read_text()
        )
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp) / "new"
            root.mkdir()
            baseline = Path(tmp) / "old"
            baseline.mkdir()
            report_cfg = dict(cfg)
            cfg = dict(cfg, variants=cfg["reuse_variants"] + cfg["variants"])
            cfg["expected_trainable"].update(
                json.loads(
                    (
                        Path(__file__).parents[1] / "configs/u_stage2_census.json"
                    ).read_text()
                )["expected_trainable"]
            )
            for seed in cfg["seeds"]:
                for idx, target in enumerate(cfg["targets"]):
                    for v, variant in enumerate(cfg["variants"]):
                        folder = (
                            (baseline if variant in cfg["reuse_variants"] else root)
                            / str(seed)
                            / target
                            / variant
                        )
                        folder.mkdir(parents=True)
                        y = np.arange(32) % 2
                        rng = np.random.default_rng(seed + idx + v)
                        p = np.clip(rng.random(32) * 0.6 + 0.2 * y, 0.01, 0.99)
                        m = dict(
                            paired_seed=seed,
                            seed=seed + idx + 3,
                            target=target,
                            variant=variant,
                            val=core.measures(y, p),
                            test=core.measures(y, p),
                            frozen_original_paths=dict(
                                passed=True, max_probability_difference=0.0
                            ),
                            optimizer_parameters=cfg["expected_trainable"][variant],
                            exclusive_parameters=cfg["expected_trainable"][variant],
                            seconds=1.0,
                            peak_cuda_bytes=1024,
                        )
                        core.dump(folder / "metrics.json", m)
                        np.savez_compressed(
                            folder / "predictions.npz",
                            val_y=y,
                            val_p=p,
                            test_y=y,
                            test_p=p,
                            val_rowid=np.arange(32),
                            test_rowid=np.arange(32),
                        )
                        _commit(core, folder)
            self.assertEqual(report(root, report_cfg, baseline), 30)
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["raw_split_checks"], 160)
            r = summary["summary"]["tri_full"]["education/auc"]["vs_mpt"]
            self.assertEqual(r["n"], 5)
            self.assertLessEqual(r["low"], r["mean"])
            self.assertGreaterEqual(r["high"], r["mean"])
            folder = baseline / str(cfg["seeds"][0]) / "education/mpt"
            m = json.loads((folder / "metrics.json").read_text())
            m["paired_seed"] += 1
            core.dump(folder / "metrics.json", m)
            _commit(core, folder)
            with self.assertRaisesRegex(ValueError, "identity"):
                report(root, report_cfg, baseline)
