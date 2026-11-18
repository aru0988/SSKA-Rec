import importlib
import importlib.util
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path


class Table2QueueTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('sska_rec.table2_baselines'),
                             'Table2 experiment entry point is missing')
        self.runner = importlib.import_module('sska_rec.table2_baselines')
        self.cfg = self.runner.default_config()

    def test_exact_current_table2_scope_and_paired_seeds(self):
        jobs = self.runner.jobs(self.cfg)
        self.assertEqual(len(jobs), 40)
        self.assertEqual(set(d for d, s, v in jobs), {'census', 'aliccp'})
        self.assertEqual(set(v for d, s, v in jobs), {'sharedbottom', 'mmoe', 'ple', 'adatt'})
        self.assertEqual(len(set(jobs)), 40)
        self.assertEqual(len(self.runner.jobs(self.cfg, dataset='census', variant='adatt')), 5)

    def test_protocol_changes_rejected(self):
        self.cfg['datasets']['census']['split_seed'] = 42
        with self.assertRaises(ValueError):
            self.runner.validate_config(self.cfg)

    def test_partial_summary_contains_observed_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.runner.report(Path(tmp), self.cfg)
            self.assertEqual(result['expected_models'], 40)
            self.assertEqual(result['completed_models'], 0)
            self.assertTrue((Path(tmp)/'tables.md').exists())

    def test_import_does_not_probe_cuda_before_fork(self):
        code = "import torch; torch.cuda.is_available=lambda: (_ for _ in ()).throw(AssertionError('CUDA probe in fork parent')); import sska_rec.table2_baselines"
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_summary_rejects_corrupt_prediction_artifact(self):
        from sska_rec import core
        from sska_rec.training import _commit
        dataset, seed, variant = self.runner.jobs(self.cfg)[0]
        with tempfile.TemporaryDirectory() as tmp:
            folder = self.runner.leaf(tmp, dataset, seed, variant)
            folder.mkdir(parents=True)
            core.dump(folder/'metrics.json', dict(dataset=dataset, paired_seed=seed,
                      variant=variant, test={'income': {'auc': .7}, 'marital': {'auc': .8}}))
            (folder/'model.pt').write_bytes(b'weights')
            (folder/'predictions.npz').write_bytes(b'probabilities')
            _commit(core,folder)
            self.assertEqual(self.runner.report(Path(tmp),self.cfg)['completed_models'],1)
            (folder/'predictions.npz').write_bytes(b'corrupt-probabilities')
            with self.assertRaises(ValueError):
                self.runner.report(Path(tmp),self.cfg)


if __name__ == '__main__':
    unittest.main()
