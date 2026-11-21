import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from sska_rec import table2_v2_runner as runner


class Table2RunnerTests(unittest.TestCase):
    def test_formal_matrix_and_dependencies(self):
        cfg = dict(order=['census', 'aliccp', 'kuairand'], formal_seeds=list(range(5)),
                   datasets={name: {'initial': ['old1', 'old2']}
                             for name in ('census', 'aliccp', 'kuairand')})
        jobs = runner.make_jobs(cfg)
        self.assertEqual(len(set(jobs)), 150)
        self.assertEqual(len([job for job in jobs if job.dataset == 'census']), 50)
        source = runner.Job('aliccp', 1, 'source')
        self.assertEqual(source.dependencies(), (runner.Job('aliccp', 1, 'base'),))
        self.assertEqual(runner.Job('aliccp', 1, 'readout', 'old2').dependencies(), (source,))
        self.assertEqual(runner.Job('aliccp', 1, 'single_task', 'old2').dependencies(), ())

    def test_pilot_manifest_must_match(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            receipts = []
            for dataset in runner.PILOT_DATASETS:
                for mode in ('sum', 'mean'):
                    folder = root / dataset / mode / 'source' / 'base'
                    names = ('base.pt', 'metrics.json', 'protocol.json')
                    for name in names:
                        runner.core.dump(folder / name, {})
                    path = folder / 'COMMITTED.json'
                    runner.core.dump(path, {name: runner.core.file_hash(folder / name) for name in names})
                    receipts.append(dict(dataset=dataset, mode=mode,
                                         relative_manifest=path.relative_to(root).as_posix(),
                                         manifest_sha256=runner.core.file_hash(path)))
            signatures = {name: 'signature' for name in runner.PILOT_DATASETS}
            runner.core.dump(root / 'decision.json', dict(state='pass', cfg_sha256='config',
                             data_signatures=signatures, receipts=receipts))
            runner.verify_gate(root, 'config', signatures)
            with self.assertRaisesRegex(ValueError, 'config'):
                runner.verify_gate(root, 'different', signatures)
            decision_path = root / 'decision.json'
            runner.core.dump(decision_path, dict(runner.read_json(decision_path), state='review'))
            review_hash = runner.core.file_hash(decision_path)
            with self.assertRaisesRegex(ValueError, 'review'):
                runner.verify_gate(root, 'config', signatures)
            runner.verify_gate(root, 'config', signatures, review_hash=review_hash)
            with self.assertRaisesRegex(ValueError, 'review'):
                runner.verify_gate(root, 'config', signatures, review_hash='other')
            runner.core.dump(path, {'different': 'content'})
            with self.assertRaisesRegex(ValueError, 'manifest'):
                runner.verify_gate(root, 'config', signatures, review_hash=review_hash)

    def test_shared_regularization_modes_and_legacy_default(self):
        from sska_rec.baseline_models import build_baseline
        from sska_rec.st_fit import build_single
        from sska_rec.table2_v2_backbone import normalized_spec
        spec = dict(dataset='census', tasks=['a', 'b'], initial=['a', 'b'], vocab={'cat': 8},
                    embed=4, xdim=5, rdim=128, hidden=[256, 128], tower=[64, 32],
                    reg_dnn=3e-5, lr=.001, batch=256)
        for mode, expected in (('sum', .006), ('mean', .006/32)):
            cfg = {'embedding_l2_mode': mode}
            models = [runner.core.build_base(normalized_spec(spec, mode)), build_single(spec, 3, cfg)]
            models.extend(build_baseline(spec, 3, variant, cfg) for variant in runner.JOINT_VARIANTS)
            self.assertEqual([model.reg_embedding for model in models], [expected]*6)
        self.assertEqual(build_single(spec, 3, {}).reg_embedding, .006/32)
        self.assertEqual(build_baseline(spec, 3, 'mmoe').reg_embedding, .006/32)

    def test_sample_variance_and_sign_in_paired_differences(self):
        result = runner.paired_summary([.75, .75, .625], [.625, .75, .75])
        self.assertEqual(result['mean'], 0.0)
        self.assertEqual(result['variance'], .015625)
        self.assertEqual((result['positive'], result['negative'], result['tie']), (1, 1, 1))

    def test_readout_report_checks_reference_predictions_and_row_ids(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'readout'
            folder.mkdir()
            ids, labels = np.arange(4), np.asarray([[0, 1], [0, 0], [1, 1], [1, 0]])
            probabilities = np.asarray([[.1, .8], [.2, .3], [.7, .9], [.8, .25]])
            spec = {'tasks': ['old1', 'old2'], 'sets': {
                split: SimpleNamespace(y=labels, row_ids=ids) for split in runner.SPLITS}}
            metrics = {'selected_epoch': 0}
            inputs, baseline = {}, {}
            for split in runner.SPLITS:
                metrics[split] = runner.core.measures(labels[:, 1], probabilities[:, 1])
                for key, value in dict(y=labels[:, 1], p=probabilities[:, 1], rowid=ids,
                                       base_p=probabilities[:, 1]).items():
                    inputs[f'{split}_{key}'] = value
                baseline[f'{split}_p'] = probabilities
            runner.core.dump(folder / 'metrics.json', metrics)
            base_path = root / 'baseline.npz'
            np.savez(base_path, **baseline)
            raw_path = folder / 'predictions.npz'
            np.savez(raw_path, **inputs)
            self.assertEqual(runner.verify_raw(folder, spec, ['old2'], base_path=base_path), metrics)

            np.savez(raw_path, **dict(inputs, test_base_p=probabilities[:, 0]))
            with self.assertRaisesRegex(ValueError, 'baseline prediction mismatch'):
                runner.verify_raw(folder, spec, ['old2'], base_path=base_path)
            np.savez(raw_path, **dict(inputs, test_rowid=ids[::-1]))
            with self.assertRaisesRegex(ValueError, 'row IDs'):
                runner.verify_raw(folder, spec, ['old2'], base_path=base_path)


if __name__ == '__main__':
    unittest.main()
