import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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
            runner.core.dump(path, {'different': 'content'})
            with self.assertRaisesRegex(ValueError, 'manifest'):
                runner.verify_gate(root, 'config', signatures)

    def test_sample_variance_and_sign_in_paired_differences(self):
        result = runner.paired_summary([.75, .75, .625], [.625, .75, .75])
        self.assertEqual(result['mean'], 0.0)
        self.assertEqual(result['variance'], .015625)
        self.assertEqual((result['positive'], result['negative'], result['tie']), (1, 1, 1))


if __name__ == '__main__':
    unittest.main()
