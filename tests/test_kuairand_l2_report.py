import hashlib
import json
import tempfile
import unittest
from pathlib import Path

try:
    from sska_rec.kuairand_l2_report import report
except ImportError:
    report = None

SEEDS = [1685480945, 1685463909, 1685477428, 1685459668, 1685496394]
CFG = {'version': 'kuairand-normalized-l2-headonly-20261120', 'datasets': {'kuairand': {'seeds': SEEDS, 'initial': ['click', 'like'], 'targets': ['follow', 'forward']}}}


def commit(folder, metrics, weights=None, raw=True):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'metrics.json').write_text(json.dumps(metrics), encoding='utf8')
    if weights:
        (folder / weights).write_bytes(b'fixture checkpoint')
    if raw:
        (folder / 'predictions.npz').write_bytes(b'fixture sealed predictions')
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.name != 'COMMITTED.json'}
    (folder / 'COMMITTED.json').write_text(json.dumps(manifest), encoding='utf8')


def scores(auc=.7):
    return {'auc': auc, 'logloss': .5}


def source(root, seed, auc=.7, ssl=False):
    folder = root / 'kuairand' / str(seed)
    commit(folder / 'source/base', {'seed': seed, 'seconds': 10.}, 'base.pt', raw=False)
    if ssl:
        commit(folder / 'source/ssl', {'seed': seed + 1, 'seconds': 8.}, 'source.pt', raw=False)
    commit(folder / 'source-evaluation', {'paired_seed': seed, 'variant': 'mpt_base', 'model_hash': 'base', 'regularization': {'mode': 'mean', 'lambda': .006}, 'data_identity': {'rows': 10}, 'val': {t: scores(auc) for t in ['click', 'like']}, 'test': {t: scores(auc) for t in ['click', 'like']}})


def model(root, seed, target, variant, auc=.8, zero=False):
    old = target in ['click', 'like']
    folder = root / 'kuairand' / str(seed) / ('old' if old else 'new') / target / variant
    m = {'paired_seed': seed, 'target': target, 'variant': 'h_conditional' if variant == 'head_only' else variant, 'val': scores(auc), 'test': scores(auc), 'seconds': 3., 'optimizer_parameters': 12, 'zero_correction_selected': zero}
    commit(folder, m, 'model.pt')
    return folder


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
    def run_report(self, refs=None):
        self.assertIsNotNone(report, 'KuaiRand report is not implemented')
        return report(self.root, CFG, refs)
    def test_empty_report_exposes_actual_counts(self):
        result = self.run_report()
        self.assertEqual((result['completed_units'], result['expected_units']), (0, 70))
        self.assertEqual(result['completed_models'], 0)
        for name in ['summary.json', 'tables.md', 'backbone-diagnostic.md']:
            self.assertTrue((self.root / name).is_file())
    def test_b_only_diagnostic_is_available_before_u_and_heads(self):
        source(self.root, SEEDS[0], .71)
        result = self.run_report()
        self.assertEqual(result['completed_units'], 1)
        self.assertEqual(result['source_counts'], {'base': 1, 'ssl': 0, 'evaluation': 1})
        row = result['groups']['mpt_new/click/auc']
        self.assertEqual((row['n'], row['values'], row['seeds']), (1, [.71], [SEEDS[0]]))
        self.assertIsNone(row['variance'])
        text = (self.root / 'backbone-diagnostic.md').read_text(encoding='utf8')
        self.assertIn('1/5', text)
        self.assertIn('0.710000', text)
    def test_full_report_keeps_five_values_and_paired_deltas(self):
        for i, seed in enumerate(SEEDS):
            source(self.root, seed, .70 + i*.01, ssl=True)
            for task in ['click', 'like']:
                model(self.root, seed, task, 'head_only', .72 + i*.01)
                model(self.root, seed, task, 'conditional_readout', .75 + i*.01, zero=i == 0)
            for task in ['follow', 'forward']:
                for variant in ['mpt', 'tri_frozen', 'random_lora', 'tri_lora']:
                    model(self.root, seed, task, variant, .8 + i*.01)
        result = self.run_report()
        self.assertEqual(result['completed_units'], 70)
        self.assertEqual(result['completed_models'], 60)
        self.assertAlmostEqual(result['groups']['head_only/click/auc']['variance'], .00025)
        self.assertAlmostEqual(result['deltas']['head_only-minus-mpt_new/click/auc']['mean'], .02)
        self.assertAlmostEqual(result['deltas']['conditional_readout-minus-head_only/click/auc']['mean'], .03)
        self.assertEqual(result['zero_correction']['conditional_readout/click'], {'n': 5, 'selected': 1})
        self.assertEqual(result['costs']['head_only/click']['seconds']['values'], [3.]*5)
        self.assertEqual(result['costs']['head_only/click']['optimizer_parameters']['values'], [12.]*5)
    def test_references_are_attributed_without_fabricating_seed_values(self):
        refs = self.root / 'references.json'
        refs.write_text(json.dumps({'version': 'sealed-v1', 'source_files': [{'path': 'prior.json', 'sha256': 'abc'}], 'rows': {'kuairand/single_task/click': {'n': 5, 'mean': .8, 'variance': .0001}, 'kuairand/mpt_old/click': {'n': 5, 'mean': .6, 'variance': .0004}}}), encoding='utf8')
        source(self.root, SEEDS[0])
        result = self.run_report(refs)
        self.assertEqual(result['references']['sha256'], hashlib.sha256(refs.read_bytes()).hexdigest())
        self.assertNotIn('values', result['groups']['single_task/click/auc'])
        text = (self.root / 'backbone-diagnostic.md').read_text(encoding='utf8')
        self.assertIn('0.800000', text)
        self.assertIn('0.600000', text)
        self.assertIn('sealed-v1', text)
    def test_committed_tampering_is_rejected_even_after_cached_report(self):
        source(self.root, SEEDS[0])
        self.run_report()
        p = self.root / 'kuairand' / str(SEEDS[0]) / 'source/base/base.pt'
        p.write_bytes(b'changed checkpoint')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            self.run_report()
    def test_uncommitted_model_is_not_counted(self):
        folder = model(self.root, SEEDS[0], 'follow', 'mpt')
        (folder / 'COMMITTED.json').unlink()
        self.assertEqual(self.run_report()['completed_models'], 0)
    def test_identity_mismatch_is_rejected(self):
        folder = model(self.root, SEEDS[0], 'click', 'head_only')
        commit(folder, {'paired_seed': SEEDS[1], 'target': 'click', 'variant': 'h_conditional', 'test': scores()}, 'model.pt')
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.run_report()
    def test_missing_checkpoint_is_rejected(self):
        folder = model(self.root, SEEDS[0], 'follow', 'mpt')
        manifest = json.loads((folder / 'COMMITTED.json').read_text())
        del manifest['model.pt']
        (folder / 'COMMITTED.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.run_report()
    def test_reference_incomplete_or_nan_is_rejected(self):
        p = self.root / 'references.json'
        for row in [{'mean': .8, 'variance': .01, 'n': 4}, {'mean': float('nan'), 'variance': .01, 'n': 5}]:
            p.write_text(json.dumps({'version': 'v', 'source_files': [], 'rows': {'kuairand/single_task/click': row}}))
            with self.assertRaisesRegex(ValueError, 'reference'):
                self.run_report(p)
    def test_output_config_version_is_checked(self):
        (self.root / 'identity.json').write_text(json.dumps({'config': {'version': 'other'}}))
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.run_report()



    def test_zero_correction_uses_the_training_selected_epoch(self):
        folder = model(self.root, SEEDS[0], 'click', 'conditional_readout')
        m = json.loads((folder / 'metrics.json').read_text())
        del m['zero_correction_selected']
        m['selected_epoch'] = 0
        commit(folder, m, 'model.pt')
        result = self.run_report()
        self.assertEqual(result['zero_correction']['conditional_readout/click'], {'n': 1, 'selected': 1})
    def test_changed_sealed_reference_is_rejected_by_execution_identity(self):
        path = self.root / 'references.json'
        path.write_text(json.dumps({'version': 'v1', 'source_files': [], 'rows': {}}))
        (self.root / 'identity.json').write_text(json.dumps({'config': CFG, 'references_sha256': 'wrong'}))
        with self.assertRaisesRegex(ValueError, 'reference identity'):
            self.run_report(path)
    def test_evaluation_source_hash_must_match_the_committed_base(self):
        source(self.root, SEEDS[0])
        folder = self.root / 'kuairand' / str(SEEDS[0]) / 'source/base'
        commit(folder, {'model_hash': 'other-base'}, 'base.pt', raw=False)
        with self.assertRaisesRegex(ValueError, 'source identity'):
            self.run_report()


    def test_readout_with_other_backbone_is_rejected(self):
        source(self.root, SEEDS[0])
        folder = model(self.root, SEEDS[0], 'click', 'conditional_readout')
        m = json.loads((folder / 'metrics.json').read_text())
        m['base_hash'] = 'wrong-base'
        commit(folder, m, 'model.pt')
        with self.assertRaisesRegex(ValueError, 'source identity'):
            self.run_report()
