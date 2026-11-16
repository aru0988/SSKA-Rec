import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class SchemeRunnerTests(unittest.TestCase):
    def test_real_tensor_both_schemes_and_completed_reuse(self):
        import importlib.util
        import torch
        from sska_rec import core, tri_sources
        from sska_rec.training import _commit
        from sska_rec.scheme1_runner import run_seed as run1
        from sska_rec.scheme2_runner import run_seed as run2
        from sska_rec.scheme_report import report
        helper = importlib.util.spec_from_file_location('scheme_source_fixture', ROOT/'tests/test_scheme_reuse.py')
        module = importlib.util.module_from_spec(helper)
        helper.loader.exec_module(module)
        fixture = module.SchemeReuseTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        spec = fixture.spec()
        owner = fixture.fixture(spec)
        base = core.build_base(spec).eval()
        base.load_state_dict(torch.load(owner/'source/base/base.pt', map_location='cpu', weights_only=True))
        with torch.no_grad():
            for split, rows in spec['sets'].items():
                features = {name: torch.tensor(rows.x[:,i]) for i,name in enumerate(rows.names)}
                x,g,ss,_ = base.get_infos(features)
                old = torch.stack(base.predict(features),1)
                for name,value in zip(['x','g','s0','s1','old'],[x,g,*ss,old]):
                    np.save(owner/f'feature-cache/{split}-{name}.npy', value.numpy())
        _commit(core, owner/'feature-cache')
        receipt = json.loads((owner/'reuse-receipt.json').read_text())
        receipt['stages']['cache'] = tri_sources._verify_stage(owner/'feature-cache',['identity.json'])
        core.dump(owner/'reuse-receipt.json',receipt)
        cfg = dict(base_epochs=30, ssl_epochs=30, decoder_epochs=1, epochs=1, patience=5,
                   learning_rate=spec['lr'],batch_size=spec['batch'])
        before = core.file_hash(owner/'source/ssl/source.pt')
        first = run1(spec,cfg,17,fixture.root/'scheme1/aliccp/17',owner)
        self.assertEqual(len(first),5)
        second = run2(spec,cfg,17,fixture.root/'scheme2/aliccp/17',owner)
        self.assertEqual(set(second),{'a','b'})
        self.assertEqual(before,core.file_hash(owner/'source/ssl/source.pt'))
        stamp = (fixture.root/'scheme2/aliccp/17/old/a/model.pt').stat().st_mtime_ns
        run2(spec,cfg,17,fixture.root/'scheme2/aliccp/17',owner)
        self.assertEqual(stamp,(fixture.root/'scheme2/aliccp/17/old/a/model.pt').stat().st_mtime_ns)
        rcfg={'variants':['mpt','tri_frozen','tri_lora','tri_ssf','tri_full'],
              'datasets':{'aliccp':dict(seeds=[17],initial=['a','b'],targets=['c'])}}
        self.assertEqual(report(fixture.root/'scheme1',rcfg,1)['completed'],5)
        self.assertEqual(report(fixture.root/'scheme2',rcfg,2)['completed'],2)

    def test_mpt_control_reuse_verifies_predictions_and_frozen_base(self):
        import torch
        from types import SimpleNamespace
        from sska_rec import core
        from sska_rec.scheme_models import build_scheme1_model
        from sska_rec.scheme_controls import reuse_control
        from sska_rec.scheme_common import source_hashes
        from sska_rec.training import _commit, _save
        torch.set_num_threads(1)
        previous = core.DEVICE
        core.DEVICE = torch.device('cpu')
        self.addCleanup(setattr, core, 'DEVICE', previous)
        spec = dict(dataset='census', xdim=6, rdim=8, tower=[4, 3], batch=4,
                    tasks=['income', 'marital', 'sex'], initial=['income', 'marital'], sets={})
        caches, envs = {}, [torch.randn(8), torch.randn(8)]
        model = build_scheme1_model(spec, None, 17, 'mpt')
        model.eval()
        for i, split in enumerate(['val', 'test']):
            y = np.asarray([[0, 1, 0], [1, 0, 1]] * 4, dtype='float32')
            spec['sets'][split] = SimpleNamespace(row_ids=np.arange(8) + i*8)
            caches[split] = {k: torch.randn(8, w) for k, w in [('x', 6), ('g', 8), ('s0', 8), ('s1', 8)]}
            caches[split]['y'] = torch.from_numpy(y)
        cfg = dict(epochs=1, patience=5, learning_rate=.001, batch_size=4)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            leaf = root/'source/census/17'
            old = leaf/'sex/mpt'
            old.mkdir(parents=True)
            core.dump(root/'source/identity.json', dict(source=source_hashes(), config={'datasets': {'census': cfg}}))
            raw, scores = {}, {}
            for split in ['val', 'test']:
                p = core.probabilities(model, caches[split], envs, spec)
                y = caches[split]['y'][:, 2].numpy()
                raw.update({f'{split}_p': p, f'{split}_y': y, f'{split}_rowid': spec['sets'][split].row_ids})
                scores[split] = core.measures(y, p)
            _save(model.state_dict(), old/'model.pt')
            core.dump(old/'metrics.json', dict(paired_seed=17, seed=20, target='sex', variant='mpt',
                epochs=1, selected_epoch=1, stop_reason='budget', learning_rate=.001, patience=5,
                model_hash=core.model_hash(model), frozen_original_paths={'base_hash': 'base'}, **scores))
            np.savez_compressed(old/'predictions.npz', **raw)
            _commit(core, old)
            self.assertTrue(reuse_control(model, spec, caches, envs, 17, 'sex', cfg,
                root/'new/sex/mpt', leaf, 'base', lambda: {'passed': True}))
            self.assertFalse((root/'new/sex/mpt/model.pt').exists())
            with self.assertRaisesRegex(ValueError, 'base'):
                reuse_control(model, spec, caches, envs, 17, 'sex', cfg,
                    root/'other/sex/mpt', leaf, 'different', lambda: {} )

    def test_scheme2_rejects_stage2_and_changed_seed_protocol(self):
        from sska_rec.scheme_common import validate_config
        cfg = json.loads((ROOT / 'configs/scheme2_three_datasets.json').read_text())
        validate_config(cfg, 2)
        changed = copy.deepcopy(cfg)
        changed['variants'] = ['tri_lora']
        with self.assertRaisesRegex(ValueError, 'stage2|variants'):
            validate_config(changed, 2)
        changed = copy.deepcopy(cfg)
        changed['datasets']['census']['seeds'] = [1]
        with self.assertRaisesRegex(ValueError, 'seed'):
            validate_config(changed, 2)

    def test_scheme_configs_keep_exact_historical_data_and_training_budget(self):
        from sska_rec.scheme_common import validate_config
        old = json.loads((ROOT / 'configs/tri_three_datasets.json').read_text())
        for scheme in [1, 2]:
            cfg = json.loads((ROOT / f'configs/scheme{scheme}_three_datasets.json').read_text())
            validate_config(cfg, scheme)
            for dataset, dc in old['datasets'].items():
                for key in ['seeds', 'initial', 'targets', 'learning_rate', 'batch_size', 'epochs', 'patience']:
                    self.assertEqual(cfg['datasets'][dataset][key], dc[key])

    def test_report_only_does_not_load_dataset_or_sources(self):
        from sska_rec.scheme2_runner import main
        with tempfile.TemporaryDirectory() as temp:
            main(['--report-only', '--output', temp])
            summary = json.loads((Path(temp) / 'summary.json').read_text())
            self.assertEqual(summary['expected'], 30)
            self.assertEqual(summary['completed'], 0)
            self.assertEqual(summary['scheme'], 2)

    def test_parallel_jobs_are_independent_and_never_schedule_scheme2_stage2(self):
        from sska_rec.scheme_parallel import build_jobs
        cfg = json.loads((ROOT / 'configs/scheme2_three_datasets.json').read_text())
        jobs = build_jobs(cfg, 2, Path('out'), Path('old'), Path('dataset'), Path('cache'))
        self.assertEqual(len(jobs), 15)
        self.assertEqual(len({tuple(job) for job in jobs}), 15)
        for job in jobs:
            self.assertIn('sska_rec.scheme2_runner', job)
            self.assertIn('--seed', job)
            self.assertNotIn('--stage2', job)

    def test_report_recomputes_raw_and_rejects_tampered_metrics(self):
        from sska_rec import core
        from sska_rec.scheme_report import report
        from sska_rec.training import _commit
        cfg = {'datasets': {'census': {'seeds': [17], 'initial': ['income'], 'targets': ['sex']}}}
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / 'census/17/old/income'
            folder.mkdir(parents=True)
            y = np.asarray([0, 1], dtype='float32')
            p = np.asarray([.1, .9], dtype='float32')
            scores = core.measures(y, p)
            contract = dict(stage='conditional_readout',dataset='census',target='income',
                dependencies=dict(base_hash='base',u_hash='u'),data={'splits':{'val':{'rows':2},'test':{'rows':2}}})
            core.dump(folder/'contract.json',contract)
            core.dump(folder / 'metrics.json', dict(paired_seed=17, target='income', variant='conditional_readout',
                val=scores, test=scores, baseline={'val': scores, 'test': scores}, seconds=1., selected_epoch=0,
                contract=contract,base_hash='base',u_hash='u'))
            np.savez(folder / 'predictions.npz', val_y=y, val_p=p, val_rowid=[1, 2], val_base_p=p,
                     test_y=y, test_p=p, test_rowid=[3, 4], test_base_p=p)
            _commit(core, folder)
            summary = report(Path(temp), cfg, 2)
            self.assertEqual(summary['completed'], 1)
            item = summary['summary']['census']['conditional_readout/income/auc']
            self.assertEqual(item['n'], 1)
            self.assertIsNone(item['variance'])
            m = json.loads((folder / 'metrics.json').read_text())
            m['test'] = dict(scores, auc=.5)
            core.dump(folder / 'metrics.json', m)
            _commit(core, folder)
            with self.assertRaisesRegex(ValueError, 'metric'):
                report(Path(temp), cfg, 2)


if __name__ == '__main__':
    unittest.main()
