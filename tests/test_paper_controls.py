import copy
import json
import unittest
from pathlib import Path


class PaperControlQueueTests(unittest.TestCase):
    def config(self):
        from sska_rec.paper_controls import default_config
        return default_config()

    def test_exact_115_training_models_and_nine_profiles(self):
        from sska_rec.paper_controls import jobs, expected_counts
        cfg = self.config()
        self.assertEqual(expected_counts(cfg), {'table4': 90, 'table5': 25, 'table6': 9})
        queue = jobs(cfg, [4, 5, 6])
        self.assertEqual(len(queue), 33)
        self.assertEqual(sum(t == 4 for t, d, s in queue), 15)
        self.assertEqual(sum(t == 5 for t, d, s in queue), 15)
        self.assertEqual(sum(t == 6 for t, d, s in queue), 3)
        self.assertEqual(len(set(queue)), len(queue))

    def test_original_protocol_is_preserved_and_kuairand_table4_present(self):
        from sska_rec.paper_controls import validate_config, jobs
        cfg = self.config()
        validate_config(cfg)
        self.assertEqual(len([j for j in jobs(cfg,[4]) if j[1]=='kuairand']), 5)
        changed=copy.deepcopy(cfg)
        changed['datasets']['aliccp']['epochs']=20
        with self.assertRaises(ValueError):validate_config(changed)

    def test_report_only_empty_output_does_not_load_datasets(self):
        import tempfile
        from sska_rec.paper_controls import main
        with tempfile.TemporaryDirectory() as folder:
            result=main(['--report-only','--output',folder])
            self.assertEqual(result['completed'], {'table4':0,'table5':0,'table6':0})
            self.assertEqual(result['expected']['table4'],90)

    def test_partial_referenced_results_are_labeled_not_rejected(self):
        from types import SimpleNamespace
        import tempfile
        from sska_rec import core
        from sska_rec.paper_controls import existing_rows
        cfg=self.config();seed=cfg['datasets']['census']['seeds'][0]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            core.dump(root/'summary.json',dict(records={f'census/{seed}/education/mpt':
                      dict(test=dict(auc=.8,logloss=.2))}))
            args=SimpleNamespace(source_root=root,readout_root=None,single_task_summary=None)
            rows=existing_rows(args,cfg)
            self.assertEqual(rows['table3/census/education/mpt']['auc']['n'],1)
            self.assertIsNone(rows['table3/census/education/mpt']['auc']['variance'])

    def test_real_tensor_all_three_tables_and_skip_completed(self):
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        import importlib.util
        import numpy as np
        import torch
        from sska_rec import core
        from sska_rec.paper_controls import run_job,completed_job,report
        torch.set_num_threads(1)
        helper=importlib.util.spec_from_file_location('paper_source_fixture',Path(__file__).parent/'test_scheme_reuse.py')
        module=importlib.util.module_from_spec(helper);helper.loader.exec_module(module)
        fixture=module.SchemeReuseTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        spec=fixture.spec();spec['scheme_data_identity']='tiny-admitted-data'
        base=core.build_base(spec).eval().requires_grad_(False)
        u=core.make_u(spec,17).eval().requires_grad_(False)
        caches={}
        with torch.no_grad():
            for split,rows in spec['sets'].items():
                features={name:torch.tensor(rows.x[:,i]) for i,name in enumerate(rows.names)}
                x,g,ss,_=base.get_infos(features)
                caches[split]=dict(x=x,g=g,s0=ss[0],s1=ss[1],old=torch.stack(base.predict(features),1),y=torch.from_numpy(rows.y))
            envs=[base.env_embedding_network(torch.tensor(i)) for i in range(2)]
        dc=dict(epochs=1,patience=5,learning_rate=spec['lr'],batch_size=spec['batch'],
                initial=spec['initial'],targets=spec['tasks'][2:],seeds=[17])
        cfg=dict(datasets={spec['dataset']:dc},order=[spec['dataset']])
        with tempfile.TemporaryDirectory() as folder:
            args=SimpleNamespace(output=Path(folder),cache_root=Path(folder)/'cache',dataset_root=Path('dataset'),
                                 source_root=Path(folder)/'old',readout_root=None,single_task_summary=None)
            with patch('sska_rec.tri_data.build_spec',return_value=spec),patch('sska_rec.scheme_reuse.load_sources',
                 return_value=(base,caches,envs,u,{'source_folder':'fixture'})):
                for table in [4,5,6]:run_job(args,cfg,table,spec['dataset'],17,'tiny-execution')
                stamp=(args.output/'table5'/spec['dataset']/'17'/spec['tasks'][2]/'random_lora/model.pt').stat().st_mtime_ns
                run_job(args,cfg,5,spec['dataset'],17,'tiny-execution')
                self.assertEqual(stamp,(args.output/'table5'/spec['dataset']/'17'/spec['tasks'][2]/'random_lora/model.pt').stat().st_mtime_ns)
            result=report(args,cfg)
            self.assertEqual(result['completed'],dict(table4=6,table5=1,table6=3))


if __name__=='__main__':unittest.main()
