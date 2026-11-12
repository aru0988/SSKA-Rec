import copy
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from sska_rec import core


class STRunnerTests(unittest.TestCase):
    def test_cuda_device_without_index_is_normalized_before_memory_queries(self):
        from sska_rec.st_runner import selected_device
        self.assertEqual(selected_device('cuda').index,0)
        self.assertEqual(selected_device('cuda:1').index,1)
        self.assertEqual(selected_device('cpu').type,'cpu')
    def test_only_explicit_census_is_allowed_to_overlap_primary(self):
        from sska_rec.st_runner import parallel_allowed
        self.assertTrue(parallel_allowed({'parallel_datasets':['census']},'census'))
        self.assertFalse(parallel_allowed({},'census'))
        self.assertFalse(parallel_allowed({'parallel_datasets':['census']},'aliccp'))
        with self.assertRaises(ValueError):parallel_allowed({'parallel_datasets':['kuairand']},'kuairand')
    def test_same_data_ignores_paths_but_rejects_changed_rows_or_labels(self):
        from sska_rec.st_runner import compare_data
        sig=dict(dataset='census',tasks=['a'],initial=['a'],features=['x'],vocab={},input_dim=1,representation_dim=4,widths=[1],inputs=[dict(path='/old/train.gz',sha256='raw',bytes=123)],splits={'train':dict(rows=8,rowid_sha256='ids',labels_sha256='labels',features_sha256='features',features_dtype='float32',labels_dtype='float32',rowid_dtype='int64')})
        other=copy.deepcopy(sig);other['inputs'][0]['path']='/new/train.gz'
        compare_data(sig,other)
        other['splits']['train']['labels_sha256']='different'
        with self.assertRaisesRegex(ValueError,'labels_sha256'):compare_data(sig,other)

    def test_primary_gate_requires_complete_matrix_and_exited_worker(self):
        from sska_rec.st_runner import primary_ready
        self.assertFalse(primary_ready({'state':'running','completed':125},False))
        self.assertFalse(primary_ready({'state':'complete','completed':124,'pid':None},False))
        self.assertFalse(primary_ready({'state':'complete','completed':125,'pid':None},True))
        self.assertTrue(primary_ready({'state':'complete','completed':125,'pid':None},False))

    def test_config_has_all_eleven_targets_and_five_seeds(self):
        import json
        c=json.loads((Path(__file__).parents[1]/'configs/st_three_datasets.json').read_text(encoding='utf8'))
        self.assertEqual(sum(len(d['st_targets'])*len(d['seeds']) for d in c['datasets'].values()),55)
        for d in c['datasets'].values():
            self.assertEqual(d['st_targets'],d['initial']+d['targets'])
            self.assertEqual(len(d['seeds']),5)
            self.assertEqual(d['epochs'],30)
            self.assertEqual(d['patience'],5)
            self.assertEqual(d['embedding_l2'],.006)

    def test_complete_status_cannot_bypass_held_primary_lock(self):
        from sska_rec.st_runner import wait_primary
        from sska_rec.locking import ExclusiveLock
        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[1]) as tmp:
            primary=Path(tmp)/'main/outputs/tri';primary.mkdir(parents=True)
            output=Path(tmp)/'st';output.mkdir()
            core.dump(primary/'status.json',dict(state='complete',completed=125,pid=None))
            lock=ExclusiveLock(primary/'worker.lock');lock.__enter__()
            before=b' '
            def release(_):
                self.assertEqual(__import__('json').loads((output/'status.json').read_text())['state'],'queued')
                lock.__exit__()
            with patch('sska_rec.st_runner.discover_primary',return_value=False),patch('sska_rec.st_runner.time.sleep',side_effect=release) as sleep:
                wait_primary(primary,output)
                sleep.assert_called_once_with(30)
            self.assertEqual((primary/'worker.lock').read_bytes(),before)

    def test_report_rejects_equal_metric_but_different_primary_rowids(self):
        import json
        import numpy as np
        from sska_rec.st_report import report
        from sska_rec.training import _commit
        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[1]) as tmp:
            root=Path(tmp);out=root/'st';main=root/'main'
            c=dict(datasets={'census':dict(seeds=[1,2,3,4,5],initial=['a','b'],st_targets=['new'])})
            y=np.array([0,1],dtype='float32');p=np.array([.2,.8],dtype='float32');ids=np.array([0,1])
            m=dict(paired_seed=1,seed=3,model_seed=3,target='new',variant='single_task',val=core.measures(y,p),test=core.measures(y,p))
            sf=out/'census/1/new';sf.mkdir(parents=True);core.dump(sf/'metrics.json',m)
            np.savez(sf/'predictions.npz',val_y=y,val_p=p,val_rowid=ids,test_y=y,test_p=p,test_rowid=ids);_commit(core,sf)
            mf=main/'census/1/new/mpt';mf.mkdir(parents=True);core.dump(mf/'metrics.json',dict(m,variant='mpt'))
            np.savez(mf/'predictions.npz',val_y=y,val_p=p,val_rowid=ids[::-1],test_y=y,test_p=p,test_rowid=ids[::-1]);_commit(core,mf)
            with self.assertRaisesRegex(ValueError,'labels/rowids'):report(out,c,main)
