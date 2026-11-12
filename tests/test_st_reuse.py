import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from sska_rec import core
from sska_rec.backbone import SingleTask
from sska_rec.training import _commit
from sska_rec.tri_sources import _object_hash


class SingleReuseTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('sska_rec.st_reuse'))
        from sska_rec import st_reuse
        return st_reuse

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.legacy = self.root / 'st'
        self.old = self.legacy / 'outputs' / '101' / 'is_follow'
        self.old.mkdir(parents=True)
        self.dest = self.root / 'new'
        self.cfg = dict(epochs=30, patience=5, embedding_l2=.006, dnn_l2=3e-5,
                        learning_rate=.001, batch_size=512,
                        legacy_source_root=str(self.root / 'source'))
        source = self.root / 'source'
        project = source / 'paper_continual_20261030'
        project.mkdir(parents=True)
        old_model = source / 'MPT-Rec-exp-universal-representation/multitaskrec/model.py'
        old_model.parent.mkdir(parents=True)
        old_model.write_bytes(Path('sska_rec/backbone.py').read_bytes())
        oldcfg = dict(seeds=[101], kuairand=dict(epochs=30, patience=5, learning_rate=.001,
                      batch_size=512, embedding_size=2, hidden=[4], tower=[2], rdim=4))
        core.dump(project / 'config.json', oldcfg)
        lock = dict(files={str(p.relative_to(source)).replace('\\', '/'):core.file_hash(p)
                           for p in [old_model, project / 'config.json']})
        core.dump(project / 'source-lock.json', lock)
        (self.legacy / 'train.py').write_text('original execution fixture', encoding='utf8')
        core.dump(self.legacy / 'source-lock.json', {'train.py':core.file_hash(self.legacy / 'train.py')})
        self.identity = dict(version='kuairand-st-normalized-l2-v2', source=_object_hash(lock),
                             execution=core.file_hash(self.legacy / 'train.py'), data='data-hash', seeds=[101],
                             embedding_l2='0.006/(2*N_E)*sum(E^2)', epochs=30, patience=5,
                             lr=.001, batch=512, shuffle=False)
        core.dump(self.legacy / 'outputs/identity.json', self.identity)
        self.spec = dict(dataset='kuairand', tasks=['click','like','follow','forward'], initial=['click','like'],
                         vocab={'category':3}, embed=2, xdim=3, rdim=4, hidden=[4], tower=[2],
                         batch=512, lr=.001, sets={})
        cache = self.root / 'cache'
        cache.mkdir()
        manifest = dict(preproc_hash='data-hash', features=dict(names=['category','dense']), splits={})
        raw = {}
        for i, split in enumerate(['train','val','test']):
            y = np.column_stack([np.arange(8)%2]*4).astype('float32')
            rows = core.Rows(np.column_stack([np.arange(8)%3, np.arange(8)]).astype('float32'), y,
                             ['category','dense'], [0,1])
            rows.row_ids = np.arange(8, dtype='int64') + i * 8
            self.spec['sets'][split] = rows
            info = dict(rows=8, sha256={})
            for kind, value in [('x',rows.x),('y',rows.y),('rowid',rows.row_ids)]:
                name = split+'-'+kind+'.npy'
                np.save(cache / name, value)
                info[kind] = name
                info['sha256'][kind] = core.file_hash(cache / name)
            manifest['splits'][split] = info
            if split != 'train':
                raw[split+'_y'] = y[:,2]
                raw[split+'_rowid'] = rows.row_ids
                raw[split+'_p'] = np.linspace(.1,.9,8,dtype='float32')
        core.dump(cache / 'manifest.json', manifest)
        self.spec['preprocessing_manifest'] = str(cache / 'manifest.json')
        self.model = SingleTask({'category':3},2,3,[4],[2],.006/6,3e-5)
        torch.save(self.model.state_dict(), self.old / 'model.pt')
        np.savez_compressed(self.old / 'predictions.npz', **raw)
        self.raw = raw
        self.metrics = dict(target='is_follow', seed=105, epochs=6, selected_epoch=1, stop_reason='patience',
                            embedding_elements=6, embedding_sum_coefficient=.006/6, dnn_coefficient=3e-5,
                            optimizer_parameters=sum(p.numel() for p in self.model.parameters()), seconds=8,
                            peak_cuda_bytes=42, history=[dict(epoch=i+1,val_auc=.7-i*.01,val_logloss=.4) for i in range(6)],
                            val=core.measures(raw['val_y'],raw['val_p']), test=core.measures(raw['test_y'],raw['test_p']))
        # The selected validation result is bound to the actual raw predictions.
        self.metrics['history'][0].update(val_auc=self.metrics['val']['auc'],val_logloss=self.metrics['val']['logloss'])
        for row in self.metrics['history'][1:]: row['val_auc']=self.metrics['val']['auc']-.01
        core.dump(self.old / 'metrics.json', self.metrics)
        _commit(core, self.old)

    def reuse(self):
        api = self.api()
        with patch.object(api, 'EXECUTION_SHA256', self.identity['execution']):
            return api.reuse_single(self.spec,101,'follow',self.model,self.dest,self.legacy,self.cfg)

    def test_keeps_raw_artifacts_external_and_rechecks_integrity(self):
        self.assertTrue(self.reuse())
        self.assertEqual(self.api().artifact_folder(self.dest),self.old.resolve())
        self.assertEqual({p.name for p in self.dest.iterdir()},{'metrics.json','REUSED.json','COMMITTED.json'})
        meta=json.loads((self.dest/'metrics.json').read_text())
        self.assertEqual((meta['variant'],meta['seed'],meta['paired_seed']),('single_task',105,101))
        self.assertEqual(meta['seconds'],8)
        self.assertIsNone(json.loads((self.dest/'REUSED.json').read_text())['base'])
        self.assertTrue(self.reuse())
        with (self.old/'model.pt').open('ab') as stream: stream.write(b'corrupt')
        with self.assertRaisesRegex(ValueError,'integrity'):
            self.api().artifact_folder(self.dest)

    def test_refuses_wrong_rows_and_changed_training_features(self):
        self.spec['sets']['val'].row_ids[0] = 123
        with self.assertRaisesRegex(ValueError,'rows'):
            self.reuse()
        self.spec['sets']['val'].row_ids[0] = 8
        self.spec['sets']['train'].x[0,1] += 1
        with self.assertRaisesRegex(ValueError,'rows'):
            self.reuse()

    def test_refuses_protocol_seed_and_early_stopping_mismatch(self):
        self.cfg['epochs']=20
        with self.assertRaisesRegex(ValueError,'protocol'):
            self.reuse()
        self.cfg['epochs']=30
        self.metrics['seed']=104
        core.dump(self.old/'metrics.json',self.metrics); _commit(core,self.old)
        with self.assertRaisesRegex(ValueError,'seed'):
            self.reuse()
        self.metrics.update(seed=105,epochs=5)
        core.dump(self.old/'metrics.json',self.metrics); _commit(core,self.old)
        with self.assertRaisesRegex(ValueError,'history'):
            self.reuse()

    def test_refuses_raw_metric_and_regularization_mismatch(self):
        self.raw['val_p'] = 1-self.raw['val_p']
        np.savez_compressed(self.old/'predictions.npz',**self.raw); _commit(core,self.old)
        with self.assertRaisesRegex(ValueError,'metric'):
            self.reuse()
        self.raw['val_p'] = 1-self.raw['val_p']
        np.savez_compressed(self.old/'predictions.npz',**self.raw); _commit(core,self.old)
        self.model.reg_embedding = .006
        with self.assertRaisesRegex(ValueError,'regularization'):
            self.reuse()

    def test_no_historical_candidate_for_census_or_aliccp(self):
        for dataset in ['census','aliccp']:
            self.spec['dataset']=dataset
            self.assertFalse(self.reuse())
        self.assertFalse(self.dest.exists())

    def test_refuses_changed_execution_or_semantic_source(self):
        (self.legacy/'train.py').write_text('tampered execution',encoding='utf8')
        with self.assertRaisesRegex(ValueError,'execution'):
            self.reuse()
        (self.legacy/'train.py').write_text('original execution fixture',encoding='utf8')
        source=Path(self.cfg['legacy_source_root'])
        path=source/'MPT-Rec-exp-universal-representation/multitaskrec/model.py'
        path.write_text(path.read_text(encoding='utf8').replace('final_output = self.tower_network(mid_output)',
                        'final_output = self.tower_network(mid_output) + 1'),encoding='utf8')
        lockpath=source/'paper_continual_20261030/source-lock.json'
        lock=json.loads(lockpath.read_text())
        lock['files']['MPT-Rec-exp-universal-representation/multitaskrec/model.py']=core.file_hash(path)
        core.dump(lockpath,lock)
        self.identity['source']=_object_hash(lock)
        core.dump(self.legacy/'outputs/identity.json',self.identity)
        with self.assertRaisesRegex(ValueError,'semantics'):
            self.reuse()

    def test_refuses_incompatible_checkpoint_without_fallback(self):
        torch.save({'unrelated.weight':torch.ones(2)},self.old/'model.pt')
        _commit(core,self.old)
        with self.assertRaisesRegex(RuntimeError,'state_dict'):
            self.reuse()
        self.assertFalse(self.dest.exists())

if __name__=='__main__': unittest.main()
