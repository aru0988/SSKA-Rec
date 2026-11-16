import gc
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sska_rec import core, tri_data, tri_sources
import numpy as np
import torch
from sska_rec.training import _commit


class SchemeReuseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        previous = core.DEVICE
        core.DEVICE = torch.device('cpu')
        self.addCleanup(setattr, core, 'DEVICE', previous)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = dict(base_epochs=30, ssl_epochs=30)

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('sska_rec.scheme_reuse'),
                             'read-only source auditor has not been implemented')
        from sska_rec import scheme_reuse
        return scheme_reuse

    def spec(self, dataset='aliccp', counts=(8, 4, 4)):
        sets, offset = {}, 0
        for split, n in zip(('train', 'val', 'test'), counts):
            x = np.column_stack((np.arange(n) / max(1, n), np.arange(n) % 3))
            y = np.column_stack([(np.arange(n) // k) % 2 for k in (1, 2, 1)]).astype('float32')
            rows = core.Rows(x, y, ['dense', 'cat'], [0, 1])
            rows.row_ids = np.arange(offset, offset + n, dtype='int64')
            rows.user_groups = rows.row_ids.copy()
            sets[split], offset = rows, offset + n
        spec = dict(dataset=dataset, sets=sets, tasks=['a', 'b', 'c'], initial=['a', 'b'],
                    vocab={'cat': 3}, embed=2, xdim=3, rdim=4, batch=8, lr=.001,
                    hidden=[8, 4], tower=[4], widths=[1, 2], env_coe=.1)
        spec['data_signature'] = tri_data.dataset_signature(spec, [dict(path='/raw/input.csv', sha256='a'*64, bytes=77, mtime_ns=1)])
        return spec

    def fixture(self, spec, owner=None):
        owner = owner or self.root / 'tri' / spec['dataset'] / '17'
        owner.mkdir(parents=True)
        identity = tri_sources.source_identity(spec, 17, self.cfg)
        core.dump(owner / 'tri-source-identity.json', identity)
        base, u = core.build_base(spec), core.make_u(spec, 17)
        stages = {}
        for name, model, checkpoint, stage in [('base', base, 'base.pt', 'source/base'), ('u', u, 'source.pt', 'source/ssl')]:
            folder = owner / stage
            folder.mkdir(parents=True)
            torch.save(model.state_dict(), folder / checkpoint)
            core.dump(folder / 'metrics.json', dict(model_hash=core.model_hash(model)))
            _commit(core, folder)
            stages[name] = tri_sources._verify_stage(folder, [checkpoint, 'metrics.json'])
        folder = owner / 'feature-cache'
        folder.mkdir()
        for split, rows in spec['sets'].items():
            for key, width in dict(x=spec['xdim'], g=spec['rdim'], s0=spec['rdim'], s1=spec['rdim'], old=2).items():
                np.save(folder / f'{split}-{key}.npy', np.zeros((len(rows), width), dtype='float32'))
        core.dump(folder / 'identity.json', dict(base_hash=core.model_hash(base), splits={s:dict(rows=len(r), rowid_sha256=tri_data._array_hash(r.row_ids)) for s,r in spec['sets'].items()}))
        core.dump(folder / 'cost.json', dict(seconds=0))
        _commit(core, folder)
        stages['cache'] = tri_sources._verify_stage(folder, ['identity.json'])
        receipt = dict(seed=17, source_root=str(owner), identity_sha256=core.file_hash(owner/'tri-source-identity.json'),
                       data_signature=identity['data_signature'], source_hashes=identity['source_hashes'],
                       model_hashes=dict(base=core.model_hash(base), u=core.model_hash(u)), stages=stages)
        core.dump(owner / 'reuse-receipt.json', receipt)
        return owner

    def load(self, spec, root, require_u=True):
        api = self.api()
        with patch.object(core.PairedTrainManager, 'train_two_task', side_effect=AssertionError('must never train')), patch.object(tri_sources, '_prepare', side_effect=AssertionError('must never prepare')):
            return api.load_sources(spec, 17, self.cfg, root, require_u=require_u)

    def test_accepts_all_root_forms_without_linear_head_and_never_writes(self):
        spec = self.spec()
        owner = self.fixture(spec)
        before = {str(p.relative_to(self.root)):core.file_hash(p) for p in self.root.rglob('*') if p.is_file()}
        for root in (owner, owner.parent, owner.parent.parent):
            base, caches, envs, u, receipt = self.load(spec, root)
            self.assertTrue(all(not m.training and all(not p.requires_grad for p in m.parameters()) for m in (base,u)))
            self.assertTrue(all(not e.requires_grad for e in envs))
            np.testing.assert_array_equal(caches['test']['y'].numpy(), spec['sets']['test'].y)
            self.assertEqual(receipt['model_hashes']['base'], core.model_hash(base))
            caches['train']['x'][0, 0] = 999
            self.assertEqual(np.load(owner/'feature-cache/train-x.npy')[0,0], 0)
            del base, caches, envs, u, receipt
            gc.collect()
        self.assertEqual(before, {str(p.relative_to(self.root)):core.file_hash(p) for p in self.root.rglob('*') if p.is_file()})

    def test_scheme1_does_not_require_or_instantiate_original_u(self):
        spec = self.spec()
        owner = self.fixture(spec)
        (owner/'source/ssl/source.pt').write_bytes(b'not needed by scheme1')
        with patch.object(core, 'make_u', side_effect=AssertionError('old U must not load')):
            result = self.load(spec, owner, require_u=False)
        self.assertIsNone(result[3])
        self.assertNotIn('u', result[-1]['model_hashes'])
        del result
        gc.collect()
        with self.assertRaisesRegex(ValueError,'integrity'):
            self.load(spec,owner,require_u=True)

    def test_complete_census_sources_accepted(self):
        spec = self.spec('census', (199523, 49881, 49881))
        owner = self.fixture(spec)
        result = self.load(spec, owner, require_u=False)
        self.assertEqual(len(result[1]['train']['y']), 199523)
        del result
        gc.collect()

    def test_relocated_raw_paths_mtime_and_transport_are_semantically_identical(self):
        spec = self.spec()
        owner = self.fixture(spec)
        spec['data_signature']['inputs'][0].update(path='D:/different/input.csv', mtime_ns=999)
        ip = owner/'tri-source-identity.json'
        identity = json.loads(ip.read_text())
        identity['runtime_pipeline'].update(num_workers=2, pin_memory=True)
        core.dump(ip, identity)
        rp = owner/'reuse-receipt.json'
        receipt = json.loads(rp.read_text())
        receipt['identity_sha256'] = core.file_hash(ip)
        core.dump(rp, receipt)
        result = self.load(spec, owner)
        del result
        gc.collect()

    def test_changed_actual_labels_features_and_rows_rejected_even_with_stale_signature(self):
        for kind in ('labels', 'features', 'rows'):
            with self.subTest(kind=kind):
                spec = self.spec()
                owner = self.fixture(spec, self.root/kind)
                rows = spec['sets']['train']
                if kind == 'labels': rows.y[0,2] = 1 - rows.y[0,2]
                elif kind == 'features': rows.x[0,0] += 1
                else: rows.row_ids[0] = 999
                with self.assertRaisesRegex(ValueError, 'actual|fingerprint|row'):
                    self.load(spec, owner)

    def test_identity_protocol_targets_seed_and_source_mismatches_rejected(self):
        for kind in ('protocol', 'target', 'seed', 'source', 'split', 'vocab'):
            with self.subTest(kind=kind):
                spec = self.spec()
                owner = self.fixture(spec, self.root/kind)
                path = owner/'tri-source-identity.json'
                identity = json.loads(path.read_text())
                if kind == 'protocol': identity['training']['base_epochs'] = 29
                elif kind == 'target': identity['data_signature']['tasks'][2] = 'wrong'
                elif kind == 'seed': identity['seed'] = 18
                elif kind == 'source': identity['source_hashes']['core.py'] = 'b'*64
                elif kind == 'split': identity['splits']['test']['n'] += 1
                else: identity['architecture']['vocab']['cat'] = 4
                core.dump(path, identity)
                with self.assertRaises(ValueError): self.load(spec, owner)

    def test_corrupt_weights_cache_and_unsafe_commit_rejected(self):
        for kind in ('weights', 'cache', 'unsafe', 'cache-owner', 'cache-rows', 'cache-shape', 'model-hash'):
            with self.subTest(kind=kind):
                spec = self.spec()
                owner = self.fixture(spec, self.root/kind)
                if kind == 'weights': (owner/'source/base/base.pt').write_bytes(b'corrupt')
                elif kind == 'cache': (owner/'feature-cache/train-g.npy').write_bytes(b'corrupt')
                elif kind == 'unsafe':
                    path = owner/'source/base/COMMITTED.json'
                    entries = json.loads(path.read_text()); entries['../outside'] = 'a'*64; core.dump(path, entries)
                elif kind.startswith('cache-'):
                    if kind in ('cache-owner','cache-rows'):
                        path = owner/'feature-cache/identity.json'; identity=json.loads(path.read_text())
                        if kind == 'cache-owner': identity['base_hash']='a'*64
                        else: identity['splits']['train']['rowid_sha256']='b'*64
                        core.dump(path,identity)
                    else: np.save(owner/'feature-cache/train-g.npy', np.zeros((8,3),dtype='float32'))
                    _commit(core, owner/'feature-cache')
                else:
                    core.dump(owner/'source/base/metrics.json',dict(model_hash='a'*64)); _commit(core,owner/'source/base')
                if kind.startswith('cache-') or kind == 'model-hash':
                    path=owner/'reuse-receipt.json'; receipt=json.loads(path.read_text())
                    key='cache' if kind.startswith('cache-') else 'base'
                    folder=owner/'feature-cache' if key == 'cache' else owner/'source/base'
                    receipt['stages'][key]=tri_sources._verify_stage(folder,['identity.json'] if key == 'cache' else ['base.pt','metrics.json'])
                    core.dump(path,receipt)
                with self.assertRaises(ValueError): self.load(spec, owner)

    def test_explicit_source_is_required_and_missing_source_never_trains(self):
        api = self.api()
        for root in (None, self.root/'absent'):
            with self.assertRaisesRegex(ValueError, 'source|Source'):
                api.load_sources(self.spec(),17,self.cfg,root)

    def test_requested_source_protocol_mismatch_is_rejected(self):
        spec = self.spec()
        owner = self.fixture(spec)
        for key,value in [('patience',4),('learning_rate',.9),('batch_size',999),('precision','float16'),('shuffle',True)]:
            with self.subTest(key=key):
                cfg = dict(self.cfg, **{key:value})
                with self.assertRaisesRegex(ValueError, 'protocol'):
                    self.api().load_sources(spec,17,cfg,owner)

    def test_audited_identity_available_to_downstream_without_hashing_arrays_again(self):
        spec = self.spec()
        owner = self.fixture(spec)
        result = self.load(spec,owner,require_u=False)
        self.assertEqual(result[1]['train']['y'].numpy().__array_interface__['data'][0],spec['sets']['train'].y.__array_interface__['data'][0])
        self.assertIn('scheme_cache_identity',spec)
        self.assertEqual(spec['scheme_cache_identity']['base_hash'],result[-1]['model_hashes']['base'])
        self.assertEqual(spec['scheme_cache_identity']['cache_files'],result[-1]['stages']['cache']['files'])
        self.assertEqual(spec['scheme_cache_identity']['data_signature_sha256'],spec['scheme_data_identity'])
        del result
        gc.collect()

    def historical_fixture(self):
        spec = self.spec()
        spec['dataset'] = spec['data_signature']['dataset'] = 'kuairand'
        owner = self.fixture(spec)
        legacy = self.root/'paper_continual_20261030'
        old = legacy/'outputs/kuairand/17'
        old.mkdir(parents=True)
        shutil.move(str(owner/'source'),str(old/'source'))
        shutil.move(str(owner/'feature-cache'),str(old/'feature-cache'))
        config = dict(seeds=[17],kuairand=dict(base_epochs=30,ssl_epochs=30,patience=5,
            learning_rate=spec['lr'],batch_size=spec['batch'],embedding_size=spec['embed'],
            rdim=spec['rdim'],hidden=spec['hidden'],tower=spec['tower'],env_coe=spec['env_coe']))
        core.dump(legacy/'config.json',config)
        layout = {
            'core.py': ('paper_complete_20261026/core.py',['build_base','make_u','batches','seed_all']),
            'training.py': ('paper_continual_20261030/training.py',['_prepare','_feature_cache']),
            'backbone.py': ('MPT-Rec-exp-universal-representation/multitaskrec/model.py',['MLP','EmbeddingNetwork','MPTRec','NewTask']),
            'expert.py': ('MPT-Rec-exp-universal-representation/universal_experiment/model.py',['UniversalExpert']),
            'base_training.py': ('MPT-Rec-exp-universal-representation/universal_experiment/paired_run.py',['PairedTrainManager']),
        }
        locked,semantics = {},{}
        for name,(old_name,nodes) in layout.items():
            src,dest = Path(core.__file__).with_name(name), self.root/old_name
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(src,dest)
            locked[old_name] = core.file_hash(dest)
            semantics[name] = dict(original_sha256=core.file_hash(dest),current_sha256=core.file_hash(src),nodes=nodes)
        locked['paper_continual_20261030/config.json'] = core.file_hash(legacy/'config.json')
        lock = dict(files=locked)
        core.dump(legacy/'source-lock.json',lock)
        manifest = dict(preproc_hash='synthetic-preproc',features=dict(names=['dense','cat']),splits={})
        arrays = {}
        data = self.root/'preprocessed'
        data.mkdir()
        for split,rows in spec['sets'].items():
            info,arrays[split] = dict(rows=len(rows)),{}
            for kind,array in [('x',rows.x),('y',rows.y),('rowid',rows.row_ids)]:
                path=data/f'{split}-{kind}.npy';np.save(path,array)
                info[kind] = path.name
                arrays[split][kind] = core.file_hash(path)
            info['sha256']=arrays[split]
            manifest['splits'][split]=info
        core.dump(data/'manifest.json',manifest)
        spec['preprocessing_manifest']=str(data/'manifest.json')
        identity = dict(seed=17,source=tri_sources._object_hash(lock),config=tri_sources._object_hash(config),data='synthetic-preproc')
        core.dump(old/'identity.json',identity)
        receipt=json.loads((owner/'reuse-receipt.json').read_text())
        receipt['source_root']=str(old)
        for key,stage,required in [('base','source/base',['base.pt','metrics.json']),('u','source/ssl',['source.pt','metrics.json']),('cache','feature-cache',['identity.json'])]:
            receipt['stages'][key]=tri_sources._verify_stage(old/stage,required)
        receipt['historical_verification']=dict(source_lock_sha256=core.file_hash(legacy/'source-lock.json'),
            identity=identity,semantic_equivalence=semantics,manifest_sha256=core.file_hash(data/'manifest.json'),
            preprocessing_arrays=arrays,cache_root=str(old/'feature-cache'))
        core.dump(owner/'reuse-receipt.json',receipt)
        return spec,owner,old

    def test_historical_receipt_resolves_old_sources_and_rechecks_recorded_semantics(self):
        spec,owner,old = self.historical_fixture()
        result=self.load(spec,owner)
        self.assertEqual(result[-1]['source_folder'],str(owner.resolve()))
        self.assertEqual(Path(result[-1]['stages']['base']['path']),old/'source/base')
        self.assertFalse((owner/'source').exists())
        self.assertIn('historical_verification',result[-1])
        del result
        gc.collect()
        receipt=json.loads((owner/'reuse-receipt.json').read_text())
        receipt['historical_verification']['semantic_equivalence']['expert.py']['current_sha256']='f'*64
        core.dump(owner/'reuse-receipt.json',receipt)
        with self.assertRaisesRegex(ValueError,'semantic'):
            self.load(spec,owner)


if __name__ == '__main__':
    unittest.main()
