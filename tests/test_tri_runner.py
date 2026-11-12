import json
from pathlib import Path

import numpy as np
import unittest
import torch


class TriRunnerTests(unittest.TestCase):
 def test_all_update_modes_support_aliccp_dimensions_without_mutating_source(self):
    from sska_rec.tri_runner import model_for
    from sska_rec.expert import UniversalExpert
    spec = dict(xdim=80, rdim=64, tower=[32, 32], widths=[5]*16)
    source = UniversalExpert(spec['widths'], rep_dim=64).requires_grad_(False)
    before = {k:v.clone() for k,v in source.state_dict().items()}
    models = [model_for(spec,source,10,v) for v in ['tri_frozen','tri_lora','tri_ssf','tri_full']]
    batch = {k:torch.randn(7,w) for k,w in [('x',80),('g',64),('s0',64),('s1',64)]}
    envs = [torch.randn(64),torch.randn(64)]
    outputs = [m(batch,envs) for m in models]
    for output in outputs[1:]:
        assert torch.equal(outputs[0],output)
    for m in models[1:]:
        for k,v in models[0].core.state_dict().items():
            assert torch.equal(v,m.core.state_dict()[k])
    for k,v in source.state_dict().items():
        assert torch.equal(v,before[k])
    assert all(not p.requires_grad for p in source.parameters())
    assert sum(p.numel() for p in models[2].u.parameters() if p.requires_grad)==640


 def test_report_rejects_tampered_raw_even_with_valid_manifest(self):
    import tempfile
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    tmp_path = Path(temporary.name)
    from sska_rec.tri_report import report
    from sska_rec import core
    from sska_rec.training import _commit
    cfg = dict(variants=['mpt','tri_frozen'],datasets={'census':dict(seeds=[1,2,3,4,5],targets=['education'])})
    folder=tmp_path/'census/1/education/mpt';folder.mkdir(parents=True)
    m=dict(paired_seed=1,target='education',variant='mpt',val={'auc':.8,'logloss':.2},test={'auc':.8,'logloss':.2})
    core.dump(folder/'metrics.json',m)
    np.savez(folder/'predictions.npz',val_y=[0,1],val_p=[.9,.1],val_rowid=[1,2],test_y=[0,1],test_p=[.9,.1],test_rowid=[3,4])
    _commit(core,folder)
    with self.assertRaisesRegex(ValueError,'metric'):
        report(tmp_path,cfg)

 def test_training_records_bce_regularization_and_throughput(self):
    from sska_rec.tri_training import train_one
    from sska_rec.tri_runner import model_for
    from sska_rec import core
    import tempfile
    temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
    root=Path(temporary.name);core.DEVICE=torch.device('cpu')
    spec=dict(xdim=6,rdim=8,tower=[4,3],batch=4,tasks=['old0','old1','new'],sets={})
    caches={}
    for split in ['train','val','test']:
        cache={k:torch.randn(8,w) for k,w in [('x',6),('g',8),('s0',8),('s1',8)]}
        cache['y']=torch.tensor([[0.,1.,0.],[1.,0.,1.]]*4)
        caches[split]=cache
        class Rows:pass
        rows=Rows();rows.row_ids=np.arange(8);spec['sets'][split]=rows
    model=model_for(spec,None,1,'mpt')
    cfg=dict(learning_rate=.001,epochs=2,patience=5)
    result=train_one(model,spec,caches,[torch.randn(8),torch.randn(8)],'new',root/'model',cfg,4,'mpt',lambda:{'passed':True},root/'status.json',paired_seed=1)
    for row in result['history']:
        assert abs(row['train_loss']-row['train_bce']-row['train_regularization'])<1e-6
        assert row['samples_per_second']>0
    assert (root/'model/COMMITTED.json').exists()
