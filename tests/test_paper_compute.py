"""Actual raw-input profiling contracts, without datasets or optimization."""

import copy
import importlib
import json
import unittest

import torch

from sska_rec.backbone import MPTRec
from sska_rec.expert import UniversalExpert
from sska_rec.stage2_models import build_model


class PaperComputeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, rdim=8):
        spec = dict(xdim=5, rdim=rdim, tower=[4, 2], tasks=['a', 'b', 'c'], dataset='tiny')
        base = MPTRec(2, {'category': 7}, 2, 5, [12, rdim], [4, 2])
        universal = UniversalExpert([1, 1, 1, 2], rep_dim=rdim, hidden=12)
        raw = {'continuous0': torch.arange(3).float(), 'continuous1': torch.ones(3),
               'continuous2': torch.zeros(3), 'category': torch.tensor([0, 1, 2])}
        envs = list(base.env_embedding_network.weight.detach().unbind())
        return spec, base, universal, envs, raw

    def test_profile_counts_real_backward_and_preserves_sources(self):
        module = importlib.import_module('sska_rec.paper_compute')
        spec, base, universal, envs, raw = self.fixture()
        base.train()
        base.shared_expert_network.eval()  # Mixed module modes must survive.
        first = next(base.parameters())
        first.grad = torch.ones_like(first)
        state = [copy.deepcopy(x.state_dict()) for x in (base, universal)]
        modes = [m.training for x in (base, universal) for m in x.modules()]
        flags = [p.requires_grad for x in (base, universal) for p in x.parameters()]
        rng = torch.random.get_rng_state().clone()
        report = module.profile(spec, base, universal, envs, raw, 17)
        json.dumps(report)
        self.assertEqual(set(report['variants']), {'mpt', 'tri_frozen', 'tri_lora'})
        for variant, row in report['variants'].items():
            self.assertEqual(row['batch_size'], 3)
            self.assertGreater(row['inference_forward_flops'], 0)
            self.assertGreater(row['training_backward_flops'], 0)
            self.assertEqual(row['training_total_flops'], row['training_forward_flops'] + row['training_backward_flops'])
            self.assertTrue(row['coverage']['backward']['counted_operators'])
            self.assertLessEqual(set(row['coverage']['forward']['counted_operators']),
                                 {'aten::mm', 'aten::addmm', 'aten::bmm', 'aten::baddbmm', 'aten::mv', 'aten::addmv'})
            self.assertIn('aten::embedding', row['coverage']['inference']['uncounted_operators'])
            self.assertEqual(row['training_total_macs'], row['training_total_flops'] / 2)
            model = build_model(spec, universal, 17, variant)
            base_count = sum(p.numel() for n, p in base.named_parameters() if not n.startswith('env_classifier.'))
            stage_count = sum(p.numel() for n, p in model.named_parameters() if not n.startswith('u.decoder.'))
            self.assertEqual(row['deployment_parameters'], base_count + stage_count)
            unused_old = sum(p.numel() for n, p in base.named_parameters()
                             if n.startswith(('gate_networks.', 'tower_networks.')))
            self.assertEqual(row['active_new_task_inference_parameters'], base_count + stage_count - unused_old)
            self.assertEqual(row['trainable_parameters'], model.trainable_parameters)
        # All dense operations, independently calculated from these tiny widths:
        # B: 3*(5*12+12*8); new: 5*4+4*8+5*2+8*4+4*2+2;
        # routing / specific mix / gate fusion: 8*2 each. Batch=3, 2 FLOPs/MAC.
        self.assertEqual(report['variants']['mpt']['inference_forward_flops'], 3720)
        for source, old_state in zip((base, universal), state):
            for name, tensor in source.state_dict().items():
                self.assertTrue(torch.equal(tensor, old_state[name]), name)
        self.assertEqual(modes, [m.training for x in (base, universal) for m in x.modules()])
        self.assertEqual(flags, [p.requires_grad for x in (base, universal) for p in x.parameters()])
        self.assertTrue(torch.equal(first.grad, torch.ones_like(first)))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))

    def test_rank4_counts_agree_for_r128_and_r64_and_five_seeds(self):
        module = importlib.import_module('sska_rec.paper_compute')
        for xdim, rdim, widths, tower in [(119, 128, [1] * 11 + [4] * 27, [64, 32]), (80, 64, [5] * 16, [32, 32])]:
            spec = dict(xdim=xdim, rdim=rdim, tower=tower, tasks=['a', 'b', 'c'], dataset='tiny')
            source = UniversalExpert(widths, rep_dim=rdim)
            base = MPTRec(2, {}, 1, xdim, [2 * rdim, rdim], tower)
            raw = {f'f{i}': torch.zeros(2) for i in range(xdim)}
            envs = list(base.env_embedding_network.weight.detach().unbind())
            results = [module.profile(spec, base, source, envs, raw, seed)['variants'] for seed in range(5)]
            for result in results:
                self.assertEqual(result['tri_lora']['trainable_parameters'] - result['tri_frozen']['trainable_parameters'],
                                 4 * (xdim + len(widths) + 256 + 256 + rdim))
                self.assertEqual(result['tri_frozen']['trainable_parameters'] - result['mpt']['trainable_parameters'], xdim)
            self.assertEqual([row['trainable_parameters'] for row in results[0].values()],
                             [row['trainable_parameters'] for row in results[-1].values()])


if __name__ == '__main__':
    unittest.main()
