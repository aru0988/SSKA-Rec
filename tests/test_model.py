import copy
import unittest
import torch
from sska_rec.expert import UniversalExpert
from sska_rec.core import Selector


class ModelTests(unittest.TestCase):
    def test_task_copy_preserves_rng_and_isolation(self):
        u = UniversalExpert([1, 2], rep_dim=4, hidden=8)
        v = copy.deepcopy(u)
        self.assertTrue(torch.equal(u.mask_rng.get_state(), v.mask_rng.get_state()))
        with torch.no_grad():
            next(v.parameters()).add_(1)
        self.assertFalse(torch.equal(next(u.parameters()), next(v.parameters())))

    def test_zero_delta_preserves_predictions(self):
        gate = Selector(16)
        p = torch.tensor([0.1, 0.3, 0.8])
        out = gate(p, torch.randn(3, 16))
        self.assertTrue(torch.equal(out, p))
        out.sum().backward()
        self.assertGreater(gate.delta[-1].weight.grad.abs().sum().item(), 0)
