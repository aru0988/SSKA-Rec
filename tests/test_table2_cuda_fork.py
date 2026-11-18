"""Explicit cloud integration check: no CUDA calls in parent, GPU work only after fork."""
import multiprocessing as mp
import os
import unittest

from sska_rec import table2_baselines as runner
from sska_rec import core
import torch


def cuda_child(queue):
    from sska_rec.baseline_models import build_baseline
    torch.set_num_threads(2)
    core.DEVICE = torch.device('cuda:0')
    spec = dict(dataset='census', vocab={'cat':3}, embed=2, xdim=3,
                hidden=[8,4], tower=[4])
    result = []
    for variant in runner.VARIANTS:
        model = build_baseline(spec, 41, variant)
        x = dict(dense=torch.tensor([.1,.2,.3],device='cuda:0'),
                 cat=torch.tensor([0,1,2],device='cuda:0'))
        p = model(x)
        loss = sum(v.mean() for v in p) + model.get_l2_reg()
        loss.backward()
        torch.cuda.synchronize()
        assert all(torch.isfinite(v).all() for v in p)
        result.append(variant)
    queue.put(result)


@unittest.skipUnless(os.environ.get('RUN_CUDA_FORK_TEST') == '1', 'Explicit cloud GPU check')
class CudaForkTests(unittest.TestCase):
    def test_four_models_train_after_linux_fork(self):
        context = mp.get_context('fork')
        queue = context.Queue()
        process = context.Process(target=cuda_child, args=(queue,))
        process.start()
        process.join(90)
        if process.is_alive():
            process.terminate()
            process.join()
            self.fail('CUDA fork child did not finish')
        self.assertEqual(process.exitcode,0)
        self.assertEqual(queue.get(timeout=5), list(runner.VARIANTS))


if __name__ == '__main__':
    unittest.main()
