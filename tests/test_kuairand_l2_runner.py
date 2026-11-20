import importlib
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class KuaiRandQueueTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('sska_rec.kuairand_l2_runner'),
                             'Normalized-L2 scheduler is missing')
        self.runner = importlib.import_module('sska_rec.kuairand_l2_runner')
        # The scheduler graph depends only on these registered fields.
        self.cfg = {'datasets': {'kuairand': {'seeds': [11, 22, 33, 44, 55],
                    'initial': ['click', 'like'], 'targets': ['follow', 'forward']}}}

    def test_registered_graph_has_65_jobs_and_70_train_units(self):
        jobs = self.runner.jobs(self.cfg)
        self.assertEqual(len(jobs), 65)
        self.assertEqual(len({j.key for j in jobs}), 65)
        self.assertEqual(sum(j.units for j in jobs), 70)
        self.assertEqual(sum(j.stage == 'source' for j in jobs), 5)
        self.assertEqual(sum(j.stage == 'readout' for j in jobs), 20)
        self.assertEqual(sum(j.stage == 'stage2' for j in jobs), 40)
        self.assertFalse(any('linear' in j.variant or 'single' in j.variant for j in jobs))

    def test_source_completion_releases_only_its_seed_dependencies(self):
        jobs = self.runner.jobs(self.cfg)
        queue = self.runner.JobQueue(jobs)
        first = queue.take()
        self.assertEqual(first.stage, 'source')
        queue.finish(first, 0)
        ready = queue.ready()
        self.assertEqual(sum(j.stage != 'source' for j in ready), 4)
        self.assertTrue(all(j.seed == first.seed for j in ready if j.stage != 'source'))

    def test_source_failure_blocks_its_dependents_without_retry(self):
        queue = self.runner.JobQueue(self.runner.jobs(self.cfg))
        first = queue.take()
        queue.finish(first, 9)
        self.assertEqual(len(queue.blocked), 12)
        self.assertEqual(queue.failures[first.key]['exitcode'], 9)
        ready = queue.ready()
        self.assertTrue(all(j.seed != first.seed for j in ready))
        self.assertNotIn(first.key, queue.pending)

    def test_readout_failure_leaves_independent_same_seed_stage2_runnable(self):
        queue = self.runner.JobQueue(self.runner.jobs(self.cfg))
        source = queue.take()
        queue.finish(source, 0)
        readout = next(j for j in queue.ready() if j.stage == 'readout' and j.variant == 'head_only')
        queue.start(readout)
        queue.finish(readout, 1)
        self.assertEqual(len(queue.blocked), 0)
        for head in [j for j in queue.ready() if j.seed == source.seed and j.variant == 'conditional_readout']:
            queue.start(head)
            queue.finish(head, 0)
        self.assertEqual(sum(j.stage == 'stage2' and j.seed == source.seed for j in queue.ready()), 8)

    def test_conditional_readout_failure_blocks_stage2_but_head_only_continues(self):
        queue = self.runner.JobQueue(self.runner.jobs(self.cfg))
        source = queue.take()
        queue.finish(source, 0)
        head = next(j for j in queue.ready() if j.stage == 'readout' and j.variant == 'conditional_readout')
        queue.start(head)
        queue.finish(head, 1)
        self.assertEqual(len(queue.blocked), 8)
        self.assertEqual(sum(j.seed == source.seed and j.variant == 'head_only' for j in queue.ready()), 2)

    def test_output_paths_keep_readout_variants_and_new_tasks_distinct(self):
        for job in self.runner.jobs(self.cfg):
            folder = self.runner.leaf(Path('output'), job)
            expected = Path('output')/'kuairand'/str(job.seed)
            if job.stage == 'source':
                self.assertEqual(folder, expected/'source')
            else:
                self.assertEqual(folder, expected/('old' if job.stage == 'readout' else 'new')/job.target/job.variant)

    def test_identity_binds_reference_content_and_rejects_changed_configuration(self):
        from sska_rec import core
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs = root/'references.json'
            refs.write_text('{}')
            first = self.runner.execution_identity(self.cfg, root/'data', root/'cache', refs)
            refs.write_text('{"changed": true}')
            second = self.runner.execution_identity(self.cfg, root/'data', root/'cache', refs)
            self.assertNotEqual(first['references_sha256'], second['references_sha256'])
            from sska_rec.scheme_common import bind_contract
            bind_contract(root/'identity.json', first)
            with self.assertRaisesRegex(ValueError, 'contract changed'):
                bind_contract(root/'identity.json', second)

    def test_import_does_not_probe_cuda_before_fork(self):
        code = "import torch; torch.cuda.is_available=lambda: (_ for _ in ()).throw(AssertionError('parent CUDA probe')); import sska_rec.kuairand_l2_runner"
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_live_worker_job_cannot_be_started_twice(self):
        queue = self.runner.JobQueue(self.runner.jobs(self.cfg))
        first = queue.take()
        with self.assertRaises(ValueError):
            queue.start(first)


if __name__ == '__main__':
    unittest.main()

class KuaiRandWorkerIntegrationTests(unittest.TestCase):
    def test_real_tensor_heads_and_stage2_reuse_keep_paired_seeds_and_frozen_sources(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import torch
        import torch._dynamo  # Keep lazy torch imports outside patch.dict(sys.modules).
        from sska_rec import core
        from sska_rec import kuairand_l2_runner as runner
        from tests.test_scheme_reuse import SchemeReuseTests
        torch.set_num_threads(1)
        fixture = SchemeReuseTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        spec = fixture.spec()
        spec['data_signature'] = {'fixture': 'tiny-normalized-kuairand'}
        spec['scheme_data_identity'] = 'tiny-admitted-data'
        base = core.build_base(spec).eval().requires_grad_(False)
        u = core.make_u(spec, 17).eval().requires_grad_(False)
        with torch.no_grad():
            next(u.encoder.parameters()).add_(.1)
        caches = {}
        with torch.no_grad():
            for split, rows in spec['sets'].items():
                features = {name: torch.tensor(rows.x[:, i]) for i, name in enumerate(rows.names)}
                x, g, ss, _ = base.get_infos(features)
                caches[split] = dict(x=x, g=g, s0=ss[0], s1=ss[1],
                                     old=torch.stack(base.predict(features), 1), y=torch.from_numpy(rows.y))
            envs = [base.env_embedding_network(torch.tensor(i)) for i in range(2)]
        source_hash = (core.model_hash(base), core.model_hash(u))

        def freeze_guard(got_base, got_caches, got_envs, got_u, got_spec, readout_root=None):
            self.assertIs(got_base, base)
            self.assertIs(got_caches, caches)
            self.assertIs(got_envs, envs)
            self.assertIs(got_u, u)
            def guard():
                self.assertEqual((core.model_hash(base), core.model_hash(u)), source_hash)
                self.assertFalse(any(p.requires_grad for m in (base, u) for p in m.parameters()))
                return {'passed': True}
            return guard

        dc = dict(epochs=1, patience=5, learning_rate=spec['lr'], batch_size=spec['batch'],
                  initial=spec['initial'], targets=spec['tasks'][2:], seeds=[17], head_seed_offset=4)
        source_module = SimpleNamespace(load_source=lambda *args: (base, caches, envs, u),
                                       prepare_source=lambda *args: self.fail('Completed source retrained'),
                                       freeze_guard=freeze_guard)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'sska_rec.kuairand_l2_sources': source_module}):
            output = Path(tmp)
            root = output/'kuairand'/'17'
            core.dump(root/'source-ready.json', {'ready': True})
            for stage in ('base', 'ssl'):
                core.dump(root/'source'/stage/'COMMITTED.json', {})
            runner.run_job(spec, runner.Job('source', 17), output, dc, output/'worker.json')
            all_jobs = [runner.Job('readout', 17, t, v) for t in spec['initial'] for v in runner.READOUTS]
            all_jobs += [runner.Job('stage2', 17, t, v) for t in dc['targets'] for v in runner.STAGE2]
            for job in all_jobs:
                runner.run_job(spec, job, output, dc, output/'worker.json')
                folder = runner.leaf(output, job)
                metrics = runner.read_json(folder/'metrics.json')
                self.assertEqual(metrics['paired_seed'], 17)
                if job.stage == 'stage2':
                    self.assertEqual(metrics['base_hash'], source_hash[0])
                    self.assertEqual(metrics['u_hash'], source_hash[1])
                    contract_path = runner._job_paths(output, job)[2]
                    self.assertEqual(metrics['job_contract_sha256'], core.file_hash(contract_path))
                    self.assertEqual(runner.read_json(folder/'contract.json'), runner.read_json(contract_path))
                    self.assertEqual(metrics['seed'], 17+dc['targets'].index(job.target)+4)
                    self.assertEqual(metrics['variant'], job.variant)
                    if job.variant in ('random_lora', 'tri_lora'):
                        self.assertGreater(metrics['u_update_parameters'], 0)
                    else:
                        self.assertEqual(metrics['u_update_parameters'], 0)
                if job.variant == 'head_only':
                    self.assertEqual(metrics['variant'], 'h_conditional')
                stamp = (folder/'model.pt').stat().st_mtime_ns
                committed_stamp = (folder/'COMMITTED.json').stat().st_mtime_ns
                runner.run_job(spec, job, output, dc, output/'worker.json')
                self.assertEqual(stamp, (folder/'model.pt').stat().st_mtime_ns)
                self.assertEqual(committed_stamp, (folder/'COMMITTED.json').stat().st_mtime_ns)
            # Recover metrics written just before COMMITTED, without optimizing again.
            (folder/'COMMITTED.json').unlink()
            runner.run_job(spec, job, output, dc, output/'worker.json')
            self.assertEqual(stamp, (folder/'model.pt').stat().st_mtime_ns)
            from sska_rec.training import _commit
            valid_metrics = runner.read_json(folder/'metrics.json')
            wrong_source = dict(valid_metrics, base_hash='foreign-source')
            core.dump(folder/'metrics.json', wrong_source)
            _commit(core, folder)
            with self.assertRaisesRegex(ValueError, 'provenance'):
                runner.run_job(spec, job, output, dc, output/'worker.json')
            (folder/'COMMITTED.json').unlink()
            with self.assertRaisesRegex(ValueError, 'provenance'):
                runner.run_job(spec, job, output, dc, output/'worker.json')
            core.dump(folder/'metrics.json', valid_metrics)
            _commit(core, folder)
            (folder/'predictions.npz').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'integrity'):
                runner.run_job(spec, job, output, dc, output/'worker.json')


class KuaiRandHealthTests(unittest.TestCase):
    def test_health_keeps_progress_when_process_exits_during_resource_probe(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        class ProcessGone(Exception):
            pass
        def gone(pid):
            raise ProcessGone(pid)
        psutil_stub = SimpleNamespace(Process=gone, Error=ProcessGone)
        from sska_rec import core
        from sska_rec import kuairand_l2_runner as runner
        with tempfile.TemporaryDirectory() as tmp:
            job = runner.Job('source', 7)
            log, status, _ = runner._job_paths(tmp, job)
            log.parent.mkdir(parents=True)
            log.write_text('epoch 2')
            core.dump(status, {'state': 'running', 'stage': 'ssl'})
            with patch.dict(sys.modules, {'psutil': psutil_stub}):
                health = runner._health(SimpleNamespace(pid=12345), job, tmp)
            self.assertEqual(health['progress']['stage'], 'ssl')
            self.assertEqual(health['log_bytes'], 7)


class KuaiRandResourceRecordTests(unittest.TestCase):
    def test_parallel_profile_preserves_observed_peak_after_workers_exit(self):
        from sska_rec import kuairand_l2_runner as runner
        with tempfile.TemporaryDirectory() as tmp:
            profile = runner.record_health(tmp, [{'pid': 1, 'rss_bytes': 100},
                        {'pid': 2, 'rss_bytes': 200}], 4, 2)
            self.assertEqual(profile['observed_max_workers'], 2)
            self.assertEqual(profile['observed_max_worker_rss_bytes'], 300)
            profile = runner.record_health(tmp, [], 4, 2)
            self.assertEqual(profile['observed_max_workers'], 2)
            self.assertEqual(profile['observed_max_worker_rss_bytes'], 300)
            self.assertEqual(profile['threads_per_worker'], 2)
            self.assertEqual(len((Path(tmp)/'health-samples.jsonl').read_text().splitlines()), 2)

if __name__ == '__main__':
    unittest.main()


