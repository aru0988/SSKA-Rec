"""MPT backbone and shared SSKA source for the two-task Table 2 protocol."""

import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import core, training
from .kuairand_l2_sources import DiagnosticTrainManager, _check_cache, _snapshot
from .scheme_common import bind_contract, digest, read_json, verify_stage


VERSION = 'table2-two-task-v2-20261121'
EPOCHS = 30
PATIENCE = 5


def normalized_spec(spec, mode):
    if mode not in ('sum', 'mean') or len(spec['initial']) != 2 or list(spec['tasks']) != list(spec['initial']):
        raise ValueError('Expected the registered two-task backbone protocol')
    return dict(spec, table2_protocol=VERSION, embedding_l2=.006,
                embedding_l2_normalization='parameter_mean' if mode == 'mean' else 'sum')


def configuration(spec, seed, mode):
    normalized_spec(spec, mode)
    count = sum(spec['vocab'].values()) * spec['embed']
    return dict(version=VERSION, dataset=spec['dataset'], seed=seed, mode=mode,
                dataset_sha256=digest(spec['data_signature']), tasks=list(spec['tasks']),
                input_dim=spec['xdim'], embedding_elements=count,
                embedding_l2=.006, coefficient=.006 / count if mode == 'mean' else .006,
                dnn_l2=3e-5, epochs=EPOCHS, patience=PATIENCE,
                learning_rate=spec['lr'], batch_size=spec['batch'],
                code={name: core.file_hash(Path(__file__).with_name(name)) for name in
                      ('core.py', 'backbone.py', 'base_training.py', 'training.py',
                       'kuairand_l2_sources.py', 'table2_v2_backbone.py')})


class LiveValidationManager(DiagnosticTrainManager):
    def evaluation_two_task(self, loader):
        scores = super().evaluation_two_task(loader)
        if loader is self.val_loader:
            row = self.epoch_records[-1]
            if not all(math.isfinite(v) for v in row.values() if isinstance(v, (float, int))):
                raise ValueError('Non-finite MPT epoch diagnostic')
            fused_bce = row['train_fused_bce_0'] + row['train_fused_bce_1']
            row['end_epoch_embedding_l2_over_fused_bce'] = row['end_epoch_embedding_l2'] / max(fused_bce, 1e-15)
            core.dump(self.live_path, dict(dataset=self.dataset, mode=self.mode, history=self.epoch_records,
                                           updated_at=training._utc()))
        return scores


def read_backbone(spec, seed, folder, mode):
    base_path = Path(folder) / 'source/base'
    verify_stage(base_path, ['base.pt', 'metrics.json', 'protocol.json'])
    expected = configuration(spec, seed, mode)
    if read_json(base_path/'protocol.json') != expected:
        raise ValueError('MPT backbone protocol identity differs')
    core.seed_all(seed)
    base = core.build_base(normalized_spec(spec, mode))
    base.load_state_dict(torch.load(base_path/'base.pt', map_location=core.DEVICE, weights_only=True))
    metrics = read_json(base_path/'metrics.json')
    if metrics['model_hash'] != core.model_hash(base):
        raise ValueError('MPT backbone weight identity differs')
    return base.eval().requires_grad_(False)


def fit_backbone(spec, seed, root_folder, mode, status_path):
    """Fit from the training split and select on validation, without evaluating test."""
    folder = Path(root_folder)
    base_path = folder/'source/base'
    if (base_path/'COMMITTED.json').is_file():
        return read_backbone(spec, seed, folder, mode)
    contract = configuration(spec, seed, mode)
    bind_contract(folder/'backbone-contract.json', contract)
    training._committed(core, base_path)
    core.seed_all(seed)
    base = core.build_base(normalized_spec(spec, mode))
    initial_hash = core.model_hash(base)
    count = sum(p.numel() for p in base.embedding_network.parameters())
    if count != contract['embedding_elements'] or base.reg_embedding != contract['coefficient']:
        raise ValueError('MPT effective embedding regularization differs from the protocol')
    loaders = {key: DataLoader(spec['sets'][key], batch_size=spec['batch'], shuffle=False)
               for key in ('train', 'val')}
    env = torch.randint(0, 2, (len(spec['sets']['train']),),
                        generator=torch.Generator().manual_seed(20260929))
    manager = LiveValidationManager(base, loaders['train'], loaders['val'], env, spec['initial'],
                                    spec['lr'], spec['batch'], .9, spec['env_coe'],
                                    epochs=EPOCHS, patience=PATIENCE)
    manager.live_path = Path(status_path)
    manager.dataset, manager.mode = spec['dataset'], mode
    torch_peak_available = core.DEVICE.type == 'cuda'
    if torch_peak_available:
        torch.cuda.reset_peak_memory_stats(core.DEVICE)
    started_at, start = training._utc(), time.perf_counter()
    manager.train_two_task()
    history = manager.epoch_records
    scores = [row['auc_val_0'] + row['auc_val_1'] for row in history]
    selected = history[max(range(len(history)), key=scores.__getitem__)]
    base.load_state_dict(manager.best_weight)
    training._save(base.state_dict(), base_path/'base.pt')
    core.dump(base_path/'protocol.json', contract)
    core.dump(base_path/'metrics.json', dict(history=history, selected_epoch=selected['epoch'],
              epochs=len(history), initial_state_hash=initial_hash, model_hash=core.model_hash(base),
              problem=contract, seconds=time.perf_counter()-start,
              peak_cuda_bytes=torch.cuda.max_memory_allocated(core.DEVICE) if torch_peak_available else 0,
              val={t: {'auc': selected[f'auc_val_{i}']} for i, t in enumerate(spec['initial'])},
              stop_reason='patience' if core.early_stop(scores, PATIENCE) else 'budget',
              started_at=started_at, ended_at=training._utc()))
    training._commit(core, base_path)
    return base.eval().requires_grad_(False)


def prepare_source(spec, seed, root_folder, status_path, mode='mean'):
    root = Path(root_folder)
    read_backbone(spec, seed, root, mode)
    if (root/'source-ready.json').is_file():
        return load_source(spec, seed, root, mode)
    base, _, _, u = training._prepare(core, normalized_spec(spec, mode), seed, root, EPOCHS)
    ready = dict(version=VERSION, seed=seed, data_sha256=digest(spec['data_signature']),
                 base_hash=core.model_hash(base), u_hash=core.model_hash(u),
                 base_protocol_sha256=core.file_hash(root/'source/base/protocol.json'),
                 ssl_manifest_sha256=core.file_hash(root/'source/ssl/COMMITTED.json'),
                 cache=_snapshot(root/'feature-cache'))
    core.dump(root/'source-ready.json', ready)
    return load_source(spec, seed, root, mode)


def load_source(spec, seed, root_folder, mode='mean'):
    folder = Path(root_folder)
    receipt = read_json(folder/'source-ready.json')
    if (receipt['version'], receipt['seed'], receipt['data_sha256']) != (VERSION, seed, digest(spec['data_signature'])):
        raise ValueError('SSKA source receipt identity differs')
    base = read_backbone(spec, seed, folder, mode)
    verify_stage(folder/'source/ssl', ['source.pt', 'metrics.json'])
    if (core.file_hash(folder/'source/base/protocol.json') != receipt['base_protocol_sha256']
            or core.file_hash(folder/'source/ssl/COMMITTED.json') != receipt['ssl_manifest_sha256']):
        raise ValueError('SSKA committed source identity differs')
    _check_cache(folder, receipt['cache'])
    u = core.make_u(spec, seed)
    u.load_state_dict(torch.load(folder/'source/ssl/source.pt', map_location=core.DEVICE, weights_only=True))
    if (core.model_hash(base), core.model_hash(u)) != (receipt['base_hash'], receipt['u_hash']):
        raise ValueError('SSKA loaded B/U hash differs')
    u.eval().requires_grad_(False)
    cache_identity = read_json(folder/'feature-cache/identity.json')
    if cache_identity['base_hash'] != receipt['base_hash']:
        raise ValueError('Representation cache belongs to another backbone')
    caches = {}
    from .tri_data import _array_hash
    for split, rows in spec['sets'].items():
        if cache_identity['splits'][split] != dict(rows=len(rows), rowid_sha256=_array_hash(rows.row_ids)):
            raise ValueError('Representation cache belongs to different rows')
        caches[split] = {k: torch.from_numpy(np.load(folder/'feature-cache'/f'{split}-{k}.npy', mmap_mode='c'))
                         for k in ('x', 'g', 's0', 's1', 'old')}
        caches[split]['y'] = torch.from_numpy(rows.y)
    spec['scheme_data_identity'] = receipt['data_sha256']
    spec['scheme_cache_identity'] = dict(version=VERSION, seed=seed, base_hash=receipt['base_hash'],
                                         cache_sha256=receipt['cache']['manifest_sha256'])
    envs = [base.env_embedding_network(torch.tensor(i, device=core.DEVICE)).detach() for i in range(2)]
    return base, caches, envs, u


def base_evaluation(spec, seed, root_folder, mode='mean'):
    root = Path(root_folder)
    path = root/'source-evaluation'
    source = read_json(root/'source-ready.json')
    if (path/'COMMITTED.json').is_file():
        verify_stage(path, ['metrics.json', 'predictions.npz'])
        result = read_json(path/'metrics.json')
        if (result['model_hash'], result['paired_seed']) != (source['base_hash'], seed):
            raise ValueError('MPT test receipt identity differs')
        return result
    _check_cache(root, source['cache'])
    training._committed(core, path)
    result = dict(paired_seed=seed, model_hash=source['base_hash'], variant='mpt',
                  protocol=configuration(spec, seed, mode))
    raw = {}
    for split in ('val', 'test'):
        rows = spec['sets'][split]
        p = np.load(root/'feature-cache'/f'{split}-old.npy', mmap_mode='r')
        y = rows.y
        result[split] = {t: core.measures(y[:, i], p[:, i]) for i, t in enumerate(spec['initial'])}
        raw.update({split+'_y': y, split+'_p': p, split+'_rowid': rows.row_ids})
    with (path/'predictions.tmp').open('wb') as stream:
        np.savez_compressed(stream, **raw)
    (path/'predictions.tmp').replace(path/'predictions.npz')
    core.dump(path/'metrics.json', result)
    training._commit(core, path)
    return result
