"""Joint initial-task baselines using the admitted table2 data and training budget."""
import copy
import json
import os
import time
from pathlib import Path

from . import core
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .training import _commit, _committed, _save, _utc
from .tri_sources import _split_identity


class InitialTasks(Dataset):
    def __init__(self, rows, indices):
        self.rows, self.indices = rows, indices

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index][-1], np.asarray(self.rows.y[index, self.indices], dtype='float32')


def selected_epoch(history):
    return max(history, key=lambda row: row['val_mean_auc'])['epoch']


def code_hashes():
    root = Path(__file__).parent
    names = ['baseline_fit.py', 'baseline_models.py', 'backbone.py', 'core.py',
             'training.py', 'tri_data.py', 'vocabulary.py', 'aliccp_data.py',
             'table2_baselines.py', 'tri_sources.py', 'locking.py', 'scheme_common.py']
    paths = [root/name for name in names]
    paths += list((root/'vendor').rglob('*'))
    return {str(p.relative_to(root)): core.file_hash(p) for p in sorted(paths)
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def protocol(spec, seed, variant, cfg):
    if set(cfg) & {'reg_embedding', 'reg_dnn'}:
        raise ValueError('Use registered embedding_l2/dnn_l2 configuration names')
    for name, expected in [('epochs', 30), ('patience', 5), ('embedding_l2', .006),
                           ('dnn_l2', 3e-5)]:
        if cfg.get(name, expected) != expected:
            raise ValueError(f'Joint baseline protocol requires {name}={expected}')
    return dict(version='table2-joint-baselines-v1', dataset=spec['dataset'],
                variant=variant, paired_seed=seed, model_seed=seed,
                tasks=list(spec['initial']), hidden=list(spec['hidden']),
                tower=list(spec['tower']), vocab=spec['vocab'], embed=spec['embed'],
                xdim=spec['xdim'], learning_rate=spec['lr'], batch_size=spec['batch'],
                epochs=30, patience=5, embedding_l2=.006, dnn_l2=3e-5,
                selection='mean-initial-task-val-auc-earliest-best',
                loss='sum-initial-task-bce-plus-l2', shuffle=False, precision='float32',
                optimizer='Adam', code=code_hashes(),
                splits={s: _split_identity(spec, s, rows) for s, rows in spec['sets'].items()})


def _check_probabilities(value):
    if not torch.isfinite(value).all() or torch.any((value < 0) | (value > 1)):
        raise ValueError('Invalid joint baseline probabilities')


def _metrics(labels, probabilities, tasks):
    return {task: core.measures(labels[:, i], probabilities[:, i]) for i, task in enumerate(tasks)}


def verify_result(folder, spec, expected, model):
    folder = Path(folder)
    result = json.loads((folder/'metrics.json').read_text())
    if result.get('protocol') != expected or not result.get('ready_for_commit'):
        raise ValueError('Completed baseline protocol mismatch')
    model.load_state_dict(torch.load(folder/'model.pt', map_location=core.DEVICE, weights_only=True))
    if core.model_hash(model) != result['model_hash']:
        raise ValueError('Completed baseline weight hash mismatch')
    indices = [spec['tasks'].index(t) for t in spec['initial']]
    with np.load(folder/'predictions.npz', allow_pickle=False) as raw:
        for split in ('val', 'test'):
            rows = spec['sets'][split]
            y, p = raw[split+'_y'], raw[split+'_p']
            if (p.shape != (len(rows), len(indices)) or y.shape != p.shape
                    or not np.array_equal(y, rows.y[:, indices])
                    or not np.array_equal(raw[split+'_rowid'], rows.row_ids)):
                raise ValueError('Completed baseline label/row-id alignment mismatch')
            _check_probabilities(torch.from_numpy(p))
            measured = _metrics(y, p, spec['initial'])
            for task in measured:
                if any(abs(measured[task][k]-result[split][task][k]) > 1e-10 for k in measured[task]):
                    raise ValueError('Completed baseline raw metrics mismatch')
    return result


def fit_baseline(spec, paired_seed, variant, folder, cfg, status_path):
    from .baseline_models import build_baseline
    folder, status_path = Path(folder), Path(status_path)
    contract = protocol(spec, paired_seed, variant, cfg)
    model = build_baseline(spec, paired_seed, variant, cfg)
    if (folder/'COMMITTED.json').exists():
        _committed(core, folder)
        return verify_result(folder, spec, contract, model)
    if (folder/'metrics.json').exists():
        pending = json.loads((folder/'metrics.json').read_text())
        if pending.get('ready_for_commit'):
            result = verify_result(folder, spec, contract, model)
            _commit(core, folder)
            return result
    _committed(core, folder)
    indices = [spec['tasks'].index(t) for t in spec['initial']]
    gpu = core.DEVICE.type == 'cuda'
    loader_options = dict(batch_size=spec['batch'], shuffle=False,
                          num_workers=2 if gpu else 0, pin_memory=gpu)
    loaders = {s: DataLoader(InitialTasks(rows, indices), **loader_options)
               for s, rows in spec['sets'].items()}

    def predict(split):
        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for x, y in loaders[split]:
                x = {k: v.to(core.DEVICE, non_blocking=gpu) for k, v in x.items()}
                p = torch.stack([v.reshape(-1) for v in model(x)], dim=1)
                _check_probabilities(p)
                probs.append(p.cpu())
                labels.append(y)
        return torch.cat(labels).numpy(), torch.cat(probs).numpy()

    optimizer = torch.optim.Adam(model.parameters(), lr=spec['lr'])
    history, best, stale, state = [], -float('inf'), 0, None
    started_at, started = _utc(), time.perf_counter()
    if gpu:
        torch.cuda.reset_peak_memory_stats(core.DEVICE)
    for epoch in range(1, 31):
        model.train()
        loss_sum, rows_seen = 0., 0
        epoch_clock = time.perf_counter()
        for x, y in loaders['train']:
            x = {k: v.to(core.DEVICE, non_blocking=gpu) for k, v in x.items()}
            y = y.to(core.DEVICE, non_blocking=gpu)
            p = torch.stack([v.reshape(-1) for v in model(x)], dim=1)
            _check_probabilities(p)
            loss = sum(F.binary_cross_entropy(p[:, i], y[:, i]) for i in range(len(indices))) + model.get_l2_reg()
            if not torch.isfinite(loss):
                raise ValueError('Non-finite joint baseline loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            rows_seen += len(y)
        if rows_seen != len(spec['sets']['train']):
            raise ValueError('Joint baseline train row count mismatch')
        train_seconds = time.perf_counter()-epoch_clock
        y, p = predict('val')
        val = _metrics(y, p, spec['initial'])
        score = float(np.mean([m['auc'] for m in val.values()]))
        row = dict(epoch=epoch, train_loss=loss_sum/rows_seen, rows_seen=rows_seen,
                   val=val, val_mean_auc=score, train_seconds=train_seconds,
                   seconds=time.perf_counter()-epoch_clock, at=_utc())
        history.append(row)
        core.dump(folder/'history-live.json', history)
        core.dump(status_path, dict(state='running', pid=os.getpid(), dataset=spec['dataset'],
                                   seed=paired_seed, variant=variant, validation=row, updated_at=_utc()))
        print(json.dumps(dict(event='epoch', dataset=spec['dataset'], seed=paired_seed,
                              variant=variant, **row)), flush=True)
        if score > best:
            best, stale, state = score, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= 5:
            break
    model.load_state_dict(state)
    raw, metrics = {}, {}
    for split in ('val', 'test'):
        y, p = predict(split)
        metrics[split] = _metrics(y, p, spec['initial'])
        raw.update({split+'_y': y, split+'_p': p, split+'_rowid': spec['sets'][split].row_ids})
    _save(state, folder/'model.pt')
    with (folder/'predictions.npz.tmp').open('wb') as stream:
        np.savez_compressed(stream, **raw)
    os.replace(folder/'predictions.npz.tmp', folder/'predictions.npz')
    result = dict(variant=variant, dataset=spec['dataset'], paired_seed=paired_seed,
                  model_seed=paired_seed, tasks=spec['initial'], protocol=contract,
                  history=history, selected_epoch=selected_epoch(history), epochs=len(history),
                  stop_reason='patience' if stale >= 5 else 'budget',
                  model_hash=core.model_hash(model), total_parameters=sum(p.numel() for p in model.parameters()),
                  started_at=started_at, ended_at=_utc(), seconds=time.perf_counter()-started,
                  peak_cuda_bytes=torch.cuda.max_memory_allocated(core.DEVICE) if gpu else 0,
                  execution=dict(threads=torch.get_num_threads(), num_workers=loader_options['num_workers']),
                  ready_for_commit=True, **metrics)
    core.dump(folder/'metrics.json', result)
    verify_result(folder, spec, contract, model)
    _commit(core, folder)
    return result
