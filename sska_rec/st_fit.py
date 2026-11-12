"""Independent full-network SingleTask baselines with normalized embedding L2."""

import copy
import json
import os
import time
from pathlib import Path

# Keep the established native-library import order used by this workspace.
from . import core
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .backbone import SingleTask
from .training import _Target, _commit, _committed, _save, _utc


def _settings(spec, cfg):
    values = dict(epochs=30, patience=5, embedding_l2=.006, dnn_l2=3e-5)
    for key, expected in values.items():
        if cfg.get(key, expected) != expected:
            raise ValueError(f'SingleTask protocol requires {key}={expected}')
    for key, expected in [('learning_rate', spec['lr']), ('batch_size', spec['batch']),
                          ('shuffle', False), ('precision', 'float32')]:
        if cfg.get(key, expected) != expected:
            raise ValueError(f'SingleTask protocol requires {key}={expected}')
    if spec['dataset'] not in ('census', 'aliccp', 'kuairand'):
        raise ValueError('Unknown SingleTask dataset')
    return values


def build_single(spec, model_seed, cfg):
    """Seed and initialize a fresh embedding, MLP, and tower; every weight trains."""
    settings = _settings(spec, cfg)
    core.seed_all(model_seed)
    elements = sum(spec['vocab'].values()) * spec['embed']
    if elements <= 0:
        raise ValueError('SingleTask requires a positive embedding element count')
    model = SingleTask(spec['vocab'], spec['embed'], spec['xdim'], spec['hidden'],
                       spec['tower'], settings['embedding_l2'] / elements,
                       settings['dnn_l2'],
                       dropout=[.1, .3] if spec['dataset'] == 'aliccp' else None)
    return model.float().to(core.DEVICE)


def _protocol(spec, target, paired_seed, settings):
    return dict(version='single-task-full-network-normalized-l2-v1',
                dataset=spec['dataset'], target=target,
                target_index=spec['tasks'].index(target), paired_seed=paired_seed,
                model_seed=paired_seed + spec['tasks'].index(target) + 2,
                learning_rate=spec['lr'], batch_size=spec['batch'],
                shuffle=False, precision='float32', optimizer='Adam',
                hidden=list(spec['hidden']), tower=list(spec['tower']),
                vocab=dict(spec['vocab']), embed=spec['embed'], xdim=spec['xdim'],
                dropout=[.1, .3] if spec['dataset'] == 'aliccp' else None,
                rows={s: len(spec['sets'][s]) for s in ('train', 'val', 'test')},
                **settings)


def _probability_check(p):
    if not torch.isfinite(p).all():
        raise ValueError('Non-finite SingleTask predictions')
    if torch.any((p < 0) | (p > 1)):
        raise ValueError('SingleTask predictions outside [0, 1]')


def _rms(model):
    return {name: float(emb.weight.detach().square().mean().sqrt())
            for name, emb in model.embedding_network.embedding_dict.items()}


def _verify_result(folder, spec, protocol, model):
    result = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
    if (result.get('protocol') != protocol or result.get('variant') != 'single_task'
            or result.get('target') != protocol['target']
            or result.get('model_seed') != protocol['model_seed']
            or result.get('paired_seed') != protocol['paired_seed']
            or result.get('seed') != protocol['model_seed']
            or not result.get('ready_for_commit')):
        raise ValueError('SingleTask stage protocol mismatch')
    model.load_state_dict(torch.load(folder / 'model.pt', map_location=core.DEVICE,
                                    weights_only=True))
    if core.model_hash(model) != result['model_hash']:
        raise ValueError('SingleTask saved model hash mismatch')
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Non-finite saved SingleTask parameters')
    idx = protocol['target_index']
    with np.load(folder / 'predictions.npz', allow_pickle=False) as raw:
        for split in ('val', 'test'):
            rows = spec['sets'][split]
            y, p = raw[split + '_y'], raw[split + '_p']
            if (y.shape != (len(rows),) or p.shape != y.shape
                    or not np.array_equal(y, rows.y[:, idx])
                    or not np.array_equal(raw[split + '_rowid'], rows.row_ids)):
                raise ValueError('SingleTask raw label/row-id alignment mismatch')
            _probability_check(torch.from_numpy(p))
            actual = core.measures(y, p)
            if any(abs(actual[k] - result[split][k]) > 1e-10 for k in actual):
                raise ValueError('SingleTask raw metric mismatch')
    return result


def fit_single(spec, paired_seed, target, folder, cfg, status_path):
    """Fit one target from scratch or verify and reuse its completed stage."""
    settings = _settings(spec, cfg)
    protocol = _protocol(spec, target, paired_seed, settings)
    model_seed = protocol['model_seed']
    model = build_single(spec, model_seed, cfg)
    folder, status_path = Path(folder), Path(status_path)
    if (folder / 'COMMITTED.json').exists():
        _committed(core, folder)
        return _verify_result(folder, spec, protocol, model)
    if (folder / 'metrics.json').exists():
        pending = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
        if pending.get('ready_for_commit'):
            result = _verify_result(folder, spec, protocol, model)
            _commit(core, folder)
            return result
    _committed(core, folder)
    idx = protocol['target_index']
    gpu = core.DEVICE.type == 'cuda'
    loader_kw = dict(batch_size=spec['batch'], shuffle=False,
                     num_workers=2 if gpu else 0, pin_memory=gpu)
    if gpu:
        loader_kw['prefetch_factor'] = 2
    loaders = {split: DataLoader(_Target(spec['sets'][split], idx), **loader_kw)
               for split in ('train', 'val', 'test')}

    def predict(split):
        model.eval()
        predictions, labels = [], []
        with torch.no_grad():
            for x, y in loaders[split]:
                p = model({k: v.to(core.DEVICE, non_blocking=gpu) for k, v in x.items()})[0].reshape(-1)
                _probability_check(p)
                predictions.append(p.cpu())
                labels.append(y)
        return torch.cat(labels).numpy(), torch.cat(predictions).numpy()

    parameters = list(model.parameters())
    if not all(p.requires_grad for p in parameters):
        raise ValueError('Every SingleTask parameter must train')
    optimizer = torch.optim.Adam(parameters, lr=spec['lr'])
    best, stale, state = -float('inf'), 0, None
    history = []
    started_at, started = _utc(), time.perf_counter()
    if gpu:
        torch.cuda.reset_peak_memory_stats(core.DEVICE)
    for epoch in range(1, settings['epochs'] + 1):
        model.train()
        totals = np.zeros(3, dtype=np.float64)
        rows_seen = positive = steps = 0
        epoch_clock = time.perf_counter()
        for x, y in loaders['train']:
            x = {k: v.to(core.DEVICE, non_blocking=gpu) for k, v in x.items()}
            y = y.to(core.DEVICE, non_blocking=gpu).float()
            p = model(x)[0].reshape(-1)
            _probability_check(p)
            bce = F.binary_cross_entropy(p, y)
            emb = model.reg_embedding * model.embedding_network.get_l2_reg()
            dnn = model.reg_dnn * (model.base_network.get_l2_reg() + model.tower_network.get_l2_reg())
            loss = bce + emb + dnn
            if not torch.isfinite(loss):
                raise ValueError('Non-finite SingleTask training loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals += np.array([float(bce.detach()), float(emb.detach()), float(dnn.detach())]) * len(y)
            rows_seen += len(y)
            positive += int(y.sum())
            steps += 1
        if rows_seen != len(spec['sets']['train']) or rows_seen == 0:
            raise ValueError('SingleTask training row count mismatch')
        train_seconds = time.perf_counter() - epoch_clock
        eval_clock = time.perf_counter()
        # Preserve the corrected legacy sequence: full train evaluation, then val.
        ty, tp = predict('train')
        vy, vp = predict('val')
        train_metrics, val = core.measures(ty, tp), core.measures(vy, vp)
        means = totals / rows_seen
        row = dict(epoch=epoch, train_bce=float(means[0]), embedding_l2=float(means[1]),
                   dnn_l2=float(means[2]), train_loss=float(means.sum()),
                   train_auc=train_metrics['auc'], train_eval_logloss=train_metrics['logloss'],
                   val_auc=val['auc'], val_logloss=val['logloss'],
                   embedding_l2_over_bce=float(means[1] / max(means[0], 1e-15)),
                   rows_seen=rows_seen, positive_n=positive, optimizer_steps=steps,
                   embedding_rms=_rms(model), train_seconds=train_seconds,
                   evaluation_seconds=time.perf_counter() - eval_clock,
                   samples_per_second=rows_seen / max(train_seconds, 1e-9), at=_utc())
        history.append(row)
        core.dump(folder / 'history-live.json', history)
        core.dump(status_path, dict(state='running', pid=os.getpid(), seed=paired_seed,
                                    model_seed=model_seed, target=target, variant='single_task',
                                    validation=row, updated_at=_utc()))
        print(json.dumps(dict(event='epoch', target=target, seed=model_seed, **row)), flush=True)
        if val['auc'] > best:
            best, stale, state = val['auc'], 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= settings['patience']:
            break
    model.load_state_dict(state)
    raw, metrics = {}, {}
    for split in ('val', 'test'):
        y, p = predict(split)
        metrics[split] = core.measures(y, p)
        raw.update({split + '_y': y, split + '_p': p,
                    split + '_rowid': np.asarray(spec['sets'][split].row_ids)})
    _save(state, folder / 'model.pt')
    with (folder / 'predictions.npz.tmp').open('wb') as stream:
        np.savez_compressed(stream, **raw)
    os.replace(folder / 'predictions.npz.tmp', folder / 'predictions.npz')
    elements = sum(p.numel() for p in model.embedding_network.parameters())
    total = sum(p.numel() for p in parameters)
    result = dict(target=target, variant='single_task', seed=model_seed, model_seed=model_seed,
                  cost_measurement_source=cfg.get('cost_measurement_source','current-ST-serial'),
                  paired_seed=paired_seed, protocol=protocol, history=history,
                  epochs=len(history), selected_epoch=max(history, key=lambda r: r['val_auc'])['epoch'],
                  stop_reason='patience' if stale >= settings['patience'] else 'budget',
                  embedding_elements=elements, embedding_sum_coefficient=model.reg_embedding,
                  dnn_coefficient=model.reg_dnn, embedding_rms=_rms(model),
                  total_parameters=total, optimizer_parameters=total,
                  exclusive_parameters=total, inference_parameters=total,
                  parameter_counts={prefix: sum(p.numel() for name, p in model.named_parameters()
                                               if name.startswith(prefix + '.'))
                                    for prefix in ('embedding_network', 'base_network', 'tower_network')},
                  trainable_tensor_bytes=sum(p.numel() * p.element_size() for p in parameters),
                  weight_file_bytes=(folder / 'model.pt').stat().st_size,
                  model_hash=core.model_hash(model), started_at=started_at, ended_at=_utc(),
                  seconds=time.perf_counter() - started,
                  peak_cuda_bytes=torch.cuda.max_memory_allocated(core.DEVICE) if gpu else 0,
                  learning_rate=spec['lr'], patience=settings['patience'], batch_size=spec['batch'],
                  num_workers=loader_kw['num_workers'], pin_memory=gpu,
                  prefetch_factor=2 if gpu else None, ready_for_commit=True, **metrics)
    core.dump(folder / 'metrics.json', result)
    _verify_result(folder, spec, protocol, model)
    _commit(core, folder)
    return result
