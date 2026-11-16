"""Resumable complementary pretraining and independent frozen-source readouts."""
import copy
import hashlib
import json
import os
import time
from pathlib import Path

from . import core
import numpy as np
import torch
from torch.nn import functional as F

from .training import _commit, _committed, _save, _utc

SPLITS = ('train', 'val', 'test')


def _array(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _digest(value):
    array = _array(value)
    digest = hashlib.sha256(str((array.shape, str(array.dtype))).encode())
    for start in range(0, len(array), 100000):
        digest.update(np.ascontiguousarray(array[start:start+100000]).tobytes())
    return digest.hexdigest()


def _data_identity(spec, caches):
    """Admitted cache owners avoid repeatedly reading multi-GB mmap features."""
    admitted = spec.get('scheme_cache_identity')
    result = {}
    all_ids = []
    for split in SPLITS:
        rows, cache = spec['sets'][split], caches[split]
        ids, y = np.asarray(rows.row_ids), _array(cache['y'])
        if ids.shape != (len(rows),) or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError(f'Invalid rowids: {split}')
        if admitted is None and len(np.unique(ids)) != len(ids):
            raise ValueError(f'Duplicate rowids: {split}')
        if y.shape != np.asarray(rows.y).shape or not np.array_equal(y, rows.y):
            raise ValueError(f'Cache label identity mismatch: {split}')
        if any(len(value) != len(ids) for value in cache.values()):
            raise ValueError(f'Cache row count mismatch: {split}')
        if admitted is None:
            all_ids.append(ids)
        result[split] = dict(rows=len(ids), rowid=_digest(ids), labels=_digest(y))
        if admitted is None:
            result[split]['cache'] = {key: _digest(value) for key, value in sorted(cache.items())}
        else:
            result[split]['cache_layout'] = {
                key: dict(shape=list(value.shape), dtype=str(value.dtype),
                          tensor_version=value._version if torch.is_tensor(value) else None)
                for key, value in sorted(cache.items())}
    if admitted is None:
        merged = np.concatenate(all_ids)
        if len(np.unique(merged)) != len(merged):
            raise ValueError('Overlapping split rowids')
    return dict(owner=admitted, data_signature=spec.get('scheme_data_identity'), splits=result)


def _settings(spec, cfg, stage):
    epoch_key = dict(decoder='decoder_epochs', complementary_u='ssl_epochs', conditional_readout='epochs')[stage]
    settings = dict(epochs=cfg.get(epoch_key, 30), patience=cfg.get('patience', 5),
                    learning_rate=cfg.get('learning_rate', spec['lr']), batch_size=cfg.get('batch_size', spec['batch']),
                    optimizer='Adam', shuffle=False, precision='float32')
    if settings['learning_rate'] != spec['lr'] or settings['batch_size'] != spec['batch']:
        raise ValueError('Stage protocol learning rate/batch size mismatch')
    if cfg.get('shuffle', False) or cfg.get('precision', 'float32') != 'float32':
        raise ValueError('Stage protocol requires ordered float32 training')
    if settings['epochs'] < 1 or settings['patience'] < 1:
        raise ValueError('Positive stage epochs and patience required')
    if stage == 'complementary_u':
        settings.update(redundancy_weight=cfg.get('redundancy_weight', .01),
                        variance_weight=cfg.get('variance_weight', .01), mask_validation_seed=20261009)
    if stage == 'conditional_readout':
        settings['rank'] = 4
    return settings


def _contract(spec, caches, cfg, seed, stage, dependencies=None, target=None):
    from . import scheme_models, expert, backbone
    return dict(version='schemes-fitting-v1', stage=stage, dataset=spec['dataset'],
                tasks=list(spec['tasks']), initial=list(spec['initial']), seed=seed,
                target=target, widths=list(spec['widths']), xdim=spec['xdim'], rdim=spec['rdim'],
                data=_data_identity(spec, caches), settings=_settings(spec, cfg, stage),
                dependencies=dependencies or {}, execution_identity=cfg.get('execution_identity'),
                code={Path(module.__file__).name: core.file_hash(module.__file__)
                      for module in (scheme_models, expert, backbone, core)} | {'scheme_fitting.py': core.file_hash(__file__)})


def _open(folder, contract):
    """Reject identity changes before preserving or modifying any existing leaf."""
    folder = Path(folder)
    path = folder/'contract.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != contract:
        raise ValueError(f'Stage contract identity mismatch: {folder}')
    metrics = folder/'metrics.json'
    if metrics.exists():
        prior = json.loads(metrics.read_text(encoding='utf-8'))
        if prior.get('contract') != contract:
            raise ValueError(f'Stage metrics contract identity mismatch: {folder}')
    if (folder/'COMMITTED.json').exists():
        if not path.exists():
            raise ValueError('Committed stage missing contract identity')
        if not metrics.exists():
            raise ValueError('Committed stage missing metrics integrity record')
        _committed(core, folder)
        return prior
    if metrics.exists() and prior.get('ready_for_commit'):
        return prior
    _committed(core, folder)  # archives an interrupted attempt without deletion
    core.dump(path, contract)
    return None


def _verify_state(folder, model, result, name):
    if not result.get('ready_for_commit'):
        raise ValueError('Stage is not ready for commit')
    path = Path(folder)/name
    if not path.is_file() or core.file_hash(path) != result.get('weight_sha256'):
        raise ValueError('Stage weight integrity hash mismatch')
    model.load_state_dict(torch.load(path, map_location=core.DEVICE, weights_only=True))
    if core.model_hash(model) != result['model_hash']:
        raise ValueError('Stage model hash mismatch')
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('Non-finite stage state')
    return model


def _finish(folder, model, result, name):
    _save(model.state_dict(), Path(folder)/name)
    result.update(model_hash=core.model_hash(model), weight_sha256=core.file_hash(Path(folder)/name), ready_for_commit=True)
    core.dump(Path(folder)/'metrics.json', result)
    _commit(core, Path(folder))


def _status(folder, status_path, row, stage, seed):
    core.dump(Path(folder)/'history-live.json', row)
    if status_path is not None:
        core.dump(status_path, dict(state='running', pid=os.getpid(), stage=stage,
                                   seed=seed, validation=row[-1], updated_at=_utc()))


def _cost(started, started_at):
    return dict(seconds=time.perf_counter()-started, started_at=started_at, ended_at=_utc(),
                peak_cuda_bytes=torch.cuda.max_memory_allocated(core.DEVICE) if core.DEVICE.type == 'cuda' else 0)


def _start():
    if core.DEVICE.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(core.DEVICE)
    return time.perf_counter(), _utc()


def _normalizer(cache, batch):
    n = 0
    sums = squares = None
    for start in range(0, len(cache['x']), batch):
        x = torch.as_tensor(cache['x'][start:start+batch]).detach().cpu().double()
        if not torch.isfinite(x).all():
            raise ValueError('Non-finite training normalizer inputs')
        sums = x.sum(0) if sums is None else sums + x.sum(0)
        squares = x.square().sum(0) if squares is None else squares + x.square().sum(0)
        n += len(x)
    if not n:
        raise ValueError('Empty normalizer training split')
    mean = sums/n
    return mean.float().to(core.DEVICE), ((squares/n-mean.square()).clamp_min(0).sqrt().clamp_min(.1)).float().to(core.DEVICE)


def _field_mse(predicted, target, widths):
    return torch.stack([v.mean(1) for v in (predicted-target).square().split(widths, 1)], 1).mean()


def _gs(batch):
    return torch.cat([batch['g'], batch['s0'], batch['s1']], 1)


def _verify_decoder_loss(decoder, spec, caches, result):
    decoder.eval()
    total = rows = 0
    with torch.no_grad():
        for batch in core.batches(caches['val'], spec['batch']):
            loss = _field_mse(decoder(_gs(batch)), decoder.normalize(batch['x']), spec['widths'])
            total += float(loss)*len(batch['x'])
            rows += len(batch['x'])
    if not rows or not np.isclose(total/rows, result['val_loss'], rtol=1e-6, atol=1e-7):
        raise ValueError('Saved decoder validation loss metric mismatch')


def fit_decoder(spec, caches, folder, cfg, seed, status_path=None):
    from .scheme_models import GSDecoder
    contract = _contract(spec, caches, cfg, seed, 'decoder')
    prior = _open(folder, contract)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed+1)
        decoder = GSDecoder(spec['rdim'], spec['xdim']).to(core.DEVICE)
    if prior is not None:
        _verify_state(folder, decoder, prior, 'model.pt')
        _verify_decoder_loss(decoder, spec, caches, prior)
        _commit(core, Path(folder))
        return decoder.eval(), prior
    decoder.set_normalizer(*_normalizer(caches['train'], spec['batch']))
    settings = contract['settings']
    optimizer = torch.optim.Adam(decoder.parameters(), lr=settings['learning_rate'])
    history, best, state, stale, selected = [], float('inf'), None, 0, None
    started, started_at = _start()
    for epoch in range(1, settings['epochs']+1):
        losses = {}
        for split in ('train', 'val'):
            decoder.train(split == 'train')
            total = rows = 0
            with torch.set_grad_enabled(split == 'train'):
                for batch in core.batches(caches[split], spec['batch']):
                    loss = _field_mse(decoder(_gs(batch)), decoder.normalize(batch['x']), spec['widths'])
                    if not torch.isfinite(loss):
                        raise ValueError('Non-finite decoder loss')
                    if split == 'train':
                        optimizer.zero_grad(); loss.backward(); optimizer.step()
                    total += float(loss.detach())*len(batch['x']); rows += len(batch['x'])
            if not rows:
                raise ValueError('Empty decoder split')
            losses[split+'_loss'] = total/rows
        history.append(dict(epoch=epoch, at=_utc(), **losses))
        _status(folder, status_path, history, 'decoder', seed)
        if losses['val_loss'] < best:
            best, state, stale, selected = losses['val_loss'], copy.deepcopy(decoder.state_dict()), 0, epoch
        else:
            stale += 1
        if stale >= settings['patience']:
            break
    decoder.load_state_dict(state)
    result = dict(variant='gs_decoder', seed=seed, paired_seed=seed, contract=contract, history=history,
                  epochs=len(history), selected_epoch=selected, val_loss=best,
                  stop_reason='patience' if stale >= settings['patience'] else 'budget',
                  optimizer_parameters=sum(p.numel() for p in decoder.parameters()), **_cost(started, started_at))
    _finish(folder, decoder, result, 'model.pt')
    return decoder.eval(), result


def _validation_mask(n, fields, offset, seed):
    generator = torch.Generator().manual_seed(seed+offset)
    ranks = torch.rand(n, fields, generator=generator).argsort(1)
    return torch.zeros(n, fields, dtype=torch.bool).scatter_(1, ranks[:, :max(1, round(.3*fields))], True)


def _verify_u_loss(u, decoder, spec, caches, settings, result):
    u.eval()
    decoder.eval()
    total = rows = 0
    with torch.no_grad():
        for batch in core.batches(caches['val'], spec['batch']):
            mask = _validation_mask(len(batch['x']), len(spec['widths']), rows,
                                    settings['mask_validation_seed'])
            losses = u.losses(batch['x'], batch['g'], [batch['s0'], batch['s1']], decoder,
                              mask=mask, redundancy_weight=settings['redundancy_weight'],
                              variance_weight=settings['variance_weight'])
            total += float(losses['total'])*len(batch['x'])
            rows += len(batch['x'])
    if not rows or not np.isclose(total/rows, result['val_loss'], rtol=1e-6, atol=1e-7):
        raise ValueError('Saved complementary U validation loss metric mismatch')


def fit_complementary_u(spec, caches, decoder, normalizer, folder, cfg, seed, status_path=None):
    from .scheme_models import ComplementaryExpert
    if len(normalizer) != 2 or any(not torch.equal(a.detach().cpu(), b.detach().cpu()) for a,b in zip(normalizer,(decoder.mean,decoder.scale))):
        raise ValueError('U normalizer differs from frozen decoder')
    decoder_hash = core.model_hash(decoder)
    deps = dict(decoder_hash=decoder_hash, normalizer=[_digest(v) for v in normalizer])
    contract = _contract(spec, caches, cfg, seed, 'complementary_u', deps)
    prior = _open(folder, contract)
    u = ComplementaryExpert(spec['widths'], rep_dim=spec['rdim'], seed=seed+1).to(core.DEVICE)
    if prior is not None:
        _verify_state(folder, u, prior, 'source.pt')
        _verify_u_loss(u, decoder, spec, caches, contract['settings'], prior)
        _commit(core, Path(folder))
        return u.eval(), prior
    u.set_normalizer(*normalizer)
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    settings = contract['settings']
    optimizer = torch.optim.Adam(u.parameters(), lr=settings['learning_rate'])
    history, best, state, stale, selected = [], float('inf'), None, 0, None
    started, started_at = _start()
    for epoch in range(1, settings['epochs']+1):
        values = {}
        for split in ('train', 'val'):
            u.train(split == 'train')
            totals, rows = {}, 0
            with torch.set_grad_enabled(split == 'train'):
                for batch in core.batches(caches[split], spec['batch']):
                    mask = None if split == 'train' else _validation_mask(len(batch['x']), len(spec['widths']), rows, settings['mask_validation_seed'])
                    losses = u.losses(batch['x'], batch['g'], [batch['s0'], batch['s1']], decoder, mask=mask,
                                      redundancy_weight=settings['redundancy_weight'], variance_weight=settings['variance_weight'])
                    if any(not torch.isfinite(v) for v in losses.values()):
                        raise ValueError('Non-finite complementary U loss')
                    if split == 'train':
                        optimizer.zero_grad(); losses['total'].backward(); optimizer.step()
                    for key, value in losses.items():
                        totals[key] = totals.get(key, 0.)+float(value.detach())*len(batch['x'])
                    rows += len(batch['x'])
            if not rows:
                raise ValueError('Empty complementary U split')
            values.update({split+'_'+key: value/rows for key,value in totals.items()})
        values.update(train_loss=values['train_total'], val_loss=values['val_total'])
        history.append(dict(epoch=epoch, at=_utc(), **values))
        _status(folder, status_path, history, 'complementary_u', seed)
        if values['val_loss'] < best:
            best, state, stale, selected = values['val_loss'], copy.deepcopy(u.state_dict()), 0, epoch
        else:
            stale += 1
        if stale >= settings['patience']:
            break
    u.load_state_dict(state)
    if core.model_hash(decoder) != decoder_hash:
        raise ValueError('Frozen decoder source changed during U training')
    result = dict(variant='complementary_u', seed=seed+1, paired_seed=seed, contract=contract,
                  history=history, epochs=len(history), selected_epoch=selected, val_loss=best,
                  decoder_hash=decoder_hash, stop_reason='patience' if stale >= settings['patience'] else 'budget',
                  optimizer_parameters=sum(p.numel() for p in u.parameters()), **_cost(started, started_at))
    _finish(folder, u, result, 'source.pt')
    return u.eval(), result


def _probabilities(p):
    if not torch.isfinite(p).all() or torch.any((p < 0) | (p > 1)):
        raise ValueError('Invalid readout probabilities')


def _verify_raw(folder, spec, caches, position, result):
    path = Path(folder)/'predictions.npz'
    if not path.is_file() or core.file_hash(path) != result.get('predictions_sha256'):
        raise ValueError('Readout raw integrity hash mismatch')
    idx = spec['tasks'].index(spec['initial'][position])
    with np.load(path, allow_pickle=False) as raw:
        for split in ('val', 'test'):
            y, p, base_p = [raw[split+'_'+key] for key in ('y','p','base_p')]
            if (y.shape != (len(spec['sets'][split]),) or p.shape != y.shape or base_p.shape != y.shape
                    or not np.array_equal(y, _array(caches[split]['y'])[:,idx])
                    or not np.array_equal(raw[split+'_rowid'], spec['sets'][split].row_ids)
                    or not np.array_equal(base_p, _array(caches[split]['old'])[:,position])):
                raise ValueError('Readout raw label/rowid/base alignment mismatch')
            _probabilities(torch.from_numpy(p))
            actual = core.measures(y,p)
            if any(abs(actual[key]-result[split][key]) > 1e-10 for key in actual):
                raise ValueError('Readout raw metrics mismatch')
            if result['selected_epoch'] == 0 and not np.array_equal(p, base_p):
                raise ValueError('Readout epoch0 prediction mismatch')


def fit_readout(base, u, spec, caches, position, folder, cfg, seed, status_path=None):
    from .scheme_models import ConditionalReadout, old_task_hidden
    if position not in (0,1):
        raise ValueError('Readout position must identify an original task')
    base_hash, u_hash = core.model_hash(base), core.model_hash(u)
    target = spec['initial'][position]
    contract = _contract(spec, caches, cfg, seed, 'conditional_readout',
                         dict(base_hash=base_hash, u_hash=u_hash), target)
    prior = _open(folder, contract)
    model_seed = seed+position+2
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(model_seed)
        head = ConditionalReadout(spec['rdim'], rank=4).to(core.DEVICE)
    if prior is not None:
        _verify_state(folder, head, prior, 'model.pt')
        _verify_raw(folder, spec, caches, position, prior)
        _commit(core, Path(folder))
        return prior
    for source in (base, u):
        source.eval()
        for p in source.parameters():
            p.requires_grad_(False)
    idx, settings = spec['tasks'].index(target), contract['settings']

    def forward(batch):
        with torch.no_grad():
            h, representation = old_task_hidden(base, batch, position), u(batch['x'])
        return head(h, representation, batch['old'][:, position]).reshape(-1)

    def predict(split):
        head.eval()
        with torch.no_grad():
            p = torch.cat([forward(batch).cpu() for batch in core.batches(caches[split], spec['batch'])]).numpy()
        _probabilities(torch.from_numpy(p))
        return _array(caches[split]['y'])[:,idx], p

    started, started_at = _start()
    vy, vp = predict('val')
    baseline = core.measures(vy, vp)
    if not np.array_equal(vp, _array(caches['val']['old'])[:,position]):
        raise ValueError('Zero readout must exactly preserve baseline')
    history = [dict(epoch=0, val_auc=baseline['auc'], val_logloss=baseline['logloss'], at=_utc())]
    best, state, stale, selected = baseline['auc'], copy.deepcopy(head.state_dict()), 0, 0
    optimizer = torch.optim.Adam(head.parameters(), lr=settings['learning_rate'])
    for epoch in range(1, settings['epochs']+1):
        head.train()
        total = rows = 0
        clock = time.perf_counter()
        for batch in core.batches(caches['train'], spec['batch']):
            p = forward(batch)
            _probabilities(p)
            loss = F.binary_cross_entropy(p, batch['y'][:,idx].float())
            if not torch.isfinite(loss):
                raise ValueError('Non-finite readout BCE')
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += float(loss.detach())*len(p); rows += len(p)
        if not rows:
            raise ValueError('Empty readout train split')
        train_seconds = time.perf_counter()-clock
        vy, vp = predict('val')
        score = core.measures(vy,vp)
        history.append(dict(epoch=epoch, train_bce=total/rows, train_loss=total/rows,
                            val_auc=score['auc'], val_logloss=score['logloss'], train_seconds=train_seconds,
                            samples_per_second=rows/max(train_seconds,1e-9), at=_utc()))
        _status(folder, status_path, history, 'conditional_readout', seed)
        if score['auc'] > best:
            best, state, stale, selected = score['auc'], copy.deepcopy(head.state_dict()), 0, epoch
        else:
            stale += 1
        if stale >= settings['patience']:
            break
    head.load_state_dict(state)
    raw, metrics = {}, {}
    for split in ('val','test'):
        y,p = predict(split)
        metrics[split] = core.measures(y,p)
        raw.update({split+'_y':y, split+'_p':p, split+'_rowid':spec['sets'][split].row_ids,
                    split+'_base_p':_array(caches[split]['old'])[:,position]})
    path = Path(folder)/'predictions.npz'
    with path.with_suffix('.npz.tmp').open('wb') as stream:
        np.savez_compressed(stream, **raw)
    os.replace(path.with_suffix('.npz.tmp'), path)
    if (core.model_hash(base),core.model_hash(u)) != (base_hash,u_hash):
        raise ValueError('Frozen B/U source changed during readout')
    count = sum(p.numel() for p in head.parameters())
    result = dict(variant='conditional_readout', target=target, position=position, seed=model_seed,
                  paired_seed=seed, model_seed=model_seed, contract=contract, history=history,
                  epochs=len(history)-1, selected_epoch=selected, baseline_val=baseline,
                  stop_reason='patience' if stale >= settings['patience'] else 'budget',
                  base_hash=base_hash, u_hash=u_hash, optimizer_parameters=count, exclusive_parameters=count,
                  inference_parameters=count, total_parameters=count,
                  trainable_tensor_bytes=sum(p.numel()*p.element_size() for p in head.parameters()),
                  predictions_sha256=core.file_hash(path), **metrics, **_cost(started, started_at))
    result['baseline'] = {split: core.measures(raw[split+'_y'], raw[split+'_base_p'])
                          for split in ('val', 'test')}
    _save(head.state_dict(), Path(folder)/'model.pt')
    result.update(model_hash=core.model_hash(head), weight_sha256=core.file_hash(Path(folder)/'model.pt'), ready_for_commit=True)
    _verify_raw(folder, spec, caches, position, result)
    core.dump(Path(folder)/'metrics.json', result)
    _commit(core, Path(folder))
    return result
