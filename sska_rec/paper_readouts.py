"""Table 4 frozen-source readout controls with independently selected heads."""
import copy
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import core
from .scheme_fitting import (
    _array, _commit, _contract, _cost, _finish, _open, _probabilities,
    _start, _status, _utc, _verify_raw, _verify_state,
)
from .scheme_models import ConditionalReadout, old_task_hidden

VARIANTS = ('h_conditional', 'random_conditional', 'linear_aligned')


class LinearAlignedReadout(nn.Module):
    """Raw U linear logit correction, exactly retaining zero-correction p."""

    def __init__(self, rdim):
        super().__init__()
        self.linear = nn.Linear(rdim, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, h, u, p):
        p = p.detach().reshape(-1)
        logits = torch.logit(p.clamp(1e-7, 1-1e-7))
        delta = self.linear(u.detach()).reshape(-1)
        difference = (logits + delta).sigmoid() - logits.sigmoid()
        return (p + difference).clamp(0, 1)


def _freeze(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def make_random_u(spec, u, seed):
    """Fresh core.make_u initialization at paired seed+1; copy only mean/scale."""
    random = core.make_u(spec, seed)
    with torch.no_grad():
        random.mean.copy_(u.mean.detach())
        random.scale.copy_(u.scale.detach())
    return _freeze(random)


def make_control_head(spec, seed, position, variant):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown readout control variant: {variant}')
    if position not in (0, 1):
        raise ValueError('Readout position must identify an original task')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed+position+2)
        head = (LinearAlignedReadout(spec['rdim']) if variant == 'linear_aligned'
                else ConditionalReadout(spec['rdim'], rank=4))
    return head.to(device=core.DEVICE, dtype=torch.float32)


def fit_control(base, u, spec, caches, position, folder, cfg, seed, variant,
                status_path=None):
    """Fit one private control head by strict target valAUC, including epoch 0."""
    head = make_control_head(spec, seed, position, variant)
    for source in (base, u):
        _freeze(source)
    base_hash, u_hash = core.model_hash(base), core.model_hash(u)
    source = make_random_u(spec, u, seed) if variant == 'random_conditional' else u
    source_hash = core.model_hash(source)
    dependencies = dict(base_hash=base_hash, u_hash=u_hash)
    if variant == 'random_conditional':
        dependencies.update(random_u_hash=source_hash, random_u_seed=seed+1,
                            random_initialization='core.make_u',
                            copied_buffers=['mean', 'scale'])
    target = spec['initial'][position]
    contract = _contract(spec, caches, cfg, seed, 'conditional_readout', dependencies, target)
    contract.update(version='paper-readouts-v1', variant=variant)
    contract['code']['paper_readouts.py'] = core.file_hash(__file__)
    contract['settings']['representation'] = {
        'h_conditional': 'ConditionalReadout(h,h,p)',
        'random_conditional': 'ConditionalReadout(h,random_u(x),p)',
        'linear_aligned': 'Linear(raw_u,1)',
    }[variant]
    prior = _open(folder, contract)
    if prior is not None:
        _verify_state(folder, head, prior, 'model.pt')
        _verify_raw(folder, spec, caches, position, prior)
        _commit(core, Path(folder))
        return prior
    idx, settings = spec['tasks'].index(target), contract['settings']

    def forward(batch):
        with torch.no_grad():
            h = old_task_hidden(base, batch, position)
            representation = h if variant == 'h_conditional' else source(batch['x'])
        return head(h, representation, batch['old'][:, position]).reshape(-1)

    def predict(split):
        head.eval()
        with torch.no_grad():
            p = torch.cat([forward(batch).cpu() for batch in
                           core.batches(caches[split], spec['batch'])]).numpy()
        _probabilities(torch.from_numpy(p))
        return _array(caches[split]['y'])[:, idx], p

    started, started_at = _start()
    vy, vp = predict('val')
    baseline = core.measures(vy, vp)
    if not np.array_equal(vp, _array(caches['val']['old'])[:, position]):
        raise ValueError('Zero readout must exactly preserve baseline')
    history = [dict(epoch=0, val_auc=baseline['auc'],
                    val_logloss=baseline['logloss'], at=_utc())]
    best, state, stale, selected = baseline['auc'], copy.deepcopy(head.state_dict()), 0, 0
    optimizer = torch.optim.Adam(head.parameters(), lr=settings['learning_rate'])
    for epoch in range(1, settings['epochs']+1):
        head.train()
        total = rows = 0
        clock = time.perf_counter()
        for batch in core.batches(caches['train'], spec['batch']):
            p = forward(batch)
            _probabilities(p)
            loss = F.binary_cross_entropy(p, batch['y'][:, idx].float())
            if not torch.isfinite(loss):
                raise ValueError('Non-finite readout BCE')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())*len(p)
            rows += len(p)
        if not rows:
            raise ValueError('Empty readout train split')
        seconds = time.perf_counter()-clock
        vy, vp = predict('val')
        score = core.measures(vy, vp)
        history.append(dict(epoch=epoch, train_bce=total/rows, train_loss=total/rows,
                            val_auc=score['auc'], val_logloss=score['logloss'],
                            train_seconds=seconds, samples_per_second=rows/max(seconds, 1e-9),
                            at=_utc()))
        _status(folder, status_path, history, variant, seed)
        if score['auc'] > best:
            best, state, stale, selected = score['auc'], copy.deepcopy(head.state_dict()), 0, epoch
        else:
            stale += 1
        if stale >= settings['patience']:
            break
    head.load_state_dict(state)
    raw, metrics = {}, {}
    for split in ('val', 'test'):
        y, p = predict(split)
        metrics[split] = core.measures(y, p)
        raw.update({split+'_y': y, split+'_p': p,
                    split+'_rowid': spec['sets'][split].row_ids,
                    split+'_base_p': _array(caches[split]['old'])[:, position]})
    path = Path(folder)/'predictions.npz'
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **raw)
    os.replace(temporary, path)
    if (core.model_hash(base), core.model_hash(u), core.model_hash(source)) != (base_hash, u_hash, source_hash):
        raise ValueError('Frozen B/U source changed during readout')
    count = sum(p.numel() for p in head.parameters())
    model_seed = seed+position+2
    result = dict(variant=variant, target=target, position=position, seed=model_seed,
                  paired_seed=seed, model_seed=model_seed, contract=contract, history=history,
                  epochs=len(history)-1, selected_epoch=selected, baseline_val=baseline,
                  stop_reason='patience' if stale >= settings['patience'] else 'budget',
                  base_hash=base_hash, u_hash=u_hash, optimizer_parameters=count,
                  exclusive_parameters=count, inference_parameters=count, total_parameters=count,
                  trainable_tensor_bytes=sum(p.numel()*p.element_size() for p in head.parameters()),
                  predictions_sha256=core.file_hash(path), **metrics, **_cost(started, started_at))
    if variant == 'random_conditional':
        result.update(random_u_hash=source_hash, random_u_seed=seed+1)
    result['baseline'] = {split: core.measures(raw[split+'_y'], raw[split+'_base_p'])
                          for split in ('val', 'test')}
    # Raw alignment and metrics must pass before writing the committed leaf.
    _verify_raw(folder, spec, caches, position, result)
    _finish(folder, head, result, 'model.pt')
    return result
