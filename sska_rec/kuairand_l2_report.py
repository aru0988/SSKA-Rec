"""Aggregate sealed KuaiRand normalized-L2 runs and attributed historical rows."""
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

from . import core
from .scheme_common import digest, read_json, short_lock, verify_stage

METRICS = ('auc', 'logloss')
OLD_TASKS = ('click', 'like')
NEW_TASKS = ('follow', 'forward')
NEW_VARIANTS = ('mpt', 'tri_frozen', 'random_lora', 'tri_lora')
# A stage is fully hashed once per unchanged filesystem snapshot in this process.
_VERIFIED = {}


def _verified(folder, required):
    folder = Path(folder).resolve()
    manifest = folder / 'COMMITTED.json'
    entries = read_json(manifest)
    snapshot = []
    for name in ['COMMITTED.json', *entries]:
        path = (folder / name).resolve()
        if not path.is_relative_to(folder) or not path.is_file():
            raise ValueError(f'Committed integrity failure: {path}')
        stat = path.stat()
        snapshot.append((name, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    key = str(folder)
    stamp = tuple(snapshot)
    if key not in _VERIFIED or _VERIFIED[key][0] != stamp:
        _VERIFIED[key] = (stamp, verify_stage(folder, required))
    receipt = _VERIFIED[key][1]
    if not set(required).issubset(receipt['files']):
        raise ValueError(f'Incomplete committed stage: {folder}')
    return receipt


def _stats(values, seeds=None):
    values = [float(v) for v in values]
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError('Reported statistics require finite completed values')
    variance = statistics.variance(values) if len(values) > 1 else None
    row = dict(n=len(values), mean=statistics.mean(values), variance=variance,
               sd=math.sqrt(variance) if variance is not None else None, values=values)
    if seeds is not None:
        row['seeds'] = list(seeds)
    return row


def _cell(row):
    if row is None:
        return '0/5'
    value = f"{row['mean']:.6f}"
    if row.get('sd') is not None:
        value += f" ± {row['sd']:.6f}; var={row['variance']:.8g}"
    return value + f" (n={row['n']}/5)"


def _references(path):
    if path is None:
        return None, {}
    path = Path(path).resolve()
    content = read_json(path)
    if not content.get('version') or not isinstance(content.get('source_files'), (dict, list)):
        raise ValueError('Invalid reference identity')
    groups = {}
    for key, source in content.get('rows', {}).items():
        dataset, model, target = key.split('/')
        if dataset != 'kuairand' or model not in ('single_task', 'mpt_old', 'tcka_old') or target not in OLD_TASKS + NEW_TASKS:
            raise ValueError(f'Unexpected reference row: {key}')
        for metric in METRICS:
            item = source.get(metric)
            if item is None and metric == 'auc' and 'mean' in source:
                item = source
            if item is None:
                continue
            if item.get('n') != 5 or not all(isinstance(item.get(k), (int, float)) and math.isfinite(item[k]) for k in ('mean', 'variance')) or item['variance'] < 0:
                raise ValueError(f'Invalid complete reference statistics: {key}/{metric}')
            row = dict(item, sd=math.sqrt(item['variance']), reference=True)
            if 'values' in row:
                actual = _stats(row['values'])
                if actual['n'] != 5 or abs(actual['mean'] - row['mean']) > 1e-10 or abs(actual['variance'] - row['variance']) > 1e-10:
                    raise ValueError(f'Inconsistent reference seed statistics: {key}')
            groups[f'{model}/{target}/{metric}'] = row
    return dict(path=str(path), sha256=core.file_hash(path), **content), groups


def report(output, cfg, references_path=None):
    """Write partial or complete summaries; evaluation never counts as training."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with short_lock(output / '.report.lock'):
        return _report(output, cfg, references_path)


def _report(output, cfg, references_path):
    dc = cfg['datasets']['kuairand']
    seeds = dc['seeds']
    if len(seeds) != 5 or len(set(seeds)) != 5 or set(cfg['datasets']) != {'kuairand'}:
        raise ValueError('KuaiRand report requires five registered seeds')
    identity_path = output / 'identity.json'
    identity = read_json(identity_path) if identity_path.exists() else None
    if identity is not None and identity.get('config') != cfg:
        raise ValueError('Report execution/config identity mismatch')
    references, groups = _references(references_path)
    if identity is not None and identity.get('references_sha256') is not None:
        if references is None or references['sha256'] != identity['references_sha256']:
            raise ValueError('Report reference identity mismatch')
    records, artifacts, costs, series = {}, {}, {}, {}
    source_counts = dict(base=0, ssl=0, evaluation=0)
    zero, model_count = {}, 0

    def add(folder, required, key):
        if not (folder / 'COMMITTED.json').is_file():
            return None
        artifacts[key] = _verified(folder, required)
        m = read_json(folder / 'metrics.json')
        contract = m.get('contract', {})
        execution_id = contract.get('execution_identity', contract.get('config', {}).get('execution_identity'))
        if identity is not None and execution_id is not None and execution_id != digest(identity):
            raise ValueError(f'Report stage execution identity mismatch: {key}')
        records[key] = m
        return m

    def collect(variant, target, seed, scores):
        for metric in METRICS:
            value = scores[metric]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'Invalid reported metric: {variant}/{target}/{metric}')
            series.setdefault(f'{variant}/{target}/{metric}', {})[seed] = value

    def cost(variant, target, seed, m):
        key = f'{variant}/{target}' if target else variant
        for name in ('seconds', 'peak_cuda_bytes', 'optimizer_parameters', 'exclusive_parameters', 'inference_parameters', 'u_update_parameters', 'weight_file_bytes', 'total_parameters', 'trainable_tensor_bytes'):
            if name in m:
                costs.setdefault(key, {}).setdefault(name, {})[seed] = m[name]

    for seed in seeds:
        folder = output / 'kuairand' / str(seed)
        for stage, weights in [('base', 'base.pt'), ('ssl', 'source.pt')]:
            m = add(folder / 'source' / stage, ['metrics.json', weights], f'{seed}/source/{stage}')
            if m is not None:
                source_counts[stage] += 1
                cost(stage, None, seed, m)
        m = add(folder / 'source-evaluation', ['metrics.json', 'predictions.npz'], f'{seed}/source-evaluation')
        if m is not None:
            if (m.get('paired_seed'), m.get('variant')) != (seed, 'mpt_base'):
                raise ValueError('Source evaluation pairing identity mismatch')
            base_metrics = records.get(f'{seed}/source/base', {})
            if base_metrics.get('model_hash') is not None and base_metrics['model_hash'] != m.get('model_hash'):
                raise ValueError('Evaluation/base source identity mismatch')
            source_counts['evaluation'] += 1
            for target in OLD_TASKS:
                collect('mpt_new', target, seed, m['test'][target])
        for kind, targets, variants in [('old', OLD_TASKS, ('head_only', 'conditional_readout')), ('new', NEW_TASKS, NEW_VARIANTS)]:
            for target in targets:
                for variant in variants:
                    key = f'{seed}/{kind}/{target}/{variant}'
                    m = add(folder / kind / target / variant, ['metrics.json', 'model.pt', 'predictions.npz'], key)
                    if m is None:
                        continue
                    expected_variant = 'h_conditional' if variant == 'head_only' else variant
                    if (m.get('paired_seed'), m.get('target'), m.get('variant')) != (seed, target, expected_variant):
                        raise ValueError(f'Reported model pairing identity mismatch: {key}')
                    base_hash = records.get(f'{seed}/source-evaluation', {}).get('model_hash')
                    if base_hash is None:
                        base_hash = records.get(f'{seed}/source/base', {}).get('model_hash')
                    u_hash = records.get(f'{seed}/source/ssl', {}).get('model_hash')
                    for field, expected_hash in [('base_hash', base_hash), ('u_hash', u_hash)]:
                        if m.get(field) is not None and expected_hash is not None and m[field] != expected_hash:
                            raise ValueError(f'Model dependency source identity mismatch: {key}/{field}')
                    model_count += 1
                    collect(variant, target, seed, m['test'])
                    cost(variant, target, seed, m)
                    if kind == 'old':
                        item = zero.setdefault(f'{variant}/{target}', dict(n=0, selected=0))
                        selected = m.get('zero_correction_selected', m.get('selected_zero_correction'))
                        if selected is None and 'selected_epoch' in m:
                            selected = m['selected_epoch'] == 0
                        if selected is not None:
                            item['n'] += 1
                            item['selected'] += int(bool(selected))
    for key, by_seed in series.items():
        available = [seed for seed in seeds if seed in by_seed]
        groups[key] = _stats([by_seed[seed] for seed in available], available)
    deltas = {}
    for target in OLD_TASKS:
        for metric in METRICS:
            for left, right in [('head_only', 'mpt_new'), ('conditional_readout', 'head_only')]:
                a = series.get(f'{left}/{target}/{metric}', {})
                b = series.get(f'{right}/{target}/{metric}', {})
                available = [s for s in seeds if s in a and s in b]
                if available:
                    deltas[f'{left}-minus-{right}/{target}/{metric}'] = _stats([a[s]-b[s] for s in available], available)
    costs = {key: {name: _stats([values[s] for s in seeds if s in values], [s for s in seeds if s in values]) for name, values in fields.items()} for key, fields in costs.items()}
    completed = model_count + source_counts['base'] + source_counts['ssl']
    result = dict(version=cfg.get('version'), completed_units=completed, expected_units=70,
                  completed_models=model_count, expected_models=60, source_counts=source_counts,
                  groups=groups, deltas=deltas, zero_correction=zero, costs=costs,
                  records=records, artifacts=artifacts, references=references,
                  updated_at=datetime.now(timezone.utc).isoformat())
    core.dump(output / 'summary.json', result)
    introduction = ['# KuaiRand normalized embedding L2', '', f'训练单元：{completed}/70；下游模型：{model_count}/60。', '', '测试集均值 ± 样本SD；var为样本方差；括号列出实际seed数。', '']
    if references:
        introduction += [f"历史参照：{references['version']}；{references['path']}；SHA256 {references['sha256']}。", '']
    diagnostic = introduction + _table(groups, OLD_TASKS, ('single_task', 'mpt_old', 'mpt_new'), 'B骨干诊断')
    lines = introduction + _table(groups, OLD_TASKS, ('single_task', 'mpt_new', 'head_only', 'conditional_readout'), '原任务')
    lines += _table(groups, NEW_TASKS, ('single_task', *NEW_VARIANTS), '新任务')
    lines += ['## 原任务配对差值', '', '| 比较 / 任务 / 指标 | 均值 ± SD；方差；n |', '|---|---:|']
    lines += [f'| {key} | {_cell(row)} |' for key, row in deltas.items()]
    lines += ['', '## 零修正选择', '', '| 模型 / 任务 | 选择次数 | 已记录数 |', '|---|---:|---:|']
    lines += [f"| {key} | {item['selected']} | {item['n']} |" for key, item in zero.items()]
    lines += ['', '## 训练成本与参数', '', '| 模型 / 任务 | 字段 | 均值 | n |', '|---|---|---:|---:|']
    lines += [f"| {key} | {name} | {row['mean']:.6f} | {row['n']} |" for key, fields in costs.items() for name, row in fields.items()]
    (output / 'tables.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    (output / 'backbone-diagnostic.md').write_text('\n'.join(diagnostic)+'\n', encoding='utf8')
    return result


def _table(groups, targets, models, title):
    labels = {'single_task': 'Single Task', 'mpt_old': 'MPT old (.006 sum)', 'mpt_new': 'MPT new (.006/N_E)', 'head_only': 'Head-only', 'conditional_readout': 'TCKA', 'mpt': 'MPT', 'tri_frozen': 'Frozen U', 'random_lora': 'Random U + LoRA', 'tri_lora': 'Pretrained U + LoRA'}
    columns = [(target, metric) for metric in METRICS for target in targets]
    lines = [f'## {title}', '', '| 模型 | ' + ' | '.join(f'{task} {metric}' for task, metric in columns) + ' |', '|---|' + '---:|'*len(columns)]
    for model in models:
        lines.append('| ' + ' | '.join([labels[model]] + [_cell(groups.get(f'{model}/{task}/{metric}')) for task, metric in columns]) + ' |')
    return lines + ['']
