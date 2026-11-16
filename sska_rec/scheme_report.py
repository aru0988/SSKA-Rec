"""Recompute committed probabilities and aggregate full paired seeds."""
from pathlib import Path

import numpy as np
from scipy.stats import t

from . import core
from .scheme_common import read_json, short_lock, verify_stage, digest
from .training import _utc


def resolve_artifacts(folder):
    folder = Path(folder)
    verify_stage(folder, ['metrics.json'])
    reference = folder / 'REUSED.json'
    if reference.exists():
        saved = read_json(reference)['artifacts']
        actual = verify_stage(saved['path'], ['metrics.json', 'model.pt', 'predictions.npz'])
        if actual != saved:
            raise ValueError('External artifacts changed after reuse')
        return Path(actual['path'])
    return folder


def statistics(values):
    a = np.asarray(values, dtype=float)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError('Statistics require finite completed values')
    variance = float(a.var(ddof=1)) if len(a) > 1 else None
    return dict(n=len(a), mean=float(a.mean()), variance=variance,
                sd=float(np.sqrt(variance)) if variance is not None else None, values=a.tolist())


def _provenance(output, folder, m, scheme, dataset, seed, target, variant, cfg):
    if scheme == 1:
        contract = read_json(folder.parent / f'{variant}-contract.json')
        source = read_json(output / dataset / str(seed) / 'complementary-source.json')
        if (contract.get('scheme'), contract.get('seed'), contract.get('target'), contract.get('variant')) != (1, seed, target, variant):
            raise ValueError('Scheme1 report contract identity mismatch')
        expected_u = source['u_model_hash'] if variant != 'mpt' else None
        if contract.get('u_hash') != expected_u or contract.get('base_hash') != source['base_source']['model_hashes']['base']:
            raise ValueError('Scheme1 report source dependency mismatch')
        data = source['base_source']['data_signature']
        if contract.get('data') != digest_data(data):
            raise ValueError('Scheme1 report data identity mismatch')
        execution_id = contract['config'].get('execution_identity')
        rows = {s:v['rows'] for s,v in data['splits'].items()}
    else:
        contract = read_json(folder / 'contract.json')
        if contract != m.get('contract') or (contract.get('stage'), contract.get('dataset'), contract.get('target')) != ('conditional_readout', dataset, target):
            raise ValueError('Scheme2 report contract identity mismatch')
        dependencies = contract['dependencies']
        if (m.get('base_hash'),m.get('u_hash')) != (dependencies['base_hash'],dependencies['u_hash']):
            raise ValueError('Scheme2 report source dependency mismatch')
        rows = {s:v['rows'] for s,v in contract['data']['splits'].items()}
        execution_id = contract.get('execution_identity')
    identity_path = output / 'identity.json'
    if identity_path.exists():
        identity = read_json(identity_path)
        if identity.get('config') != cfg or execution_id != digest(identity):
            raise ValueError('Report execution/config identity mismatch')
    for split,n in cfg['datasets'][dataset].get('sizes',{}).items():
        if rows.get(split) != n:
            raise ValueError('Reported data is not the registered full split')
    return rows


def digest_data(data):
    from .scheme_reuse import canonical_data
    return digest(canonical_data(data))


def report(output, cfg, scheme):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with short_lock(output / '.report.lock'):
        return _report(output, cfg, scheme)


def _report(output, cfg, scheme):
    records, expected, checks = {}, 0, 0
    summaries = {}
    lines = [f'# 方案{scheme}实验汇总', '', '指标为测试集均值±样本SD；不足5seed时标明完成数。', '']
    for dataset, dc in cfg['datasets'].items():
        targets = dc['targets'] if scheme == 1 else dc['initial']
        variants = cfg.get('variants', ['conditional_readout']) if scheme == 1 else ['conditional_readout']
        expected += len(targets) * len(variants) * len(dc['seeds'])
        local = {}
        aligned = {}
        for seed in dc['seeds']:
            for target in targets:
                for variant in variants:
                    folder = output / dataset / str(seed)
                    folder = folder / target / variant if scheme == 1 else folder / 'old' / target
                    if not (folder / 'COMMITTED.json').is_file():
                        continue
                    artifact = resolve_artifacts(folder)
                    m = read_json(folder / 'metrics.json')
                    if (m.get('paired_seed'), m.get('target'), m.get('variant')) != (seed, target, variant):
                        raise ValueError('Reported model pairing identity mismatch')
                    expected_rows = _provenance(output, folder, m, scheme, dataset, seed, target, variant, cfg)
                    with np.load(artifact / 'predictions.npz', allow_pickle=False) as raw:
                        for split in ['val', 'test']:
                            y, p, ids = [raw[f'{split}_{key}'] for key in ['y', 'p', 'rowid']]
                            if y.shape != p.shape or y.shape != ids.shape or len(y) != expected_rows[split]:
                                raise ValueError('Raw prediction shape mismatch')
                            actual = core.measures(y, p)
                            if any(abs(actual[k] - m[split][k]) > 1e-10 for k in actual):
                                raise ValueError('Raw metric mismatch')
                            # AUC/logloss all use the same paired evaluation rows.
                            import hashlib
                            pair = (hashlib.sha256(y.tobytes()).hexdigest(), hashlib.sha256(ids.tobytes()).hexdigest())
                            key = (seed, target, split)
                            if key in aligned and aligned[key] != pair:
                                raise ValueError('Paired raw labels/rowids differ')
                            aligned[key] = pair
                            if scheme == 2:
                                base = core.measures(y, raw[f'{split}_base_p'])
                                if any(abs(base[k] - m['baseline'][split][k]) > 1e-10 for k in base):
                                    raise ValueError('Baseline raw metric mismatch')
                            checks += 1
                    records[f'{dataset}/{seed}/{target}/{variant}'] = m
                    local[seed, target, variant] = m
        summaries[dataset] = {}
        for metric in ['auc', 'logloss']:
            lines += [f'## {dataset} {metric}', '', '| 模型 | ' + ' | '.join(targets) + ' |',
                      '|---|' + '---:|' * len(targets)]
            for variant in variants:
                cells = []
                for target in targets:
                    available = [s for s in dc['seeds'] if (s, target, variant) in local]
                    if not available:
                        cells.append('0/5')
                        continue
                    values = [local[s, target, variant]['test'][metric] for s in available]
                    item = statistics(values)
                    deltas = []
                    for seed in available:
                        if scheme == 2:
                            reference = local[seed, target, variant]['baseline']['test'][metric]
                        elif (seed, target, 'mpt') in local:
                            reference = local[seed, target, 'mpt']['test'][metric]
                        else:
                            continue
                        deltas.append(local[seed, target, variant]['test'][metric] - reference)
                    if deltas:
                        delta = statistics(deltas)
                        margin = float(t.ppf(.975, len(deltas)-1) * delta['sd'] / np.sqrt(len(deltas))) if len(deltas) > 1 else None
                        delta.update(low=delta['mean']-margin if margin is not None else None,
                                     high=delta['mean']+margin if margin is not None else None)
                        item['vs_mpt'] = delta
                    summaries[dataset][f'{variant}/{target}/{metric}'] = item
                    value = f"{item['mean']:.6f} ± {item['sd']:.6f}" if item['sd'] is not None else f"{item['mean']:.6f}"
                    cells.append(value + (f' ({len(available)}/5)' if len(available) != 5 else ''))
                lines.append('| ' + ' | '.join([variant] + cells) + ' |')
            lines.append('')
        lines += ['| 模型 | 目标 | 完成 | 平均训练秒 | 平均峰值MiB | 成本来源 |', '|---|---|---:|---:|---:|---|']
        for variant in variants:
            for target in targets:
                ms = [m for (s, name, v), m in local.items() if name == target and v == variant]
                if ms:
                    sources = ','.join(sorted({m.get('cost_measurement_source','current-scheme-concurrent') for m in ms}))
                    lines.append(f"| {variant} | {target} | {len(ms)}/5 | {np.mean([m.get('seconds',0) for m in ms]):.3f} | {np.mean([m.get('peak_cuda_bytes',0)/1024**2 for m in ms]):.3f} | {sources} |")
        lines.append('')
    result = dict(scheme=scheme, completed=len(records), expected=expected, raw_split_checks=checks,
                  updated_at=_utc(), summary=summaries, records=records)
    core.dump(output / 'summary.json', result)
    (output / 'table.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    return result
