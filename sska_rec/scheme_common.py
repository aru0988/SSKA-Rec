"""Shared CLI, execution identity and immutable stage checks for the two schemes."""
import argparse
import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from . import core
from .locking import ExclusiveLock
from .training import _utc

PROJECT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def source_hashes():
    return {p.name: core.file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def validate_config(cfg, scheme):
    if cfg.get('scheme') != scheme:
        raise ValueError('Config scheme mismatch')
    old = read_json(PROJECT / 'configs/tri_three_datasets.json')
    if set(cfg['datasets']) != set(old['datasets']):
        raise ValueError('Three registered datasets are required')
    for dataset, previous in old['datasets'].items():
        for key, value in previous.items():
            if cfg['datasets'][dataset].get(key) != value:
                raise ValueError(f'Historical data/training protocol differs: {dataset}/{key}')
        if scheme == 1 and cfg['datasets'][dataset].get('decoder_epochs') != 30:
            raise ValueError('Scheme1 decoder_epochs must equal 30')
    expected = old['variants'] if scheme == 1 else ['conditional_readout']
    if cfg.get('variants') != expected:
        raise ValueError('Invalid variants; scheme2 has no stage2 variants')
    if cfg.get('interaction_rank', 4) != 4:
        raise ValueError('Scheme2 interaction rank is fixed at four')


def parser(scheme):
    p = argparse.ArgumentParser(description=f'Scheme {scheme}: ' +
        ('complementary U and new-task routing' if scheme == 1 else 'stage1 original-task frozen-U readout only'))
    p.add_argument('--config', type=Path, default=PROJECT / f'configs/scheme{scheme}_three_datasets.json')
    p.add_argument('--dataset-root', type=Path)
    p.add_argument('--source-root', type=Path, help='Audited existing outputs/tri; read-only')
    p.add_argument('--cache-root', type=Path, default=Path('cache'))
    p.add_argument('--output', type=Path, default=Path(f'outputs/scheme{scheme}'))
    p.add_argument('--dataset', choices=['census', 'kuairand', 'aliccp'])
    p.add_argument('--seed', type=int)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--report-only', action='store_true')
    if scheme == 1:
        p.add_argument('--pretrain-only', action='store_true')
    return p


@contextmanager
def short_lock(path, timeout=10):
    deadline = time.monotonic() + timeout
    while True:
        lock = ExclusiveLock(path)
        try:
            lock.__enter__()
            break
        except RuntimeError:
            if time.monotonic() > deadline:
                raise
            time.sleep(.05)
    try:
        yield
    finally:
        lock.__exit__(None, None, None)


def bind_execution(output, cfg, source_root, dataset_root):
    identity = dict(config=cfg, source=source_hashes(), source_root=str(Path(source_root).resolve()),
                    dataset_root=str(Path(dataset_root).resolve()))
    path = Path(output) / 'identity.json'
    with short_lock(Path(output) / '.identity.lock'):
        if path.exists() and read_json(path) != identity:
            raise ValueError('Execution identity changed; use a new output version')
        core.dump(path, identity)
    return digest(identity)


def bind_contract(path, value):
    path = Path(path)
    if path.exists() and read_json(path) != value:
        raise ValueError(f'Stage contract changed: {path}')
    core.dump(path, value)


def verify_stage(folder, required=()):
    folder = Path(folder).resolve()
    manifest = folder / 'COMMITTED.json'
    if not manifest.is_file():
        raise ValueError(f'Missing committed stage: {folder}')
    entries = read_json(manifest)
    if not set(required).issubset(entries):
        raise ValueError(f'Incomplete committed stage: {folder}')
    for name, sha in entries.items():
        p = (folder / name).resolve()
        if not p.is_relative_to(folder) or not p.is_file() or core.file_hash(p) != sha:
            raise ValueError(f'Committed integrity failure: {p}')
    return dict(path=str(folder), files=entries, committed_sha256=core.file_hash(manifest))


def status(path, state, **kwargs):
    core.dump(path, dict(state=state, pid=os.getpid() if state == 'running' else None,
                         updated_at=_utc(), **kwargs))


def selected_jobs(cfg, dataset=None, seed=None):
    found = []
    for name in cfg.get('order', list(cfg['datasets'])):
        if dataset is not None and name != dataset:
            continue
        for current in cfg['datasets'][name]['seeds']:
            if seed is None or current == seed:
                found.append((name, current))
    if not found:
        raise ValueError('Requested dataset/seed is not in the registered protocol')
    return found
