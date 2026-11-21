"""Run the validation gate and, if it passes, the matching formal queue."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from sska_rec.locking import ExclusiveLock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'dataset-root', 'cache-root', 'source-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    common = ['--config', str(args.config.resolve()), '--dataset-root', str(args.dataset_root.resolve()),
              '--cache-root', str(args.cache_root.resolve()), '--source-root', str(args.source_root.resolve())]
    env = dict(os.environ, SSKA_DEVICE='cpu', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with ExclusiveLock(out/'.supervisor.lock'):
        pilot = out/'pilot'
        subprocess.run([sys.executable, '-m', 'sska_rec.table2_v2_pilot',
                        *common, '--output', str(pilot)], check=True, env=env)
        state = json.loads((pilot/'decision.json').read_text(encoding='utf8'))['state']
        if state == 'review':
            print('Pilot review required. Formal queue is waiting.', flush=True)
            return
        if state != 'pass':
            raise ValueError(f'Unexpected pilot decision: {state}')
        subprocess.run([sys.executable, '-m', 'sska_rec.table2_v2_runner', *common,
                        '--output', str(out/'formal'), '--pilot-root', str(pilot)],
                       check=True, env=env)


if __name__ == '__main__':
    main()
