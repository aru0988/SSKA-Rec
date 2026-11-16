"""Read status only. Never starts, repairs, or stops an experiment worker."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    records = []
    for path in sorted(a.output.glob('*/*/status.json')):
        value = json.loads(path.read_text(encoding='utf8'))
        pid, alive = value.get('pid'), False
        if pid:
            try:
                os.kill(pid, 0)
                cmd = Path(f'/proc/{pid}/cmdline')
                alive = (any(name in cmd.read_bytes() for name in
                    [b'sska_rec.scheme1_runner', b'sska_rec.scheme2_runner']) if cmd.exists() else True)
            except (OSError, ProcessLookupError):
                pass
        log = path.parent/'runner.log'
        records.append(dict(folder=str(path.parent), actual_pid=pid, alive=alive, status=value,
            status_age_seconds=round(time.time()-path.stat().st_mtime,1),
            log_age_seconds=round(time.time()-log.stat().st_mtime,1) if log.exists() else None))
    summary = a.output/'summary.json'
    result = dict(jobs=records)
    if summary.exists():
        data = json.loads(summary.read_text(encoding='utf8'))
        result['summary'] = {k:data[k] for k in ['scheme','completed','expected','updated_at']}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
