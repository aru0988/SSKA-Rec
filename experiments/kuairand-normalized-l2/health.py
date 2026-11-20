"""Read queue state and observed worker log progress without starting work."""
import json
import os
import time
from pathlib import Path

root=Path(__file__).resolve().parents[2]
path=root/'outputs/kuairand-l2/status.json'
result=json.loads(path.read_text()) if path.exists() else {'state':'not_started'}
def alive(pid):
    if not pid:return False
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False
result['scheduler_alive']=alive(result.get('pid'))
for worker in result.get('running_jobs',[]):
    worker['alive']=alive(worker.get('pid'))
    log=Path(worker['log'])
    if log.exists():
        worker['log_age_seconds']=time.time()-log.stat().st_mtime
        with log.open('rb') as stream:
            stream.seek(max(0,log.stat().st_size-2500))
            worker['tail']=stream.read().decode('utf8',errors='replace')
print(json.dumps(result,ensure_ascii=False))
