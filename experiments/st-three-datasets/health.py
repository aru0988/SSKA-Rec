"""Read-only status of the queued/all-target SingleTask worker."""
import json
import os
import time
from pathlib import Path

root=Path(__file__).resolve().parents[2]
out=root/'outputs/st'
status=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {'state':'not_started'}
pid=status.get('pid')
try:
    os.kill(pid,0) if pid else None
    status['alive']=bool(pid)
except (ProcessLookupError,PermissionError):status['alive']=False
commits=list(out.glob('*/*/*/COMMITTED.json'))
status['committed_models']=len(commits)
status['reused_models']=sum((p.parent/'REUSED.json').exists() for p in commits)
status['expected']=55
curves=list(out.glob('*/*/*/history-live.json'))
if curves:
    latest=max(curves,key=lambda p:p.stat().st_mtime)
    status['latest_curve']=str(latest.relative_to(root));status['curve_age_seconds']=time.time()-latest.stat().st_mtime
    history=json.loads(latest.read_text());status['latest_epoch']=history[-1] if history else None
print(json.dumps(status,ensure_ascii=False))
