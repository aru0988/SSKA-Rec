"""Read-only health for the exact three-dataset worker and its latest curve."""
import json
import os
import time
from pathlib import Path

root=Path(__file__).resolve().parents[2]
out=root/'outputs/tri'
status=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {'state':'not_started'}
pid=status.get('pid')
try:
    os.kill(pid,0) if pid else None
    alive=bool(pid)
except (ProcessLookupError,PermissionError):alive=False
status['alive']=alive
variants=['mpt','tri_frozen','tri_lora','tri_ssf','tri_full']
status['committed_models']=sum(1 for p in out.glob('*/*/*/*/COMMITTED.json') if p.parent.name in variants)
status['expected']=125
log=root/'runner.log'
status['log_age_seconds']=time.time()-log.stat().st_mtime if log.exists() else None
curves=list(out.glob('*/*/*/*/history-live.json'))+list(out.glob('*/*/linear/history-live.json'))
if curves:
    latest=max(curves,key=lambda p:p.stat().st_mtime)
    status['latest_curve']=str(latest.relative_to(root))
    status['curve_age_seconds']=time.time()-latest.stat().st_mtime
    history=json.loads(latest.read_text())
    status['latest_epoch']=history[-1] if history else None
print(json.dumps(status,ensure_ascii=False))
