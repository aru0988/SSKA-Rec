"""Launch this registered queue once on the existing GPUHub instance."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timezone

root=Path(__file__).resolve().parents[2]
out=root/'outputs/kuairand-l2'
out.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
with (out/'launch.lock').open('w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=out/'status.json'
    receipt=root/'training-launch.json'
    for p in [status,receipt]:
        if p.exists():
            old=json.loads(p.read_text());pid=old.get('pid')
            if pid:
                try:
                    cmdline=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
                    if 'sska_rec.kuairand_l2_runner' in cmdline:
                        raise RuntimeError(f'Worker already running: {pid}')
                except FileNotFoundError:pass
    cfg=root/'configs/kuairand_normalized_l2_20261120.json'
    refs=root/'references.json'
    if not refs.is_file():raise ValueError('Missing audited references')
    env=dict(os.environ,SSKA_DEVICE='cpu',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
             OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    args=[sys.executable,'-m','sska_rec.kuairand_l2_runner','--config',str(cfg),
          '--dataset-root',str(Path.home()/'shared/dataset'),'--cache-root',str(Path.home()/'shared/cache'),
          '--output',str(out),'--references',str(refs),'--workers','4','--threads','2','--device','cuda:0']
    with (root/'runner.log').open('ab') as log,(root/'runner.err.log').open('ab') as err:
        process=subprocess.Popen(args,cwd=root,env=env,stdin=subprocess.DEVNULL,
                                 stdout=log,stderr=err,start_new_session=True)
    result=dict(pid=process.pid,command=args,cwd=str(root),started_at=datetime.now(timezone.utc).isoformat(),
                config_sha256=sha(cfg),references_sha256=sha(refs),
                code={p.name:sha(p) for p in sorted((root/'sska_rec').glob('*.py'))},
                concurrency=dict(workers=4,threads_per_worker=2),
                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),
                cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
                memory_limit=Path('/sys/fs/cgroup/memory.max').read_text().strip())
    receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    time.sleep(2)
    if process.poll() is not None:raise RuntimeError('Queue exited; inspect runner.err.log')
    print(json.dumps(result,ensure_ascii=False))
