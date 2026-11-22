"""Gate new jobs on real host RAM usage without signalling training workers."""
import argparse
import fcntl
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path


GIB = 1024 ** 3
PAUSE_GIB = 48
RESUME_GIB = 44


def clean_file_cache_bytes(fields):
    legacy = 'total_inactive_file' in fields
    prefix = 'total_' if legacy else ''
    file_lru = fields.get(prefix + 'active_file', 0) + fields.get(prefix + 'inactive_file', 0)
    cache = fields.get('total_cache' if legacy else 'file', file_lru)
    shared = fields.get(prefix + 'shmem', 0)
    dirty = fields.get('total_dirty' if legacy else 'file_dirty', 0)
    writeback = fields.get('total_writeback' if legacy else 'file_writeback', 0)
    return max(0, min(file_lru, cache - shared) - dirty - writeback)


def memory_sample():
    for root, counter in (('/sys/fs/cgroup', 'memory.current'),
                          ('/sys/fs/cgroup/memory', 'memory.usage_in_bytes')):
        root = Path(root)
        usage = root/counter
        if usage.is_file():
            charged = int(usage.read_text())
            fields = {name: int(value) for name, value in
                      (row.split() for row in (root/'memory.stat').read_text().splitlines())}
            cached = min(charged, clean_file_cache_bytes(fields))
            return dict(charged_gib=charged/GIB, clean_file_cache_gib=cached/GIB,
                        protected_gib=(charged-cached)/GIB)
    raise RuntimeError('Cannot find the experiment cgroup memory counters')


def process_info(proc):
    try:
        fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return None
    if fields[0] != 'Z':
        return fields[19], fields[0]
    return None


def guard(launch, status):
    resume = json.loads(launch.read_text())
    pid = resume['pid']
    proc = Path(f'/proc/{pid}')
    if proc.stat().st_uid != os.geteuid():
        raise RuntimeError('Queue supervisor belongs to a different user')
    token = os.fsencode(resume['execution_script'])
    if token not in (proc/'cmdline').read_bytes().split(b'\0'):
        raise RuntimeError('Queue supervisor identity differs from resume receipt')
    initial = process_info(proc)
    if initial is None:
        raise RuntimeError('Queue supervisor has already exited')
    start = initial[0]
    shutdown = False
    owns_pause = False

    def stop(*_):
        nonlocal shutdown
        shutdown = True

    def current():
        info = process_info(proc)
        return info if info and info[0] == start else None

    def notify(sig):
        if current() is None:
            return False
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            return False
        print(f'{datetime.now(timezone.utc).isoformat()} {sig.name} queue PID {pid}', flush=True)
        return True

    def publish(state, sample):
        data = dict(state=state, runner_pid=pid, guard_pid=os.getpid(), pause_gib=PAUSE_GIB,
                    resume_gib=RESUME_GIB, updated_at=datetime.now(timezone.utc).isoformat(), **sample)
        tmp = status.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, indent=2)+'\n')
        tmp.replace(status)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    sample = {}
    try:
        while not shutdown and (info := current()):
            sample = memory_sample()
            if sample['protected_gib'] > PAUSE_GIB and not owns_pause and info[1] not in ('T', 't'):
                owns_pause = notify(signal.SIGSTOP)
            if owns_pause and sample['protected_gib'] < RESUME_GIB:
                notify(signal.SIGCONT)
                owns_pause = False
            publish('admission_paused' if owns_pause else 'watching', sample)
            time.sleep(5)
    finally:
        if owns_pause:
            notify(signal.SIGCONT)
        publish('guard_stopped' if current() else 'queue_exited', sample)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-receipt', type=Path, required=True)
    parser.add_argument('--status', type=Path, required=True)
    args = parser.parse_args()
    with args.status.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard(args.resume_receipt, args.status)


if __name__ == '__main__':
    main()
