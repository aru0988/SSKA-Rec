import os
from pathlib import Path


class ExclusiveLock:
    """OS lock is released on a crash; the file contents alone never claim liveness."""

    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("a+b")
        self.f.seek(0)
        self.f.write(b" ")
        self.f.flush()
        self.f.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            self.f.close()
            raise RuntimeError("Another worker holds the experiment lock")
        return self

    def __exit__(self, *args):
        self.f.close()
