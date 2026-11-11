import json
import os
from pathlib import Path

root = Path(__file__).resolve().parents[2] / "outputs/u-access"
status = root / "status.json"
if not status.exists():
    print(json.dumps({"state": "not_started"}))
else:
    d = json.loads(status.read_text())
    pid = d.get("pid")
    alive = False
    if pid:
        try:
            os.kill(pid, 0)
            alive = True
        except ProcessLookupError:
            pass
    d["alive"] = alive
    d["committed_models"] = sum(
        (root / str(seed) / name / "COMMITTED.json").exists()
        for seed in [1685480945, 1685463909, 1685477428, 1685459668, 1685496394]
        for name in ["mpt", "linear", "gated_linear", "original", "bottleneck"]
    )
    log = root.parent.parent / "runner.log"
    if log.exists():
        d["log_age_seconds"] = __import__("time").time() - log.stat().st_mtime
    print(json.dumps(d, ensure_ascii=False))
