import json
import os
import time
from pathlib import Path

project = Path(__file__).resolve().parents[2]
root = project / "outputs/u-peft"
path = root / "status.json"
if path.exists():
    d = json.loads(path.read_text())
    pid = d.get("pid")
    d["alive"] = False
    if pid:
        try:
            os.kill(pid, 0)
            d["alive"] = True
        except ProcessLookupError:
            pass
    cfg = json.loads((project / "configs/u_peft_census.json").read_text())
    d["committed_models"] = sum(
        (root / str(seed) / target / variant / "COMMITTED.json").exists()
        for seed in cfg["seeds"]
        for target in cfg["targets"]
        for variant in cfg["variants"]
    )
    log = project / "runner.log"
    if log.exists():
        d["log_age_seconds"] = time.time() - log.stat().st_mtime
    print(json.dumps(d))
else:
    print(json.dumps({"state": "not_started"}))
