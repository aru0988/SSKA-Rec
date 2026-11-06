"""Train independent task copies from one frozen backbone and SSL source."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
import torch
from . import core
from .data import prepare_kuairand
from .locking import ExclusiveLock
from .training import train_seed


def kuairand_spec(manifest_path, cfg):
    m = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    names = m["features"]["names"]
    vocab = {k: v["cardinality"] for k, v in m["features"]["categorical"].items()}
    sets = {}
    for split in ["train", "val", "test"]:
        rows = core.Rows(
            np.load(root / f"{split}-x.npy", mmap_mode="c"),
            np.load(root / f"{split}-y.npy", mmap_mode="c"),
            names,
            [0, 1],
        )
        rows.row_ids = np.load(root / f"{split}-rowid.npy", mmap_mode="r")
        sets[split] = rows
    e = cfg["embedding_size"]
    numeric = [n for n in names if n not in vocab]
    return dict(
        sets=sets,
        tasks=["is_click", "is_like", "is_follow", "is_forward"],
        initial=["is_click", "is_like"],
        vocab=vocab,
        xdim=len(numeric) + len(vocab) * e,
        rdim=cfg["rdim"],
        batch=cfg["batch_size"],
        lr=cfg["learning_rate"],
        env_coe=cfg["env_coe"],
        hidden=cfg["hidden"],
        tower=cfg["tower"],
        embed=e,
        widths=[1] * len(numeric) + [e] * len(vocab),
        dataset="kuairand",
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    p.add_argument("--cache-root", type=Path, default=Path("cache"))
    p.add_argument("--output", type=Path, default=Path("outputs"))
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument(
        "--models",
        nargs="+",
        choices=["sska", "no_u", "random_u", "fixed_fusion", "single_task"],
    )
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    a = p.parse_args(argv)
    cfg = json.loads(a.config.read_text())
    core.DATA = a.dataset_root.resolve()
    core.CACHE = a.cache_root.resolve() / "aliccp"
    core.CACHE.mkdir(parents=True, exist_ok=True)
    core.DEVICE = torch.device(a.device)
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu for checks")
    torch.set_num_threads(4)
    dataset = cfg["dataset"]
    if dataset == "kuairand":
        k = cfg["kuairand"]
        keys = [
            "log_files",
            "categorical_features",
            "numeric_features",
            "split_fractions",
            "min_positive_per_split",
        ]
        manifest = prepare_kuairand(
            core.DATA / "KuaiRand-Pure/data",
            a.cache_root / "kuairand",
            {key: k[key] for key in keys},
        )
        m = json.loads(manifest.read_text())
        if {Path(v["path"]).name: v["sha256"] for v in m["inputs"]} != k[
            "expected_inputs_sha256"
        ]:
            raise ValueError("KuaiRand input identity mismatch")
        spec = kuairand_spec(manifest, k)
        data_id = m["preproc_hash"]
    else:
        spec = (
            core.census(True, ["income", "marital"])
            if dataset == "census"
            else core.aliccp(["CTR", "CVR"])
        )
        data_id = {str(path): core.file_hash(path) for path in spec["data_files"]}
    spec.update(epochs1=cfg["base_epochs"], epochs2=cfg["epochs"])
    if a.prepare_only:
        return
    seeds = a.seeds or cfg["seeds"]
    models = a.models or cfg["models"]
    source = {
        str(f.relative_to(Path(__file__).parent)): core.file_hash(f)
        for f in sorted(Path(__file__).parent.glob("*.py"))
    }
    a.output.mkdir(parents=True, exist_ok=True)
    with ExclusiveLock(a.output / "worker.lock"):
        core.dump(
            a.output / "environment.json",
            dict(
                python=sys.version,
                platform=platform.platform(),
                torch=torch.__version__,
                cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name() if a.device == "cuda" else None,
            ),
        )
        for seed in seeds:
            folder = a.output / dataset / str(seed)
            identity = dict(
                source=source, config=cfg, data=data_id, seed=seed, models=models
            )
            if (folder / "identity.json").exists() and json.loads(
                (folder / "identity.json").read_text()
            ) != identity:
                raise ValueError("Output identity mismatch; choose another output")
            core.dump(folder / "identity.json", identity)
            train_seed(core, spec, seed, folder, cfg["epochs"], models)


if __name__ == "__main__":
    main()
