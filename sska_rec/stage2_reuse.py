"""Read the committed U-access sources without training or changing artifacts."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import core
from .u_access import AccessHead


EXPECTED_SIZES = {"train": 20000, "val": 5000, "test": 10000}
RAW_SHA256 = {
    "train.gz": "e2f2d57925a8020d23fcd05802284b3bc1d91fcdd451f1eaaefd683e76d1b2dc",
    "test.gz": "8de690375028ac531de37fcbb117bd6527c5f3542d3bfead8975226e45100cec",
}


def _read(path):
    return json.loads(path.read_text(encoding="utf8"))


def _row_hash(ids):
    return hashlib.sha256(ids.tobytes()).hexdigest()


def _verify_loaded_sources(hashes):
    # The code that interprets checkpoints and data must match the old run too.
    for name, digest in hashes.items():
        path = Path(core.__file__).parent / name
        if not path.is_file() or core.file_hash(path) != digest:
            raise ValueError(f"Loaded source hash mismatch: {path}")


def _verify_stage(folder, required):
    manifest_path = folder / "COMMITTED.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing committed source stage: {folder}")
    manifest = _read(manifest_path)
    if not set(required) <= set(manifest):
        raise ValueError(f"Incomplete committed source stage: {folder}")
    for name, digest in manifest.items():
        path = (folder / name).resolve()
        if (
            not path.is_relative_to(folder.resolve())
            or not path.is_file()
            or core.file_hash(path) != digest
        ):
            raise ValueError(f"Committed stage integrity failure: {path}")
    return dict(
        path=str(folder), manifest_sha256=core.file_hash(manifest_path), files=manifest
    )


def _subset(source_root, dataset_root, identity):
    cfg, data = identity["protocol"], identity["data"]
    if (
        cfg.get("protocol") != "census-original-task-u-access-v1-20261111"
        or cfg.get("sizes") != EXPECTED_SIZES
        or cfg.get("split_seed") != 20260929
        or cfg.get("subset_seed") != 20261111
        or cfg.get("input_dim") != 119
        or cfg.get("batch_size") != 256
        or cfg.get("validation_fallback") is not False
    ):
        raise ValueError("Old source protocol mismatch")
    if _read(source_root / "subset/subset.json") != data:
        raise ValueError("Subset identity mismatch")
    if data.get("split_seed") != 20260929 or data.get("subset_seed") != 20261111:
        raise ValueError("Subset seed identity mismatch")
    raw = {
        name: core.file_hash(dataset_root / "Census-income" / name)
        for name in RAW_SHA256
    }
    if raw != RAW_SHA256 or raw != data["raw_sha256"]:
        raise ValueError("Census raw dataset hash mismatch")
    core.DATA = dataset_root
    spec = core.census(True, ["income", "marital"])
    if (
        spec["tasks"] != ["income", "marital", "education", "sex"]
        or spec["xdim"] != 119
    ):
        raise ValueError("Four-target Census schema mismatch")
    ntrain = len(spec["sets"]["train"])
    vi, ti = core.make_split(
        len(spec["sets"]["val"]) + len(spec["sets"]["test"]), 20260929
    )
    all_ids = dict(train=np.arange(ntrain), val=ntrain + vi, test=ntrain + ti)
    with np.load(source_root / "subset/indices.npz", allow_pickle=False) as saved:
        for split, rows in spec["sets"].items():
            ix = saved[split + "_indices"]
            ids = saved[split + "_rowid"]
            if (
                ix.ndim != 1
                or ix.dtype.kind not in "iu"
                or len(ix) != EXPECTED_SIZES[split]
                or len(np.unique(ix)) != len(ix)
                or np.any(ix < 0)
                or np.any(ix >= len(rows))
                or np.any(np.diff(ix.astype("int64")) <= 0)
            ):
                raise ValueError(f"Invalid subset indices: {split}")
            info = data["splits"][split]
            if (
                not np.array_equal(ids, all_ids[split][ix])
                or _row_hash(ids) != info["rowid_sha256"]
                or info["n"] != len(ix)
            ):
                raise ValueError(f"Subset rowid mismatch: {split}")
            new = core.Rows(rows.x[ix], rows.y[ix], rows.names, [0, 1])
            new.row_ids = ids.copy()
            counts = np.bincount(
                (new.y[:, 0] * 2 + new.y[:, 1]).astype(int), minlength=4
            ).tolist()
            if (
                new.y.shape != (len(ix), 4)
                or list(rows.names) != data["features"]
                or counts != info["joint_label_counts"]
            ):
                raise ValueError(f"Subset labels/features mismatch: {split}")
            spec["sets"][split] = new
    spec.update(
        batch=cfg["batch_size"],
        lr=cfg["learning_rate"],
        epochs1=cfg["base_epochs"],
        epochs2=cfg["access_epochs"],
    )
    return spec


def _freeze(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def original_predictions(base, u, linear_heads, caches, spec, split):
    """Reproduce the old two-task linear U path in its original batch order."""
    predictions = []
    with torch.no_grad():
        for batch in core.batches(caches[split], spec["batch"]):
            features = u(batch["x"])
            predictions.append(
                torch.stack(
                    [
                        linear_heads[i](
                            dict(u=features, p=batch["old"][:, i]),
                            base.tower_networks[i],
                        )
                        for i in range(2)
                    ],
                    1,
                ).cpu()
            )
    return torch.cat(predictions).numpy()


def load_reused_seed(source_root, dataset_root, seed, device):
    """Return the fixed subset, frozen sources, mmap caches and provenance receipt."""
    source_root, dataset_root = (
        Path(source_root).resolve(),
        Path(dataset_root).resolve(),
    )
    identity = _read(source_root / "identity.json")
    if seed not in identity["protocol"]["seeds"]:
        raise ValueError(f"Seed absent from old source protocol: {seed}")
    if not identity["source"]:
        raise ValueError("Missing old source hashes")
    for name, digest in identity["source"].items():
        path = source_root.parents[1] / "sska_rec" / name
        if not path.is_file() or core.file_hash(path) != digest:
            raise ValueError(f"Old source hash mismatch: {path}")
    _verify_loaded_sources(identity["source"])
    core.DEVICE = torch.device(device)
    spec = _subset(source_root, dataset_root, identity)
    folder = source_root / str(seed)
    stages = {
        "base": _verify_stage(folder / "source/base", ["base.pt", "metrics.json"]),
        "u": _verify_stage(folder / "source/ssl", ["source.pt", "metrics.json"]),
        "linear": _verify_stage(
            folder / "linear", ["model.pt", "metrics.json", "predictions.npz"]
        ),
        "cache": _verify_stage(
            folder / "feature-cache",
            [
                "identity.json",
                *[
                    f"{split}-{key}.npy"
                    for split in spec["sets"]
                    for key in ["x", "g", "s0", "s1", "old"]
                ],
            ],
        ),
    }
    base = core.build_base(spec)
    base.load_state_dict(
        torch.load(
            folder / "source/base/base.pt", map_location=core.DEVICE, weights_only=True
        )
    )
    u = core.make_u(spec, seed)
    u.load_state_dict(
        torch.load(
            folder / "source/ssl/source.pt", map_location=core.DEVICE, weights_only=True
        )
    )
    heads = nn.ModuleList([AccessHead("linear"), AccessHead("linear")]).to(core.DEVICE)
    heads.load_state_dict(
        torch.load(
            folder / "linear/model.pt", map_location=core.DEVICE, weights_only=True
        )
    )
    hashes = dict(
        base=core.model_hash(base), u=core.model_hash(u), linear=core.model_hash(heads)
    )
    if (
        hashes["base"] != _read(folder / "source/base/metrics.json")["model_hash"]
        or hashes["u"] != _read(folder / "source/ssl/metrics.json")["model_hash"]
    ):
        raise ValueError("Source model hash mismatch")
    linear_metrics = _read(folder / "linear/metrics.json")
    if (
        hashes["base"] != linear_metrics["frozen_base_hash"]
        or hashes["u"] != linear_metrics["frozen_u_hash"]
    ):
        raise ValueError("Linear source dependencies mismatch")
    base, u, heads = map(_freeze, [base, u, heads])
    cache_root = folder / "feature-cache"
    expected_cache_identity = dict(
        base_hash=hashes["base"],
        splits={
            s: dict(rows=len(rows), rowid_sha256=_row_hash(rows.row_ids))
            for s, rows in spec["sets"].items()
        },
    )
    if _read(cache_root / "identity.json") != expected_cache_identity:
        raise ValueError("Feature cache base/rowid identity mismatch")
    caches = {}
    for split, rows in spec["sets"].items():
        caches[split] = {}
        for key, width in dict(
            x=spec["xdim"], g=spec["rdim"], s0=spec["rdim"], s1=spec["rdim"], old=2
        ).items():
            array = np.load(
                cache_root / f"{split}-{key}.npy", mmap_mode="c", allow_pickle=False
            )
            if array.shape != (len(rows), width) or array.dtype != np.float32:
                raise ValueError(f"Feature cache schema mismatch: {split}-{key}")
            caches[split][key] = torch.from_numpy(array)
        caches[split]["y"] = torch.from_numpy(rows.y)
    with torch.no_grad():
        envs = [
            base.env_embedding_network(torch.tensor(i, device=core.DEVICE)).detach()
            for i in range(2)
        ]
    raw_path = folder / "linear/predictions.npz"
    with np.load(raw_path, allow_pickle=False) as raw:
        for split in ["val", "test"]:
            if not np.array_equal(
                raw[split + "_rowid"], spec["sets"][split].row_ids
            ) or not np.array_equal(raw[split + "_y"], spec["sets"][split].y[:, :2]):
                raise ValueError(f"Original prediction rowid/label mismatch: {split}")
    receipt = dict(
        source_root=str(source_root),
        dataset_root=str(dataset_root),
        seed=seed,
        identity_sha256=core.file_hash(source_root / "identity.json"),
        subset_indices_sha256=core.file_hash(source_root / "subset/indices.npz"),
        source_hashes=identity["source"],
        raw_sha256=identity["data"]["raw_sha256"],
        model_hashes=hashes,
        stages=stages,
        original_predictions_npz=str(raw_path),
        dependencies=dict(
            base=["subset", "raw_dataset"],
            u=["base", "cache"],
            linear=["base", "u", "cache"],
            new_stages=["base", "u", "linear", "cache"],
        ),
        split_rowid_sha256={s: _row_hash(r.row_ids) for s, r in spec["sets"].items()},
    )
    return spec, base, u, heads, caches, envs, receipt
