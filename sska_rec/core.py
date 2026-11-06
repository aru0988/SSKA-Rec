import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from .vocabulary import CensusIncome_Vocabulary_Size
from .backbone import MPTRec, NewTask
from .expert import UniversalExpert
from .base_training import PairedTrainManager

DATA = Path("dataset")
CACHE = Path("cache/aliccp")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_split(n_test: int, split_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """按索引切分 test.gz，返回排序后的 (val_idx, test_idx)：与 train_test_split(dataset, ...,
    random_state=split_seed) 同一排列，但可落盘、可校验。"""
    val_idx, test_idx = train_test_split(
        np.arange(n_test), test_size=0.5, random_state=split_seed
    )
    return np.sort(val_idx), np.sort(test_idx)


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        a = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(a.dtype).encode())
        digest.update(str(tuple(a.shape)).encode())
        digest.update(a.numpy().tobytes())
    return digest.hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rank_auc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if not np.isfinite(p).all() or len(np.unique(y)) != 2:
        raise ValueError("AUC needs finite probabilities and both labels")
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_p)) + 1]
    ends = np.r_[starts[1:], len(p)]
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.repeat((starts + ends + 1) / 2, ends - starts)
    n1 = int(np.count_nonzero(y == 1))
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def measures(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=np.float64)
    q = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "auc": rank_auc(y, p),
        "logloss": float(-(y * np.log(q) + (1 - y) * np.log1p(-q)).mean()),
    }


def early_stop(curve, patience=5):
    best, stale = -math.inf, 0
    for value in curve:
        if value > best:
            best, stale = value, 0
        else:
            stale += 1
    return stale >= patience


class Selector(nn.Module):
    def __init__(self, width, fixed=False):
        super().__init__()
        self.fixed = fixed
        self.gate = nn.Sequential(nn.Linear(width, 32), nn.ReLU(), nn.Linear(32, 1))
        self.delta = nn.Sequential(nn.Linear(width, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, probability, features, enabled=True):
        if not enabled:
            return probability
        alpha = 1.0 if self.fixed else self.gate(features).sigmoid().flatten()
        logits = torch.logit(probability.clamp(1e-7, 1 - 1e-7))
        shifted = (logits + alpha * self.delta(features).flatten()).sigmoid()
        # Subtract the same floating-point reference before adding the residual:
        # zero delta exactly preserves p while retaining a nonzero delta gradient.
        return (probability + (shifted - logits.sigmoid())).clamp(0, 1)


class TaskPredictor(nn.Module):
    def __init__(self, original, universal, xdim, rdim, variant):
        super().__init__()
        self.core = original
        self.u = universal
        self.variant = variant
        self.selector = (
            None
            if variant == "no_u"
            else Selector(4 * rdim, fixed=variant == "fixed_fusion")
        )

    def forward(self, batch, envs):
        p = self.core(batch["x"], batch["g"], [batch["s0"], batch["s1"]], envs).reshape(
            -1
        )
        if self.selector is None:
            return p
        u = self.u(batch["x"])
        r = torch.cat([batch["g"], batch["s0"], batch["s1"], u], 1)
        return self.selector(p, r)


class Rows(Dataset):
    def __init__(self, features, labels, names, initial):
        self.x, self.y, self.names, self.initial = features, labels, names, initial

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        features = {key: self.x[i, k] for k, key in enumerate(self.names)}
        # Third field is ignored by the original two-task manager.
        return (
            self.y[i, self.initial[0]],
            self.y[i, self.initial[1]],
            self.y[i, 0],
            features,
        )


def census(continuous, initial, tiny=False):
    names = ["income", "marital", "education", "sex"]
    frames = [
        pd.read_csv(DATA / "Census-income" / f"{s}.gz") for s in ["train", "test"]
    ]
    all_y = [
        np.column_stack(
            [d.label_income, d.label_marital, (d.education == 9).astype(int), d.sex]
        ).astype("float32")
        for d in frames
    ]
    remove = ["label_income", "label_marital", "education"] + (
        ["sex"] if continuous else []
    )
    feature_names = [c for c in frames[0] if c not in remove]
    assert not set(feature_names) & {
        "income",
        "marital",
        "label_income",
        "label_marital",
        "education",
    }
    if continuous:
        assert "sex" not in feature_names
    vi, ti = make_split(len(frames[1]), 20260929)
    split_idx = {
        "train": (0, np.arange(len(frames[0]))),
        "val": (1, vi),
        "test": (1, ti),
    }
    vocab = {
        k: v for k, v in CensusIncome_Vocabulary_Size.items() if k in feature_names
    }
    xdim = len(feature_names) - len(vocab) + len(vocab) * 4
    sets = {}
    for split, (part, idx) in split_idx.items():
        if tiny:
            idx = idx[:1024]
        x = frames[part][feature_names].to_numpy()[idx]
        sets[split] = Rows(
            x, all_y[part][idx], feature_names, [names.index(t) for t in initial]
        )
    return {
        "sets": sets,
        "tasks": names,
        "initial": initial,
        "vocab": vocab,
        "xdim": xdim,
        "rdim": 128,
        "batch": 256,
        "lr": 0.001,
        "env_coe": 0.1,
        "hidden": [256, 128],
        "tower": [64, 32],
        "embed": 4,
        "widths": [1 for k in feature_names if k not in vocab] + [4] * len(vocab),
        "epochs1": 20,
        "epochs2": 30,
        "dataset": "census",
        "data_files": [
            str(DATA / "Census-income" / f"{s}.gz") for s in ["train", "test"]
        ],
    }


def aliccp(initial, tiny=False, data_hashes=None):
    # On-disk int32 arrays avoid 10M Python lists and all-split RAM residency.
    from . import aliccp_data as A

    limits = [10000000, 1000000, 10000000] if not tiny else [20000, 5000, 10000]
    sets, files = {}, []
    for split, count in zip(["train", "val", "test"], limits):
        path = DATA / A.DATA_FILES[split]
        files.append(str(path))
        dest = CACHE / f"aliccp-{split}-{count}.npy"
        dest.parent.mkdir(exist_ok=True)
        cache_meta = dest.with_suffix(".identity.json")
        fingerprint = {
            "file": str(path),
            "sha256": data_hashes[str(path)] if data_hashes else file_hash(path),
            "rows": count,
        }
        if dest.exists():
            if (
                not cache_meta.exists()
                or json.loads(cache_meta.read_text()) != fingerprint
            ):
                raise ValueError(f"AliCCP array cache identity mismatch: {dest}")
        if not dest.exists():
            head = pd.read_csv(path, nrows=1)
            width = len(head.columns)
            temp = dest.with_suffix(".partial.npy")
            mm = np.lib.format.open_memmap(
                temp, mode="w+", dtype="int32", shape=(count, width)
            )
            offset = 0
            for chunk in pd.read_csv(
                path, dtype="int32", chunksize=100000, nrows=count
            ):
                mm[offset : offset + len(chunk)] = chunk.to_numpy()
                offset += len(chunk)
            if offset != count:
                raise ValueError(f"Not enough AliCCP rows: {split} {offset}/{count}")
            mm.flush()
            del mm
            os.replace(temp, dest)
            dump(cache_meta, fingerprint)
        a = np.load(dest, mmap_mode="r")
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        feature_names = columns[3:-1]
        labels = np.column_stack([a[:, 0], a[:, 1], (a[:, -1] != 2)]).astype("float32")
        sets[split] = Rows(a[:, 3:-1], labels, feature_names, [0, 1])
        # User id 101 is excluded from the model but retained as evaluation grouping metadata.
        sets[split].user_groups = a[:, columns.index("101")]
    vocab = A.build_vocab()
    if set(feature_names) != set(vocab):
        raise ValueError("AliCCP vocabulary mismatch")
    return {
        "sets": sets,
        "tasks": ["CTR", "CVR", "BSI"],
        "initial": initial,
        "vocab": vocab,
        "xdim": 80,
        "rdim": 64,
        "batch": 2000,
        "lr": 0.0001,
        "env_coe": 0.3,
        "hidden": [128, 64],
        "tower": [32, 32],
        "embed": 5,
        "widths": [5] * 16,
        "epochs1": 30,
        "epochs2": 60,
        "dataset": "aliccp",
        "data_files": files,
    }


def build_base(spec):
    kw = {"dropout": [0.1, 0.3]} if spec["dataset"] == "aliccp" else {}
    return MPTRec(
        2,
        spec["vocab"],
        spec["embed"],
        spec["xdim"],
        spec["hidden"],
        spec["tower"],
        reg_embedding=0.006,
        reg_dnn=3e-5,
        device=DEVICE,
        **kw,
    ).to(DEVICE)


def batches(cache, batch, indices=None):
    n = len(cache["y"]) if indices is None else len(indices)
    for start in range(0, n, batch):
        idx = (
            slice(start, start + batch)
            if indices is None
            else indices[start : start + batch]
        )
        yield {k: v[idx].to(DEVICE) for k, v in cache.items()}


def make_u(spec, seed):
    return UniversalExpert(spec["widths"], rep_dim=spec["rdim"], seed=seed + 1).to(
        DEVICE
    )


def probabilities(model, cache, envs, spec, indices=None):
    model.eval()
    with torch.no_grad():
        return torch.cat(
            [
                model(b, envs).detach().cpu()
                for b in batches(cache, spec["batch"], indices)
            ]
        ).numpy()


def train_old(spec, caches, source, seed, variant, folder):
    folder.mkdir(parents=True, exist_ok=True)
    result = {}
    models = {}
    previous = (
        json.loads((folder / "metrics.json").read_text())
        if (folder / "metrics.json").exists()
        else {}
    )
    if source is not None:
        for p in source.parameters():
            p.requires_grad_(False)
    for pos, target in enumerate(spec["initial"]):
        seed_all(seed + 20 + pos)
        model = Selector(4 * spec["rdim"], fixed=variant == "fixed_fusion").to(DEVICE)

        def predict(cache):
            model.eval()
            ps = []
            with torch.no_grad():
                for b in batches(cache, spec["batch"]):
                    r = (
                        torch.cat([b["g"], b["s0"], b["s1"], source(b["x"])], 1)
                        if source is not None
                        else None
                    )
                    ps.append(
                        (
                            model(b["old"][:, pos], r)
                            if r is not None
                            else b["old"][:, pos]
                        ).cpu()
                    )
            return torch.cat(ps).numpy()

        idx = spec["tasks"].index(target)
        dest = folder / f"{target}.pt"
        enabled = variant != "no_u"
        if target in previous and (dest.exists() or variant == "no_u"):
            if dest.exists():
                model.load_state_dict(
                    torch.load(dest, map_location=DEVICE, weights_only=False)
                )
            result[target] = previous[target]
            models[target] = (model, source, pos, previous[target]["selected_enhanced"])
            continue
        records = []
        start = time.perf_counter()
        if enabled:
            if dest.exists():
                model.load_state_dict(
                    torch.load(dest, map_location=DEVICE, weights_only=False)
                )
            else:
                opt = torch.optim.Adam(model.parameters(), lr=spec["lr"])
                best = -math.inf
                stale = 0
                state = None
                for epoch in range(spec["epochs2"]):
                    model.train()
                    for b in batches(caches["train"], spec["batch"]):
                        with torch.no_grad():
                            r = torch.cat([b["g"], b["s0"], b["s1"], source(b["x"])], 1)
                        p = model(b["old"][:, pos], r)
                        loss = F.binary_cross_entropy(p, b["y"][:, idx])
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                    score = rank_auc(
                        caches["val"]["y"][:, idx].numpy(), predict(caches["val"])
                    )
                    records.append({"epoch": epoch + 1, "val_auc": score})
                    if score > best:
                        best, stale, state = score, 0, copy.deepcopy(model.state_dict())
                    else:
                        stale += 1
                    if stale >= 5:
                        break
                model.load_state_dict(state)
                torch.save(state, dest)
            y = caches["val"]["y"][:, idx].numpy()
            candidate = measures(y, predict(caches["val"]))
            baseline = measures(y, caches["val"]["old"][:, pos].numpy())
            enabled = (
                candidate["auc"] >= baseline["auc"]
                and candidate["logloss"] <= baseline["logloss"]
            )
        info = {
            "selected_enhanced": enabled,
            "history": records,
            "seconds": time.perf_counter() - start,
        }
        raw = {}
        for split in ["val", "test"]:
            y = caches[split]["y"][:, idx].numpy()
            p0 = caches[split]["old"][:, pos].numpy()
            pc = predict(caches[split]) if variant != "no_u" else p0
            p = pc if enabled else p0
            info[split] = measures(y, p)
            info[f"{split}_baseline"] = measures(y, p0)
            info[f"{split}_candidate"] = measures(y, pc)
            raw.update(
                {
                    f"{split}_y": y,
                    f"{split}_p": p,
                    f"{split}_candidate": pc,
                    f"{split}_baseline": p0,
                }
            )
        np.savez_compressed(folder / f"{target}-predictions.npz", **raw)
        result[target] = info
        models[target] = (model, source, pos, enabled)
    dump(folder / "metrics.json", result)
    return models, result
