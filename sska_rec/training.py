"""Shared source preparation and isolated continual task training."""

import copy
import json
import os
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _rowids(rows):
    return np.asarray(getattr(rows, "row_ids", np.arange(len(rows))))


def _save(state, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def _committed(core, folder):
    manifest = folder / "COMMITTED.json"
    if manifest.exists():
        for name, digest in json.loads(manifest.read_text()).items():
            path = folder / name
            if not path.exists() or core.file_hash(path) != digest:
                raise ValueError(f"Committed stage integrity failure: {path}")
        return True
    if folder.exists() and any(folder.iterdir()):
        # Preserve the interrupted attempt, then restart only this leaf stage.
        archive = folder.with_name(
            folder.name
            + ".interrupted-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
        folder.rename(archive)
        print(f"Restarting interrupted stage {folder}; preserved {archive}", flush=True)
    folder.mkdir(parents=True, exist_ok=True)
    return False


def _commit(core, folder):
    core.dump(
        folder / "COMMITTED.json",
        {
            str(p.relative_to(folder)): core.file_hash(p)
            for p in sorted(folder.rglob("*"))
            if p.is_file() and p.name != "COMMITTED.json"
        },
    )


class _Target(Dataset):
    def __init__(self, rows, idx):
        self.rows, self.idx = rows, idx

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i][-1], self.rows.y[i, self.idx]


def _fit(core, model, folder, budget, train_batches, predict, regularization=lambda: 0):
    if _committed(core, folder):
        model.load_state_dict(
            torch.load(
                folder / "model.pt", map_location=core.DEVICE, weights_only=False
            )
        )
        result = json.loads((folder / "metrics.json").read_text())
        if core.model_hash(model) != result["model_hash"]:
            raise ValueError("Model hash mismatch")
        return result
    if budget < 1:
        raise ValueError("Positive epoch budget required")
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=train_batches.lr
    )
    best = -float("inf")
    stale = 0
    state = None
    history = []
    started = time.perf_counter()
    started_at = _utc()
    if core.DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(core.DEVICE)
    for epoch in range(budget):
        model.train()
        losses = []
        for prediction, y in train_batches(model):
            loss = (
                F.binary_cross_entropy(prediction.reshape(-1), y.float())
                + regularization()
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            y, p = predict(model, "val")
        score = core.measures(y, p)
        history.append(
            dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), **score)
        )
        print(f'{folder.name} epoch={epoch+1} val_auc={score["auc"]:.8f}', flush=True)
        if score["auc"] > best:
            best, stale, state = score["auc"], 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= 5:
            break
    model.load_state_dict(state)
    model.eval()
    result = dict(
        history=history,
        epochs=len(history),
        selected_epoch=max(history, key=lambda r: r["auc"])["epoch"],
        stop_reason="patience" if stale >= 5 else "budget",
        seconds=time.perf_counter() - started,
        total_parameters=sum(p.numel() for p in model.parameters()),
        optimizer_parameters=sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        peak_cuda_bytes=(
            torch.cuda.max_memory_allocated(core.DEVICE)
            if core.DEVICE.type == "cuda"
            else 0
        ),
        model_hash=core.model_hash(model),
    )
    result.update(started_at=started_at, seed=getattr(train_batches, "seed", None))
    raw = {}
    with torch.no_grad():
        for split in ["val", "test"]:
            y, p = predict(model, split)
            result[split] = core.measures(y, p)
            raw[split + "_y"] = y
            raw[split + "_p"] = p
            rowids = getattr(predict, "row_ids", {}).get(split, np.arange(len(y)))
            if len(rowids) != len(y):
                raise ValueError("Prediction row-id mismatch")
            raw[split + "_rowid"] = rowids
    _save(state, folder / "model.pt")
    with open(folder / "predictions.npz.tmp", "wb") as f:
        np.savez_compressed(f, **raw)
    os.replace(folder / "predictions.npz.tmp", folder / "predictions.npz")
    result["ended_at"] = _utc()
    core.dump(folder / "metrics.json", result)
    _commit(core, folder)
    return result


def train_single(core, spec, seed, target, folder, budget):
    from .backbone import SingleTask

    core.seed_all(seed)
    model = SingleTask(
        spec["vocab"],
        spec["embed"],
        spec["xdim"],
        spec["hidden"],
        spec["tower"],
        spec.get("reg_embedding", 0.006)
        / (
            sum(spec["vocab"].values()) * spec["embed"]
            if spec["dataset"] == "kuairand"
            else 1
        ),
        spec.get("reg_dnn", 3e-5),
    ).to(core.DEVICE)
    idx = spec["tasks"].index(target)
    loaders = {
        s: DataLoader(_Target(rows, idx), batch_size=spec["batch"], shuffle=False)
        for s, rows in spec["sets"].items()
    }

    def train(m):
        for x, y in loaders["train"]:
            yield m({k: v.to(core.DEVICE) for k, v in x.items()})[0], y.to(core.DEVICE)

    train.lr = spec["lr"]
    train.seed = seed

    def pred(m, split):
        ps = []
        ys = []
        for x, y in loaders[split]:
            ps.append(
                m({k: v.to(core.DEVICE) for k, v in x.items()})[0].reshape(-1).cpu()
            )
            ys.append(y)
        return torch.cat(ys).numpy(), torch.cat(ps).numpy()

    pred.row_ids = {s: _rowids(rows) for s, rows in spec["sets"].items()}
    return _fit(core, model, Path(folder), budget, train, pred, model.get_l2_reg)


def _feature_cache(core, base, spec, folder):
    """Persist one streaming, memory-mapped representation of each split."""
    path = folder / "feature-cache"
    identity = {
        "base_hash": core.model_hash(base),
        "splits": {
            s: {
                "rows": len(r),
                "rowid_sha256": hashlib.sha256(_rowids(r).tobytes()).hexdigest(),
            }
            for s, r in spec["sets"].items()
        },
    }
    ready = _committed(core, path)
    if ready and json.loads((path / "identity.json").read_text()) != identity:
        raise ValueError("Feature cache base/row identity mismatch")
    if not ready:
        start = time.perf_counter()
        started_at = _utc()
        with torch.no_grad():
            for split, rows in spec["sets"].items():
                widths = {
                    "x": spec["xdim"],
                    "g": spec["rdim"],
                    "s0": spec["rdim"],
                    "s1": spec["rdim"],
                    "old": 2,
                }
                arrays = {
                    k: np.lib.format.open_memmap(
                        path / f"{split}-{k}.npy",
                        mode="w+",
                        dtype="float32",
                        shape=(len(rows), w),
                    )
                    for k, w in widths.items()
                }
                offset = 0
                for *_, features in DataLoader(
                    rows, batch_size=spec["batch"], shuffle=False
                ):
                    features = {k: v.to(core.DEVICE) for k, v in features.items()}
                    x, g, s, _ = base.get_infos(features)
                    old = torch.stack(base.predict(features), 1)
                    for k, v in zip(["x", "g", "s0", "s1", "old"], [x, g, *s, old]):
                        arrays[k][offset : offset + len(x)] = v.cpu().numpy()
                    offset += len(x)
                if offset != len(rows):
                    raise ValueError("Incomplete feature cache")
                for value in arrays.values():
                    value.flush()
                del value, arrays
        core.dump(path / "identity.json", identity)
        core.dump(
            path / "cost.json",
            dict(
                seconds=time.perf_counter() - start,
                started_at=started_at,
                ended_at=_utc(),
            ),
        )
        _commit(core, path)
    caches = {
        s: {
            k: torch.from_numpy(np.load(path / f"{s}-{k}.npy", mmap_mode="c"))
            for k in ["x", "g", "s0", "s1", "old"]
        }
        for s in spec["sets"]
    }
    for split, rows in spec["sets"].items():
        caches[split]["y"] = torch.from_numpy(rows.y)
    envs = [
        base.env_embedding_network(torch.tensor(k, device=core.DEVICE)).detach()
        for k in range(2)
    ]
    return caches, envs


def _prepare(core, spec, seed, folder, budget):
    """One fixed B and one offline SSL U0; no label expert or probes."""
    source_dir = folder / "source"
    base_dir = source_dir / "base"
    ssl_dir = source_dir / "ssl"
    ready = _committed(core, base_dir)
    core.seed_all(seed)
    base = core.build_base(spec)
    if ready:
        base.load_state_dict(
            torch.load(
                base_dir / "base.pt", map_location=core.DEVICE, weights_only=False
            )
        )
    else:
        started = time.perf_counter()
        started_at = _utc()
        loaders = {
            s: DataLoader(r, batch_size=spec["batch"], shuffle=False)
            for s, r in spec["sets"].items()
        }
        env = torch.randint(
            0,
            2,
            (len(spec["sets"]["train"]),),
            generator=torch.Generator().manual_seed(20260929),
        )
        manager = core.PairedTrainManager(
            base,
            loaders["train"],
            loaders["val"],
            env,
            spec["initial"],
            spec["lr"],
            spec["batch"],
            0.9,
            spec["env_coe"],
            epochs=budget,
            patience=5,
        )
        manager.train_two_task()
        base.load_state_dict(manager.best_weight)
        _save(base.state_dict(), base_dir / "base.pt")
        records = manager.epoch_records
        selected = max(records, key=lambda r: r["auc_val_0"] + r["auc_val_1"])["epoch"]
        scores = [r["auc_val_0"] + r["auc_val_1"] for r in records]
        core.dump(
            base_dir / "metrics.json",
            dict(
                history=records,
                epochs=len(records),
                selected_epoch=selected,
                stop_reason="patience" if core.early_stop(scores, 5) else "budget",
                seconds=time.perf_counter() - started,
                model_hash=core.model_hash(base),
                started_at=started_at,
                ended_at=_utc(),
            ),
        )
        _commit(core, base_dir)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    caches, envs = _feature_cache(core, base, spec, folder)
    ready = _committed(core, ssl_dir)
    u = core.make_u(spec, seed)
    if ready:
        u.load_state_dict(
            torch.load(
                ssl_dir / "source.pt", map_location=core.DEVICE, weights_only=False
            )
        )
    else:
        started = time.perf_counter()
        started_at = _utc()
        sums = torch.zeros(spec["xdim"], device=core.DEVICE, dtype=torch.float64)
        sq = sums.clone()
        n = 0
        for b in core.batches(caches["train"], spec["batch"]):
            x = b["x"].double()
            sums += x.sum(0)
            sq += x.square().sum(0)
            n += len(x)
        mean = sums / n
        u.set_normalizer(
            mean.float(), (sq / n - mean.square()).clamp_min(0).sqrt().float()
        )
        opt = torch.optim.Adam(u.parameters(), lr=spec["lr"])
        best = float("inf")
        stale = 0
        state = None
        history = []
        core.seed_all(seed + 2)
        for epoch in range(budget):
            u.train()
            losses = []
            for b in core.batches(caches["train"], spec["batch"]):
                loss = u.losses(b["x"], b["g"])["total"]
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
            u.eval()
            vals = []
            with torch.no_grad():
                for b in core.batches(caches["val"], spec["batch"]):
                    gen = torch.Generator().manual_seed(20261026 + len(vals))
                    ranks = torch.rand(
                        len(b["x"]), len(u.widths), generator=gen
                    ).argsort(1)
                    mask = torch.zeros_like(ranks, dtype=torch.bool).scatter_(
                        1, ranks[:, : max(1, round(0.3 * len(u.widths)))], True
                    )
                    vals.append(float(u.losses(b["x"], b["g"], mask=mask)["total"]))
            score = float(np.mean(vals))
            history.append(
                dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), val_loss=score)
            )
            print(f"SSL epoch={epoch+1} val_loss={score:.8f}", flush=True)
            if score < best:
                best, stale, state = score, 0, copy.deepcopy(u.state_dict())
            else:
                stale += 1
            if stale >= 5:
                break
        u.load_state_dict(state)
        _save(state, ssl_dir / "source.pt")
        core.dump(
            ssl_dir / "metrics.json",
            dict(
                history=history,
                seconds=time.perf_counter() - started,
                selected_epoch=min(history, key=lambda r: r["val_loss"])["epoch"],
                stop_reason="patience" if stale >= 5 else "budget",
                model_hash=core.model_hash(u),
                started_at=started_at,
                ended_at=_utc(),
            ),
        )
        _commit(core, ssl_dir)
    u.eval()
    for p in u.parameters():
        p.requires_grad_(False)
    return base, caches, envs, u


def _old_predictions(core, spec, caches, old):
    current = {}
    for t, (selector, u, pos, enabled) in old.items():
        current[t] = {}
        selector.eval()
        if u is not None:
            u.eval()
        with torch.no_grad():
            for split in ["val", "test"]:
                ps = []
                for b in core.batches(caches[split], spec["batch"]):
                    p = b["old"][:, pos]
                    if enabled:
                        p = selector(
                            p, torch.cat([b["g"], b["s0"], b["s1"], u(b["x"])], 1)
                        )
                    ps.append(p.cpu())
                current[t][split] = torch.cat(ps).numpy()
    return current


def train_seed(
    core, spec, seed, folder, budget, models=("sska", "no_u", "single_task")
):
    if budget < 1:
        raise ValueError("Positive epoch budget required")
    spec = dict(spec, epochs2=budget)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if set(models) == {"single_task"}:
        targets = [t for t in spec["tasks"] if t not in spec["initial"]]
        result = {
            "single_task": [
                train_single(
                    core, spec, seed + i + 3, t, folder / "single_task" / t, budget
                )
                for i, t in enumerate(targets, 1)
            ]
        }
        core.dump(folder / "report.json", result)
        return result
    base, caches, envs, u0 = _prepare(core, spec, seed, folder, spec["epochs1"])
    hb, hu = core.model_hash(base), core.model_hash(u0)
    targets = [t for t in spec["tasks"] if t not in spec["initial"]]
    result = {}
    if "single_task" in models:
        result["single_task"] = [
            train_single(
                core, spec, seed + i + 3, t, folder / "single_task" / t, budget
            )
            for i, t in enumerate(targets, 1)
        ]
    for strategy in [m for m in models if m != "single_task"]:
        variant = "full" if strategy == "sska" else strategy
        initial = folder / strategy / "initial"
        ready = _committed(core, initial)
        initial_started_at = _utc()
        old, old_metrics = core.train_old(
            spec, caches, None if strategy == "no_u" else u0, seed, variant, initial
        )
        if not ready:
            for info in old_metrics.values():
                info.update(started_at=initial_started_at, ended_at=_utc())
            core.dump(initial / "metrics.json", old_metrics)
            _commit(core, initial)
        references = _old_predictions(core, spec, caches, old)
        deployed = {}
        steps = []
        cumulative = sum(x["seconds"] for x in old_metrics.values())
        for step, target in enumerate(targets, 1):
            core.seed_all(seed + step + 3)
            head = core.NewTask(
                spec["xdim"], spec["rdim"], spec["tower"], 3e-5, core.DEVICE
            ).to(core.DEVICE)
            private = (
                None
                if strategy == "no_u"
                else copy.deepcopy(
                    core.make_u(spec, seed) if strategy == "random_u" else u0
                )
            )
            if private is not None:
                for p in private.parameters():
                    p.requires_grad_(True)
                for p in private.decoder.parameters():
                    p.requires_grad_(False)
            model = core.TaskPredictor(
                head, private, spec["xdim"], spec["rdim"], variant
            ).to(core.DEVICE)
            idx = spec["tasks"].index(target)

            def train(m):
                for b in core.batches(caches["train"], spec["batch"]):
                    yield m(b, envs), b["y"][:, idx]

            train.lr = spec["lr"]
            train.seed = seed + step + 3

            def pred(m, split):
                return caches[split]["y"][:, idx].numpy(), core.probabilities(
                    m, caches[split], envs, spec
                )

            pred.row_ids = {s: _rowids(rows) for s, rows in spec["sets"].items()}
            metrics = _fit(
                core,
                model,
                folder / strategy / f"stage{step}",
                budget,
                train,
                pred,
                head.get_l2_reg,
            )
            cumulative += metrics["seconds"]
            deployed[target] = model
            current = {
                t: {
                    s: core.probabilities(m, caches[s], envs, spec)
                    for s in ["val", "test"]
                }
                for t, m in deployed.items()
            }
            current.update(_old_predictions(core, spec, caches, old))
            maxdiff = 0.0
            worst = 0.0
            matrix = {}
            raw = {}
            for split in ["val", "test"]:
                raw[split + "_rowid"] = _rowids(spec["sets"][split])
            for t, split_probs in current.items():
                matrix[t] = {}
                for split, p in split_probs.items():
                    y = caches[split]["y"][:, spec["tasks"].index(t)].numpy()
                    matrix[t][split] = core.measures(y, p)
                    raw[f"{t}_{split}_y"] = y
                    raw[f"{t}_{split}_p"] = p
                if t in references:
                    p = split_probs["test"]
                    y = caches["test"]["y"][:, spec["tasks"].index(t)].numpy()
                    maxdiff = max(
                        maxdiff, float(np.max(np.abs(p - references[t]["test"])))
                    )
                    worst = min(
                        worst,
                        matrix[t]["test"]["auc"]
                        - core.measures(y, references[t]["test"])["auc"],
                    )
                    if any(
                        not np.array_equal(split_probs[s], references[t][s])
                        for s in ["val", "test"]
                    ):
                        raise ValueError("Frozen historical path changed")
            references.update(current)
            raw_path = folder / strategy / f"stage{step}-all-tasks.npz"
            with open(raw_path.with_suffix(".npz.tmp"), "wb") as f:
                np.savez_compressed(f, **raw)
            os.replace(raw_path.with_suffix(".npz.tmp"), raw_path)
            steps.append(
                dict(
                    stage=step,
                    new_target=target,
                    new=metrics,
                    metrics=matrix,
                    historical_max_probability_difference=maxdiff,
                    historical_worst_auc_delta=worst,
                    adaptation_cumulative_seconds=cumulative,
                )
            )
            core.dump(folder / strategy / "matrix.json", steps)
            if core.model_hash(base) != hb or core.model_hash(u0) != hu:
                raise ValueError("Fixed source changed")
        result[strategy] = steps
    core.dump(folder / "report.json", result)
    return result
