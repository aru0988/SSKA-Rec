"""Matched stage2 optimizer with explicit loss components and throughput."""

import argparse
import copy
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from . import core
from .locking import ExclusiveLock
from .training import _committed, _commit, _utc, _save


def train_one(
    model,
    spec,
    caches,
    envs,
    target,
    folder,
    cfg,
    seed,
    variant,
    guard,
    status_path,
    paired_seed=None,
):
    folder = Path(folder)
    guard()
    if (folder / "COMMITTED.json").exists():
        _committed(core, folder)
        prior = json.loads((folder / "metrics.json").read_text())
        assert (
            prior["target"] == target
            and prior["variant"] == variant
            and prior["seed"] == seed
        )
        return prior
    # Completed weights/metrics can survive a crash just before final commit.
    # Verify them and finish the commit without repeating optimization.
    if (folder / "metrics.json").exists():
        pending = json.loads((folder / "metrics.json").read_text())
        if pending.get("ready_for_commit"):
            assert (
                pending["target"] == target
                and pending["variant"] == variant
                and pending["seed"] == seed
            )
            model.load_state_dict(
                torch.load(
                    folder / "model.pt", map_location=core.DEVICE, weights_only=True
                )
            )
            if core.model_hash(model) != pending["model_hash"]:
                raise ValueError("Pending model hash mismatch")
            idx = spec["tasks"].index(target)
            with np.load(folder / "predictions.npz") as raw:
                for split in ["val", "test"]:
                    if not np.array_equal(
                        raw[split + "_y"], caches[split]["y"][:, idx].numpy()
                    ) or not np.array_equal(
                        raw[split + "_rowid"], spec["sets"][split].row_ids
                    ):
                        raise ValueError("Pending raw alignment mismatch")
                    actual = core.measures(raw[split + "_y"], raw[split + "_p"])
                    if any(abs(actual[k] - pending[split][k]) > 1e-10 for k in actual):
                        raise ValueError("Pending metrics mismatch")
            guard()
            _commit(core, folder)
            return pending
    _committed(core, folder)
    idx = spec["tasks"].index(target)
    live = []

    def train(m):
        for b in core.batches(caches["train"], spec["batch"]):
            p = m(b, envs)
            if not torch.isfinite(p).all():
                raise ValueError("Non-finite training predictions")
            yield p, b["y"][:, idx]

    train.lr = cfg["learning_rate"]
    train.seed = seed

    def predict(m, split):
        y = caches[split]["y"][:, idx].numpy()
        p = core.probabilities(m, caches[split], envs, spec)
        if split == "val":
            row = dict(check=len(live) + 1, **core.measures(y, p), at=_utc())
            live.append(row)
            core.dump(folder / "validation-live.json", live)
            core.dump(
                status_path,
                dict(
                    state="running",
                    pid=os.getpid(),
                    seed=paired_seed if paired_seed is not None else seed,
                    model_seed=seed,
                    target=target,
                    variant=variant,
                    validation=row,
                    updated_at=_utc(),
                ),
            )
        return y, p

    started_at = _utc()
    started = time.perf_counter()
    history = []
    if core.DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(core.DEVICE)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=cfg["learning_rate"]
    )
    best = -float("inf")
    stale = 0
    state = None
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        loss_sum = 0.0
        bce_sum = reg_sum = data_wait = 0.0
        epoch_started = time.perf_counter()
        previous_end = epoch_started
        rows = 0
        for prediction, y in train(model):
            data_wait += time.perf_counter() - previous_end
            bce = F.binary_cross_entropy(prediction.reshape(-1), y.float())
            regularization = model.get_l2_reg()
            loss = bce + regularization
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            bce_sum += float(bce.detach()) * len(y)
            reg_sum += float(regularization.detach()) * len(y)
            rows += len(y)
            previous_end = time.perf_counter()
        train_seconds = time.perf_counter() - epoch_started
        model.eval()
        with torch.no_grad():
            y, p = predict(model, "val")
        metrics = core.measures(y, p)
        row = dict(epoch=epoch, train_loss=loss_sum / rows, train_bce=bce_sum / rows, train_regularization=reg_sum / rows, train_seconds=train_seconds, samples_per_second=rows / max(train_seconds,1e-9), batch_fetch_and_forward_seconds=data_wait, **metrics)
        history.append(row)
        core.dump(folder / "history-live.json", history)
        print(
            json.dumps(dict(target=target, variant=variant, seed=seed, **row)),
            flush=True,
        )
        if metrics["auc"] > best:
            best = metrics["auc"]
            stale = 0
            state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= cfg["patience"]:
            break
    model.load_state_dict(state)
    model.eval()
    result = dict(
        history=history,
        epochs=len(history),
        selected_epoch=max(history, key=lambda r: r["auc"])["epoch"],
        stop_reason="patience" if stale >= cfg["patience"] else "budget",
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
        seed=seed,
        started_at=started_at,
        ended_at=_utc(),
    )
    raw = {}
    with torch.no_grad():
        for split in ["val", "test"]:
            y, p = predict(model, split)
            result[split] = core.measures(y, p)
            raw.update(
                {
                    split + "_y": y,
                    split + "_p": p,
                    split + "_rowid": spec["sets"][split].row_ids,
                }
            )
    _save(state, folder / "model.pt")
    with (folder / "predictions.npz.tmp").open("wb") as f:
        np.savez_compressed(f, **raw)
    os.replace(folder / "predictions.npz.tmp", folder / "predictions.npz")
    frozen = guard()
    result.update(
        u_update_parameters=sum(p.numel() for n,p in model.named_parameters() if n.startswith('u.') and p.requires_grad),
        trainable_tensor_bytes=sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad),
        weight_file_bytes=(folder / 'model.pt').stat().st_size,
        target=target,
        variant=variant,
        exclusive_parameters=model.exclusive_parameters,
        inference_parameters=model.inference_parameters,
        frozen_original_paths=frozen,
        validation_fallback=False,
        learning_rate=cfg["learning_rate"],
        patience=cfg["patience"],
        paired_seed=paired_seed if paired_seed is not None else seed,
        ready_for_commit=True,
    )
    core.dump(folder / "metrics.json", result)
    _commit(core, folder)
    return result
