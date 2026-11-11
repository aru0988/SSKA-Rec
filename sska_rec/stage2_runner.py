"""Independent new-task training with read-only original-task sources."""

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
        rows = 0
        for prediction, y in train(model):
            loss = (
                F.binary_cross_entropy(prediction.reshape(-1), y.float())
                + model.get_l2_reg()
            )
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            rows += len(y)
        model.eval()
        with torch.no_grad():
            y, p = predict(model, "val")
        metrics = core.measures(y, p)
        row = dict(epoch=epoch, train_loss=loss_sum / rows, **metrics)
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


def main(argv=None):
    from .stage2_models import build_model
    from .stage2_reuse import load_reused_seed, original_predictions
    from .stage2_report import report

    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/u-stage2"))
    p.add_argument("--config", type=Path, default=Path("configs/u_stage2_census.json"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--report-only", action="store_true")
    a = p.parse_args(argv)
    cfg = json.loads(a.config.read_text())
    a.output.mkdir(parents=True, exist_ok=True)
    if a.report_only:
        report(a.output, cfg)
        return
    core.DEVICE = torch.device(a.device)
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4)
    code = {
        f.name: core.file_hash(f) for f in sorted(Path(__file__).parent.glob("*.py"))
    }
    identity = dict(
        config=cfg, execution_source=code, source_root=str(a.source_root.resolve())
    )
    status = a.output / "status.json"
    with ExclusiveLock(a.output / "worker.lock"):
        dest = a.output / "identity.json"
        if dest.exists() and json.loads(dest.read_text()) != identity:
            raise ValueError("Execution identity changed; select a new output version")
        core.dump(dest, identity)
        core.dump(
            a.output / "environment.json",
            dict(
                python=sys.version,
                platform=platform.platform(),
                torch=torch.__version__,
                cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name() if a.device == "cuda" else None,
                git_commit=os.environ.get("SSKA_COMMIT"),
            ),
        )
        try:
            for seed in cfg["seeds"]:
                core.dump(
                    status,
                    dict(
                        state="running",
                        pid=os.getpid(),
                        seed=seed,
                        stage="verify_reuse",
                        updated_at=_utc(),
                    ),
                )
                spec, base, u, heads, caches, envs, receipt = load_reused_seed(
                    a.source_root, a.dataset_root, seed, core.DEVICE
                )
                spec.update(batch=cfg["batch_size"], lr=cfg["learning_rate"])
                core.dump(a.output / str(seed) / "reuse-receipt.json", receipt)
                hashes = [core.model_hash(x) for x in [base, u, heads]]
                references = {
                    s: original_predictions(base, u, heads, caches, spec, s)
                    for s in ["val", "test"]
                }
                with np.load(
                    a.source_root / str(seed) / "linear/predictions.npz"
                ) as old:
                    for split in references:
                        if not np.array_equal(references[split], old[split + "_p"]):
                            raise ValueError(
                                "Original task source predictions do not reproduce"
                            )

                def guard():
                    if hashes != [core.model_hash(x) for x in [base, u, heads]]:
                        raise ValueError("Frozen original task parameters changed")
                    for split, expected in references.items():
                        actual = original_predictions(
                            base, u, heads, caches, spec, split
                        )
                        if not np.array_equal(actual, expected):
                            raise ValueError("Frozen original task predictions changed")
                    return dict(
                        passed=True,
                        base_hash=hashes[0],
                        u_hash=hashes[1],
                        linear_heads_hash=hashes[2],
                        max_probability_difference=0.0,
                    )

                for pos, target in enumerate(cfg["targets"]):
                    init_seed = seed + pos
                    for variant in cfg["variants"]:
                        core.dump(
                            status,
                            dict(
                                state="running",
                                pid=os.getpid(),
                                seed=seed,
                                target=target,
                                variant=variant,
                                updated_at=_utc(),
                            ),
                        )
                        model = build_model(spec, u, init_seed, variant).to(core.DEVICE)
                        actual = sum(
                            p.numel() for p in model.parameters() if p.requires_grad
                        )
                        if actual != cfg["expected_trainable"][variant]:
                            raise ValueError(f"Parameter mismatch {variant}: {actual}")
                        train_one(
                            model,
                            spec,
                            caches,
                            envs,
                            target,
                            a.output / str(seed) / target / variant,
                            cfg,
                            init_seed + 3,
                            variant,
                            guard,
                            status,
                            paired_seed=seed,
                        )
                        del model
                        report(a.output, cfg)
                del base, u, heads, caches, references
                if a.device == "cuda":
                    torch.cuda.empty_cache()
            core.dump(
                status,
                dict(
                    state="complete",
                    pid=None,
                    completed=report(a.output, cfg),
                    expected=50,
                    finished_at=_utc(),
                ),
            )
        except BaseException as exc:
            core.dump(
                status,
                dict(
                    state="paused" if isinstance(exc, KeyboardInterrupt) else "failed",
                    pid=None,
                    error=str(exc),
                    updated_at=_utc(),
                ),
            )
            raise


if __name__ == "__main__":
    main()
