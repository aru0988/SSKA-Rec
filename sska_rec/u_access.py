"""Fixed-subset original-task U access experiment; no validation fallback."""

import argparse
import copy
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from scipy.stats import t

from . import core
from .locking import ExclusiveLock
from .training import _prepare, _committed, _commit, _utc

NAMES = ["income", "marital"]
COUNTS = {
    "mpt": 0,
    "linear": 129,
    "gated_linear": 258,
    "original": 32898,
    "bottleneck": 2441,
}


class AccessHead(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        if variant == "linear":
            self.delta = nn.Linear(128, 1)
            nn.init.zeros_(self.delta.weight)
            nn.init.zeros_(self.delta.bias)
        elif variant == "gated_linear":
            self.delta = nn.Linear(128, 1)
            self.gate = nn.Linear(128, 1)
            nn.init.zeros_(self.delta.weight)
            nn.init.zeros_(self.delta.bias)
        elif variant == "original":
            self.selector = core.Selector(512)
        elif variant == "bottleneck":
            self.down = nn.Linear(128, 8)
            self.up = nn.Linear(8, 128)
            self.gate = nn.Linear(256, 1)
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)
        else:
            raise ValueError(variant)

    def forward(self, b, tower):
        if self.variant == "original":
            return self.selector(
                b["p"], torch.cat([b["g"], b["s0"], b["s1"], b["u"]], 1)
            )
        if self.variant == "bottleneck":
            residual = self.up(F.relu(self.down(b["u"])))
            alpha = self.gate(torch.cat([b["h"], b["u"]], 1)).sigmoid()
            reference = tower(b["h"]).reshape(-1)
            candidate = tower(b["h"] + alpha * residual).reshape(-1)
        else:
            shift = self.delta(b["u"]).reshape(-1)
            if self.variant == "gated_linear":
                shift = shift * self.gate(b["u"]).sigmoid().reshape(-1)
            logits = torch.logit(b["p"].clamp(1e-7, 1 - 1e-7))
            reference = logits.sigmoid()
            candidate = (logits + shift).sigmoid()
        return (b["p"] + (candidate - reference)).clamp(0, 1)


def stratified_indices(y, n, seed):
    labels = (y[:, 0] * 2 + y[:, 1]).astype("int64")
    selected, _ = train_test_split(
        np.arange(len(y)), train_size=n, stratify=labels, random_state=seed
    )
    return np.sort(selected)


def paired_interval(values, reference):
    delta = np.asarray(values) - np.asarray(reference)
    if len(delta) != 5:
        raise ValueError("Five paired seeds required")
    mean = float(delta.mean())
    sd = float(delta.std(ddof=1))
    margin = float(t.ppf(0.975, 4) * sd / np.sqrt(5))
    return dict(
        n=5,
        mean=mean,
        sd=sd,
        low=mean - margin,
        high=mean + margin,
        values=delta.tolist(),
    )


def subset_spec(dataset_root, folder, cfg):
    core.DATA = Path(dataset_root).resolve()
    hashes = {
        name: core.file_hash(core.DATA / "Census-income" / name)
        for name in ["train.gz", "test.gz"]
    }
    expected = [
        "e2f2d57925a8020d23fcd05802284b3bc1d91fcdd451f1eaaefd683e76d1b2dc",
        "8de690375028ac531de37fcbb117bd6527c5f3542d3bfead8975226e45100cec",
    ]
    if list(hashes.values()) != expected:
        raise ValueError("Census input hash mismatch")
    spec = core.census(True, NAMES)
    assert spec["xdim"] == cfg["input_dim"] == 119
    assert not set(spec["sets"]["train"].names) & {
        "label_income",
        "label_marital",
        "education",
        "sex",
    }
    ntrain = len(spec["sets"]["train"])
    nval = len(spec["sets"]["val"])
    ntest = len(spec["sets"]["test"])
    vi, ti = core.make_split(nval + ntest, cfg["split_seed"])
    ids = {"train": np.arange(ntrain), "val": ntrain + vi, "test": ntrain + ti}
    indices = {}
    info = {}
    for split, rows in spec["sets"].items():
        ix = stratified_indices(rows.y, cfg["sizes"][split], cfg["subset_seed"])
        indices[split + "_indices"] = ix
        indices[split + "_rowid"] = ids[split][ix]
        new = core.Rows(rows.x[ix], rows.y[ix], rows.names, [0, 1])
        new.row_ids = ids[split][ix]
        spec["sets"][split] = new
        info[split] = dict(
            n=len(ix),
            joint_label_counts=np.bincount(
                (new.y[:, 0] * 2 + new.y[:, 1]).astype(int), minlength=4
            ).tolist(),
            rowid_sha256=hashlib.sha256(new.row_ids.tobytes()).hexdigest(),
        )
    meta = dict(
        raw_sha256=hashes,
        split_seed=cfg["split_seed"],
        subset_seed=cfg["subset_seed"],
        splits=info,
        features=spec["sets"]["train"].names,
        input_dim=119,
    )
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "subset.json").exists() and json.loads(
        (folder / "subset.json").read_text()
    ) != meta:
        raise ValueError("Subset identity changed")
    if not (folder / "indices.npz").exists():
        np.savez_compressed(folder / "indices.npz", **indices)
    else:
        with np.load(folder / "indices.npz") as old:
            if any(not np.array_equal(old[k], v) for k, v in indices.items()):
                raise ValueError("Subset indices changed")
    core.dump(folder / "subset.json", meta)
    spec.update(
        batch=cfg["batch_size"],
        lr=cfg["learning_rate"],
        epochs1=cfg["base_epochs"],
        epochs2=cfg["access_epochs"],
    )
    return spec, meta


def access_cache(base, u, caches, spec):
    result = {}
    with torch.no_grad():
        for split, cache in caches.items():
            pieces = {key: [] for key in ["u", "h0", "h1"]}
            for b in core.batches(cache, spec["batch"]):
                pieces["u"].append(u(b["x"]).cpu())
                for pos in range(2):
                    gate = base.gate_networks[pos](b["x"])
                    env = base.env_embedding_network(
                        torch.tensor(pos, device=core.DEVICE)
                    )
                    reps = torch.stack([b["s" + str(pos)] * env, b["g"]], dim=2)
                    pieces["h" + str(pos)].append(
                        torch.matmul(reps, gate.unsqueeze(2)).squeeze(2).cpu()
                    )
            result[split] = dict(cache, **{k: torch.cat(v) for k, v in pieces.items()})
    return result


def predict(heads, base, cache, spec):
    ps = []
    with torch.no_grad():
        for b in core.batches(cache, spec["batch"]):
            ps.append(
                torch.stack(
                    [
                        heads[i](
                            dict(b, h=b["h" + str(i)], p=b["old"][:, i]),
                            base.tower_networks[i],
                        )
                        for i in range(2)
                    ],
                    1,
                ).cpu()
            )
    return torch.cat(ps).numpy()


def score(y, p):
    return {name: core.measures(y[:, i], p[:, i]) for i, name in enumerate(NAMES)}


def run_access(base, u, cache, spec, seed, variant, folder, cfg, status_path):
    resumed = _committed(core, folder)
    if resumed:
        return json.loads((folder / "metrics.json").read_text())
    before = (core.model_hash(base), core.model_hash(u))
    started = _utc()
    clock = time.perf_counter()
    if core.DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(core.DEVICE)
    if variant == "mpt":
        heads = None
        history = []
        selected = 0
        stop = "not_trained"
        peak = 0
    else:
        modules = []
        for pos in range(2):
            core.seed_all(seed + 20 + pos)
            modules.append(AccessHead(variant).to(core.DEVICE))
        heads = nn.ModuleList(modules)
        assert sum(p.numel() for p in heads.parameters()) == 2 * COUNTS[variant]
        for split in ["val"]:
            if not np.array_equal(
                predict(heads, base, cache[split], spec), cache[split]["old"].numpy()
            ):
                raise ValueError("Initialization changed baseline predictions")
        opt = torch.optim.Adam(heads.parameters(), lr=cfg["learning_rate"])
        history = []
        best = -float("inf")
        state = None
        stale = 0
        for epoch in range(1, cfg["access_epochs"] + 1):
            heads.train()
            total = 0.0
            n = 0
            for b in core.batches(cache["train"], spec["batch"]):
                pred = torch.stack(
                    [
                        heads[i](
                            dict(b, h=b["h" + str(i)], p=b["old"][:, i]),
                            base.tower_networks[i],
                        )
                        for i in range(2)
                    ],
                    1,
                )
                loss = F.binary_cross_entropy(pred, b["y"][:, :2])
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite adaptation loss")
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(pred)
                n += len(pred)
            heads.eval()
            metrics = score(
                cache["val"]["y"].numpy(), predict(heads, base, cache["val"], spec)
            )
            mean = float(np.mean([v["auc"] for v in metrics.values()]))
            row = dict(
                epoch=epoch,
                train_bce=total / n,
                val=metrics,
                val_mean_auc=mean,
                at=_utc(),
            )
            history.append(row)
            core.dump(folder / "history-live.json", history)
            core.dump(
                status_path,
                dict(
                    state="running",
                    pid=os.getpid(),
                    seed=seed,
                    stage=variant,
                    last_epoch=row,
                    updated_at=_utc(),
                ),
            )
            print(json.dumps(dict(seed=seed, model=variant, **row)), flush=True)
            if mean > best:
                best = mean
                stale = 0
                state = copy.deepcopy(heads.state_dict())
            else:
                stale += 1
            if stale >= cfg["patience"]:
                break
        heads.load_state_dict(state)
        heads.eval()
        torch.save(state, folder / "model.pt")
        selected = max(history, key=lambda r: r["val_mean_auc"])["epoch"]
        stop = "patience" if stale >= cfg["patience"] else "budget"
        peak = (
            torch.cuda.max_memory_allocated(core.DEVICE)
            if core.DEVICE.type == "cuda"
            else 0
        )
    raw = {}
    metrics = {}
    for split in ["val", "test"]:
        y = cache[split]["y"].numpy()[:, :2]
        p = (
            cache[split]["old"].numpy()
            if heads is None
            else predict(heads, base, cache[split], spec)
        )
        raw.update(
            {
                split + "_y": y,
                split + "_p": p,
                split + "_rowid": spec["sets"][split].row_ids,
            }
        )
        metrics[split] = score(y, p)
    assert before == (core.model_hash(base), core.model_hash(u))
    np.savez_compressed(folder / "predictions.npz", **raw)
    result = dict(
        seed=seed,
        variant=variant,
        history=history,
        selected_epoch=selected,
        stop_reason=stop,
        epochs=len(history),
        access_parameters_per_task=COUNTS[variant],
        access_parameters_total=2 * COUNTS[variant],
        seconds=time.perf_counter() - clock,
        peak_cuda_bytes=peak,
        started_at=started,
        ended_at=_utc(),
        frozen_base_hash=before[0],
        frozen_u_hash=before[1],
        validation_fallback=False,
        **metrics,
    )
    core.dump(folder / "metrics.json", result)
    _commit(core, folder)
    return result


def report(output, cfg):
    records = {}
    checks = 0
    for seed in cfg["seeds"]:
        for variant in cfg["models"]:
            folder = output / str(seed) / variant
            if not (folder / "COMMITTED.json").exists():
                continue
            assert _committed(core, folder)
            m = json.loads((folder / "metrics.json").read_text())
            with np.load(folder / "predictions.npz") as raw:
                for split in ["val", "test"]:
                    y = raw[split + "_y"]
                    p = raw[split + "_p"]
                    actual = {
                        name: dict(
                            auc=float(roc_auc_score(y[:, i], p[:, i])),
                            logloss=float(
                                log_loss(
                                    y[:, i],
                                    np.clip(p[:, i].astype("float64"), 1e-7, 1 - 1e-7),
                                    labels=[0, 1],
                                )
                            ),
                        )
                        for i, name in enumerate(NAMES)
                    }
                    for name in NAMES:
                        for metric in ["auc", "logloss"]:
                            if (
                                abs(actual[name][metric] - m[split][name][metric])
                                > 1e-10
                            ):
                                raise ValueError("Raw metric mismatch")
                    checks += 1
            records[(seed, variant)] = m
    summary = {}
    lines = [
        "# Census原任务U接入：5seed均值±样本SD",
        "",
        "| 模型 | Income AUC | Marital AUC | Income Logloss | Marital Logloss |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in cfg["models"]:
        rows = [
            records[(seed, variant)]
            for seed in cfg["seeds"]
            if (seed, variant) in records
        ]
        cells = []
        summary[variant] = {}
        for metric in ["auc", "logloss"]:
            for name in NAMES:
                values = [r["test"][name][metric] for r in rows]
                cells.append(
                    f"{np.mean(values):.6f} ± {np.std(values,ddof=1):.6f}"
                    if len(rows) == 5
                    else f"待完成 {len(rows)}/5"
                )
                if len(rows) == 5:
                    item = dict(
                        mean=float(np.mean(values)),
                        sd=float(np.std(values, ddof=1)),
                        values=values,
                    )
                    for ref in ["mpt", "original"]:
                        if all((seed, ref) in records for seed in cfg["seeds"]):
                            item["vs_" + ref] = paired_interval(
                                values,
                                [
                                    records[(seed, ref)]["test"][name][metric]
                                    for seed in cfg["seeds"]
                                ],
                            )
                    summary[variant][name + "/" + metric] = item
        lines.append("| " + " | ".join([variant, *cells]) + " |")
    if len(records) == 25:
        lines += [
            "",
            "## 配对差值及95% t区间",
            "",
            "| 模型 | 参照 | 任务 | 指标 | 差值均值 | 95% CI |",
            "|---|---|---|---|---:|---|",
        ]
        for variant in cfg["models"]:
            for ref in ["mpt", "original"]:
                if variant == ref:
                    continue
                for metric in ["auc", "logloss"]:
                    for name in NAMES:
                        r = summary[variant][name + "/" + metric]["vs_" + ref]
                        lines.append(
                            f"| {variant} | {ref} | {name} | {metric} | {r['mean']:+.6f} | [{r['low']:+.6f}, {r['high']:+.6f}] |"
                        )
        lines += [
            "",
            "## 接入成本（两个任务合计）",
            "",
            "| 模型 | 参数/任务 | 耗时秒均值±SD | 峰值显存MiB均值±SD |",
            "|---|---:|---:|---:|",
        ]
        for variant in cfg["models"]:
            rows = [records[(seed, variant)] for seed in cfg["seeds"]]
            seconds = [r["seconds"] for r in rows]
            memory = [r["peak_cuda_bytes"] / 1024**2 for r in rows]
            lines.append(
                f"| {variant} | {COUNTS[variant]} | {np.mean(seconds):.3f} ± {np.std(seconds,ddof=1):.3f} | {np.mean(memory):.3f} ± {np.std(memory,ddof=1):.3f} |"
            )
    (output / "table.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    core.dump(
        output / "summary.json",
        dict(
            completed=len(records),
            expected=25,
            raw_split_checks=checks,
            summary=summary,
        ),
    )
    return len(records)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    p.add_argument("--output", type=Path, default=Path("outputs/u-access"))
    p.add_argument("--config", type=Path, default=Path("configs/u_access_census.json"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--prepare-only", action="store_true")
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
    source = {
        str(f.relative_to(Path(__file__).parent)): core.file_hash(f)
        for f in sorted(Path(__file__).parent.glob("*.py"))
    }
    with ExclusiveLock(a.output / "worker.lock"):
        spec, data = subset_spec(a.dataset_root, a.output / "subset", cfg)
        identity = dict(protocol=cfg, source=source, data=data)
        path = a.output / "identity.json"
        if path.exists() and json.loads(path.read_text()) != identity:
            raise ValueError(
                "Execution/data identity changed; use a new output version"
            )
        core.dump(path, identity)
        if a.prepare_only:
            return
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
        status = a.output / "status.json"
        try:
            for seed in cfg["seeds"]:
                folder = a.output / str(seed)
                core.dump(
                    status,
                    dict(
                        state="running",
                        pid=os.getpid(),
                        seed=seed,
                        stage="base_and_ssl",
                        updated_at=_utc(),
                    ),
                )
                if a.device == "cuda":
                    torch.cuda.reset_peak_memory_stats(core.DEVICE)
                clock = time.perf_counter()
                base, caches, envs, u = _prepare(
                    core, spec, seed, folder, cfg["base_epochs"]
                )
                if not (folder / "source-cost.json").exists():
                    core.dump(
                        folder / "source-cost.json",
                        dict(
                            seconds=time.perf_counter() - clock,
                            peak_cuda_bytes=(
                                torch.cuda.max_memory_allocated(core.DEVICE)
                                if a.device == "cuda"
                                else 0
                            ),
                            base_parameters=sum(p.numel() for p in base.parameters()),
                            u_parameters=sum(p.numel() for p in u.parameters()),
                        ),
                    )
                assert all(
                    not v.requires_grad for v in [*base.parameters(), *u.parameters()]
                )
                cache = access_cache(base, u, caches, spec)
                for variant in cfg["models"]:
                    core.dump(
                        status,
                        dict(
                            state="running",
                            pid=os.getpid(),
                            seed=seed,
                            stage=variant,
                            updated_at=_utc(),
                        ),
                    )
                    run_access(
                        base,
                        u,
                        cache,
                        spec,
                        seed,
                        variant,
                        folder / variant,
                        cfg,
                        status,
                    )
                    report(a.output, cfg)
                del base, u, caches, cache
                if a.device == "cuda":
                    torch.cuda.empty_cache()
            core.dump(
                status,
                dict(
                    state="complete",
                    pid=None,
                    completed=report(a.output, cfg),
                    expected=25,
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
