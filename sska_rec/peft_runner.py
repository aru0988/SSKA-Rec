"""Run new PEFT models with read-only original sources and old controls."""

import argparse, json, os, platform, sys
from pathlib import Path
import numpy as np
import torch
from . import core
from .locking import ExclusiveLock
from .training import _utc
from .stage2_runner import train_one


def verify_baseline(root, source, cfg):
    root = Path(root).resolve()
    meta = json.loads((root / "identity.json").read_text())
    if (
        meta["config"]["seeds"] != cfg["seeds"]
        or meta["config"]["targets"] != cfg["targets"]
    ):
        raise ValueError("Baseline pairing mismatch")
    if Path(meta["source_root"]).resolve() != Path(source).resolve():
        raise ValueError("Baseline B/U source mismatch")
    for name, sha in meta["execution_source"].items():
        if core.file_hash(root.parents[1] / "sska_rec" / name) != sha:
            raise ValueError("Baseline execution source changed")
    if json.loads((root / "status.json").read_text())["completed"] != 50:
        raise ValueError("Baseline matrix incomplete")


def main(argv=None):
    from .peft_models import build_model
    from .stage2_reuse import load_reused_seed, original_predictions
    from .peft_report import report as build_report

    p = argparse.ArgumentParser()
    p.add_argument("--baseline-root", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/u-peft"))
    p.add_argument("--config", type=Path, default=Path("configs/u_peft_census.json"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--report-only", action="store_true")
    a = p.parse_args(argv)
    cfg = json.loads(a.config.read_text())

    def report(output, cfg):
        return build_report(output, cfg, a.baseline_root)

    verify_baseline(a.baseline_root, a.source_root, cfg)
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
        config=cfg,
        execution_source=code,
        source_root=str(a.source_root.resolve()),
        baseline_root=str(a.baseline_root.resolve()),
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
                for split, cache in caches.items():
                    for key in list(cache):
                        if key != "y":
                            cache[key] = cache[key].to(core.DEVICE)
                    pieces = []
                    with torch.no_grad():
                        for b in core.batches(cache, spec["batch"]):
                            pieces.append(u(b["x"]))
                    cache["u0"] = torch.cat(pieces).detach()
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
                    expected=30,
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
