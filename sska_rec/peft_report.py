"""Complete paired-seed metrics, paired differences and costs."""

import json
from pathlib import Path
import numpy as np
from scipy.stats import t
from sklearn.metrics import roc_auc_score, log_loss
from . import core
from .training import _committed


def stats(values):
    a = np.asarray(values)
    return dict(
        n=len(a), mean=float(a.mean()), sd=float(a.std(ddof=1)), values=a.tolist()
    )


def difference(a, b):
    r = stats(np.asarray(a) - np.asarray(b))
    margin = float(t.ppf(0.975, 4) * r["sd"] / np.sqrt(5))
    return dict(r, low=r["mean"] - margin, high=r["mean"] + margin)


def report(output, cfg, baseline_root):
    cfg = dict(cfg, variants=cfg["reuse_variants"] + cfg["variants"])
    baseline_root = Path(baseline_root)
    output = Path(output)
    results = {}
    checks = 0
    for seed in cfg["seeds"]:
        for target in cfg["targets"]:
            for variant in cfg["variants"]:
                folder = (
                    (baseline_root if variant in cfg["reuse_variants"] else output)
                    / str(seed)
                    / target
                    / variant
                )
                if not (folder / "COMMITTED.json").exists():
                    continue
                _committed(core, folder)
                m = json.loads((folder / "metrics.json").read_text())
                expected_seed = seed + cfg["targets"].index(target) + 3
                if (m["paired_seed"], m["seed"], m["target"], m["variant"]) != (
                    seed,
                    expected_seed,
                    target,
                    variant,
                ):
                    raise ValueError("Paired model identity mismatch")
                with np.load(folder / "predictions.npz") as raw:
                    for split in ["val", "test"]:
                        y, p = raw[split + "_y"], raw[split + "_p"]
                        auc = float(roc_auc_score(y, p))
                        ll = float(
                            log_loss(
                                y,
                                np.clip(p.astype(float), 1e-7, 1 - 1e-7),
                                labels=[0, 1],
                            )
                        )
                        if (
                            abs(auc - m[split]["auc"]) > 1e-10
                            or abs(ll - m[split]["logloss"]) > 1e-10
                        ):
                            raise ValueError("Independent raw metric check failed")
                        checks += 1
                assert (
                    m["frozen_original_paths"]["passed"]
                    and m["frozen_original_paths"]["max_probability_difference"] == 0.0
                )
                results[(seed, target, variant)] = m
    summary = {}
    lines = [
        "# Census轻量U微调：复用对照与新增方案",
        "",
        "| 结构 | Education AUC | Sex AUC | Education Logloss | Sex Logloss |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in cfg["variants"]:
        cells = []
        summary[variant] = {}
        for metric in ["auc", "logloss"]:
            for target in cfg["targets"]:
                rows = [
                    results[(seed, target, variant)]
                    for seed in cfg["seeds"]
                    if (seed, target, variant) in results
                ]
                if len(rows) < 5:
                    cells.append(f"待完成 {len(rows)}/5")
                    continue
                values = [r["test"][metric] for r in rows]
                item = stats(values)
                for ref in cfg["variants"]:
                    if ref == variant:
                        continue
                    if all((seed, target, ref) in results for seed in cfg["seeds"]):
                        item["vs_" + ref] = difference(
                            values,
                            [
                                results[(seed, target, ref)]["test"][metric]
                                for seed in cfg["seeds"]
                            ],
                        )
                summary[variant][target + "/" + metric] = item
                cells.append(f"{item['mean']:.6f} ± {item['sd']:.6f}")
        lines.append("| " + " | ".join([variant, *cells]) + " |")
    if len(results) == 80:
        lines += [
            "",
            "## 配对差值与95% t区间",
            "",
            "| 结构 | 参照 | 目标 | 指标 | 差值 | 95% CI |",
            "|---|---|---|---|---:|---|",
        ]
        for variant in cfg["variants"]:
            refs = ["mpt", "tri_frozen", "tri_lora", "tri_full", "residual"]
            for ref in dict.fromkeys(refs):
                if ref == variant:
                    continue
                for target in cfg["targets"]:
                    for metric in ["auc", "logloss"]:
                        r = summary[variant][target + "/" + metric]["vs_" + ref]
                        lines.append(
                            f"| {variant} | {ref} | {target} | {metric} | {r['mean']:+.6f} | [{r['low']:+.6f}, {r['high']:+.6f}] |"
                        )
        lines += [
            "",
            "## 每新任务成本（旧组CPU缓存；新组GPU常驻缓存，峰值包含缓存）",
            "",
            "| 结构 | 目标 | 训练参数 | 独占参数 | 耗时秒均值±SD | 峰值MiB均值±SD |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for variant in cfg["variants"]:
            for target in cfg["targets"]:
                rows = [results[(seed, target, variant)] for seed in cfg["seeds"]]
                sec = stats([r["seconds"] for r in rows])
                mem = stats([r["peak_cuda_bytes"] / 1024**2 for r in rows])
                lines.append(
                    f"| {variant} | {target} | {rows[0]['optimizer_parameters']} | {rows[0]['exclusive_parameters']} | {sec['mean']:.3f} ± {sec['sd']:.3f} | {mem['mean']:.3f} ± {mem['sd']:.3f} |"
                )
    (output / "table.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    core.dump(
        output / "summary.json",
        dict(
            completed=len(results),
            expected=80,
            completed_new=sum(v not in cfg["reuse_variants"] for _, _, v in results),
            expected_new=30,
            reuse_root=str(baseline_root),
            raw_split_checks=checks,
            summary=summary,
        ),
    )
    return sum(v not in cfg["reuse_variants"] for _, _, v in results)
