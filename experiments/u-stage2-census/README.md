# 二阶段融合与U更新范围

配置：[`configs/u_stage2_census.json`](../../configs/u_stage2_census.json)。计划：[`2026-11-11-u-stage2-census.md`](../../docs/superpowers/plans/2026-11-11-u-stage2-census.md)。

直接复用原任务U线性接入实验的固定子集、五份B/U和原任务线性头。五结构×Education/Sex×五seed，共50模型。

| 结构 | U更新 | 训练参数/新任务 |
|---|---|---:|
| mpt | 不使用U | 26799 |
| residual | 全编码器；历史表示残差 | 116528 |
| tri_frozen | 冻结 | 26918 |
| tri_lora | 两层权重rank4 LoRA | 30106 |
| tri_full | 全编码器 | 100262 |

```bash
export SOURCE_ROOT=$HOME/experiments/u-access-census-20261111/outputs/u-access
export DATASET_ROOT=$HOME/shared/dataset
export PYTHON_BIN=/opt/conda/envs/pytorch-2.1.1/bin/python
bash experiments/u-stage2-census/run.sh
```

状态：`outputs/u-stage2/status.json`。表格：`outputs/u-stage2/table.md`。

```bash
python -m sska_rec.stage2_runner --source-root "$SOURCE_ROOT" --dataset-root "$DATASET_ROOT" --report-only
```

模型实现：`sska_rec/stage2_models.py`；源复用：`stage2_reuse.py`；训练：`stage2_runner.py`；统计：`stage2_report.py`。
