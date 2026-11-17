# 表4–6补充实验

沿用三数据集完整划分和原5seed。表4为3组读取对照×6个原目标×5seed＝90个头；表5为随机U＋LoRA×5个新增目标×5seed＝25个模型；表6为3数据集×3结构＝9组统计。

| 表 | 模型 | 数据与源 | 训练内容 |
|---|---|---|---|
| 4 | 原表示＋条件头 | h来自冻结MPT的G/S融合，ConditionalReadout(h,h,p) | 条件头 |
| 4 | 随机U＋条件头 | 相同U结构，seed+1随机初始化，沿用训练集归一化统计 | 条件头，随机U冻结 |
| 4 | 预训练U＋线性头 | 原U、原B直接复用 | Linear(r,1)，每任务独立选点、包含epoch0 |
| 5 | 随机U＋LoRA | 与原U＋LoRA相同新头、三路融合、rank4/alpha4和选点 | 新任务模块、LoRA |
| 6 | MPT／固定U／LoRA | 首seed，原始特征输入，正式batch大小 | 参数、前向/反向矩阵FLOPs，不更新权重 |

表4使用原学习率、batch、FP32、30轮、patience5，每任务独立验证AUC选最早最佳，零修正为第0轮候选。表5保持原新任务协议，无第0轮回退，不重新预训练B/U。已有表3和表4/5结果通过来源路径与摘要引用，旧选点协议线性头不混入新组。

## 云端并行

在新实验目录检出本分支代码，直接引用公共dataset/cache和原产物：

```bash
export PYTHON_BIN=/opt/conda/envs/pytorch-2.1.1/bin/python
export SOURCE_ROOT="$HOME/experiments/tri-three-datasets-20261112/outputs/tri"
export READOUT_ROOT="$HOME/experiments/schemes-1-2-20261116/outputs/scheme2"
export DATASET_ROOT="$HOME/shared/dataset"
export CACHE_ROOT="$HOME/shared/cache"
export WORKERS=4 THREADS_PER_WORKER=2
bash experiments/paper-controls-20261117/run.sh
```

统一任务池共33个调度单元：表4有15个dataset×seed（每单元6头）、表5有15个dataset×seed（每单元1或2目标）、表6有3个数据集单元（每单元3结构）。先入队表4，再表5和表6；空位立即接下一项，不固定划分两个并行队列。4进程×2线程沿用已有8核/30GB/12GB实例记录，不另做吞吐实验。科学配置不随并发改变。

只跑一部分：

```bash
bash experiments/paper-controls-20261117/run.sh --tables 4
bash experiments/paper-controls-20261117/run.sh --tables 5 --dataset aliccp
python -m sska_rec.paper_controls --table 6 --dataset census --source-root "$SOURCE_ROOT" \
  --dataset-root "$DATASET_ROOT" --cache-root "$CACHE_ROOT" --output outputs/paper-controls
```

参数统计同时给出共享B＋单新增目标模块的总参数、实际新增目标路径参数、可训练参数。共享B总量包含原MPT旧塔，活跃路径数排除未执行的旧塔/旧门控。两者都排除U解码器和B对抗分类器；旧条件头不计入新增目标路径。

FLOPs采用乘加＝2的约定，统计原始输入经过嵌入、冻结G/S提取、新目标融合与预测的前向，以及BCE实际反向；不乘固定经验系数。字段名标注`FLOPs_matmul_only`：计入profiler支持的矩阵算子，嵌入查询、激活、归一化、softmax、BCE、逐元素算子和优化器不计入，完整覆盖清单随JSON保存。每批计算量同时换算到每样本。

## 汇总与复用

```bash
python -m sska_rec.paper_controls --report-only --output outputs/paper-controls \
  --source-root "$SOURCE_ROOT" --readout-root "$READOUT_ROOT" \
  --single-task-summary /path/to/auc-mean-variance.json
```

生成`summary.json`和`tables.md`，包含已有表3行、完整条件头、无U/冻结U/LoRA及新对照。Single Task直接读取已核验的聚合JSON，不需上传dataset。新权重、原始概率、rowid、标签、训练曲线、停止原因、UTC时间、成本、来源收据和COMMITTED保留在新输出目录。已完成调度单元直接跳过；中断单元复用其内部已完成模型，不重训整组。

计划见`docs/superpowers/plans/2026-11-17-paper-controls.md`。本次仅准备实现，尚未启动云端训练。
