# KuaiRand归一化嵌入正则与Head-only对照（2026-11-20）

## 修改

仅将MPT主干的嵌入正则从`.006/2*sum(E²)`改为`.006/(2*N_E)*sum(E²)`，与已有Single Task一致。DNN正则、GRL实现、数据、初始化、优化器和预算保持原配置。旧结果单独保留。

## 执行矩阵

|阶段|模型|目标|5seed训练单元|
|---|---|---|---:|
|源模型|B、SSL U|Click、Like联合主干|10|
|原任务读取|条件低秩头、Head-only|Click、Like|20|
|新增任务|MPT、冻结U、随机U+LoRA、预训练U+LoRA|Follow、Forward|40|
|合计|||70|

Single Task四目标共20个已有模型经原始预测、检查点、数据划分核验后直接复用。

Head-only使用同一个`ConditionalReadout`，将U输入替换为h，保留无仿射LayerNorm、rank=4、1541个参数、w/b零初始化和零修正候选。报告head-only−MPT、TCKA−head-only及零修正被选次数。

## 固定配置

- 5seed：1685480945、1685463909、1685477428、1685459668、1685496394。
- 复用公共dataset及原时间划分；四种行为反馈均不进入输入。
- FP32、Adam、LR=.001、batch=512、上限30轮、patience=5。
- 主干按两任务验证AUC之和选最早最佳；U按验证自监督损失选点；读取头逐任务验证AUC选点（含零修正）；新增任务逐任务验证AUC选点。
- Follow、Forward独立从同一U0初始化。LoRA rank=4/alpha=4，增量零初始化。
- 冻结B、源U和已部署原任务头；新增任务的U更新仅作用于私有副本。

## 并行与保存

使用4进程×2线程，源准备完成后释放同seed依赖；新增任务等待两个条件读取头完成。共享只读数据，源缓存每seed一份。完成阶段校验复用。

保存权重、原始预测/标签/rowid、曲线、停止原因、UTC时间、损失分量、正则与嵌入RMS、参数、耗时、显存、并发资源记录。报告分别给出新版与旧版来源，AUC/Logloss均值、样本方差和SD；不覆盖旧结果。

## 启动

在项目根目录运行：

```bash
python -m sska_rec.kuairand_l2_runner --config configs/kuairand_normalized_l2_20261120.json --dataset-root ~/shared/dataset --cache-root ~/shared/cache --output outputs/kuairand-l2 --references references.json --workers 4 --threads 2 --device cuda:0
```

同命令加`--report-only`汇总；`python experiments/kuairand-normalized-l2/health.py`检查实际PID和单任务日志。
