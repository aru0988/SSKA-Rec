# 三数据集：U线性接入与三路融合

配置：`configs/tri_three_datasets.json`。Census完整划分、Ali-CCP、KuaiRand-Pure；五seed，每个新增目标比较MPT、冻结U、LoRA、SSF、全量U，共125份二阶段结果。

原任务采用冻结B/U后的零初始化线性logit修正。二阶段冻结旧路径，新增目标从同seed同一U0独立接入。Census全部重训；KuaiRand已有B/U及匹配MPT结果通过源码、数据、权重和raw核验后只读复用。Ali-CCP线性头与二阶段均30轮；其他训练也30轮，patience5。

```bash
bash experiments/tri-three-datasets/run.sh
python experiments/tri-three-datasets/health.py
python -m sska_rec.tri_runner --config configs/tri_three_datasets.json --dataset-root "$HOME/shared/dataset" --output outputs/tri --report-only
```

输出：`outputs/tri/status.json`、`summary.json`、`table.md`。每seed记录源复用收据、完整原任务预测和五组新增目标权重/raw/曲线/停止原因/成本/UTC时间。原始dataset使用共享目录，不随实验复制。

旧小划分Census结果保持在旧实验目录；本版不读取为新结果。旧控制的耗时沿用原测量口径并标明来源；缓存和B/U源成本单列。
