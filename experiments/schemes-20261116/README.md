# 方案1与方案2

两套独立输出，沿用三数据集完整划分、原5seed、FP32、原学习率/batch、30轮上限和patience=5。

| 方案 | 入口 | 新训练 | 复用 |
|---|---|---|---|
| 1 | `python -m sska_rec.scheme1_runner` | G/S重建解码器、剩余信息U、三路表示路由下的新增任务模型 | B、G/S缓存；通过依赖/原始预测核验的MPT对照 |
| 2 | `python -m sska_rec.scheme2_runner` | 每个原任务的条件低秩读出头 | 原B、原U、G/S缓存、原MPT概率 |

方案2仅一阶段，每旧任务新增Census/KuaiRand 1,541、Ali-CCP 773参数。没有二阶段选项，没有U微调；每个任务独立选验证AUC检查点，包含epoch0原预测。方案1使用新的补充U和单层三路门控，不增加独立U开关，原任务保留原MPT路径。

## 云端运行

从本分支检出代码，在已有实例内使用公共dataset和旧产物；无需复制dataset或feature-cache。

```bash
export SOURCE_ROOT="$HOME/experiments/tri-three-datasets-20261112/outputs/tri"
export DATASET_ROOT="$HOME/shared/dataset"
export CACHE_ROOT="$HOME/shared/cache"
export PYTHON_BIN=/opt/conda/envs/pytorch-2.1.1/bin/python
export WORKERS=2
export THREADS_PER_WORKER=2
bash experiments/schemes-20261116/run.sh 1
bash experiments/schemes-20261116/run.sh 2
```

两条命令分别运行独立队列。默认队列内部同时运行2个dataset/seed任务；若同时运行两套队列，需要把总并发一起计入CPU/内存/显存预算。先根据实例配额设置WORKERS和THREADS_PER_WORKER。每个任务有OS锁，已存活worker不会重复启动。健康检查只读取状态。

```bash
python experiments/schemes-20261116/health.py --output outputs/scheme2
python -m sska_rec.scheme2_runner --report-only --output outputs/scheme2
python -m sska_rec.scheme1_runner --report-only --output outputs/scheme1
```

限定一个独立任务或仅准备方案1源：

```bash
bash experiments/schemes-20261116/run.sh 2 --dataset census --seed 1685480945
python -m sska_rec.scheme1_runner --dataset-root "$DATASET_ROOT" --source-root "$SOURCE_ROOT" \
  --cache-root "$CACHE_ROOT" --output outputs/scheme1 --dataset census --seed 1685480945 --pretrain-only
```

## 复用和版本

- `--source-root`支持原`outputs/tri`、单数据集目录或单seed目录；显式来源不完整/校验失败时直接报错，不回退重训B/U。
- Census必须是完整划分，不能使用旧20k/5k/10k源。
- 每次核验data_signature、原始数据与划分、源模型/实现、COMMITTED文件和缓存owner；原KuaiRand的历史引用继续检查实际外部目录，不复制权重。
- 方案1复用B/cache，不把旧U编码器或旧含U模型当作新方案结果；正常需要训练15个解码器、15个U源和100个含U模型，25个MPT结果通过核验后引用。
- 方案2复用原B/U，只训练6个原目标×5seed=30个读出头，不运行新增目标或探针。
- 已完成新阶段复用前重新核验contract、权重和raw；失败未完成阶段保留中断快照。更换代码/源/配置时新建输出版本。
- 新权重只保存新模块；B/U和dataset使用来源收据引用。Scheme1的新任务完整state保存其私有适配模块及U状态，沿用现有checkpoint格式。
- 汇总对val/test原始概率重算AUC/Logloss，检查配对标签/rowid，输出5seed均值、样本SD/方差、配对差值95%t区间，以及来源、参数、时间和成本。有效负结果完整保留。

设计记录见`docs/superpowers/specs/2026-11-16-scheme-1-complementary-knowledge-onboarding.md`和`2026-11-16-scheme-2-frozen-u-old-task-readout.md`。实现中没有启动云端训练。
