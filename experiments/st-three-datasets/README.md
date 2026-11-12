# 三数据集全部目标Single Task

Census四目标、Ali-CCP三目标、KuaiRand四目标，各五seed，共55份结果。每份ST从头训练完整Embedding、MLP和预测塔；与五结构实验共享完整样本/字段/划分。

统一嵌入L2为`.006/(2*N_E)*sum(E²)`，DNN L2为`3e-5`。30轮、patience5、验证AUC最早最佳；LR/batch分别为Census`.001/256`、Ali-CCP`.0001/2000`、KuaiRand`.001/512`。只读复用已核验的KuaiRand Follow/Forward十份修正版ST，其余预计新增45份。

```bash
bash experiments/st-three-datasets/run.sh
python experiments/st-three-datasets/health.py
python -m sska_rec.st_runner --config configs/st_three_datasets.json --dataset-root "$HOME/shared/dataset" --cache-root "$HOME/shared/cache" --comparison-root "$HOME/experiments/tri-three-datasets-20261112/outputs/tri" --output outputs/st --report-only
```

并行版本允许一个Census ST进程与主队列同时使用GPU；Census结束后，KuaiRand和Ali-CCP的ST等待主队列125组完成、worker退出并释放OS锁后运行。每个Census模型启动前检查至少2GiB显存余量，CPU训练线程设为2，所有训练协议保持原设置。独立输出`outputs/st`；不改动主队列代码或配置，不复制dataset/cache。对齐主队列数据指纹、raw labels/rowids后生成六模型表和相对ST的配对统计。

保存完整曲线、BCE/L2分项、train/val指标、raw预测、权重、停止原因、成本、UTC时间及COMMITTED。资源释放需等待主队列和本ST队列均完成并备份核验。
