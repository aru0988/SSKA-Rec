# Census原任务U接入实验

固定协议：`configs/u_access_census.json`。5个配对seed，五模型，不按测试结果修改结构、训练预算或筛选seed。

- Income/Marital初始目标；从第一步排除Income、Marital、Education、Sex四标签字段，嵌入119维。
- 既有训练/验证/测试划分内，按Income×Marital组合标签固定分层抽取20,000/5,000/10,000条。划分种子20260929，抽样种子20261111；仅保存索引与身份，不复制dataset。
- 每seed训练一份B及一份自监督U，各30轮上限、patience5、Adam .001、batch256。B按验证两任务平均AUC（等价于原manager两任务AUC之和）选最早最佳；U按验证自监督损失选最早最佳。
- 冻结B、原预测塔与共享U源。每模型两个任务各有自己的接入头；同时优化两个头的平均BCE，不额外添加接入L2。按验证两任务平均AUC选最早最佳，30轮上限、patience5。固定时序批次，不动态调LR。不使用验证双指标回退。
- mpt：原预测。linear：U→1修正129参数/任务。gated_linear：两条U→1共258。original：[G,S0,S1,U]两条512→32→1，共32898。bottleneck：U的128→8→128（ReLU），[h,U]→1门控后残差加入冻结预测塔前，共2441。
- 修正输出层初始化0；门控默认PyTorch初始化。每任务初始化种子为模型seed+20+任务下标，各模型一致；零输出保证初始概率逐位等于MPT。
- h使用MPT原来的任务门控融合G和环境感知S，与原主干保持相同运算顺序。U预训练方法不变。
- 保存全部有效结果、检查点、val/test原始概率和行ID、曲线、参数、停止原因、UTC时间、耗时与峰值显存。源身份或数据身份改变禁止静默复用；中断仅从原seed重启未完成阶段，不声称优化器续训。
- 以全5seed输出每任务AUC/Logloss均值±样本SD，以及相对mpt/original的配对差值95% t区间（df4）。统计仅解释固定小划分；不推断全量数据/新任务效果。

启动：

```bash
python -m sska_rec.u_access --dataset-root /path/to/shared/dataset --output outputs/u-access
```

汇总：在同一命令后添加`--report-only`。状态：`outputs/u-access/status.json`。原始dataset不提交仓库；本分支不改主分支。
