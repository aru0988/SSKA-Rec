# 三数据集全部目标Single Task补充基线

2026-11-12用户确认：覆盖初始任务和新增目标，不中断已启动的五结构实验。

| 数据集 | 单任务目标 | seed数 | 结果数 |
|---|---|---:|---:|
| Census-income | Income、Marital、Education、Sex | 5 | 20 |
| Ali-CCP | CTR、CVR、BSI | 5 | 15 |
| KuaiRand-Pure | Click、Like、Follow、Forward | 5 | 20 |

总计55份单任务结果。每个目标独立从头训练完整Embedding、MLP和预测塔，不读取MPT的B/U表示或权重。

## 数据与模型

沿用五结构实验的完整数据、输入字段、编码、目标定义及划分。Census为199523/49881/49881；Ali-CCP为10M/1M/10M；KuaiRand使用同一全曝光完整自然日划分。保持原有各数据集五seed，记录paired_seed和实际model_seed。模型种子为paired_seed+目标在完整任务列表中的序号+2，使已完成KuaiRand Follow/Forward的seed偏移仍为+4/+5。

网络采用现有SingleTask完整网络。隐藏层、嵌入维度及预测塔宽度对应各数据集B的基础配置；Ali-CCP采用同B的dropout[.1,.3]，其余无dropout。

三个数据集ST统一采用已修正的正则尺度：embedding L2=.006/(2*N_E)*sum(E²)，DNN L2=3e-5。Adam、FP32、按既有顺序训练、30轮上限、patience5、独立目标验证AUC最早最佳。学习率和batch沿用数据集原设置：Census .001/256、KuaiRand .001/512、Ali-CCP .0001/2000。不根据测试表现修改配置。

## 复用

- KuaiRand Follow/Forward已有修正ST各5seed，检查源码、训练配置、原始数据和预处理身份、完整权重、原始概率、rowid/labels和COMMITTED后只读引用，预计复用10份。
- 旧Census ST嵌入L2未归一化，与本版不一致，保留旧记录，本版重新训练。
- Census20、Ali-CCP15、KuaiRand初始任务10，预计新增45份。发现其他完全匹配结果则登记后扣除，不重复训练。

## 执行

使用独立GitHub分支experiment/single-task-three-datasets-20261112和独立云端目录~/experiments/single-task-three-datasets-20261112。不覆盖正在运行的~/experiments/tri-three-datasets-20261112。

独立ST队列等待五结构主队列完成且实际worker退出后启动GPU训练；不与当前worker争用显存/算力。等待期间只记录队列状态，不能重复启动。若主队列失败/暂停，ST保持等待，直到主队列完成或用户另行指定。

ST运行前逐数据集核对主队列data-signature中的原始SHA、特征顺序、每划分样本数、rowid、labels及输入哈希；差异时停止定位，不换划分或静默覆盖。复用公共dataset/cache，不复制数据集。

## 记录与交付

每份模型保存训练/验证曲线、train AUC及Logloss、BCE/嵌入L2/DNN L2、嵌入RMS、最优与实际轮次、停止原因、val/test原始概率和rowid/labels、权重及哈希、参数、耗时/显存、UTC时间。

汇总55份ST的五seed均值和样本SD，并与五结构主队列按目标/paired_seed合并为六模型表。原任务使用MPT原预测及含U组冻结的原任务线性路径；新增任务使用对应接入预测。报告每个模型相对ST的AUC/Logloss配对差值及95%t区间，不以单个seed决定负迁移。

全部主队列和ST结果备份核验结束后，再释放计算资源；保留持久存储。

## 并行修订

用户要求尝试并行。GPU六次采样4%—8%、显存4459/12032MiB，允许一个Census ST进程与原主进程并行，CPU线程2、DataLoader worker2。模型/优化/batch/FP32不变，保存并发计时来源。每份新模型前显存空闲至少2GiB；Census完成后等待主队列退出再运行KuaiRand/Ali ST。旧等待队列只在确认尚无训练结果时停止；新版本独立目录，不覆盖原主代码。全部有效完成结果保留并复用。
