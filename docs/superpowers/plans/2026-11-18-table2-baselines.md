# 表2初始任务基线实现与云端运行

用户已确定模型为Single Task、SharedBottom、MMoE、PLE、AdaTT、MPT-Rec、SSKA-Rec，并授权复用代码、补齐实现和云端运行。

## 设计

表2保持Census-income（Income、Marital）和Ali-CCP（CTR、CVR）四列。新增SharedBottom、MMoE、PLE、AdaTT各2数据集×5seed，共40个联合模型、80个任务AUC。Single Task、MPT-Rec和SSKA-Rec引用现有匹配产物，不重训。

复用`MPT-Rec/multitaskrec/model.py`中的SharedBottom、MMOE、CGC、PLE及其依赖，保存来源文件哈希；通过局部提取形成可独立部署的基线模块。AdaTT采用Meta原作者实现的AdaTT-sp，保留任务本地专家残差与全任务门控，记录原代码提交和许可证。统一接入已有`tri_data.build_spec`和公共dataset/cache。

SharedBottom沿用数据集hidden/tower；MMoE为3专家；PLE为2层、每任务1专家及1共享专家，各层专家宽度为数据集hidden对应的单层宽度；AdaTT-sp为2融合层、每任务1专家，层宽同hidden，启用本地残差。各模型使用同一字段、嵌入维度、Adam、原学习率/batch、FP32、不打乱、30轮/patience5。嵌入L2为0.006除以嵌入元素总数，DNN L2为3e-5，继承已采用的归一化正则方式。联合模型用两个初始任务的平均验证AUC选择最早最佳同一检查点；训练损失为两个任务BCE之和加正则，不使用新增目标标签。

## 执行

- [x] 添加模型测试，检查四种模型输出、末批单样本、反向梯度与AdaTT原作者等价性；先观察缺少模块的失败。
- [x] 复用三种现有基线，接入AdaTT原作者模块。
- [x] 添加训练与调度测试，检查仅初始标签参与训练、平均验证选点、提交哈希核验、已完成跳过和队列40任务。
- [x] 实现训练入口、原始预测/标签/rowid/曲线/权重/时间与来源记录、OS锁、统一worker池和汇总。
- [x] 在云端现有PyTorch环境运行定向测试及微型端到端检查。
- [x] 更新正文表2，将SharedBottom放在Single Task后面；同步比较说明。
- [x] 核对无存活重复worker与数据身份；沿用8CPU/30GiB/12GiB、4进程×2线程启动正式训练。
- [x] 记录实际资源占用及启动收据，检查正式epoch日志与任务状态。
- [ ] 40个模型全部完成后备份并核验产物。

代码位于当前已有隔离worktree，原有未跟踪临时文件不纳入提交。新增实验输出使用独立目录，不复制dataset。

## 正式启动记录

北京时间2026-11-18 15:24:45启动，调度器PID3245；代码提交c02f5c9a97c2f4c7f0816c4d7539a99618d5c2ea。Census和Ali-CCP的数据签名均与原主实验一致。15:27首批四模型已分别进入SharedBottom第7轮、MMoE第5轮、PLE第2轮、AdaTT第5轮，队列无失败。实际配额8CPU/30GiB/12032MiB，4进程×2线程，容器内存约9.37GiB、显存1576MiB。启动收据和资源观察保存在工作区cc-run-logs/table2-launch-20261118.json及table2-baselines-parallel-profile-20261118.json。
