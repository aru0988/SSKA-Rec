# 表2初始任务基线

正文顺序：Single Task、SharedBottom、MMoE、PLE、AdaTT、MPT-Rec、SSKA-Rec。

新增训练为SharedBottom、MMoE、PLE、AdaTT，覆盖Census-income的Income/Marital和Ali-CCP的CTR/CVR。每数据集沿用原5个seed，共40个联合模型。已有ST、MPT、SSKA结果通过`references.json`引用，来源路径及SHA256一并保存。

## 实现与协议

SharedBottom、MMoE和PLE的专家/门控实现提取自现有`MPT-Rec/multitaskrec/model.py`，仅将无参数`squeeze()`改为`squeeze(-1)`以保持单样本批次维度。MMoE用3专家；PLE用2层、每任务1专家及1共享专家，两层宽度对应原hidden。AdaTT采用原作者提交`0efdd821179029bbd599043e353841c1afd3fcfa`的AdaTT-sp，源码及MIT许可证保存在`sska_rec/vendor/`，每任务每层1专家，2层宽度对应原hidden，保留本地残差。来源、SHA256及网络配置见`sska_rec/vendor/baseline-provenance.json`。

输入字段、标签定义、完整训练/验证/测试划分、嵌入维度、原学习率和batch均经现有`tri_data.build_spec`加载，并与主实验`data-signature.json`逐项比较。两个初始任务BCE相加，Adam、FP32、不打乱、最多30轮、patience5；以两个初始任务的平均验证AUC选择同一最早最佳检查点。归一化嵌入L2为0.006/嵌入元素数，DNN L2为3e-5。Census无dropout；Ali-CCP沿用SharedBottom/MMoE的0.1/0.3，PLE两层分别0.1/0.3；AdaTT保留原作者实现，不添加提取层dropout。

## 运行

```bash
/opt/conda/envs/pytorch-2.1.1/bin/python -m sska_rec.table2_baselines \
  --dataset-root /home/u2025140625/shared/dataset \
  --cache-root /home/u2025140625/shared/cache \
  --source-root /home/u2025140625/experiments/tri-three-datasets-20261112/outputs/tri \
  --output outputs/table2 --workers 4 --threads 2 --device cuda:0
```

主进程仅在CPU核验数据，复用公共缓存；Linux fork子进程共享只读数据页，各模型独立初始化CUDA。通过调度锁和模型锁防止重复worker；已完成模型核验产物及协议后跳过。`core`支持显式`SSKA_DEVICE=cpu`以跳过导入时的CUDA探测，调度入口设置该变量，子进程再按`--device`设置设备。

每模型保存权重、验证/测试原始概率、两个任务标签、rowid、逐轮曲线、训练时间、选点/停止原因和COMMITTED哈希；总目录保存数据匹配收据、身份、队列状态、`summary.json`和`tables.md`。未完成模型的中断产物保留，恢复时只重启未完成模型。

## 核验

云端PyTorch 2.1.1环境已通过15项模型/训练/队列定向测试，以及1项独立进程CUDA fork测试。覆盖原代码等价性、AdaTT逐层融合和本地残差、单样本批次、反向梯度、归一化正则、标签隔离、验证选点、原始结果对齐、完成结果复用和损坏产物拒绝。

并发配置沿用现有8核CPU/30GiB内存/12GiB显存实例的4进程×2线程；正式训练后记录实际资源占用，不另跑并发实验。队列完成后核验并备份结果，再处理资源。
