# Census-income轻量U微调
新增BitFit、SSF、输出Adapter三组，Education/Sex各5seed，共30模型。已有五组50模型只读复用，合并报告80模型。
配置：configs/u_peft_census.json。B/U、旧线性头、固定20k/5k/10k索引、共享数据和缓存均复用前轮。
BitFit：两层偏置384，总27302；SSF：线性层后、既有ReLU之前缩放/平移初值1/0，新增768，总27686；Adapter：128→8→128，降维默认、升维零初始化，新增2184，总29102。
新任务三路头初始化与前轮一致；Adam .001、batch256、30轮、patience5，验证AUC最早最佳；原塔L2=3e-5，无验证回退。
性能：每seed固定表示常驻GPU，按原批次缓存冻结U输出供Adapter读取；BitFit/SSF保留可训练变换的前向与反向。FP32和批次顺序不变，逐模型独立计时/显存，旧耗时保留原来源。
启动：bash experiments/u-peft-census/run.sh。输出：outputs/u-peft。模型peft_models.py，训练peft_runner.py，报告peft_report.py。
