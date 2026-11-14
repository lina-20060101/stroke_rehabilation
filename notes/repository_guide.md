# 仓库文件说明

本说明按当前 GitHub `main` 分支整理。仓库目前是研究原型代码骨架，不包含患者原始数据，也尚未形成从 FIF/SNIRF 到训练评估的端到端流水线。

## 当前主线

当前实验主线是“事件时间对齐的逐时间片质量门控运动反应识别”，即预测刺激任务中的运动反应/无运动反应。模型按时间片调整 EEG/fNIRS 权重。FMA-UE 暂时是可选的 session 级辅助标签，不是主模型输入。

## 文件目录

```text
.
├── .gitignore
├── README.md
├── pyproject.toml
├── configs/
│   └── default.yaml
├── data/
│   └── manifests/
│       ├── clinical_scales_template.csv
│       └── recordings.csv
├── notes/
│   ├── experiment_plan.md
│   ├── temporal_quality_fusion.md
│   └── archive/
│       └── fma_scale_prediction_experiment_plan.md
├── scripts/
│   └── build_manifest.py
└── src/stroke_qafusion/
    ├── __init__.py
    ├── models/
    │   ├── __init__.py
    │   ├── quality_aware_temporal.py
    │   ├── quality_gated.py
    │   ├── temporal_quality_fusion.py
    │   └── temporal_losses.py
    └── quality/
        ├── __init__.py
        ├── common.py
        ├── eeg.py
        └── fnirs.py
```

## 根目录文件

- `.gitignore`：排除原始数据、处理后数据、模型权重、日志和环境文件；保留数据清单 CSV。
- `README.md`：项目目标、数据限制、主要模型和环境安装说明。
- `pyproject.toml`：Python 包信息、依赖和可选安装组。PyTorch 属于 `train` 可选依赖。

## 配置与数据清单

- `configs/default.yaml`：主任务、分组划分、时间对齐、模型规模、损失权重和评估指标的默认配置。窗口长度和 bin 宽仍待检查真实数据后填写。
- `data/manifests/recordings.csv`：run 级文件索引模板/占位清单，列出 subject、session、target hand、EEG/fNIRS/event 路径。
- `data/manifests/clinical_scales_template.csv`：临床量表关联模板。只记录真实测量时点；不要给中间 session 复制 FMA/ARAT 分数。

原始数据和处理后的大型文件应保存在仓库外；Git 中只放代码、配置和空白模板。

## 实验与方法说明

- `notes/experiment_plan.md`：当前运动反应识别主线的对照、鲁棒性和患者分组评估方案。
- `notes/temporal_quality_fusion.md`：逐时间片模型结构、张量形状、时间对齐原则、损失与限制。
- `notes/archive/fma_scale_prediction_experiment_plan.md`：早期 FMA/ARAT 结局预测方案存档，仅作历史记录。

## 脚本

- `scripts/build_manifest.py`：扫描解压后的数据文件并生成 run 级 CSV 清单；不读写信号内容，也不处理临床标签。

## Python 包

### 质量指标：`src/stroke_qafusion/quality/`

- `common.py`：质量指标函数共用的有限值比例、鲁棒标准化等工具。
- `eeg.py`：EEG 有效样本比例、平直通道比例、振幅异常、高频功率代理量和工频噪声指标。
- `fnirs.py`：fNIRS 有效样本比例、平直通道比例、导数尖峰及振幅异常代理量。
- `__init__.py`：质量指标子包说明。

目前质量指标函数按给定窗口汇总；要用于逐时间片门控，需在每个事件时间片分别调用/扩展这些指标。

### 模型：`src/stroke_qafusion/models/`

- `temporal_quality_fusion.py`：**当前主模型**。接收事件时间片特征、局部质量、事件相对时间；逐片输出 EEG/fNIRS 权重，经 GRU 汇总并输出 trial 分类。
- `temporal_losses.py`：当前主模型的分类、分支辅助、权重校准、局部损坏排序、输出一致性和轻度平滑损失。
- `quality_aware_temporal.py`：早期 FMA 分数回归网络；不包含逐时间片门控，作为比较基线保留。
- `quality_gated.py`：临床基线预测加 EEG/fNIRS 残差的融合基线；需要临床特征作为输入，不是当前信号独立主模型。
- `__init__.py`：模型子包说明。

## 当前尚缺的模块

1. FIF/SNIRF 数据读取和数据集专用预处理。
2. 事件标记与 target hand/患侧信息的标签生成、EMG 标签核验。
3. 将 EEG/fNIRS 各自映射到事件相对时间片特征的生成器。
4. 逐时间片质量指标计算器。
5. 按患者分组的训练、验证、测试循环和指标报告脚本。
6. 模型权重与实验日志保存流程。

因此，当前模型文件是可调用的网络框架，但还不能直接对原始公开数据一键训练。上述缺失部分应在数据包解压并审阅预处理示例后实现。
