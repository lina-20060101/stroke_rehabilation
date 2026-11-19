# 卒中康复中的 EEG-fNIRS 时间质量门控

仓库文件地图：[notes/repository_guide.md](notes/repository_guide.md)。

## 患者级康复结局数据准备

目标是用基线 FMA-UE 及早期 EEG/fNIRS 预测疗程结束的 FMA-UE。解压官方
`EEG_FIF.zip` 和 `FNIRS_FIF.zip` 到同一数据根目录下的两个子目录，并准备官方
`Patients scales.xlsx`。以下命令从仓库根目录执行：

```powershell
python -m pip install -e '.[signals,clinical]'
python scripts/build_manifest.py --data-root D:/Downloads/stroke-dataset-reference/fif-data --output data/processed/recordings.csv
python scripts/prepare_recovery_data.py --manifest data/processed/recordings.csv --scales D:/Downloads/stroke-dataset-reference/Patients_scales.xlsx --max-session 2 --output-dir data/processed/recovery
```

将路径替换为实际路径。使用 `--max-session 1` 可生成仅含 session 1 的队列，请给两种
截止点指定不同的输出目录。截止点按原始 session 编号判定。某患者缺少 session 1 时，
截止 session 2 的方案允许使用 session 2。量表工作簿提供前后测分数，缺少逐人的量表时间戳，
所以这里用数据集的疗程序号定义早期窗口。

- `patient_features.csv` 每位符合条件的患者一行，包含基线 FMA-UE、信号及质量特征。
- `clinical_outcomes.csv` 保留全体患者的原始量表缺失状态、末次 FMA-UE 和前后变化量。
  监督学习时按 `subject_id` 连接末次 FMA-UE，显式选择输入列。
- `cohort_audit.csv` 记录队列纳入情况和早期有效记录覆盖。`run_features.csv` 记录患手 run
  的特征、触发数和排除原因。

处理以两路 FIF 自带的 `flash_l` / `flash_r` 触发为基准，核对触发序列及相对间隔。
EEG 选取患手对侧的可用感觉运动通道，计算 α/β 频带相对刺激前的对数功率变化。
fNIRS 光密度经 Beer–Lambert 转换和 0.02–0.1 Hz 滤波后，提取刺激后 4–8 秒相对刺激前
2 秒的 HbO 变化。`qc_` 列为整段质量代理量。特征在 session 内取中位数、跨 session
等权平均。这些窗口和频带是预先指定的探索性基线。目标刺激表示患手任务指令，
动作实际执行情况需另用 EMG 核验。训练时按患者划分，标准化、插补和特征选择均在训练折内拟合。

## 逐时间片运动反应识别

本项目探索：EEG 和 fNIRS 的质量会随时间变化时，模型能否在事件相关时间片上动态调整模态权重，提高卒中康复运动任务识别在信号退化时的稳定性。

## 当前主模型设想

主任务暂定为 trial 级“运动反应 / 无运动反应”分类。模型接收各自预处理后的 EEG/fNIRS 时间片特征、局部质量指标和相对事件时间；门控逐时间片输出 EEG/fNIRS 权重，再由 GRU 汇总时序并完成分类。

- 模型：`src/stroke_qafusion/models/temporal_quality_fusion.py`
- 训练损失：`src/stroke_qafusion/models/temporal_losses.py`
- 设计说明与张量接口：`notes/temporal_quality_fusion.md`
- 已有 EEG/fNIRS 整段质量代理指标：`src/stroke_qafusion/quality/`

新模型接收预计算的特征 token，不直接读 FIF/SNIRF，也尚未包含数据集专用的数据加载和局部特征提取代码。时间片宽度、事件标签生成和质量指标需在检查原始数据后确定。旧的 `quality_aware_temporal.py` 是 FMA 分数回归基线，不包含逐时间片门控。

## 时间对齐原则

公开数据中 EEG 采样率为 500 Hz，fNIRS 为 13.56 Hz；两种设备有共同事件触发，但生理响应时间尺度不同。每种模态应独立预处理，并基于事件相对时间形成特征 token；不要将原始 fNIRS 插值到 EEG 采样率后逐点融合。细节见模型设计说明和数据论文。

## 数据集与标签注意事项

拟用公开队列包含 16 名患者和 84 次康复记录。任务采用刺激-反应范式；文件/事件信息提供刺激侧和目标手条件，EEG 文件还含双侧 EMG，可用于核对实际运动反应。最终标签规则应检查数据随附脚本与事件文件后再固定。

FMA/ARAT 主要为康复前后量表，并非每个 trial/session 都有分数。因此：
- 不要复制端点量表分数给中间记录。
- 如作为辅助目标，只在真实对应的 session 计算量表损失。
- 按患者而非 trial 随机划分；16 名患者意味着泛化结论有限，当前设计只能作探索性方法验证。

数据集 DOI：<https://doi.org/10.6084/m9.figshare.28904942>。引用时请同时引用数据论文。原始患者数据不得提交到 Git。

## 训练与验证路线

1. 检查数据文件、预处理示例、刺激事件、目标手信息和 EMG。
2. 按事件切出 EEG/fNIRS trial，独立预处理并生成共同事件时间网格上的特征与局部质量指标。
3. 固定“运动反应/无反应”标签规则，确认 EMG 可用于标签核对。
4. 比较 EEG、fNIRS、普通融合、整段质量门控和逐时间片质量门控。
5. 对局部时间片施加可控损坏，评估受损区段权重是否下降、未受损区段是否保持、最终预测是否稳定。
6. 使用按患者分组的交叉验证，报告 balanced accuracy、macro-F1、AUROC、Brier/ECE 和不同损坏强度下的性能。
7. 只有在更多患者数据上验证后，才讨论临床应用。

## 环境

Python 3.10 或更高版本。PyTorch 需结合服务器 CUDA 环境安装；当前服务器尚未安装 PyTorch，模型也尚未在真实数据上运行。

```bash
python -m pip install -e '.[signals]'
python -m pip install -e '.[train]'
```

原始数据和中间结果放在 Git 工作区以外。
