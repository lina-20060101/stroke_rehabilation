# 卒中康复中的 EEG-fNIRS 时间质量门控

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