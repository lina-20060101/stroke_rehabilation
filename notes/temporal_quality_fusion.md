# 事件时间对齐的逐时间片质量门控模型

## 目的

模型的主任务暂定为识别一次刺激后的“运动反应 / 无运动反应”。每个 EEG/fNIRS 模态在每个事件时间片都有自己的质量和预测不确定性；融合权重按时间片计算，而不是整条记录只算一次。

这是待验证的研究原型，不是已确认的新颖算法或临床工具。

## 数据流

```text
EEG 原始流   -> EEG 独立预处理 -> 事件时间片特征 h_e(t) 与质量 q_e(t) --┐
                                                                            ├-> 局部门控 -> 融合时间序列 -> GRU -> trial 分类
fNIRS 原始流 -> fNIRS 独立预处理 -> 事件时间片特征 h_f(t) 与质量 q_f(t) --┘
                                                        时间 t -> 时间编码
```

每个时间片的门控分数由质量、分支不确定性和刺激相对时间共同决定：

```text
u_m(t) = 该模态在时间片 t 的归一化预测熵
score_m(t) = quality_score_m(q_m(t), time_embedding(t)) - beta * u_m(t)
alpha(t) = softmax([score_EEG(t), score_fNIRS(t)])
z(t) = alpha_EEG(t) * h_EEG(t) + alpha_fNIRS(t) * h_fNIRS(t)
```

质量较差或分支不确定性较高时，预期该模态权重下降。实际是否成立，须通过损坏实验验证；目前网络并没有数学保证质量分数变差时权重必定单调下降。

## 为什么不用原始采样点直接融合

EEG 和 fNIRS 采样率不同，且血流响应比电活动慢。数据论文报告 EEG 为 500 Hz，fNIRS 为 13.56 Hz，并通过共同触发事件同步设备。应当：

1. 根据刺激/任务触发事件切出 trial。
2. EEG 与 fNIRS 各自使用合适的滤波、伪迹处理和时间窗。
3. 在共同的“相对事件时间”网格上计算特征 token 和对应局部质量指标。
4. 保留 fNIRS 的延迟响应；不要把原始 fNIRS 插值到 EEG 采样率后逐点融合。
5. 具体事件窗与 bin 宽度须检查随数据提供的预处理脚本和标签后确定，不在未检查数据时硬编码。

模型模块接收已经计算好的特征序列：`[batch, time_bins, feature_dim]`；不负责读取 FIF/SNIRF 文件，也不负责原始信号重采样。

## 标签

数据集事件记录包含左右刺激标记，文件名/记录信息包含目标手和患侧。可据此生成目标运动反应与非目标无反应标签，并使用 EEG 文件中的 EMG 通道核验实际肌肉活动。标签定义要经过数据文件检查；模型输入不使用 EMG，除非研究问题明确改成多模态 EEG-fNIRS-EMG。

FMA/ARAT 不作为本模型的 trial 级输入或主标签。若后续要把 FMA 用作辅助目标，只能在真实量表对应的 session 上计算辅助回归损失；无对应量表的 session 必须屏蔽，不能复制前测/后测分数。

## 训练损失

`src/stroke_qafusion/models/temporal_losses.py` 提供了以下损失项：

- trial 级分类损失；
- EEG/fNIRS 单模态分支辅助分类损失；
- 门控权重与两分支实际分类误差的校准损失；
- 相邻时间片权重的轻度平滑项；
- 局部信号损坏后的权重下降排序损失；
- 轻度损坏前后分类输出的一致性损失。

当前实现不含数据集读取、局部特征提取或真实数据训练脚本。需在完成数据检查后再确定局部质量指标、时间片宽度和扰动范围。

## 输入/输出契约

模型文件：`src/stroke_qafusion/models/temporal_quality_fusion.py`

输入：

- `eeg_features`: `[B,T,D_e]`
- `fnirs_features`: `[B,T,D_f]`
- `eeg_quality`: `[B,T,Q_e]`
- `fnirs_quality`: `[B,T,Q_f]`
- `event_time`: `[B,T]`，相对事件时间，秒
- 可选 `available`: `[B,T,2]`，按时间片标记模态是否可用

输出：

- `trial_logits`: 每条 trial 的类别 logits
- `weights`: `[B,T,2]`，逐时间片权重，最后一维顺序 EEG、fNIRS
- 单模态 trial logits 和时间片 logits，供辅助损失与诊断使用

简要调用方式：

```python
from stroke_qafusion.models.temporal_quality_fusion import TemporalQualityAwareFusion
from stroke_qafusion.models.temporal_losses import temporal_quality_objective

model = TemporalQualityAwareFusion(
    eeg_feature_dim=..., fnirs_feature_dim=...,
    eeg_quality_dim=..., fnirs_quality_dim=...,
    num_classes=2,
)
outputs = model(eeg_features, fnirs_features, eeg_quality, fnirs_quality, event_time)
losses = temporal_quality_objective(outputs, labels)
loss = losses["total"]
```

## 评价设计

- 训练、验证、测试按患者分组，患者的所有 session/run/trial 必须留在同一折。
- 比较单模态、普通拼接、全 trial 质量门控、逐时间片质量门控。
- 对局部时间区段分别施加噪声、通道丢失、幅度剪切或 fNIRS 漂移/尖峰。
- 验证受损区段的对应模态权重是否下降、未受损区段是否基本保持、分类性能是否更稳。
- 报告 balanced accuracy、macro-F1、AUROC、Brier/ECE，以及不同损坏强度下的性能变化。
- 多 trial 不等于多患者；该公开数据只有 16 名患者，结果只能作为探索性验证。

## 当前限制

- 尚未在服务器安装 PyTorch，也未用真实数据跑通。
- 逐时间片特征/质量计算尚需结合实际数据字段实现。
- 时间网格需要依据事件标记和预处理文件确定。
- softmax 门控不是天然单调质量函数；排序扰动损失与压力测试用于检验这一点，不能保证临床泛化。
