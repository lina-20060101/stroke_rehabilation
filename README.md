# 入组量表与早期 EEG/fNIRS 预测卒中上肢运动结局

**主结局**为疗程结束时的 FMA-UE 运动评分（0–66 分）。主模型联合入组 FMA-UE、
入组 ARAT 和截止 session 2 的患手任务 EEG/fNIRS。主比较评估在双量表基础上加入
脑信号后，留出患者的末次 FMA-UE 平均绝对误差（MAE）改变多少分。

## 完整实验与新患者预测

将官方 FIF 压缩包解压到数据根目录后，在仓库根目录运行：

```powershell
python -m pip install -e '.[signals,clinical]'
python scripts/build_manifest.py --data-root D:/Downloads/stroke-dataset-reference/fif-data --output data/processed/recordings.csv
python scripts/recovery_experiment.py fit --manifest data/processed/recordings.csv --scales D:/Downloads/stroke-dataset-reference/Patients_scales.xlsx --max-session 2 --output-dir outputs/recovery-experiment/session2
```

每名患者留出一次，同一患者的所有试验和 session 一起留出。所有对照在同一批有双量表、
末次 FMA-UE 和早期配对信号的患者上评估。每折单独拟合质量判别器、特征插补、标准化及
岭回归（alpha=10）。主要模型估计相对入组 FMA-UE 的变化，再加回入组分数得到末次预测。
分析 session 1 时，使用 `--max-session 1` 并指定单独输出目录。

| 比较 | 回答的问题 |
|---|---|
| `joint_local` 对 `scales_only` | 已知入组双量表，加入时间窗加权的脑信号能否降低误差？ |
| `joint_direct` 对 `signals_only` | 同样直接回归末次评分，在脑信号上加入双量表有多少增益？ |
| `scales_only` 对 `fma_only` | 已知入组 FMA-UE，加入入组 ARAT 有多少增益？ |
| `scales_eeg`、`scales_fnirs` | 双量表分别加入一种信号时表现如何？ |
| `joint_local`、`joint_equal`、`joint_run_shared` | 联合模型中局部时间门控有什么作用？ |
| `carry_forward`、`mean_gain` | 直接结转入组评分，或加上训练患者平均改善，能达到什么误差？ |

运行结果包含队列和 run 审计、逐位留出患者的预测、各模型 MAE、RMSE、偏差及变化量 R²。
`paired_comparisons.csv` 的 MAE 减少量为正代表该模型误差更低。区间由固定折外预测
按患者配对重采样得到，描述本队列的个体差异。`protocol.json` 保存运行参数。
最后在全部合格患者上重拟合联合模型和双量表模型，保存为 `recovery_model.joblib`。

在官方队列的 session 1–2、seed=42 实验中，13 名患者的双量表模型 MAE 为 1.365 分，
局部门控联合模型为 1.542 分，入组 FMA-UE 加训练患者平均改善为 1.346 分。
在相同的直接末次评分回归中，从仅脑信号到加入双量表，MAE 从 21.643 降至
8.560 分。上述结果支持入组量表对末次评分的预测价值。联合模型对个体改善量的
折外 R² 为 −0.522，加入脑信号对改善幅度的预测价值仍需独立验证。

预测新患者时准备基线 CSV，表头为 `subject_id,fma_ue_before,arat_before`。
ID 应与该患者早期记录清单一致。下面命令读取本实验生成的模型：

```powershell
python scripts/recovery_experiment.py predict --model outputs/recovery-experiment/session2/recovery_model.joblib --baselines new_baselines.csv --manifest new_recordings.csv --output-dir outputs/new-predictions
```

`predictions.csv` 给出双量表预测、联合预测和信号可用状态。预测截止点取自保存的模型。
仅有入组量表时可省略 `--manifest`，获得双量表预测。新患者评估需要独立队列，
项目内报告的是原始小样本的留一患者验证。

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

### 时间窗质量判别器与留一患者评估

完成上面的清单步骤后，使用相同环境直接从 FIF 运行：

```powershell
python scripts/evaluate_window_quality.py --manifest data/processed/recordings.csv --scales D:/Downloads/stroke-dataset-reference/Patients_scales.xlsx --max-session 2 --output-dir outputs/window-quality
```

EEG 分析 0.5–1.5、1.5–2.5 秒的对侧 α/β 功率，fNIRS 分析 4–6、6–8 秒的 HbO。
两者各用刺激前 2 秒作基线。判别器读取相应窗的电压或光密度，计算失效通道比例、
相邻采样点持平比例和异常跳变幅度。突跳大小以同一早期 run 的典型采样差为参照。
这套处理用于整次记录完成后的分析。

每折用训练患者的原始窗与其加扰副本（削顶、通道失效或阶跃）训练两个独立的
线性排序器。默认随机种子为 42，可用 `--seed` 指定。模型为质量损失指标拟合非负系数，
再以 sigmoid 输出相对权重。原始窗没有人工质量标签。每个响应的最终权重取该窗和
同模态基线窗权重的较小值。`local` 使用局部权重，`run_shared` 对同一 run
同一模态共享局部权重均值，`equal` 全部等权。三种方案分别汇总相同的六项脑信号特征。

使用同一批患者逐人留出，训练折内插补、标准化并拟合固定的岭回归（alpha=10）。
回归目标为前后变化量，加回已知基线并限制在 0–66 分得到末次预测。`clinical` 只输入
基线 FMA-UE，其他三种方案额外输入脑信号特征。

- `endpoint_metrics.csv` 和 `heldout_predictions.csv` 给出四种方法的 MAE 和逐患者预测。
- `heldout_window_weights.csv` 给出留出患者每个响应窗的分数、基线分数和最终权重，
  并保存仅基线加扰时的权重。定位实验在每次刺激选一个响应窗加扰，
  `stress_target` 指示选中的窗，`stress_drop` 是相对原始权重的下降量。
- `localization_metrics.csv` 按模态、扰动类型汇总选中窗的降权幅度和比例。
  该实验固定原始 run 的尺度参照，只检验判别器的局部评分。结局 MAE 使用原始记录。
- `run_audit.csv` 记录候选 run 的处理状态和排除原因。

## 逐时间片运动反应识别

本项目探索：EEG 和 fNIRS 的质量会随时间变化时，模型能否在事件相关时间片上动态调整模态权重，提高卒中康复运动任务识别在信号退化时的稳定性。

### 运动反应识别模型

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
