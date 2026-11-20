# 入组量表与早期 EEG/fNIRS 预测卒中上肢运动结局

**主结局**为疗程结束时的 FMA-UE 运动评分（0–66 分）。主模型使用入组 FMA-UE、ARAT
和康复期间截至第 2 次实验测量的患手任务 EEG/fNIRS。两个一维 CNN 提取局部波形特征，
时间窗质量判别器参与 Transformer 注意力计算。主比较使用相同留出患者的末次评分
平均绝对误差（MAE），考察联合模型相对仅用双量表的改善。

## 完整实验与新患者预测

将官方 FIF 压缩包解压到数据根目录后，在仓库根目录运行：

```powershell
python -m pip install -e '.[signals,clinical,train]'
python scripts/build_manifest.py --data-root D:/Downloads/stroke-dataset-reference/fif-data --output data/processed/recordings.csv
python scripts/recovery_experiment.py fit --architecture cnn-transformer --manifest data/processed/recordings.csv --scales D:/Downloads/stroke-dataset-reference/Patients_scales.xlsx --max-session 2 --output-dir outputs/cnn-transformer/session2
```

默认的 `fit` 架构是 `cnn-transformer`。`--architecture ridge` 可单独复现原来的岭回归流程。
分析 session 1 时使用 `--max-session 1`，各次实验分别指定输出目录。

### CNN 局部判断与时间质量注意力

1. 每次患手目标刺激产生四个波形窗。EEG 取提示后 0.5–1.5、1.5–2.5 秒的对侧感觉运动区电压，
   fNIRS 取 4–6、6–8 秒的长距通道 HbO。各通道减去自身提示前 2 秒的均值，再取有效通道中位波形。
   每个 EEG 窗重采样至 128 点，每个 HbO 窗至 28 点，分别以 µV 和 µM 作固定的 arcsinh 幅度变换。
2. 每种模态有独立的一维 CNN。每个窗输出 12 维表示，以及一个只读取本窗波形的任务证据分数
   `cnn_evidence`。该分数由患者级结局损失学习，取值在 −1 至 1 之间，尚未用动作或伪迹标签校准。
3. Transformer 接收一个双量表向量和四个带模态、刺激相对时间的窗向量。在每一层、每个头中，
   对某个窗的注意力得分为 `QK / √d + cnn_evidence + log(q)`，再对所有输入取 softmax。
   `q` 取同一模态的刺激前质量和响应窗质量中的较小值。它来自训练患者原始窗与加扰窗的排序。
   `q=0` 或缺失波形对应的注意力为零。量表向量的额外偏置为零。
4. 每个刺激取经过两层自注意力更新的量表向量，在 run 内等权汇总刺激、在 session 内等权汇总 run，
   再在患者内等权汇总 session。患者级读出层估计 FMA-UE 改善量，加回入组 FMA-UE 并限定在 0–66 分。
   一名患者只贡献一次结局损失。

每次留出一名患者及其全部记录。质量判别器、CNN、Transformer、岭回归对照的插补和缩放都在训练折内拟合。
默认网络宽度 12、2 层、每层 2 个头，CPU 上用 AdamW 固定训练 20 轮。可用 `--epochs` 指定轮数。
深度模型的三种注意力设置共享队列、初始权重、训练轮数和缺失窗掩码。

| 比较 | 回答的问题 |
|---|---|
| `cnn_transformer` 对 `cnn_transformer_no_quality_bias` | 可用窗的质量先验全部设为 1 时，误差如何变化？两者都保留 CNN 局部判断。 |
| `cnn_transformer` 对 `cnn_transformer_run_shared` | 同一 run 内各模态的总先验权重保持一致，逐窗分配比整段共享有多少增益？ |
| `cnn_transformer` 对 `quality_routed`、`scales_only`、`mean_gain` | 与原联合岭回归、双量表和训练患者平均改善相比，末次评分误差如何？ |
| `joint_local` 对 `scales_only` | 已知入组双量表，加入时间窗加权的脑信号能否降低误差？ |
| `joint_direct` 对 `signals_only` | 同样直接回归末次评分，在脑信号上加入双量表有多少增益？ |
| `scales_only` 对 `fma_only` | 已知入组 FMA-UE，加入入组 ARAT 有多少增益？ |
| `scales_eeg`、`scales_fnirs` | 双量表分别加入一种信号时表现如何？ |
| `joint_local`、`joint_equal`、`joint_run_shared` | 联合模型中局部时间门控有什么作用？ |
| `quality_routed` 对 `scales_only` | 双模态经局部门控后均有信息时用联合模型，否则回退至双量表，这套规则的误差是多少？ |
| `carry_forward`、`mean_gain` | 直接结转入组评分，或加上训练患者平均改善，能达到什么误差？ |

运行结果包含队列和 run 审计、逐位留出患者的预测、各模型 MAE、RMSE、偏差及变化量 R²。
`paired_comparisons.csv` 的 MAE 减少量为正代表该模型误差更低。区间由固定折外预测
按患者配对重采样得到，描述本队列的个体差异。`protocol.json` 保存运行参数。
最后在全部合格患者上重拟合联合模型和双量表模型，保存为 `recovery_model.joblib`。

在官方数据的 session 1–2、seed=42、20 轮实验中，局部质量 CNN＋Transformer 对 13 名患者的
MAE 为 1.210 分，整段共享质量为 1.212 分，关闭质量先验为 1.241 分。原联合岭回归为 1.542 分，
双量表岭回归为 1.365 分，入组 FMA-UE 加训练患者平均改善为 1.346 分。新模型相对双量表的 MAE 减少量为 0.155 分
（患者配对 bootstrap 95% 区间 −0.213 至 0.495），相对整段共享质量的减少量约 0.002 分。
逐窗质量的独立增益和对新队列的泛化能力仍需验证。

`heldout_attention.csv` 记录主模型每位留出患者、每次刺激、每个窗的质量先验、
CNN 局部分数及末层量表向量分配给该窗的注意力。四个窗的注意力加上一次
`clinical_attention` 等于 1。注意力描述模型如何使用输入。

预测新患者时准备基线 CSV，表头为 `subject_id,fma_ue_before,arat_before`。
ID 应与该患者早期记录清单一致。下面命令读取本实验生成的模型：

```powershell
python scripts/recovery_experiment.py predict --model outputs/cnn-transformer/session2/recovery_model.joblib --baselines new_baselines.csv --manifest new_recordings.csv --output-dir outputs/new-predictions
```

`predictions.csv` 的 `predicted_fma` 是实际采用的预测，`prediction_source` 记录来源。
预测命令从模型文件自动选择架构，同时兼容原岭回归模型。CNN 要求两种模态各至少有一个
质量为正、波形有效的窗。个别缺窗在注意力中屏蔽；整种模态不可用时，实际预测取双量表模型，
联合预测列留空。`joint_status` 区分配对信号缺失与加权后失效，`inference_attention.csv` 保存 CNN 的逐窗记录。
这个规则检测完全不可用的模态，质量分数本身只是原始窗与加扰窗的相对排序。
本队列 13 位留出患者都满足双模态条件，因此本次实验无法估计真实回退患者的误差；
整种模态失效时的分支由合成输入测试覆盖。预测截止点取自保存的模型。
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

此处是早期的辅助任务设计，目标为 trial 级“运动反应 / 无运动反应”分类。模型接收各自预处理后的 EEG/fNIRS 时间片特征、局部质量指标和相对事件时间；门控逐时间片输出 EEG/fNIRS 权重，再由 GRU 汇总时序并完成分类。患者级末次 FMA-UE 实验使用上文的 CNN＋Transformer 和岭回归对照。

- 模型：`src/stroke_qafusion/models/temporal_quality_fusion.py`
- 训练损失：`src/stroke_qafusion/models/temporal_losses.py`
- 设计说明与张量接口：`notes/temporal_quality_fusion.md`
- 已有 EEG/fNIRS 整段质量代理指标：`src/stroke_qafusion/quality/`

这个辅助分类原型接收预计算的特征 token，需要另行提供任务标签和数据加载代码。旧的 `quality_aware_temporal.py` 是早期的 FMA 分数回归原型。当前康复结局网络的实现位于 `src/stroke_qafusion/models/recovery_transformer.py`。

## 时间对齐原则

公开数据中 EEG 采样率为 500 Hz，fNIRS 为 13.56 Hz；两种设备有共同事件触发，但生理响应时间尺度不同。每种模态应独立预处理，并基于事件相对时间形成特征 token；不要将原始 fNIRS 插值到 EEG 采样率后逐点融合。细节见模型设计说明和数据论文。

## 数据集与标签注意事项

拟用公开队列包含 16 名患者和 84 次康复记录。任务采用刺激-反应范式；文件/事件信息提供刺激侧和目标手条件，EEG 文件还含双侧 EMG，可用于核对实际运动反应。最终标签规则应检查数据随附脚本与事件文件后再固定。

FMA/ARAT 主要为康复前后量表，并非每个 trial/session 都有分数。因此：
- 不要复制端点量表分数给中间记录。
- 如作为辅助目标，只在真实对应的 session 计算量表损失。
- 按患者而非 trial 随机划分；16 名患者意味着泛化结论有限，当前设计只能作探索性方法验证。

数据集 DOI：<https://doi.org/10.6084/m9.figshare.28904942>。引用时请同时引用数据论文。原始患者数据不得提交到 Git。

## 运动反应分类的训练与验证路线

1. 检查数据文件、预处理示例、刺激事件、目标手信息和 EMG。
2. 按事件切出 EEG/fNIRS trial，独立预处理并生成共同事件时间网格上的特征与局部质量指标。
3. 固定“运动反应/无反应”标签规则，确认 EMG 可用于标签核对。
4. 比较 EEG、fNIRS、普通融合、整段质量门控和逐时间片质量门控。
5. 对局部时间片施加可控损坏，评估受损区段权重是否下降、未受损区段是否保持、最终预测是否稳定。
6. 使用按患者分组的交叉验证，报告 balanced accuracy、macro-F1、AUROC、Brier/ECE 和不同损坏强度下的性能。
7. 只有在更多患者数据上验证后，才讨论临床应用。

## 环境

Python 3.10 或更高版本。患者级 CNN＋Transformer 使用 PyTorch 2.2 或更高版本，支持 CPU。

```bash
python -m pip install -e '.[signals]'
python -m pip install -e '.[train]'
```

原始数据和中间结果放在 Git 工作区以外。
