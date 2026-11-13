# 卒中康复中的质量感知 EEG-fNIRS 融合

本项目用于探索：在预测卒中后上肢功能结局时，EEG 和 fNIRS 的信号质量指标能否帮助模型动态调整各模态的贡献，并在已有临床信息的基础上提供额外预测价值。

## 研究问题

输入基线临床量表以及同步采集的 EEG/fNIRS 运动任务信号；提取各次记录的质量特征；检验质量门控融合是否比临床单模态、EEG、fNIRS 和普通融合更稳健。

## 当前状态

当前为研究代码初始骨架，不包含任何患者信号。拟使用的公开队列包括 16 名卒中患者和 84 次康复记录。临床量表主要为康复前后测量，并非每次信号记录都配有量表。因此，该数据适合开展探索性信号质量分析和方法流程验证，不足以单独证明可泛化的个体康复轨迹预测模型。

质量特征是信号可靠性的代理指标，不是临床严重程度，也不是人工标注的伪迹真值。

## 开发步骤

1. 检查公开数据文件，生成患者/康复 session/run 清单。
2. 实现 EEG 与 fNIRS 的模态专属预处理和质量指标。
3. 按任务和 run 提取 EEG、fNIRS 及跨模态特征。
4. 建立临床基线、EEG、fNIRS、普通融合对照模型。
5. 加入质量门控融合，并评估自然低质量记录、模拟信号损坏和模态缺失下的稳健性。
6. 按患者划分训练与验证数据，评估对随访 FMA-UE/ARAT 的增量预测价值。

## 数据管理

原始信号不得提交到 Git。仅提交代码、配置、空白数据清单和许可允许的衍生结果。临床信息按患者编号与记录清单关联，随访量表不能作为预测输入。

公开数据集 DOI：<https://doi.org/10.6084/m9.figshare.28904942>。数据集采用 CC BY 4.0；引用时应同时引用数据集论文。

## Linux 环境

预期使用 Python 3.10 或更高版本。建议先在服务器创建独立虚拟环境，再安装本项目。GPU 版 PyTorch 需根据服务器 CUDA 版本单独选择；CUDA 未确认前，不锁定 PyTorch 版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[signals]'
```

待 EEG/fNIRS 处理流程确定后，再安装训练依赖：`python -m pip install -e '.[train]'`。

## 生成文件清单

先在 Git 工作区以外解压数据，再运行：

```bash
python scripts/build_manifest.py --data-root /path/to/extracted_dataset
```

生成的文件清单默认写入 `data/manifests/recordings.csv`。临床量表单独维护在 `data/manifests/clinical_scales_template.csv`，核对原始表格字段后按 `subject_id` 关联。

## 质量指标解释

当前 EEG 指标包含有效样本比例、平直通道比例、振幅异常比例、高频功率比例代理量和工频噪声比例。高频功率只能作为肌电/噪声代理量；没有 EOG/EMG 标注时，不能把它解释为已确认的伪迹标签。

当前 fNIRS 指标包含有效样本比例、平直通道比例、导数尖峰比例代理量和振幅异常比例。原始光密度数据的光源-探测器接触质量、波长特异信噪比等指标，需要结合 SNIRF 中的波长与光极元数据后再实现。
