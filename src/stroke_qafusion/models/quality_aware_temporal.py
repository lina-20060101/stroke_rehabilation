"""原始 EEG/fNIRS 时间窗的质量门控多模态回归模型。

输入信号张量格式：
    EEG:   [batch, eeg_channels, time]
    fNIRS: [batch, fnirs_channels, time]
           若同时包含 HbO/HbR，可将 HbO 与 HbR 通道拼接到 channel 维。
临床特征: [batch, clinical_features]
质量特征: [batch, 对应模态的质量指标数]

门控网络只读取质量特征，不读取神经信号嵌入。模型返回各模态权重和分支预测，
便于分析质量变化时模型对不同模态的信任是否发生变化。

注意：该模型是研究原型；小样本下应优先用手工特征/低维输入对照，并按患者划分。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class TemporalCNNEncoder(nn.Module):
    """小型 1D CNN：卷积沿时间轴，输入通道对应 EEG 电极或 fNIRS 通道。"""

    def __init__(
        self,
        in_channels: int,
        embedding_dim: int = 32,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels 必须大于 0。")

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=9, stride=2, padding=4, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=7, stride=1, padding=3, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(32, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("时间信号输入形状应为 [batch, channels, time]。")
        return self.features(x)


class FeatureMLPEncoder(nn.Module):
    """临床特征编码器。输入应只包含预测时点已经可获得的信息。"""

    def __init__(
        self,
        in_features: int,
        embedding_dim: int = 32,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QualityGate(nn.Module):
    """仅根据每种模态的质量指标产生三个归一化权重。"""

    def __init__(
        self,
        eeg_quality_dim: int,
        fnirs_quality_dim: int,
        clinical_quality_dim: int,
        gate_hidden_dim: int = 8,
    ) -> None:
        super().__init__()

        def quality_score(dim: int) -> nn.Module:
            if dim < 1:
                raise ValueError("每种模态至少需要一个质量/完整度特征。")
            return nn.Sequential(
                nn.Linear(dim, gate_hidden_dim),
                nn.GELU(),
                nn.Linear(gate_hidden_dim, 1),
            )

        self.eeg_score = quality_score(eeg_quality_dim)
        self.fnirs_score = quality_score(fnirs_quality_dim)
        self.clinical_score = quality_score(clinical_quality_dim)

    def forward(
        self,
        quality_eeg: torch.Tensor,
        quality_fnirs: torch.Tensor,
        quality_clinical: torch.Tensor,
        available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = torch.cat(
            [
                self.eeg_score(quality_eeg),
                self.fnirs_score(quality_fnirs),
                self.clinical_score(quality_clinical),
            ],
            dim=1,
        )

        if available is not None:
            if available.shape != logits.shape:
                raise ValueError("available 形状应为 [batch, 3]，顺序为 EEG、fNIRS、临床。")
            available = available.bool()
            if torch.any(~available.any(dim=1)):
                raise ValueError("每个样本至少要有一个可用模态。")
            logits = logits.masked_fill(~available, torch.finfo(logits.dtype).min)

        return torch.softmax(logits, dim=1)


class QualityAwareTemporalFusion(nn.Module):
    """质量门控 EEG-fNIRS-临床信息多模态回归模型。

    输出连续结局（例如下一时间点 FMA-UE/ARAT，或相对当前值的变化量）。
    """

    def __init__(
        self,
        eeg_channels: int,
        fnirs_channels: int,
        clinical_dim: int,
        eeg_quality_dim: int,
        fnirs_quality_dim: int,
        clinical_quality_dim: int,
        embedding_dim: int = 32,
        output_dim: int = 2,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.eeg_encoder = TemporalCNNEncoder(
            eeg_channels, embedding_dim=embedding_dim, dropout=dropout
        )
        self.fnirs_encoder = TemporalCNNEncoder(
            fnirs_channels, embedding_dim=embedding_dim, dropout=dropout
        )
        self.clinical_encoder = FeatureMLPEncoder(
            clinical_dim, embedding_dim=embedding_dim, dropout=dropout
        )

        # 每个模态单独预测结局，再由质量门控加权，便于检查模态贡献。
        self.eeg_head = nn.Linear(embedding_dim, output_dim)
        self.fnirs_head = nn.Linear(embedding_dim, output_dim)
        self.clinical_head = nn.Linear(embedding_dim, output_dim)

        self.gate = QualityGate(
            eeg_quality_dim=eeg_quality_dim,
            fnirs_quality_dim=fnirs_quality_dim,
            clinical_quality_dim=clinical_quality_dim,
        )

    def forward(
        self,
        eeg: torch.Tensor,
        fnirs: torch.Tensor,
        clinical: torch.Tensor,
        quality_eeg: torch.Tensor,
        quality_fnirs: torch.Tensor,
        quality_clinical: torch.Tensor,
        available: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        z_eeg = self.eeg_encoder(eeg)
        z_fnirs = self.fnirs_encoder(fnirs)
        z_clinical = self.clinical_encoder(clinical)

        pred_eeg = self.eeg_head(z_eeg)
        pred_fnirs = self.fnirs_head(z_fnirs)
        pred_clinical = self.clinical_head(z_clinical)

        weights = self.gate(
            quality_eeg=quality_eeg,
            quality_fnirs=quality_fnirs,
            quality_clinical=quality_clinical,
            available=available,
        )

        branch_predictions = torch.stack(
            [pred_eeg, pred_fnirs, pred_clinical], dim=1
        )  # [batch, 3, output_dim]
        prediction = (branch_predictions * weights.unsqueeze(-1)).sum(dim=1)

        return {
            "prediction": prediction,
            "weights": weights,  # 列顺序：EEG、fNIRS、临床
            "branch_predictions": branch_predictions,
        }
