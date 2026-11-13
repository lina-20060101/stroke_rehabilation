"""仅用 EEG/fNIRS 预测临床量表的质量感知时序模型。

量表分数是训练标签 y，不是模型输入。推理阶段只需要 EEG、fNIRS 以及对应的质量指标。
门控网络只读取信号质量特征，输出 EEG/fNIRS 两个模态的归一化权重。

输入：
    eeg:            [batch, eeg_channels, time]
    fnirs:          [batch, fnirs_channels, time]
                    若同时输入 HbO/HbR，可沿通道维拼接。
    quality_eeg:    [batch, eeg_quality_dim]
    quality_fnirs:  [batch, fnirs_quality_dim]
    available:      可选 [batch, 2]，标记 EEG/fNIRS 是否可用。

输出：
    prediction: [batch, output_dim]，例如 FMA-UE、ARAT 或单个目标。
    weights:    [batch, 2]，列顺序为 EEG、fNIRS。
    branch_predictions: 两个模态各自的预测，供消融和诊断使用。

本模型是研究原型，不是已验证的临床替代工具。训练和验证必须按患者分组，
且不能把基线/末次量表复制到中间信号 session 作为标签。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class TemporalCNNEncoder(nn.Module):
    """小型 1D CNN：卷积沿时间轴，输入通道对应电极或 fNIRS 通道。"""

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
            nn.Conv1d(16, 32, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 32, kernel_size=5, padding=2, bias=False),
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
            raise ValueError("输入形状应为 [batch, channels, time]。")
        return self.features(x)


class QualityGate(nn.Module):
    """只根据 EEG/fNIRS 质量特征生成两个模态权重。"""

    def __init__(
        self,
        eeg_quality_dim: int,
        fnirs_quality_dim: int,
        hidden_dim: int = 8,
    ) -> None:
        super().__init__()
        if eeg_quality_dim < 1 or fnirs_quality_dim < 1:
            raise ValueError("EEG 和 fNIRS 各自至少需要一个质量特征。")

        self.eeg_score = nn.Sequential(
            nn.Linear(eeg_quality_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.fnirs_score = nn.Sequential(
            nn.Linear(fnirs_quality_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        quality_eeg: torch.Tensor,
        quality_fnirs: torch.Tensor,
        available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = torch.cat(
            [self.eeg_score(quality_eeg), self.fnirs_score(quality_fnirs)],
            dim=1,
        )

        if available is not None:
            if available.shape != logits.shape:
                raise ValueError("available 形状应为 [batch, 2]，列顺序为 EEG、fNIRS。")
            available = available.bool()
            if torch.any(~available.any(dim=1)):
                raise ValueError("每个样本至少需要一个可用信号模态。")
            logits = logits.masked_fill(~available, torch.finfo(logits.dtype).min)

        return torch.softmax(logits, dim=1)


class QualityAwareTemporalFusion(nn.Module):
    """从 EEG/fNIRS 估计临床量表分数；量表只作为监督标签。"""

    def __init__(
        self,
        eeg_channels: int,
        fnirs_channels: int,
        eeg_quality_dim: int,
        fnirs_quality_dim: int,
        embedding_dim: int = 32,
        output_dim: int = 1,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.eeg_encoder = TemporalCNNEncoder(
            eeg_channels, embedding_dim=embedding_dim, dropout=dropout
        )
        self.fnirs_encoder = TemporalCNNEncoder(
            fnirs_channels, embedding_dim=embedding_dim, dropout=dropout
        )

        # 每个信号分支独立估计量表，质量门控再融合两个预测。
        self.eeg_head = nn.Linear(embedding_dim, output_dim)
        self.fnirs_head = nn.Linear(embedding_dim, output_dim)
        self.gate = QualityGate(
            eeg_quality_dim=eeg_quality_dim,
            fnirs_quality_dim=fnirs_quality_dim,
        )

    def forward(
        self,
        eeg: torch.Tensor,
        fnirs: torch.Tensor,
        quality_eeg: torch.Tensor,
        quality_fnirs: torch.Tensor,
        available: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        z_eeg = self.eeg_encoder(eeg)
        z_fnirs = self.fnirs_encoder(fnirs)

        pred_eeg = self.eeg_head(z_eeg)
        pred_fnirs = self.fnirs_head(z_fnirs)

        weights = self.gate(
            quality_eeg=quality_eeg,
            quality_fnirs=quality_fnirs,
            available=available,
        )
        branch_predictions = torch.stack([pred_eeg, pred_fnirs], dim=1)
        prediction = (branch_predictions * weights.unsqueeze(-1)).sum(dim=1)

        return {
            "prediction": prediction,
            "weights": weights,  # 列顺序：EEG、fNIRS
            "branch_predictions": branch_predictions,
        }
