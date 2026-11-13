"""质量条件化残差融合基线。

临床分支提供基准预测。EEG 和 fNIRS 分支分别预测对该基准的修正量；
记录质量特征控制各修正量的权重。本模型是研究基线，未经临床验证。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class QualityGatedResidualFusion(nn.Module):
    """使用显式质量门控融合临床、EEG 和 fNIRS 特征。

    门控输入仅包含质量指标，因此模态权重可检查。临床估计作为回退基准保留；
    信号质量差时应减小该模态的修正量，而不是压低临床基准预测。
    """

    def __init__(
        self,
        clinical_dim: int,
        eeg_dim: int,
        fnirs_dim: int,
        eeg_quality_dim: int,
        fnirs_quality_dim: int,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.clinical_head = nn.Sequential(
            nn.Linear(clinical_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.eeg_residual_head = nn.Sequential(
            nn.Linear(eeg_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.fnirs_residual_head = nn.Sequential(
            nn.Linear(fnirs_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.eeg_quality_gate = nn.Sequential(
            nn.Linear(eeg_quality_dim, max(4, hidden_dim // 4)),
            nn.ReLU(),
            nn.Linear(max(4, hidden_dim // 4), 1),
            nn.Sigmoid(),
        )
        self.fnirs_quality_gate = nn.Sequential(
            nn.Linear(fnirs_quality_dim, max(4, hidden_dim // 4)),
            nn.ReLU(),
            nn.Linear(max(4, hidden_dim // 4), 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        clinical: torch.Tensor,
        eeg: torch.Tensor,
        fnirs: torch.Tensor,
        eeg_quality: torch.Tensor,
        fnirs_quality: torch.Tensor,
    ) -> dict[str, Any]:
        clinical_prediction = self.clinical_head(clinical)
        eeg_weight = self.eeg_quality_gate(eeg_quality)
        fnirs_weight = self.fnirs_quality_gate(fnirs_quality)
        eeg_correction = self.eeg_residual_head(eeg)
        fnirs_correction = self.fnirs_residual_head(fnirs)
        prediction = (
            clinical_prediction
            + eeg_weight * eeg_correction
            + fnirs_weight * fnirs_correction
        )
        return {
            "prediction": prediction,
            "clinical_prediction": clinical_prediction,
            "eeg_weight": eeg_weight,
            "fnirs_weight": fnirs_weight,
            "eeg_correction": eeg_correction,
            "fnirs_correction": fnirs_correction,
        }
