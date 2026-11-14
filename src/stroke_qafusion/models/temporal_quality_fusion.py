"""事件时间对齐的 EEG-fNIRS 逐时间片质量门控模型。

本模型接收预处理后的事件相对时间特征序列，不直接对齐原始 EEG/fNIRS 采样点。
EEG 与 fNIRS 应先各自预处理，再映射到共同的事件相对时间片；不要将 fNIRS 原始信号
插值到 EEG 的采样率后逐点融合。

张量接口：
    eeg_features:   [B, T, D_e]
    fnirs_features: [B, T, D_f]
    eeg_quality:    [B, T, Q_e]
    fnirs_quality:  [B, T, Q_f]
    event_time:     [B, T]，单位秒，相对于刺激/事件触发点
    available:      可选 [B, T, 2]，最后一维顺序为 EEG、fNIRS

每个时间片的权重由局部质量、该模态分支的预测熵和事件相对时间共同决定。
该模型是研究原型，时间片定义须结合实际事件标记和数据预处理流程确定。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class TokenEncoder(nn.Module):
    """对每个事件相对时间片的预计算特征进行映射。"""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("每个模态的 input_dim 必须大于 0。")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("模态特征形状必须为 [batch, time_bins, features]。")
        return self.net(x)


class TemporalQualityGate(nn.Module):
    """在每个事件时间片分别计算 EEG 与 fNIRS 的信任权重。"""

    def __init__(
        self,
        eeg_quality_dim: int,
        fnirs_quality_dim: int,
        hidden_dim: int,
        initial_uncertainty_penalty: float = 1.0,
    ) -> None:
        super().__init__()
        if eeg_quality_dim < 1 or fnirs_quality_dim < 1:
            raise ValueError("每种模态至少需要一个局部质量指标。")

        self.time_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.eeg_quality_score = nn.Sequential(
            nn.Linear(eeg_quality_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.fnirs_quality_score = nn.Sequential(
            nn.Linear(fnirs_quality_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # softplus 保证不确定性惩罚系数非负。
        raw_beta = torch.log(torch.expm1(torch.tensor(initial_uncertainty_penalty)))
        self.raw_uncertainty_penalty = nn.Parameter(raw_beta)

    def forward(
        self,
        eeg_quality: torch.Tensor,
        fnirs_quality: torch.Tensor,
        eeg_uncertainty: torch.Tensor,
        fnirs_uncertainty: torch.Tensor,
        event_time: torch.Tensor,
        available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if eeg_quality.ndim != 3 or fnirs_quality.ndim != 3:
            raise ValueError("质量指标形状必须为 [batch, time_bins, quality_features]。")
        if event_time.ndim != 2:
            raise ValueError("event_time 形状必须为 [batch, time_bins]。")

        time_embedding = self.time_encoder(event_time.unsqueeze(-1))
        eeg_input = torch.cat([eeg_quality, time_embedding], dim=-1)
        fnirs_input = torch.cat([fnirs_quality, time_embedding], dim=-1)

        beta = F.softplus(self.raw_uncertainty_penalty)
        eeg_logit = self.eeg_quality_score(eeg_input).squeeze(-1) - beta * eeg_uncertainty
        fnirs_logit = self.fnirs_quality_score(fnirs_input).squeeze(-1) - beta * fnirs_uncertainty
        logits = torch.stack([eeg_logit, fnirs_logit], dim=-1)

        if available is not None:
            if available.shape != logits.shape:
                raise ValueError("available 形状必须为 [batch, time_bins, 2]。")
            available = available.bool()
            if torch.any(~available.any(dim=-1)):
                raise ValueError("每个时间片至少要有一个可用模态。")
            logits = logits.masked_fill(~available, torch.finfo(logits.dtype).min)

        return torch.softmax(logits, dim=-1)


class TemporalQualityAwareFusion(nn.Module):
    """输出 trial 级运动反应分类，并返回每个时间片的模态权重。

    主任务标签可定义为运动反应/无运动反应。FMA 等量表不作为本模型输入；
    如需使用，应在有真实 session 量表的样本上另设辅助头和掩码损失。
    """

    def __init__(
        self,
        eeg_feature_dim: int,
        fnirs_feature_dim: int,
        eeg_quality_dim: int,
        fnirs_quality_dim: int,
        num_classes: int = 2,
        hidden_dim: int = 32,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes 至少为 2。")

        self.eeg_encoder = TokenEncoder(eeg_feature_dim, hidden_dim, dropout)
        self.fnirs_encoder = TokenEncoder(fnirs_feature_dim, hidden_dim, dropout)
        self.eeg_head = nn.Linear(hidden_dim, num_classes)
        self.fnirs_head = nn.Linear(hidden_dim, num_classes)

        self.gate = TemporalQualityGate(
            eeg_quality_dim=eeg_quality_dim,
            fnirs_quality_dim=fnirs_quality_dim,
            hidden_dim=hidden_dim,
        )
        self.temporal_model = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.trial_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    @staticmethod
    def _normalized_entropy(logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
        return entropy / torch.log(
            torch.tensor(float(logits.shape[-1]), device=logits.device, dtype=logits.dtype)
        ).clamp_min(1e-8)

    def forward(
        self,
        eeg_features: torch.Tensor,
        fnirs_features: torch.Tensor,
        eeg_quality: torch.Tensor,
        fnirs_quality: torch.Tensor,
        event_time: torch.Tensor,
        available: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if eeg_features.shape[:2] != fnirs_features.shape[:2]:
            raise ValueError(
                "EEG 与 fNIRS 特征必须先映射到同一事件时间片网格；"
                "不要按原始采样点直接拼接。"
            )
        if eeg_features.shape[:2] != event_time.shape:
            raise ValueError("event_time 必须与特征的 [batch, time_bins] 维度匹配。")
        if eeg_quality.shape[:2] != event_time.shape or fnirs_quality.shape[:2] != event_time.shape:
            raise ValueError("质量序列必须与事件时间片网格对齐。")

        eeg_tokens = self.eeg_encoder(eeg_features)
        fnirs_tokens = self.fnirs_encoder(fnirs_features)

        eeg_token_logits = self.eeg_head(eeg_tokens)
        fnirs_token_logits = self.fnirs_head(fnirs_tokens)

        # 熵作为不确定性代理；门控不通过熵反向操纵分支置信度。
        eeg_uncertainty = self._normalized_entropy(eeg_token_logits).detach()
        fnirs_uncertainty = self._normalized_entropy(fnirs_token_logits).detach()

        weights = self.gate(
            eeg_quality=eeg_quality,
            fnirs_quality=fnirs_quality,
            eeg_uncertainty=eeg_uncertainty,
            fnirs_uncertainty=fnirs_uncertainty,
            event_time=event_time,
            available=available,
        )

        fused_tokens = (
            weights[..., 0:1] * eeg_tokens + weights[..., 1:2] * fnirs_tokens
        )
        temporal_tokens, _ = self.temporal_model(fused_tokens)
        trial_logits = self.trial_head(temporal_tokens[:, -1])

        eeg_trial_logits = eeg_token_logits.mean(dim=1)
        fnirs_trial_logits = fnirs_token_logits.mean(dim=1)

        if available is None:
            branch_available = torch.ones(
                eeg_features.shape[0], 2, dtype=torch.bool, device=eeg_features.device
            )
        else:
            branch_available = available.bool().any(dim=1)

        return {
            "branch_available": branch_available,
            "trial_logits": trial_logits,
            "weights": weights,  # [B, T, 2]，最后一维顺序 EEG、fNIRS
            "eeg_trial_logits": eeg_trial_logits,
            "fnirs_trial_logits": fnirs_trial_logits,
            "eeg_token_logits": eeg_token_logits,
            "fnirs_token_logits": fnirs_token_logits,
            "eeg_uncertainty": eeg_uncertainty,
            "fnirs_uncertainty": fnirs_uncertainty,
        }
