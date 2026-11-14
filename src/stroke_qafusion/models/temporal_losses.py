"""逐时间片质量融合的训练目标。"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _branch_error(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """返回每个样本的分支交叉熵，形状 [B]。"""
    return F.cross_entropy(logits, target.long(), reduction="none")


def temporal_quality_objective(
    outputs: dict[str, Any],
    target: torch.Tensor,
    *,
    corrupted_outputs: dict[str, Any] | None = None,
    corrupted_modal_mask: torch.Tensor | None = None,
    branch_weight: float = 0.25,
    calibration_weight: float = 0.10,
    smoothness_weight: float = 0.005,
    ranking_weight: float = 0.20,
    consistency_weight: float = 0.10,
    corruption_margin: float = 0.05,
) -> dict[str, torch.Tensor]:
    """计算主任务、专家辅助、权重校准与局部损坏鲁棒性损失。

    Args:
        outputs: 原始批次的模型输出。
        target: trial 级类别标签，形状 [B]。
        corrupted_outputs: 可选，同一批样本只在局部时间段被轻度损坏后的模型输出。
        corrupted_modal_mask: [B, T, 2] 布尔值；标记被人为损坏的时间片和模态，
            最后一维顺序为 EEG、fNIRS。只对该掩码中的权重施加下降约束。
        branch_weight: 单模态专家分类辅助损失权重。
        calibration_weight: 让平均模态权重对应分支实际误差的校准损失权重。
        smoothness_weight: 相邻时间片权重总变差惩罚，建议较小。
        ranking_weight: 局部损坏后对应模态权重下降的排序损失权重。
        consistency_weight: 轻度损坏前后 trial 预测保持稳定的损失权重。
        corruption_margin: 局部损坏后对应权重至少下降的间隔。

    Returns:
        含 total 和各分项的字典，供记录训练曲线。
    """
    labels = target.long().view(-1)
    fused_logits = outputs["trial_logits"]
    weights = outputs["weights"]
    branch_available = outputs.get("branch_available")

    task_loss = F.cross_entropy(fused_logits, labels)

    if branch_available is None:
        branch_available = torch.ones(
            labels.shape[0], 2, dtype=torch.bool, device=labels.device
        )
    else:
        branch_available = branch_available.bool()

    eeg_error = _branch_error(outputs["eeg_trial_logits"], labels)
    fnirs_error = _branch_error(outputs["fnirs_trial_logits"], labels)
    branch_errors = torch.stack([eeg_error, fnirs_error], dim=-1)

    available_logits = (-branch_errors).masked_fill(
        ~branch_available, torch.finfo(branch_errors.dtype).min
    )
    target_reliability = torch.softmax(available_logits.detach(), dim=-1)
    branch_loss_per_sample = (
        branch_errors * branch_available.to(branch_errors.dtype)
    ).sum(dim=-1) / branch_available.sum(dim=-1).clamp_min(1)
    branch_loss = branch_loss_per_sample.mean()

    average_weights = weights.mean(dim=1).clamp_min(1e-8)
    calibration_loss = F.kl_div(
        average_weights.log(),
        target_reliability,
        reduction="batchmean",
    )

    if weights.shape[1] > 1:
        weight_change = (weights[:, 1:] - weights[:, :-1]).abs()
        availability = outputs.get("available")
        if availability is None:
            transition_mask = torch.ones(
                weights.shape[0], weights.shape[1] - 1,
                dtype=torch.bool, device=weights.device
            )
        else:
            availability = availability.bool()
            # 模态可用状态发生变化时，允许门控权重快速切换，不做平滑惩罚。
            transition_mask = availability[:, 1:].all(dim=-1) & availability[:, :-1].all(dim=-1)
        if transition_mask.any():
            smoothness_loss = (
                weight_change * transition_mask.unsqueeze(-1).to(weight_change.dtype)
            ).sum() / (transition_mask.sum() * weights.shape[-1]).clamp_min(1)
        else:
            smoothness_loss = weights.new_zeros(())
    else:
        smoothness_loss = weights.new_zeros(())

    ranking_loss = weights.new_zeros(())
    consistency_loss = weights.new_zeros(())

    if corrupted_outputs is not None:
        if corrupted_modal_mask is None:
            raise ValueError(
                "提供 corrupted_outputs 时，必须同时提供 corrupted_modal_mask。"
            )
        corrupted_weights = corrupted_outputs["weights"]
        if corrupted_modal_mask.shape != weights.shape:
            raise ValueError("corrupted_modal_mask 必须与 weights 同形状 [B, T, 2]。")

        affected = corrupted_modal_mask.bool()
        if affected.any():
            # 受损模态在受损时间片的权重应比干净版本至少下降 margin。
            decrease = weights - corrupted_weights
            ranking_loss = F.relu(corruption_margin - decrease[affected]).mean()

        clean_prob = torch.softmax(fused_logits.detach(), dim=-1)
        corrupted_log_prob = torch.log_softmax(
            corrupted_outputs["trial_logits"], dim=-1
        )
        consistency_loss = F.kl_div(
            corrupted_log_prob,
            clean_prob,
            reduction="batchmean",
        )

    total = (
        task_loss
        + branch_weight * branch_loss
        + calibration_weight * calibration_loss
        + smoothness_weight * smoothness_loss
        + ranking_weight * ranking_loss
        + consistency_weight * consistency_loss
    )

    return {
        "total": total,
        "task": task_loss.detach(),
        "branch": branch_loss.detach(),
        "calibration": calibration_loss.detach(),
        "smoothness": smoothness_loss.detach(),
        "corruption_ranking": ranking_loss.detach(),
        "corruption_consistency": consistency_loss.detach(),
    }
