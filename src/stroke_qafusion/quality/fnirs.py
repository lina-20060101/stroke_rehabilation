"""fNIRS 浓度时间序列的质量代理指标。"""

from __future__ import annotations

import numpy as np

from .common import finite_fraction, robust_zscore


def fnirs_quality_features(data: np.ndarray) -> dict[str, float]:
    """汇总一个 fNIRS 窗口的质量特征。

    Args:
        data: HbO/HbR 浓度或光密度数据，形状为通道数 x 采样点数。
              若 HbO 和 HbR 都有，应分别计算后再汇总。

    Returns:
        质量代理指标。原始光密度信噪比和头皮耦合指数需要含波长信息的
        光源-探测器数据；本函数不会推断这些指标。
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("data 应为至少包含两个采样点的‘通道数 × 采样点数’二维数组。")

    valid = np.isfinite(x)
    clean = np.where(valid, x, np.nan)
    channel_std = np.nanstd(clean, axis=1)
    finite_std = channel_std[np.isfinite(channel_std)]
    reference_std = np.median(finite_std[finite_std > 0]) if np.any(finite_std > 0) else 0.0
    flat_threshold = max(reference_std * 1e-3, np.finfo(float).eps)

    filled = np.nan_to_num(x, nan=0.0)
    derivative = np.diff(filled, axis=1)
    derivative_z = np.abs(robust_zscore(derivative.reshape(-1)))
    amplitude_z = np.abs(robust_zscore(filled.reshape(-1)))
    return {
        "valid_sample_fraction": finite_fraction(valid),
        "flat_channel_fraction": float(np.mean(channel_std <= flat_threshold)),
        "motion_spike_fraction_proxy": float(np.mean(derivative_z > 5)),
        "amplitude_outlier_fraction_proxy": float(np.mean(amplitude_z > 8)),
    }
