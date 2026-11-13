"""透明、逐记录质量指标的通用辅助函数。"""

from __future__ import annotations

import numpy as np


def robust_zscore(values: np.ndarray) -> np.ndarray:
    """计算基于中位数/MAD 的稳健 z 分数，并安全处理常数数组。"""
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    if not np.isfinite(mad) or mad == 0:
        return np.zeros_like(values)
    return 0.6745 * (values - median) / mad


def finite_fraction(values: np.ndarray) -> float:
    """计算有限值样本占比；输入为空时返回 NaN。"""
    values = np.asarray(values)
    return float(np.isfinite(values).mean()) if values.size else float("nan")
