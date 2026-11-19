"""EEG 记录质量代理指标；这些指标不是临床生物标志物。"""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch

from .common import finite_fraction, robust_zscore


def eeg_quality_features(data: np.ndarray, sfreq: float, line_frequency: float = 50.0) -> dict[str, float]:
    """汇总一个 EEG 记录窗口的质量特征。

    Args:
        data: 通道数 x 采样点数，使用一致的物理单位（建议伏特）。
        sfreq: 采样频率，单位 Hz。
        line_frequency: 当地电源工频，通常为 50 或 60 Hz。

    Returns:
        可解释的质量代理指标。高频功率仅是肌电/噪声代理量；没有 EOG/EMG 标注时，
        不能把它称作已测得的伪迹标签。
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2 or sfreq <= 0:
        raise ValueError("data 应为‘通道数 × 采样点数’二维数组，且 sfreq 必须大于 0。")

    valid = np.isfinite(x)
    x = np.where(valid, x, np.nan)
    channel_std = np.nanstd(x, axis=1)
    finite_channels = np.isfinite(channel_std)
    positive_std = channel_std[finite_channels & (channel_std > 0)]
    reference_std = np.median(positive_std) if positive_std.size else 0.0
    flat_threshold = max(reference_std * 1e-3, np.finfo(float).eps)

    clean = np.nan_to_num(x, nan=0.0)
    robust_amplitude = np.abs(robust_zscore(clean.reshape(-1)))
    nperseg = min(clean.shape[1], max(8, int(sfreq * 2)))
    freqs, psd = welch(clean, fs=sfreq, nperseg=nperseg, axis=-1)
    total = trapezoid(psd, freqs, axis=-1)
    hf_mask = (freqs >= 30) & (freqs <= min(45, sfreq / 2 - 1e-6))
    line_mask = np.abs(freqs - line_frequency) <= 1.0
    hf = trapezoid(psd[:, hf_mask], freqs[hf_mask], axis=-1) if hf_mask.sum() > 1 else np.zeros_like(total)
    line = trapezoid(psd[:, line_mask], freqs[line_mask], axis=-1) if line_mask.sum() > 1 else np.zeros_like(total)

    ratios_hf = np.divide(hf, total, out=np.zeros_like(hf), where=total > 0)
    ratios_line = np.divide(line, total, out=np.zeros_like(line), where=total > 0)
    return {
        "valid_sample_fraction": finite_fraction(valid),
        "flat_channel_fraction": float(np.mean(channel_std <= flat_threshold)),
        "amplitude_outlier_fraction_proxy": float(np.mean(robust_amplitude > 8)),
        "high_frequency_power_ratio_proxy": float(np.nanmedian(ratios_hf)),
        "line_noise_power_ratio": float(np.nanmedian(ratios_line)),
    }
