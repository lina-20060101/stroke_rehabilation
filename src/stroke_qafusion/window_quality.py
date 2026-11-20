"""按刺激时间窗提取质量证据，并学习原始窗与加扰窗的成对排序。"""

import mne
import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import minimize
from scipy.signal import resample_poly, welch
from scipy.special import expit
from sklearn.preprocessing import StandardScaler

from .recovery_signals import (
    SENSORIMOTOR,
    filter_hbo,
    paired_target_epochs,
    prepare_haemoglobin,
    samples_in_window,
)


QC_COLUMNS = ["unusable_fraction", "plateau_fraction", "jump_excess"]
DAMAGED_COLUMNS = [f"damaged_{name}" for name in QC_COLUMNS]
RUN = ["subject_id", "session_id", "target_hand", "run_id"]
TRIAL = RUN + ["trial"]
WINDOWS = {
    "eeg": ((0.5, 1.5), (1.5, 2.5)),
    "fnirs": ((4.0, 6.0), (6.0, 8.0)),
}
WAVEFORM_SAMPLES = {"eeg": 128, "fnirs": 28}


def step_reference(data):
    """每通道整段典型相邻采样差，供同一 run 的局部突跳检测使用。"""
    reference = np.ones(len(data))
    for i, channel in enumerate(data):
        delta = np.abs(np.diff(channel))
        positive = delta[np.isfinite(delta) & (delta > 0)]
        if positive.size:
            reference[i] = np.median(positive)
    return reference


def quality_evidence(window, reference):
    usable = np.isfinite(window).all(axis=1) & (np.ptp(window, axis=1) > 0)
    if not usable.any():
        return np.array([1.0, 1.0, 0.0])
    delta = np.abs(np.diff(window[usable], axis=1))
    # 平台包括连续削顶点。突跳按每个通道的整段参照缩放。
    excess = np.clip(delta / reference[usable, None] - 8, 0, 100)
    return np.array([1 - usable.mean(), (delta == 0).mean(), excess.mean()])


def perturb(window, reference, kind, rng):
    """在复制的原始电压或光密度窗口上施加可重现的局部干扰。"""
    modified = window.copy()
    usable = np.flatnonzero(np.isfinite(window).all(axis=1) & (np.ptp(window, axis=1) > 0))
    if not usable.size:
        return modified
    count = max(1, int(np.ceil(len(usable) * rng.uniform(0.25, 0.75))))
    channels = rng.choice(usable, size=count, replace=False)
    selected = modified[channels]
    if kind == "dropout":
        modified[channels] = selected[:, :1]
    elif kind == "clipping":
        low, high = np.quantile(selected, [0.2, 0.8], axis=1)
        modified[channels] = np.clip(selected, low[:, None], high[:, None])
    elif kind == "step":
        n = window.shape[1]
        start = rng.integers(1, n // 2)
        width = max(1, n // 4)
        modified[channels, start : start + width] += reference[channels, None] * rng.uniform(25, 80)
    else:
        raise ValueError(f"未知干扰类型：{kind}")
    return modified


class WindowQualityRanker:
    """成对 logistic 排序。质量损失指标增大时，分数只能下降。"""

    def fit(self, original, damaged):
        original, damaged = np.asarray(original), np.asarray(damaged)
        self.scaler = StandardScaler().fit(np.vstack([original, damaged]))
        self.center = np.median(original, axis=0)
        difference = (damaged - original) / self.scaler.scale_

        def objective(coef):
            margin = difference @ coef
            loss = np.logaddexp(0, -margin).mean() + 0.005 * (coef @ coef)
            gradient = -(difference.T @ expit(-margin)) / len(margin) + 0.01 * coef
            return loss, gradient

        result = minimize(
            objective,
            np.zeros(difference.shape[1]),
            jac=True,
            method="L-BFGS-B",
            bounds=[(0, None)] * difference.shape[1],
        )
        if not result.success:
            raise RuntimeError(f"质量排序器未收敛：{result.message}")
        self.coef = result.x
        return self

    def score(self, evidence):
        evidence = np.asarray(evidence)
        distance = ((evidence - self.center) / self.scaler.scale_) @ self.coef
        scores = expit(-distance)
        scores[evidence[:, 0] == 1] = 0  # 整窗不可用
        return scores


def band_powers(epochs, fs):
    """基线和响应使用相同的一秒 Welch 段长，保持频率网格一致。"""
    freq, psd = welch(epochs, fs=fs, nperseg=round(fs), axis=-1)
    powers = []
    for low, high in ((8, 13), (13, 30)):
        mask = (freq >= low) & (freq < high)
        powers.append(trapezoid(psd[..., mask], freq[mask], axis=-1))
    return powers


def response_features(modality, baseline, response, fs):
    if modality == "fnirs":
        difference = (response.mean(axis=-1) - baseline.mean(axis=-1)) * 1e6
        return {"hbo_um": np.median(difference, axis=1)}
    features = {}
    for name, base, current in zip(
        ("alpha", "beta"), band_powers(baseline, fs), band_powers(response, fs)
    ):
        values = np.full(len(base), np.nan)
        for i, (before, after) in enumerate(zip(base, current)):
            valid = np.isfinite(before) & np.isfinite(after) & (before > 0) & (after > 0)
            if valid.any():
                values[i] = np.median(np.log10(after[valid]) - np.log10(before[valid]))
        features[name] = values
    return features


def response_waveforms(modality, baseline, response):
    """按通道减去本 trial 的基线，取通道中位波形，保留各模态独立的时间网格。"""
    length = WAVEFORM_SAMPLES[modality]
    waveforms = []
    for before, after in zip(baseline, response):
        valid = (
            np.isfinite(before).all(axis=1)
            & (np.ptp(before, axis=1) > 0)
            & np.isfinite(after).all(axis=1)
            & (np.ptp(after, axis=1) > 0)
        )
        if not valid.any():
            waveforms.append(None)
            continue
        corrected = after[valid] - before[valid].mean(axis=1, keepdims=True)
        wave = np.median(corrected, axis=0)
        wave = resample_poly(wave, length, len(wave), padtype="line")
        # 固定物理尺度（µV 或 µM），保留符号与幅度，无需从其他患者估计尺度。
        waveforms.append(np.arcsinh(wave * 1e6).astype(np.float32))
    return waveforms


def modality_windows(modality, signal, qc_data, triggers, fs, rng, *, include_waveforms=False):
    reference = step_reference(qc_data)
    baseline = samples_in_window(signal, triggers, fs, -2, 0)
    records = []
    for phase, (start, end) in enumerate(((-2.0, 0.0), *WINDOWS[modality])):
        raw_windows = samples_in_window(qc_data, triggers, fs, start, end)
        features = {}
        if phase:
            response = samples_in_window(signal, triggers, fs, start, end)
            features = response_features(modality, baseline, response, fs)
            if include_waveforms:
                features["waveform"] = response_waveforms(modality, baseline, response)
        for trial, block in enumerate(raw_windows):
            kind = rng.choice(["dropout", "clipping", "step"])
            modified = perturb(block, reference, kind, rng)
            records.append(
                {
                    "trial": trial,
                    "modality": modality,
                    "phase": phase,
                    "start_s": start,
                    "end_s": end,
                    "perturbation": kind,
                    **dict(zip(QC_COLUMNS, quality_evidence(block, reference))),
                    **dict(zip(DAMAGED_COLUMNS, quality_evidence(modified, reference))),
                    **{name: values[trial] for name, values in features.items()},
                }
            )
    return records


def extract_quality_windows(eeg_path, fnirs_path, hand, rng, *, include_waveforms=False):
    eeg = mne.io.read_raw_fif(eeg_path, verbose="ERROR")
    fnirs = mne.io.read_raw_fif(fnirs_path, verbose="ERROR")
    eeg_triggers, fnirs_triggers = paired_target_epochs(eeg, fnirs, hand)
    eeg.pick("eeg", exclude="bads")
    roi = [name for name in eeg.ch_names if name in SENSORIMOTOR[hand]]
    if not roi:
        raise ValueError("缺少患手对侧感觉运动区 EEG 通道")
    eeg_data = eeg.get_data(picks=roi)
    records = modality_windows(
        "eeg",
        eeg_data,
        eeg_data,
        eeg_triggers,
        eeg.info["sfreq"],
        rng,
        include_waveforms=include_waveforms,
    )

    fs = fnirs.info["sfreq"]
    hbo, od, _ = prepare_haemoglobin(fnirs)
    records.extend(
        modality_windows(
            "fnirs",
            filter_hbo(hbo, fs),
            od,
            fnirs_triggers,
            fs,
            rng,
            include_waveforms=include_waveforms,
        )
    )
    return records
