"""从公开数据集的 FIF 配对记录提取康复结局基线特征。"""

from pathlib import Path

import mne
import numpy as np
from mne.preprocessing.nirs import beer_lambert_law, source_detector_distances
from scipy.integrate import trapezoid
from scipy.signal import butter, sosfiltfilt, welch

from .quality.eeg import eeg_quality_features
from .quality.fnirs import fnirs_quality_features


SENSORIMOTOR = {"left": {"C4", "FCC4", "CCP4"}, "right": {"C3", "FCC3", "CCP3"}}
FLASH_IDS = {"flash_l": 1, "flash_r": 2}


def flash_events(raw):
    events, _ = mne.events_from_annotations(raw, event_id=FLASH_IDS, verbose="ERROR")
    if not len(events):
        raise ValueError("没有 flash_l / flash_r 触发事件")
    return events[:, 0] - raw.first_samp, events[:, 2]


def usable_channels(data):
    """剔除含非有限值和完全平坦的通道，保留原始通道顺序。"""
    return np.isfinite(data).all(axis=1) & (np.ptp(data, axis=1) > 0)


def paired_target_epochs(eeg, fnirs, hand):
    """以各自的采样点为事件零点，核对触发顺序和相对间隔。"""
    eeg_samples, eeg_codes = flash_events(eeg)
    fnirs_samples, fnirs_codes = flash_events(fnirs)
    if not np.array_equal(eeg_codes, fnirs_codes):
        raise ValueError("EEG 和 fNIRS 的刺激触发序列不同")

    eeg_time = eeg_samples / eeg.info["sfreq"]
    fnirs_time = fnirs_samples / fnirs.info["sfreq"]
    # 两台设备的记录起点不同，核对的是触发间相对时间。
    drift = (eeg_time - eeg_time[0]) - (fnirs_time - fnirs_time[0])
    if np.max(np.abs(drift)) > 0.25:
        raise ValueError("两路刺激触发的相对时间差超过 0.25 秒")

    target_code = FLASH_IDS["flash_l" if hand == "left" else "flash_r"]
    eeg_fs, fnirs_fs = eeg.info["sfreq"], fnirs.info["sfreq"]
    valid = (
        (eeg_codes == target_code)
        & (eeg_samples >= round(2 * eeg_fs))
        & (eeg_samples + round(2.5 * eeg_fs) <= eeg.n_times)
        & (fnirs_samples >= round(2 * fnirs_fs))
        & (fnirs_samples + round(8 * fnirs_fs) <= fnirs.n_times)
        & (np.diff(np.r_[-np.inf, fnirs_time]) >= 8)
        & (np.diff(np.r_[fnirs_time, np.inf]) >= 8)
    )
    if valid.sum() < 2:
        raise ValueError("不足两次具有完整分析窗的患手目标刺激")
    return eeg_samples[valid], fnirs_samples[valid]


def samples_in_window(data, triggers, fs, start, stop):
    start_offset, stop_offset = round(start * fs), round(stop * fs)
    return np.stack([
        data[:, sample + start_offset:sample + stop_offset] for sample in triggers
    ])


def eeg_features(raw, triggers, hand):
    fs = raw.info["sfreq"]
    raw.pick("eeg", exclude="bads")
    data = raw.get_data()
    if not len(data):
        raise ValueError("没有可用的 EEG 通道")
    usable = usable_channels(data)
    sensorimotor = np.isin(raw.ch_names, sorted(SENSORIMOTOR[hand]))
    if not (usable & sensorimotor).any():
        raise ValueError("没有可用的对侧感觉运动区 EEG 通道")
    selected = data[usable & sensorimotor]
    baseline = samples_in_window(selected, triggers, fs, -2, 0)
    response = samples_in_window(selected, triggers, fs, 0.5, 2.5)
    freqs, base_psd = welch(baseline, fs=fs, nperseg=baseline.shape[-1])
    _, response_psd = welch(response, fs=fs, nperseg=response.shape[-1])
    features = {}
    for band, low, high in (("alpha", 8, 13), ("beta", 13, 30)):
        indices = (freqs >= low) & (freqs < high)
        base_power = trapezoid(base_psd[..., indices], freqs[indices], axis=-1)
        response_power = trapezoid(response_psd[..., indices], freqs[indices], axis=-1)
        defined = (base_power > 0) & (response_power > 0)
        if not defined.any():
            raise ValueError(f"EEG {band} 频带功率无法计算")
        change = np.log10(response_power[defined]) - np.log10(base_power[defined])
        features[f"eeg_{band}_log_power_change"] = float(np.median(change))

    quality = eeg_quality_features(data[usable], fs)
    quality["usable_channel_fraction"] = float(usable.mean())
    return features | {f"qc_eeg_{name}": value for name, value in quality.items()}


def prepare_haemoglobin(raw):
    """返回可用的长距 HbO、原始光密度和可转换通道比例。"""
    if set(raw.get_channel_types()) != {"fnirs_od"}:
        raise ValueError("fNIRS FIF 的预期通道类型为光密度 fnirs_od")

    optical_density = raw.get_data()
    usable = usable_channels(optical_density)
    # Beer–Lambert 需要同一光源-探测器对的两个波长。
    pairs = {}
    for index, channel in enumerate(raw.ch_names):
        pairs.setdefault(channel.rsplit(" ", 1)[0], []).append(index)
    picks = [
        index for pair in pairs.values()
        if usable[pair].all() and not any(raw.ch_names[i] in raw.info["bads"] for i in pair)
        for index in pair
    ]
    if not picks:
        raise ValueError("没有可转换为血红蛋白浓度的有效通道对")

    haemoglobin = beer_lambert_law(raw.pick(sorted(picks)), ppf=2.0)
    distances = source_detector_distances(haemoglobin.info)
    hbo = np.array(haemoglobin.get_channel_types()) == "hbo"
    selected = np.flatnonzero(hbo & (distances >= 0.02) & (distances <= 0.045))
    if not selected.size:
        raise ValueError("没有 20–45 mm 的 HbO 通道")
    data = haemoglobin.get_data(picks=selected)
    data = data[usable_channels(data)]
    if not len(data):
        raise ValueError("没有可用的 HbO 浓度序列")
    return data, optical_density, len(picks) / len(optical_density)


def filter_hbo(data, fs):
    sos = butter(3, (0.02, 0.1), btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def fnirs_features(raw, triggers):
    fs = raw.info["sfreq"]
    data, _, usable_fraction = prepare_haemoglobin(raw)
    quality = fnirs_quality_features(data)
    quality["usable_od_channel_fraction"] = usable_fraction
    filtered = filter_hbo(data, fs)
    baseline = samples_in_window(filtered, triggers, fs, -2, 0)
    response = samples_in_window(filtered, triggers, fs, 4, 8)
    change_um = (response.mean(axis=-1) - baseline.mean(axis=-1)) * 1e6
    return {"fnirs_hbo_change_um": float(np.median(change_um))} | {
        f"qc_fnirs_{name}": value for name, value in quality.items()
    }


def extract_run(eeg_path: Path, fnirs_path: Path, hand: str) -> dict[str, float]:
    """提取患手任务的三项信号特征、局部取窗计数和整段质量指标。"""
    eeg = mne.io.read_raw_fif(eeg_path, verbose="ERROR")
    fnirs = mne.io.read_raw_fif(fnirs_path, verbose="ERROR")
    eeg_triggers, fnirs_triggers = paired_target_epochs(eeg, fnirs, hand)
    return {
        "n_target_epochs": len(eeg_triggers),
        **eeg_features(eeg, eeg_triggers, hand),
        **fnirs_features(fnirs, fnirs_triggers),
    }
