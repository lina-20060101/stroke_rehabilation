"""患者级 CNN＋Transformer 实验的数据组织、训练和预测。"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .models.recovery_transformer import TOKEN_WINDOWS, RecoveryTransformer
from .recovery_prediction import SCALES
from .window_quality import RUN, TRIAL, WAVEFORM_SAMPLES


LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
ATTENTION_COLUMNS = TRIAL + [
    "modality",
    "phase",
    "start_s",
    "end_s",
    "quality",
    "cnn_evidence",
    "attention",
    "clinical_attention",
]


@dataclass
class PatientBatch:
    trials: pd.DataFrame
    baseline: pd.Series
    eeg: torch.Tensor
    fnirs: torch.Tensor
    quality: torch.Tensor
    clinical: torch.Tensor
    pool: torch.Tensor

    @property
    def available(self):
        eeg = (self.quality[:, :2] > 0).any(dim=1)
        fnirs = (self.quality[:, 2:] > 0).any(dim=1)
        return (self.pool @ torch.stack((eeg, fnirs), dim=1).float() > 0).all(dim=1)

    def inputs(self, prior_mode="local"):
        prior = self.quality
        if prior_mode == "none":
            prior = (prior > 0).float()
        elif prior_mode == "run_shared":
            # 保持同一 run、同模态的先验权重总和及硬掩码，只消去局部时间差异。
            prior = self.quality.clone()
            for rows in self.trials.groupby(RUN).indices.values():
                local = self.quality[rows].reshape(-1, 2, 2)
                count = (local > 0).sum(dim=(0, 2))
                mean = local.sum(dim=(0, 2)) / count.clamp_min(1)
                prior[rows] = ((local > 0) * mean[None, :, None]).reshape(-1, 4)
        elif prior_mode != "local":
            raise ValueError(f"未知注意力先验：{prior_mode}")
        return self.eeg, self.fnirs, prior, self.clinical, self.pool


def patient_batch(response, scales):
    """同一刺激配对。每个有效 run 在 session 内等权，每个有效 session 在患者内等权。"""
    rows = response.loc[response.subject_id.isin(scales.index)]
    if rows.duplicated(TRIAL + ["modality", "phase"]).any():
        raise ValueError("同一患者、刺激、模态和时间窗出现重复波形")
    trials = rows[TRIAL].drop_duplicates().sort_values(TRIAL).reset_index(drop=True)
    trial_index = pd.MultiIndex.from_frame(trials)
    waves = {
        mod: np.zeros((len(trials), 2, length), dtype=np.float32)
        for mod, length in WAVEFORM_SAMPLES.items()
    }
    quality = np.zeros((len(trials), len(TOKEN_WINDOWS)), dtype=np.float32)
    for token, (mod, phase, _, _) in enumerate(TOKEN_WINDOWS):
        block = rows.loc[rows.modality.eq(mod) & rows.phase.eq(phase)]
        block = block.set_index(TRIAL).reindex(trial_index)
        for i, (q, wave) in enumerate(zip(block.local, block.waveform)):
            if q > 0 and isinstance(wave, np.ndarray) and np.isfinite(wave).all():
                quality[i, token] = q
                waves[mod][i, phase - 1] = wave

    keep = quality.any(axis=1)
    trials = trials.loc[keep].reset_index(drop=True)
    patient = scales.index.get_indexer(trials.subject_id)
    weights = 1.0 / trials.groupby(RUN).trial.transform("size")
    weights /= weights.groupby([trials.subject_id, trials.session_id]).transform("sum")
    weights /= weights.groupby(trials.subject_id).transform("sum")
    pool = np.zeros((len(scales), len(trials)), dtype=np.float32)
    pool[patient, np.arange(len(trials))] = weights
    clinical = scales[SCALES].to_numpy(dtype=np.float32) / np.array([66, 57], dtype=np.float32)
    return PatientBatch(
        trials,
        scales.fma_ue_before,
        torch.from_numpy(waves["eeg"][keep]),
        torch.from_numpy(waves["fnirs"][keep]),
        torch.from_numpy(quality[keep]),
        torch.from_numpy(clinical[patient]),
        torch.from_numpy(pool),
    )


def fit_transformer(batch, changes, *, epochs, seed, prior_mode="local"):
    available = batch.available
    if not available.any():
        raise ValueError("训练患者均缺少质量加权后可用的双模态波形")
    if epochs < 1:
        raise ValueError("epochs 至少为 1")
    target = torch.from_numpy(changes.loc[batch.baseline.index].to_numpy(dtype=np.float32))
    inputs = batch.inputs(prior_mode)
    # 各消融分支使用相同的初始权重和整批患者，训练轮数固定。
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = RecoveryTransformer()
        model.mean_gain.copy_(target[available].mean())
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        for _ in range(epochs):
            optimizer.zero_grad()
            estimated, _, _ = model(*inputs)
            loss = torch.nn.functional.mse_loss(estimated[available], target[available])
            if not torch.isfinite(loss):
                raise FloatingPointError("CNN＋Transformer 训练损失出现非有限值")
            loss.backward()
            optimizer.step()
    return model.eval()


@torch.no_grad()
def predict_transformer(model, batch, *, prior_mode="local"):
    predictions = pd.Series(np.nan, index=batch.baseline.index, dtype=float)
    available = batch.available.numpy()
    if not available.any():
        return predictions, pd.DataFrame(columns=ATTENTION_COLUMNS)
    changes, weights, evidence = model(*batch.inputs(prior_mode))
    endpoints = batch.baseline.to_numpy(dtype=float) + changes.numpy()
    if not np.isfinite(endpoints[available]).all():
        raise FloatingPointError("有效双模态波形的 FMA 预测出现非有限值")
    predictions.loc[available] = np.clip(endpoints[available], 0, 66)

    frames = []
    for token, (mod, phase, start, end) in enumerate(TOKEN_WINDOWS):
        frames.append(
            batch.trials.assign(
                modality=mod,
                phase=phase,
                start_s=start,
                end_s=end,
                quality=batch.quality[:, token].numpy(),
                cnn_evidence=np.where(
                    batch.quality[:, token].numpy() > 0, evidence[:, token].numpy(), np.nan
                ),
                attention=weights[:, token + 1].numpy(),
                clinical_attention=weights[:, 0].numpy(),
            )
        )
    attention = pd.concat(frames, ignore_index=True)
    return predictions, attention.loc[
        attention.subject_id.isin(predictions.dropna().index), ATTENTION_COLUMNS
    ]


def restore_transformer(state):
    model = RecoveryTransformer()
    model.load_state_dict({name: torch.from_numpy(values) for name, values in state.items()})
    return model.eval()
