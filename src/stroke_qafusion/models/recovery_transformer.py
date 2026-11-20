"""从单次刺激的四个波形窗提取表示，汇总到患者后回归 FMA-UE 改善量。"""

import torch
from torch import nn

from ..window_quality import WINDOWS


WIDTH = 12
HEADS = 2
DEPTH = 2
TOKEN_WINDOWS = [
    (modality, phase, start, end)
    for modality in ("eeg", "fnirs")
    for phase, (start, end) in enumerate(WINDOWS[modality], start=1)
]


def window_encoder():
    # 保留幅度信息。卷积只沿一个模态自身的时间轴运行。
    return nn.Sequential(
        nn.Conv1d(1, 6, kernel_size=7, stride=2, padding=3),
        nn.GELU(),
        nn.Conv1d(6, WIDTH, kernel_size=5, stride=2, padding=2),
        nn.GELU(),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(1),
    )


class QualityAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(WIDTH)
        self.attention = nn.MultiheadAttention(WIDTH, HEADS, batch_first=True)
        self.norm2 = nn.LayerNorm(WIDTH)
        self.ffn = nn.Sequential(
            nn.Linear(WIDTH, 2 * WIDTH), nn.GELU(), nn.Linear(2 * WIDTH, WIDTH)
        )

    def forward(self, tokens, key_bias):
        x = self.norm1(tokens)
        # 浮点 key_padding_mask 按样本、按 key 加到每个头的注意力 logits。
        update, weights = self.attention(x, x, x, key_padding_mask=key_bias)
        tokens = tokens + update
        return tokens + self.ffn(self.norm2(tokens)), weights


class RecoveryTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.eeg_cnn = window_encoder()
        self.fnirs_cnn = window_encoder()
        self.local_evidence = nn.Linear(WIDTH, 1)
        self.clinical = nn.Linear(2, WIDTH)
        self.modality = nn.Embedding(2, WIDTH)
        self.event_time = nn.Linear(1, WIDTH, bias=False)
        centres = [(start + end) / 2 for _, _, start, end in TOKEN_WINDOWS]
        self.register_buffer("centres_s", torch.tensor(centres).reshape(-1, 1))
        self.register_buffer("mean_gain", torch.tensor(0.0))
        self.blocks = nn.ModuleList([QualityAttentionBlock() for _ in range(DEPTH)])
        self.readout = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, 1))

    def forward(self, eeg, fnirs, prior, clinical, patient_pool):
        """波形 [trial,2,samples]，质量先验 [trial,4]，量表 [trial,2]，汇总矩阵 [patient,trial]。"""
        n_trials = eeg.shape[0]
        eeg = eeg.masked_fill(prior[:, :2, None] == 0, 0)
        fnirs = fnirs.masked_fill(prior[:, 2:, None] == 0, 0)
        eeg_tokens = self.eeg_cnn(eeg.reshape(-1, 1, eeg.size(-1))).reshape(n_trials, 2, WIDTH)
        fnirs_tokens = self.fnirs_cnn(fnirs.reshape(-1, 1, fnirs.size(-1))).reshape(
            n_trials, 2, WIDTH
        )
        windows = torch.cat((eeg_tokens, fnirs_tokens), dim=1)
        # CNN 初判只读取局部波形，学习该窗对预测任务的相对权重，无逐窗 FMA 标签。
        evidence = self.local_evidence(windows).squeeze(-1).tanh()
        windows = windows + self.modality.weight.repeat_interleave(2, dim=0)
        windows = windows + self.event_time(self.centres_s / 8.0)
        tokens = torch.cat((self.clinical(clinical).unsqueeze(1), windows), dim=1)

        # 每层每头的 logits = QK/sqrt(d) + CNN 证据 + log(质量先验)；量表 key 偏置为零。
        key_bias = torch.cat((torch.zeros_like(prior[:, :1]), evidence + prior.log()), dim=1)
        for block in self.blocks:
            tokens, weights = block(tokens, key_bias)
        patient_repr = patient_pool @ tokens[:, 0]
        change = self.mean_gain + self.readout(patient_repr).squeeze(-1)
        return change, weights[:, 0], evidence  # 末层量表 query 对五个 key 的注意力，按头平均
