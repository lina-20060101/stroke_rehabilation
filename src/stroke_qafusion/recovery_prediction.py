"""患者级预测的输入契约与固定岭回归。"""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCALES = ["fma_ue_before", "arat_before"]
EEG = ["alpha_phase1", "alpha_phase2", "beta_phase1", "beta_phase2"]
FNIRS = ["hbo_um_phase1", "hbo_um_phase2"]
JOINT = SCALES + EEG + FNIRS


class EndpointRegressor:
    """有入组 FMA 时用其作为已知起点，无量表对照直接估计末次分数。"""

    def __init__(self, *, direct=False):
        self.direct = direct

    def fit(self, features, outcome):
        if not features.columns.is_unique or not set(features).issubset(JOINT):
            raise ValueError("输入列须为已声明的入组量表或早期脑信号特征")
        self.columns = list(features)
        target = outcome.loc[features.index].astype(float)
        self.use_baseline_offset = "fma_ue_before" in self.columns and not self.direct
        if self.use_baseline_offset:
            target = target - features.fma_ue_before
        self.pipeline = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=10),
        )
        self.pipeline.fit(features, target)
        return self

    def predict(self, features):
        ordered = features.loc[:, self.columns]
        prediction = self.pipeline.predict(ordered)
        if self.use_baseline_offset:
            prediction = prediction + ordered.fma_ue_before.to_numpy(dtype=float)
        return np.clip(prediction, 0, 66)
