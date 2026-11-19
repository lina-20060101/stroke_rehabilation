"""时间门控回归检查，使用合成的逐窗指标。"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from stroke_qafusion.window_quality import DAMAGED_COLUMNS, QC_COLUMNS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_window_quality import patient_features, score_fold  # noqa: E402


def example_windows():
    records = []
    for subject in ("L_01", "R_02", "R_03"):
        for modality in ("eeg", "fnirs"):
            for trial in range(4):
                for phase in range(3):
                    original = [0.0, 0.01 * trial, 0.02 * phase]
                    records.append(
                        {
                            "subject_id": subject,
                            "session_id": 1,
                            "run_id": 1,
                            "target_hand": "left" if subject.startswith("L") else "right",
                            "trial": trial,
                            "modality": modality,
                            "phase": phase,
                            "alpha": 2.0,
                            "beta": 3.0,
                            "hbo_um": 4.0,
                            **dict(zip(QC_COLUMNS, original)),
                            **dict(zip(DAMAGED_COLUMNS, [1.0, 1.0, 0.0])),
                        }
                    )
    return pd.DataFrame(records)


class TestWindowQuality(unittest.TestCase):
    def test_held_out_windows_do_not_train_ranker(self):
        windows = example_windows()
        original = score_fold(windows, "L_01")
        windows.loc[windows.subject_id.eq("L_01"), QC_COLUMNS + DAMAGED_COLUMNS] = 0.7
        changed = score_fold(windows, "L_01")
        train = original.subject_id.ne("L_01")
        np.testing.assert_allclose(original.loc[train, "local"], changed.loc[train, "local"])
        np.testing.assert_allclose(original.loc[train, "score"], changed.loc[train, "score"])

    def test_local_damage_and_unusable_baseline(self):
        windows = example_windows()
        dead_base = windows.subject_id.eq("L_01") & windows.modality.eq("eeg") & windows.phase.eq(0)
        windows.loc[dead_base, QC_COLUMNS] = [1, 1, 0]
        response = score_fold(windows, "L_01")
        eeg = response.subject_id.eq("L_01") & response.modality.eq("eeg")
        self.assertTrue(response.loc[eeg, "local"].eq(0).all())
        self.assertTrue(response.loc[~eeg, "local"].gt(0).all())
        self.assertTrue(response.after_baseline_damage.eq(0).all())
        self.assertTrue(response.loc[response.stress_target, "stress_score"].eq(0).all())
        controls = response[~response.stress_target]
        np.testing.assert_equal(controls.local.to_numpy(), controls.stress_score.to_numpy())
        features = patient_features(response, "local")
        self.assertTrue(np.isnan(features.loc["L_01", "alpha_phase1"]))
        self.assertAlmostEqual(features.loc["L_01", "hbo_um_phase1"], 4)


if __name__ == "__main__":
    unittest.main()
