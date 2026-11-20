"""质量先验在注意力内生效、整人留出、模型往返及回退的回归检查。"""

from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from test_window_quality import example_windows
from evaluate_window_quality import score_fold
from recovery_experiment import cross_validate, fit, predict
from stroke_qafusion.recovery_prediction import SCALES
from stroke_qafusion.window_quality import WAVEFORM_SAMPLES, response_waveforms

if find_spec("torch") is not None:
    import torch
    from stroke_qafusion.models.recovery_transformer import WIDTH, RecoveryTransformer
    from stroke_qafusion.neural_recovery import patient_batch


def example_inputs():
    windows = example_windows()
    windows["waveform"] = [
        np.sin(np.linspace(0, 6, WAVEFORM_SAMPLES[r.modality]) + r.phase).astype(np.float32)
        if r.phase
        else None
        for r in windows.itertuples()
    ]
    clinical = pd.DataFrame(
        {"fma_ue_before": [10, 20, 30], "arat_before": [5, 8, 22], "fma_ue_after": [12, 24, 32]},
        index=pd.Index(["L_01", "R_02", "R_03"], name="subject_id"),
    )
    return windows, clinical


@unittest.skipUnless(find_spec("torch") is not None, "需要 train 可选依赖")
class TestRecoveryTransformer(unittest.TestCase):
    def test_quality_bias_inside_attention_and_masked_waveform(self):
        torch.manual_seed(10)
        model = RecoveryTransformer()
        # 将内容 QK 项固定为零，可精确核对两种窗分数在 softmax 内的作用。
        with torch.no_grad():
            for layer in model.blocks:
                layer.attention.in_proj_weight[: 2 * WIDTH].zero_()
                layer.attention.in_proj_bias[: 2 * WIDTH].zero_()
        eeg = torch.randn(2, 2, 128)
        fnirs = torch.randn(2, 2, 28)
        q = torch.tensor([[0.25, 0.75, 0.0, 0.5], [0.5, 1.0, 0.0, 0.75]])
        clinical = torch.rand(2, 2)
        changes, weights, evidence = model(eeg, fnirs, q, clinical, torch.eye(2))
        odds = torch.cat((torch.ones(2, 1), evidence.exp() * q), dim=1)
        torch.testing.assert_close(weights, odds / odds.sum(dim=1, keepdim=True))
        # 被屏蔽的 key 对预测、其他窗注意力和梯度均不能产生污染。
        fnirs_masked = fnirs.clone()
        fnirs_masked[:, 0] = torch.nan
        changed, weights_masked, _ = model(eeg, fnirs_masked, q, clinical, torch.eye(2))
        torch.testing.assert_close(changes, changed)
        torch.testing.assert_close(weights, weights_masked)
        changes.sum().backward()
        for parameter in (
            model.eeg_cnn[0].weight,
            model.fnirs_cnn[0].weight,
            model.local_evidence.weight,
        ):
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_patient_pool_and_run_prior(self):
        windows, clinical = example_inputs()
        rows = score_fold(windows, "L_01")
        rows["local"] = (1 + rows.trial) / 5
        rows.loc[rows.subject_id.eq("R_02") & rows.modality.eq("fnirs"), "waveform"] = None
        trial = rows[rows.subject_id.eq("L_01") & rows.trial.eq(0)]
        rows = pd.concat([rows, trial.assign(run_id=2), trial.assign(session_id=2)])
        batch = patient_batch(rows.sample(frac=1, random_state=10), clinical[SCALES])
        self.assertEqual(batch.available.tolist(), [True, False, True])
        torch.testing.assert_close(batch.pool.sum(dim=1), torch.ones(3))
        keys = batch.trials
        np.testing.assert_allclose(batch.pool[0, keys.subject_id.ne("L_01").to_numpy()], 0)
        baseline_run = (
            keys.subject_id.eq("L_01") & keys.session_id.eq(1) & keys.run_id.eq(1)
        ).to_numpy()
        self.assertAlmostEqual(batch.pool[0, baseline_run].sum().item(), 0.25)
        shared = batch.inputs("run_shared")[2]
        torch.testing.assert_close(shared[baseline_run], torch.full((4, 4), 0.5))
        torch.testing.assert_close(shared > 0, batch.quality > 0)
        torch.testing.assert_close(batch.inputs("none")[2], (batch.quality > 0).float())

    def test_fit_predict_and_heldout_outcome_is_hidden(self):
        windows, clinical = example_inputs()
        with TemporaryDirectory() as folder:
            root = Path(folder)
            args = SimpleNamespace(
                architecture="cnn-transformer",
                epochs=1,
                scales=None,
                manifest=None,
                max_session=2,
                seed=42,
                output_dir=root,
            )
            audit = windows[["subject_id"]].drop_duplicates().assign(status="usable", reason="")
            with (
                patch("recovery_experiment.read_clinical", return_value=clinical.reset_index()),
                patch("recovery_experiment.load_windows", return_value=(windows, audit)),
            ):
                fit(args)
            oof = pd.read_csv(root / "heldout_predictions.csv")
            altered = clinical.copy()
            altered.at["L_01", "fma_ue_after"] = 60
            other = cross_validate(windows, altered, cnn_epochs=1, seed=42)

            def held(frame):
                return frame.loc[
                    frame.subject_id.eq("L_01") & frame.method.str.startswith("cnn_transformer"),
                    "predicted",
                ]

            np.testing.assert_allclose(held(oof), held(other), rtol=0, atol=1e-6)
            self.assertEqual(len(pd.read_csv(root / "heldout_attention.csv")), 3 * 4 * 4)

            clinical[SCALES].to_csv(root / "baselines.csv")
            infer = SimpleNamespace(
                model=root / "recovery_model.joblib",
                baselines=root / "baselines.csv",
                manifest=root / "manifest.csv",
                output_dir=root,
            )
            windows.loc[
                windows.subject_id.eq("L_01") & windows.modality.eq("fnirs"), "waveform"
            ] = None
            outage = cross_validate(windows, clinical, cnn_epochs=1, seed=42)
            held_scores = outage.loc[outage.subject_id.eq("L_01")].set_index("method")
            self.assertEqual(held_scores.at["cnn_transformer", "prediction_source"], "scales_only")
            self.assertEqual(
                held_scores.at["cnn_transformer", "predicted"],
                held_scores.at["scales_only", "predicted"],
            )
            with patch("recovery_experiment.load_windows", return_value=(windows, audit)):
                predict(infer)
            result = pd.read_csv(root / "predictions.csv", index_col="subject_id")
            self.assertEqual(
                result.prediction_source.tolist(),
                ["scales_only", "cnn_transformer", "cnn_transformer"],
            )
            self.assertEqual(
                result.at["L_01", "predicted_fma"], result.at["L_01", "predicted_fma_scales"]
            )
            self.assertTrue(np.isnan(result.at["L_01", "predicted_fma_joint"]))
            attention = pd.read_csv(root / "inference_attention.csv")
            self.assertEqual(set(attention.subject_id), {"R_02", "R_03"})
            infer.manifest = None
            predict(infer)
            self.assertTrue(
                pd.read_csv(root / "predictions.csv").prediction_source.eq("scales_only").all()
            )

    def test_cnn_waveform_baseline_and_modality_grid(self):
        signal = np.linspace(-1e-6, 1e-6, 40)[None, None, :]
        for mod, length in WAVEFORM_SAMPLES.items():
            (original,) = response_waveforms(mod, signal, signal)
            (shifted,) = response_waveforms(mod, signal + 2e-6, signal + 2e-6)
            self.assertEqual(original.shape, (length,))
            np.testing.assert_allclose(original, shifted, atol=1e-6)
            self.assertEqual(response_waveforms(mod, np.zeros_like(signal), signal), [None])
