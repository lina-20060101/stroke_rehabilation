"""卒中上肢运动结局实验：患者级验证、训练和新患者预测。"""

import argparse
import json
from pathlib import Path

import joblib
import mne
import numpy as np
import pandas as pd

from evaluate_window_quality import fit_rankers, load_windows, patient_features, score_windows
from prepare_recovery_data import read_clinical
from stroke_qafusion.recovery_prediction import EEG, FNIRS, JOINT, SCALES, EndpointRegressor


# 主模型和对照在运行前固定。各模型共用患者、预测时点和岭惩罚强度。
MODELS = {
    "signals_only": ("local", EEG + FNIRS),
    "fma_only": ("local", ["fma_ue_before"]),
    "scales_only": ("local", SCALES),
    "scales_eeg": ("local", SCALES + EEG),
    "scales_fnirs": ("local", SCALES + FNIRS),
    "joint_equal": ("equal", JOINT),
    "joint_run_shared": ("run_shared", JOINT),
    "joint_local": ("local", JOINT),
    "joint_direct": ("local", JOINT),
}


def signal_coverage(windows, subjects):
    """使用结果未知时也能核验的纳入条件。允许个别窗或特征缺失。"""
    response = windows.loc[windows.phase.gt(0)]
    coverage = []
    for modality, names in (("eeg", ["alpha", "beta"]), ("fnirs", ["hbo_um"])):
        rows = response[response.modality.eq(modality)]
        valid = np.isfinite(rows[names]).any(axis=1).groupby(rows.subject_id).any()
        coverage.append(valid.reindex(subjects, fill_value=False).rename(f"has_{modality}"))
    return pd.concat(coverage, axis=1)


def make_design(response, scales, weighting):
    signals = patient_features(response, weighting)
    return scales[SCALES].astype(float).join(signals, validate="one_to_one")


def cross_validate(windows, clinical):
    outcome = clinical.fma_ue_after
    records = []
    for held_out in clinical.index:
        train_ids = clinical.index.drop(held_out)
        rankers = fit_rankers(windows[windows.subject_id.isin(train_ids)])
        response = score_windows(windows, rankers)
        designs = {
            weight: make_design(response, clinical, weight)
            for weight in ("local", "equal", "run_shared")
        }
        baseline = float(clinical.at[held_out, "fma_ue_before"])
        # 将该患者的入组分数直接结转，及加上训练患者的平均改善，作为简单参照。
        mean_gain = (outcome.loc[train_ids] - clinical.loc[train_ids, "fma_ue_before"]).mean()
        estimates = {"carry_forward": baseline, "mean_gain": np.clip(baseline + mean_gain, 0, 66)}
        for method, (weight, columns) in MODELS.items():
            design = designs[weight][columns]
            model = EndpointRegressor(direct=method == "joint_direct")
            model.fit(design.loc[train_ids], outcome.loc[train_ids])
            estimates[method] = model.predict(design.loc[[held_out]])[0]
        actual = float(outcome.at[held_out])
        for method, predicted in estimates.items():
            records.append(
                {
                    "subject_id": held_out,
                    "method": method,
                    "fma_ue_before": baseline,
                    "actual": actual,
                    "predicted": predicted,
                    "actual_change": actual - baseline,
                    "absolute_error": abs(predicted - actual),
                }
            )
    return pd.DataFrame(records)


def summarize(predictions, seed):
    metrics = []
    for method, group in predictions.groupby("method", sort=False):
        errors = group.predicted - group.actual
        squared = (errors**2).sum()
        # 变化量 R² 的参照是队列实际改善幅度的方差，用于识别基线严重度的影响。
        change_ss = ((group.actual_change - group.actual_change.mean()) ** 2).sum()
        metrics.append(
            {
                "method": method,
                "patients": len(group),
                "mae": errors.abs().mean(),
                "rmse": np.sqrt((errors**2).mean()),
                "bias": errors.mean(),
                "change_r2": 1 - squared / change_ss if change_ss > 0 else np.nan,
            }
        )
    errors = predictions.pivot(index="subject_id", columns="method", values="absolute_error")
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(errors), size=(10000, len(errors)))
    comparisons = []
    pairs = [
        ("joint_local", "scales_only"),
        ("joint_direct", "signals_only"),
        ("scales_only", "fma_only"),
        ("joint_local", "joint_equal"),
        ("joint_local", "joint_run_shared"),
        ("joint_local", "mean_gain"),
    ]
    for model, reference in pairs:
        gain = (errors[reference] - errors[model]).to_numpy()
        lower, upper = np.quantile(gain[samples].mean(axis=1), [0.025, 0.975])
        comparisons.append(
            {
                "model": model,
                "reference": reference,
                "patients": len(gain),
                "mae_reduction": gain.mean(),
                "paired_bootstrap_low": lower,
                "paired_bootstrap_high": upper,
                "patients_with_lower_error": (gain > 1e-10).sum(),
            }
        )
    return pd.DataFrame(metrics), pd.DataFrame(comparisons)


def fit(args):
    clinical = read_clinical(args.scales).set_index("subject_id")
    windows, audit = load_windows(args.manifest, args.max_session, args.seed)
    unknown = set(audit.subject_id) - set(clinical.index)
    if unknown:
        raise ValueError(f"记录清单中有量表未收录的患者：{sorted(unknown)}")
    if windows.empty:
        raise ValueError(
            f"没有可用的早期配对记录。排除原因：{audit.reason.value_counts().to_dict()}"
        )
    cohort = signal_coverage(windows, clinical.index)
    for column in [*SCALES, "fma_ue_after"]:
        cohort[f"has_{column}"] = clinical[column].notna()
    cohort["eligible"] = cohort.all(axis=1)
    clinical = clinical.loc[cohort.eligible]
    if len(clinical) < 3:
        raise ValueError(f"同时具有入组双量表、末次 FMA 和配对信号的患者仅 {len(clinical)} 名")
    windows = windows[windows.subject_id.isin(clinical.index)]
    predictions = cross_validate(windows, clinical)
    metrics, comparisons = summarize(predictions, args.seed)

    rankers = fit_rankers(windows)
    design = make_design(score_windows(windows, rankers), clinical, "local")
    joint = EndpointRegressor().fit(design[JOINT], clinical.fma_ue_after)
    scale_model = EndpointRegressor().fit(design[SCALES], clinical.fma_ue_after)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(output / "cohort.csv")
    audit.to_csv(output / "run_audit.csv", index=False)
    predictions.to_csv(output / "heldout_predictions.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    comparisons.to_csv(output / "paired_comparisons.csv", index=False)
    protocol = {
        "endpoint": "fma_ue_after",
        "primary_model": "joint_local",
        "primary_reference": "scales_only",
        "metric": "mae",
        "max_session": args.max_session,
        "seed": args.seed,
        "validation": "leave_one_patient_out",
        "ridge_alpha": 10,
        "clinical_inputs": SCALES,
        "signal_inputs": EEG + FNIRS,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    joblib.dump(
        {
            "version": 1,
            "rankers": rankers,
            "joint": joint,
            "scales": scale_model,
            "max_session": args.max_session,
            "seed": args.seed,
        },
        output / "recovery_model.joblib",
    )
    print(metrics.to_string(index=False, float_format=lambda n: f"{n:.3f}"))
    print(comparisons.to_string(index=False, float_format=lambda n: f"{n:.3f}"))
    print(f"输出目录 {output.resolve()}")


def read_baselines(path):
    frame = pd.read_csv(path, usecols=["subject_id", *SCALES], dtype={"subject_id": "string"})
    ids = frame.subject_id.str.strip().str.upper()
    if (
        frame.empty
        or not ids.str.fullmatch(r"[LR]_\d{2,}", na=False).all()
        or ids.duplicated().any()
    ):
        raise ValueError("基线 CSV 需要唯一、有效的患者 ID")
    frame["subject_id"] = ids
    for name, maximum in zip(SCALES, (66, 57)):
        score = pd.to_numeric(frame[name], errors="coerce")
        if not (score.between(0, maximum) & score.mod(1).eq(0)).all():
            raise ValueError(f"{name} 需要 0–{maximum} 的整数")
        frame[name] = score.astype(float)
    return frame.set_index("subject_id")


def predict(args):
    bundle = joblib.load(args.model)
    if bundle["version"] != 1:
        raise ValueError("不支持该模型文件的版本")
    scales = read_baselines(args.baselines)
    windows, audit = pd.DataFrame(), pd.DataFrame(columns=["subject_id", "status", "reason"])
    if args.manifest is not None:
        windows, audit = load_windows(
            args.manifest, bundle["max_session"], bundle["seed"], scales.index
        )
    result = scales.copy()
    result["predicted_fma_scales"] = bundle["scales"].predict(scales)
    result["predicted_fma_joint"] = np.nan
    result["joint_status"] = "missing_early_paired_signal"
    if not windows.empty:
        coverage = signal_coverage(windows, scales.index).all(axis=1)
        if coverage.any():
            response = score_windows(
                windows[windows.subject_id.isin(scales.index[coverage])], bundle["rankers"]
            )
            design = make_design(response, scales.loc[coverage], "local")
            result.loc[coverage, "predicted_fma_joint"] = bundle["joint"].predict(design)
            result.loc[coverage, "joint_status"] = "predicted"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "predictions.csv")
    audit.to_csv(args.output_dir / "inference_run_audit.csv", index=False)
    print(f"量表预测 {len(result)} 人，联合预测 {result.predicted_fma_joint.notna().sum()} 人")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("fit", help="运行留一患者评估并拟合固定的最终模型")
    train.add_argument("--scales", type=Path, required=True)
    train.add_argument("--max-session", type=int, choices=[1, 2], default=2)
    train.add_argument("--seed", type=int, default=42)
    infer = commands.add_parser("predict", help="仅用早期记录和入组量表预测")
    infer.add_argument("--model", type=Path, required=True)
    infer.add_argument("--baselines", type=Path, required=True)
    for command in (train, infer):
        command.add_argument("--manifest", type=Path, required=command is train)
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    mne.set_log_level("ERROR")
    if args.command == "fit":
        fit(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
