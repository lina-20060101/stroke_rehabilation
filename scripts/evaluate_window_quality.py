"""留一患者验证刺激时间窗门控及其对末次 FMA-UE 预测的影响。"""

import argparse
from pathlib import Path
from zlib import crc32

import mne
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from prepare_recovery_data import read_clinical, read_manifest
from stroke_qafusion.window_quality import (
    DAMAGED_COLUMNS,
    QC_COLUMNS,
    WindowQualityRanker,
    extract_quality_windows,
)


RUN = ["subject_id", "session_id", "target_hand", "run_id"]
TRIAL = RUN + ["trial"]


def load_windows(manifest, max_session, seed):
    runs = read_manifest(manifest)
    hand = runs.paretic_side.map({"L": "left", "R": "right"})
    early = runs[runs.session_id.le(max_session) & runs.target_hand.eq(hand)]
    records, audit = [], []
    for run in early.itertuples(index=False):
        key = {name: getattr(run, name) for name in RUN}
        # 固定到 run 标识，其他文件的顺序和排除情况不会改变该 run 的扰动。
        run_seed = crc32("/".join(map(str, key.values())).encode(), seed)
        try:
            paths = [Path(run.eeg_path), Path(run.fnirs_path)]
            if not all(p.is_file() and p.suffix.lower() == ".fif" for p in paths):
                raise ValueError("缺少配对 FIF 文件")
            windows = extract_quality_windows(
                *paths, run.target_hand, np.random.default_rng(run_seed)
            )
        except (OSError, ValueError, RuntimeError) as exc:
            audit.append(dict(key, status="excluded", reason=str(exc)))
        else:
            records.extend(key | window for window in windows)
            audit.append(dict(key, status="usable", reason=""))
    return pd.DataFrame(records), pd.DataFrame(audit, columns=RUN + ["status", "reason"])


def score_fold(windows, held_out):
    parts = []
    for modality, block in windows.groupby("modality"):
        train = block[block.subject_id.ne(held_out) & block.unusable_fraction.lt(1)]
        if train.empty:
            raise ValueError(f"训练折缺少可用的 {modality} 时间窗")
        ranker = WindowQualityRanker().fit(train[QC_COLUMNS], train[DAMAGED_COLUMNS])
        parts.append(
            block.assign(
                score=ranker.score(block[QC_COLUMNS]),
                damaged_score=ranker.score(block[DAMAGED_COLUMNS]),
            )
        )
    scored = pd.concat(parts, ignore_index=True)
    key = TRIAL + ["modality"]
    baseline = scored[scored.phase.eq(0)][key + ["score", "damaged_score"]].rename(
        columns={"score": "baseline_score", "damaged_score": "damaged_baseline_score"}
    )
    response = scored[scored.phase.gt(0)].merge(baseline, on=key, validate="many_to_one")
    response["local"] = np.minimum(response.score, response.baseline_score)
    response["after_baseline_damage"] = np.minimum(response.score, response.damaged_baseline_score)
    response["equal"] = 1.0
    response["run_shared"] = response.groupby(RUN + ["modality"]).local.transform("mean")

    # 每个 trial 的四个响应窗轮流选一个加扰；其余窗作为同次刺激内的对照。
    selected = (response.trial + response.run_id + response.session_id).mod(4)
    target = response.phase.eq(selected.mod(2) + 1) & response.modality.eq(
        np.where(selected.lt(2), "eeg", "fnirs")
    )
    response["stress_target"] = target
    response["stress_score"] = np.minimum(
        response.score.where(~target, response.damaged_score), response.baseline_score
    )
    return response


def patient_features(response, weighting):
    """每个 run 先获得相等的试验总权重，再应用候选质量权重；跨 session 等权。"""
    columns = []
    for modality, block in response.groupby("modality"):
        feature_names = ["alpha", "beta"] if modality == "eeg" else ["hbo_um"]
        n_trials = block.groupby(RUN + ["phase"]).trial.transform("size")
        weights = block[weighting] / n_trials
        for name in feature_names:
            values = block[name]
            valid_weights = weights.where(values.notna(), 0)
            keys = [block.subject_id, block.session_id, block.phase]
            numerator = (values * valid_weights).groupby(keys).sum(min_count=1)
            denominator = valid_weights.groupby(keys).sum()
            session = numerator.div(denominator.where(denominator > 0))
            patient = session.groupby(level=["subject_id", "phase"]).mean().unstack()
            patient.columns = [f"{name}_phase{phase}" for phase in patient.columns]
            columns.append(patient)
    return pd.concat(columns, axis=1)


def predict_endpoint(design, clinical, held_out):
    # 预先指定的模型预测变化量，加回已知的入组分数得到末次 FMA-UE。
    train = design.index != held_out
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=10),
    )
    change = clinical.fma_ue_after - clinical.fma_ue_before
    model.fit(design.loc[train], change.loc[design.index[train]].astype(float))
    gain = model.predict(design.loc[[held_out]])[0]
    return float(np.clip(clinical.at[held_out, "fma_ue_before"] + gain, 0, 66))


def evaluate(windows, clinical):
    subjects = clinical.dropna(subset=["fma_ue_before", "fma_ue_after"]).index
    subjects = subjects.intersection(windows.subject_id.unique()).sort_values()
    if len(subjects) < 3:
        raise ValueError("至少需要三名具有配对记录及完整 FMA 的患者进行留一验证")
    windows = windows[windows.subject_id.isin(subjects)]
    baseline = clinical.loc[subjects, ["fma_ue_before"]].astype(float)
    predictions, diagnostics = [], []
    for held_out in subjects:
        response = score_fold(windows, held_out)
        diagnostics.append(response[response.subject_id.eq(held_out)].copy())
        for method in ("clinical", "equal", "run_shared", "local"):
            design = baseline
            if method != "clinical":
                design = baseline.join(patient_features(response, method), validate="one_to_one")
            prediction = predict_endpoint(design, clinical, held_out)
            actual = float(clinical.at[held_out, "fma_ue_after"])
            predictions.append(
                {
                    "subject_id": held_out,
                    "method": method,
                    "actual": actual,
                    "predicted": prediction,
                    "absolute_error": abs(actual - prediction),
                }
            )
    return pd.DataFrame(predictions), pd.concat(diagnostics, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--scales", required=True, type=Path)
    parser.add_argument("--max-session", type=int, choices=(1, 2), default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/window-quality"))
    args = parser.parse_args()
    mne.set_log_level("ERROR")
    clinical = read_clinical(args.scales).set_index("subject_id")
    windows, audit = load_windows(args.manifest, args.max_session, args.seed)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output / "run_audit.csv", index=False)
    if windows.empty:
        raise SystemExit("没有可用时间窗，详见 run_audit.csv")
    unknown = set(windows.subject_id) - set(clinical.index)
    if unknown:
        raise ValueError(f"量表中找不到患者 ID：{sorted(unknown)}")
    predictions, weights = evaluate(windows, clinical)
    metrics = predictions.groupby("method").absolute_error.agg(patients="size", mae="mean")
    weights["stress_drop"] = weights.local - weights.stress_score
    target = weights[weights.stress_target].copy()
    target["weight_decreased"] = target.stress_drop.gt(1e-8)
    stress = target.groupby(["modality", "perturbation"]).agg(
        windows=("stress_drop", "size"),
        mean_drop=("stress_drop", "mean"),
        fraction_decreased=("weight_decreased", "mean"),
    )
    predictions.to_csv(output / "heldout_predictions.csv", index=False)
    metrics.to_csv(output / "endpoint_metrics.csv")
    weights.to_csv(output / "heldout_window_weights.csv", index=False)
    stress.to_csv(output / "localization_metrics.csv")
    print(metrics.to_string())
    print(stress.to_string())
    print(f"run 纳入 {audit.status.eq('usable').sum()}/{len(audit)}。输出目录 {output.resolve()}")


if __name__ == "__main__":
    main()
