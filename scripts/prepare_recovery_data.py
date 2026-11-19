"""使用前几个 session 的患手任务记录准备患者级 FMA-UE 预测数据。"""

import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from openpyxl import load_workbook

from stroke_qafusion.recovery_signals import extract_run


CLINICAL_COLUMNS = {
    "ID": "subject_id",
    "Fugl-Meyer UE before": "fma_ue_before",
    "Fugl-Meyer UE after": "fma_ue_after",
    "ARAT before": "arat_before",
    "ARAT after": "arat_after",
}
KEY_COLUMNS = ["subject_id", "session_id", "target_hand", "run_id"]
MANIFEST_COLUMNS = KEY_COLUMNS + ["paretic_side", "eeg_path", "fnirs_path"]


def read_clinical(path: Path) -> pd.DataFrame:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = workbook.active.values
        header = [str(value).strip() for value in next(rows, ())]
        missing = CLINICAL_COLUMNS.keys() - set(header)
        if missing:
            raise ValueError(f"量表缺少列：{', '.join(sorted(missing))}")
        positions = {name: header.index(name) for name in CLINICAL_COLUMNS}
        clinical = pd.DataFrame([
            {column: row[positions[source]] for source, column in CLINICAL_COLUMNS.items()}
            for row in rows if row[positions["ID"]] is not None
        ])
    finally:
        workbook.close()

    if clinical.empty:
        raise ValueError("量表没有患者记录")
    clinical["subject_id"] = clinical["subject_id"].astype(str).str.strip().str.upper()
    if not clinical["subject_id"].str.fullmatch(r"[LR]_\d{2,}").all():
        raise ValueError("量表含无法识别的患者 ID")
    if clinical["subject_id"].duplicated().any():
        raise ValueError("量表存在重复患者 ID")
    for name in clinical.columns.drop("subject_id"):
        raw = clinical[name].astype("string").str.strip()
        missing = raw.isna() | raw.str.fullmatch(r"[-–—−]*", na=False)
        scores = pd.to_numeric(raw.mask(missing), errors="coerce")
        upper = 66 if name.startswith("fma") else 57
        invalid = ~missing & (scores.isna() | ~scores.between(0, upper) | (scores % 1 != 0))
        if invalid.any():
            raise ValueError(f"{name} 有无效评分，量表行：{list(clinical.index[invalid] + 2)}")
        clinical[name] = scores.astype("Int64")
    clinical["fma_ue_change"] = clinical["fma_ue_after"] - clinical["fma_ue_before"]
    return clinical.sort_values("subject_id")


def read_manifest(path: Path) -> pd.DataFrame:
    runs = pd.read_csv(path, keep_default_na=False)
    missing = set(MANIFEST_COLUMNS) - set(runs)
    if missing:
        raise ValueError(f"记录清单缺少列：{', '.join(sorted(missing))}")
    if runs.empty:
        raise ValueError("记录清单为空，请先运行 build_manifest.py")
    ids_valid = runs["subject_id"].str.fullmatch(r"[LR]_\d{2,}")
    sides_match = runs["subject_id"].str[0] == runs["paretic_side"]
    if not (ids_valid & sides_match & runs["target_hand"].isin(["left", "right"])).all():
        raise ValueError("记录清单的患者 ID、患侧或目标手不合法")
    for column in ("session_id", "run_id"):
        values = pd.to_numeric(runs[column], errors="coerce")
        if not (values.ge(1) & values.mod(1).eq(0)).all():
            raise ValueError(f"{column} 必须为正整数")
        runs[column] = values.astype(int)
    if runs.duplicated(KEY_COLUMNS).any():
        raise ValueError("记录清单含重复 run")
    return runs[MANIFEST_COLUMNS].sort_values(KEY_COLUMNS)


def process_runs(runs: pd.DataFrame) -> pd.DataFrame:
    processed = []
    for run in runs.itertuples(index=False):
        record = {name: getattr(run, name) for name in KEY_COLUMNS}
        paths = (Path(run.eeg_path), Path(run.fnirs_path))
        try:
            if not all(path.is_file() and path.suffix.lower() == ".fif" for path in paths):
                raise ValueError("缺少可读取的 EEG/fNIRS FIF 双模态配对")
            features = extract_run(*paths, run.target_hand)
            if not np.isfinite(list(features.values())).all():
                raise ValueError("特征含非有限值")
        except (OSError, ValueError, RuntimeError) as error:
            record.update(status="excluded", reason=str(error))
        else:
            record.update(status="usable", reason="", **features)
        processed.append(record)
    if not processed:
        return pd.DataFrame(columns=KEY_COLUMNS + ["status", "reason"])
    return pd.DataFrame(processed)


def prepare(manifest: Path, scales: Path, output: Path, max_session: int) -> None:
    runs = read_manifest(manifest)
    clinical = read_clinical(scales)
    unknown = set(runs["subject_id"]) - set(clinical["subject_id"])
    if unknown:
        raise ValueError(f"记录清单含量表未收录的患者 ID：{', '.join(sorted(unknown))}")

    paretic_hand = runs["paretic_side"].map({"L": "left", "R": "right"})
    early = runs[(runs["session_id"] <= max_session) & (runs["target_hand"] == paretic_hand)]
    run_data = process_runs(early)
    usable = run_data[run_data["status"] == "usable"]
    feature_columns = [
        name for name in usable if name.startswith(("eeg_", "fnirs_", "qc_"))
    ]
    # 同一 session 内取中位数，随后按 session 等权汇总。
    sessions = usable.groupby(["subject_id", "session_id"])[feature_columns].median()
    patient_features = sessions.groupby(level="subject_id").mean()

    audit = clinical[["subject_id"]].set_index("subject_id")
    audit["first_recorded_session"] = runs.groupby("subject_id")["session_id"].min()
    audit["early_paretic_runs"] = early.groupby("subject_id").size().reindex(audit.index, fill_value=0)
    audit["usable_runs"] = usable.groupby("subject_id").size().reindex(audit.index, fill_value=0)
    audit["usable_sessions"] = (
        usable.groupby("subject_id")["session_id"].nunique().reindex(audit.index, fill_value=0)
    )
    audit["max_session"] = max_session
    valid_fma = clinical.set_index("subject_id")[["fma_ue_before", "fma_ue_after"]].notna().all(axis=1)
    audit["fma_complete"] = valid_fma
    audit["eligible_for_fma"] = audit["fma_complete"] & audit["usable_runs"].gt(0)
    patient_features = patient_features.reindex(audit.index[audit["eligible_for_fma"]])
    patient_features.insert(0, "fma_ue_before", clinical.set_index("subject_id")["fma_ue_before"])

    output.mkdir(parents=True, exist_ok=True)
    clinical.to_csv(output / "clinical_outcomes.csv", index=False)
    audit.to_csv(output / "cohort_audit.csv")
    run_data.to_csv(output / "run_features.csv", index=False)
    patient_features.to_csv(output / "patient_features.csv")
    print(f"量表患者 {len(clinical)} 名，FMA-UE 前后齐全 {valid_fma.sum()} 名。")
    print(f"session 1–{max_session} 患手候选 {len(early)} 个 run，可用 {len(usable)} 个。")
    print(f"可进行 FMA-UE 结局分析 {len(patient_features)} 名。输出：{output.resolve()}")
    if patient_features.empty:
        raise SystemExit("当前截止点没有可用于 FMA-UE 分析的患者，请查看审计 CSV。")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scales", type=Path, required=True, help="官方患者量表 XLSX")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/recovery"))
    parser.add_argument(
        "--max-session", type=int, choices=[1, 2], default=2,
        help="固定预测截止点，按官方 session 编号",
    )
    args = parser.parse_args()
    mne.set_log_level("ERROR")
    prepare(args.manifest, args.scales, args.output_dir, args.max_session)


if __name__ == "__main__":
    main()
