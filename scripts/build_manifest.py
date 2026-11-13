"""从解压后的 Figshare EEG/fNIRS 数据中生成 run 级数据清单。

本脚本只索引文件，不读取或修改信号，也不把临床评分写入清单；
临床量表表格应另行维护，并通过 subject_id 关联。
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


FILENAME_KEY = re.compile(
    r"(?P<paretic>[LR])_(?P<subject>\d+)_session(?P<session>\d+)_"
    r"(?P<hand>left|right)(?P<run>\d+)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, order=True)
class RunKey:
    subject_id: str
    paretic_side: str
    session_id: int
    target_hand: str
    run_id: int


def parse_run_key(path: Path) -> RunKey | None:
    """解析公开数据集文档定义的患者/session/任务标识。"""
    match = FILENAME_KEY.search(path.name)
    if not match:
        return None
    groups = match.groupdict()
    return RunKey(
        subject_id=f"{groups['paretic'].upper()}_{int(groups['subject']):02d}",
        paretic_side=groups["paretic"].upper(),
        session_id=int(groups["session"]),
        target_hand=groups["hand"].lower(),
        run_id=int(groups["run"]),
    )


def modality_for(path: Path) -> str | None:
    suffix = path.suffix.lower()
    parent_tokens = {part.lower() for part in path.parts}
    if suffix == ".txt" and "event" in path.name.lower():
        return "event"
    if suffix == ".snirf" or "fnirs" in parent_tokens:
        return "fnirs"
    if suffix == ".edf" or "eeg" in parent_tokens:
        return "eeg"
    return None


def choose_preferred(existing: str, candidate: Path, modality: str) -> str:
    """当数据包同时提供多种格式时，优先选择 FIF 文件。"""
    if not existing:
        return str(candidate)
    current = Path(existing)
    preferred = ".fif"
    if candidate.suffix.lower() == preferred and current.suffix.lower() != preferred:
        return str(candidate)
    return existing


def build_manifest(data_root: Path) -> tuple[list[dict[str, str]], list[Path]]:
    rows: dict[RunKey, dict[str, str]] = {}
    unparsed: list[Path] = []

    for path in sorted(p for p in data_root.rglob("*") if p.is_file()):
        modality = modality_for(path)
        if modality is None:
            continue
        key = parse_run_key(path)
        if key is None:
            unparsed.append(path)
            continue
        row = rows.setdefault(
            key,
            {
                "subject_id": key.subject_id,
                "paretic_side": key.paretic_side,
                "session_id": str(key.session_id),
                "run_id": str(key.run_id),
                "target_hand": key.target_hand,
                "eeg_path": "",
                "fnirs_path": "",
                "event_path": "",
            },
        )
        column = f"{modality}_path"
        row[column] = choose_preferred(row[column], path, modality)

    return [rows[key] for key in sorted(rows)], unparsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="解压后的数据集根目录")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/recordings.csv"),
        help="输出 CSV 文件路径",
    )
    args = parser.parse_args()
    if not args.data_root.exists():
        raise SystemExit(f"数据目录不存在：{args.data_root}")

    rows, unparsed = build_manifest(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject_id",
        "paretic_side",
        "session_id",
        "run_id",
        "target_hand",
        "eeg_path",
        "fnirs_path",
        "event_path",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    paired = sum(bool(row["eeg_path"] and row["fnirs_path"]) for row in rows)
    print(f"已索引 {len(rows)} 个 run；其中 {paired} 个同时包含 EEG 和 fNIRS。")
    print(f"数据清单已写入：{args.output}")
    if unparsed:
        print(f"有 {len(unparsed)} 个候选文件名未识别，已跳过：")
        for path in unparsed[:10]:
            print(f"  {path}")


if __name__ == "__main__":
    main()
