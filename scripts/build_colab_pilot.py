from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TRAIN_QUOTAS = {0: 2000, 1: 11000, 2: 2000}
DEFAULT_VAL_QUOTAS = {0: 500, 1: 1000, 2: 500}


def _sample_rows(frame: pd.DataFrame, quotas: dict[int, int], seed: int) -> pd.DataFrame:
    parts = []
    for label_id, target in quotas.items():
        pool = frame[frame["label_id"] == label_id]
        if len(pool) < target:
            raise ValueError(
                f"Not enough rows for label {label_id}: requested {target}, available {len(pool)}"
            )
        parts.append(pool.sample(n=target, random_state=seed + label_id))
    result = pd.concat(parts, ignore_index=True)
    return result.sample(frac=1.0, random_state=seed + 100).reset_index(drop=True)


def _materialize_waveforms(frame: pd.DataFrame, ecg_root: Path, output_root: Path) -> dict:
    copied_files = 0
    copied_bytes = 0
    missing = []
    for record_path in frame["record_path"].astype(str).unique():
        source_base = ecg_root / record_path
        candidates = sorted(source_base.parent.glob(source_base.name + ".*"))
        if not candidates:
            missing.append(record_path)
            continue
        destination_dir = output_root / source_base.parent.relative_to(ecg_root)
        destination_dir.mkdir(parents=True, exist_ok=True)
        for source in candidates:
            destination = destination_dir / source.name
            if not destination.exists():
                shutil.copy2(source, destination)
            copied_files += 1
            copied_bytes += destination.stat().st_size
    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"Missing waveform files for {len(missing)} records. First examples: {preview}"
        )
    return {"copied_files": copied_files, "copied_bytes": copied_bytes}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a patient-disjoint Colab pilot subset entirely from the official training split."
    )
    parser.add_argument(
        "--split-csv",
        default="data/processed/precomputed_ecg_potassium_cohort_split.csv",
    )
    parser.add_argument(
        "--ecg-root",
        default="/home/bdm0162/hypok-data/mimic-iv-ecg/1.0",
    )
    parser.add_argument("--output-dir", default="colab_pilot")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--val-subject-fraction", type=float, default=0.20)
    args = parser.parse_args()

    split_csv = Path(args.split_csv).expanduser().resolve()
    ecg_root = Path(args.ecg_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    waveform_root = output_dir / "ecg"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(split_csv)
    required = {"split", "subject_id", "study_id", "record_path", "label_id", "potassium"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Split CSV missing required columns: {sorted(missing)}")

    official_train = frame[frame["split"] == "train"].copy()
    subjects = official_train["subject_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(subjects)
    n_val_subjects = max(1, int(round(len(subjects) * args.val_subject_fraction)))
    val_subjects = set(subjects[:n_val_subjects].tolist())

    val_pool = official_train[official_train["subject_id"].isin(val_subjects)].copy()
    train_pool = official_train[~official_train["subject_id"].isin(val_subjects)].copy()

    pilot_train = _sample_rows(train_pool, DEFAULT_TRAIN_QUOTAS, args.seed)
    pilot_val = _sample_rows(val_pool, DEFAULT_VAL_QUOTAS, args.seed + 1000)
    pilot_train["split"] = "train"
    pilot_val["split"] = "validation"
    pilot = pd.concat([pilot_train, pilot_val], ignore_index=True)

    train_subjects = set(pilot_train["subject_id"])
    val_subjects_used = set(pilot_val["subject_id"])
    leakage = train_subjects & val_subjects_used
    if leakage:
        raise RuntimeError(f"Patient leakage detected: {len(leakage)} subjects")

    pilot_csv = output_dir / "pilot_split.csv"
    pilot.to_csv(pilot_csv, index=False)

    materialized = _materialize_waveforms(pilot, ecg_root, waveform_root)
    summary = {
        "source_split_csv": str(split_csv),
        "source_ecg_root": str(ecg_root),
        "seed": args.seed,
        "official_source_split": "train only",
        "records": int(len(pilot)),
        "subjects": int(pilot["subject_id"].nunique()),
        "train_records": int(len(pilot_train)),
        "validation_records": int(len(pilot_val)),
        "train_subjects": int(pilot_train["subject_id"].nunique()),
        "validation_subjects": int(pilot_val["subject_id"].nunique()),
        "patient_leakage": 0,
        "train_class_counts": {
            str(k): int(v) for k, v in pilot_train["label_id"].value_counts().sort_index().items()
        },
        "validation_class_counts": {
            str(k): int(v) for k, v in pilot_val["label_id"].value_counts().sort_index().items()
        },
        **materialized,
    }
    summary["copied_gib"] = materialized["copied_bytes"] / (1024 ** 3)
    with (output_dir / "pilot_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Pilot CSV: {pilot_csv}")
    print(f"Waveforms: {waveform_root}")


if __name__ == "__main__":
    main()
