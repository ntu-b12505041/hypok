from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import write_json


def _patient_stratum(labels: pd.Series) -> str:
    """Prioritize rare dyskalemia exposure when stratifying patients."""
    values = set(int(x) for x in labels)
    if 2 in values:
        return "has_hyperk"
    if 0 in values:
        return "has_hypok"
    return "nk_only"


def _split_ids(
    ids: np.ndarray,
    strata: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    counts = pd.Series(strata).value_counts()
    stratify = strata if len(counts) > 1 and counts.min() >= 2 else None
    return train_test_split(
        ids,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )


def make_patient_level_splits(
    cohort: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 20260723,
    group_column: str = "subject_id",
    label_column: str = "label_id",
) -> tuple[pd.DataFrame, dict]:
    """Create mutually exclusive patient splits with rare-class stratification."""
    if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("Split ratios must sum to one")
    group_profile = (
        cohort.groupby(group_column, sort=True)[label_column]
        .apply(_patient_stratum)
        .rename("stratum")
        .reset_index()
    )
    ids = group_profile[group_column].to_numpy()
    strata = group_profile["stratum"].to_numpy()
    train_ids, temp_ids = _split_ids(
        ids, strata, test_size=validation_ratio + test_ratio, seed=seed
    )

    temp = group_profile[group_profile[group_column].isin(temp_ids)]
    test_fraction_of_temp = test_ratio / (validation_ratio + test_ratio)
    val_ids, test_ids = _split_ids(
        temp[group_column].to_numpy(),
        temp["stratum"].to_numpy(),
        test_size=test_fraction_of_temp,
        seed=seed + 1,
    )
    mapping = {
        **{int(x): "train" for x in train_ids},
        **{int(x): "validation" for x in val_ids},
        **{int(x): "test" for x in test_ids},
    }
    result = cohort.copy()
    result["split"] = result[group_column].map(mapping)
    if result["split"].isna().any():
        raise AssertionError("Every patient must be assigned to exactly one split")

    sets = {
        name: set(result.loc[result["split"] == name, group_column].unique())
        for name in ("train", "validation", "test")
    }
    if (sets["train"] & sets["validation"]) or (sets["train"] & sets["test"]):
        raise AssertionError("Patient leakage detected")
    if sets["validation"] & sets["test"]:
        raise AssertionError("Patient leakage detected")

    expected_classes = set(int(value) for value in result[label_column].unique())
    if expected_classes != {0, 1, 2}:
        raise ValueError(
            "The cohort must contain all three labels 0/1/2; "
            f"found {sorted(expected_classes)}"
        )
    for split_name in ("train", "validation", "test"):
        observed = set(
            int(value)
            for value in result.loc[result["split"] == split_name, label_column].unique()
        )
        if observed != expected_classes:
            raise ValueError(
                f"{split_name} is missing classes {sorted(expected_classes - observed)}. "
                "Increase cohort size or revise the prespecified grouped split."
            )

    summary: dict = {"seed": int(seed), "group_column": group_column, "splits": {}}
    for split_name in ("train", "validation", "test"):
        part = result[result["split"] == split_name]
        counts = part[label_column].value_counts().sort_index()
        summary["splits"][split_name] = {
            "records": int(len(part)),
            "subjects": int(part[group_column].nunique()),
            "class_counts": {str(int(k)): int(v) for k, v in counts.items()},
        }
    return result, summary


def write_splits(config: dict) -> tuple[pd.DataFrame, dict]:
    cohort_path = Path(config["data"]["cohort_csv"]).expanduser().resolve()
    output_path = Path(config["data"]["split_csv"]).expanduser().resolve()
    cohort = pd.read_csv(cohort_path)
    section = config["split"]
    result, summary = make_patient_level_splits(
        cohort,
        train_ratio=float(section["train_ratio"]),
        validation_ratio=float(section["validation_ratio"]),
        test_ratio=float(section["test_ratio"]),
        seed=int(config["project"]["seed"]),
        group_column=section["group_column"],
        label_column=section["label_column"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    write_json(output_path.with_suffix(".summary.json"), summary)
    return result, summary
