#!/usr/bin/env python3
"""Exercise the report pipeline without training or using protected patient data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hypok_ecg.config import load_config  # noqa: E402
from hypok_ecg.metrics import (  # noqa: E402
    bootstrap_confidence_intervals,
    classification_metrics,
    target_is_met,
)
from hypok_ecg.reporting import create_validation_report  # noqa: E402
from hypok_ecg.utils import write_json  # noqa: E402


def _predictions(seed: int = 20260723) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    study_id = 10_000_000
    # Balanced synthetic patients keep all three classes present in bootstrap samples.
    for label in range(3):
        for patient in range(45):
            subject_id = 1_000_000 + label * 1_000 + patient
            for _ in range(int(rng.integers(1, 4))):
                predicted = label
                if rng.random() < 0.08:
                    predicted = int(rng.choice([item for item in range(3) if item != label]))
                probabilities = np.full(3, 0.04, dtype=float)
                probabilities[predicted] = 0.88
                probabilities += rng.uniform(0.0, 0.02, size=3)
                probabilities /= probabilities.sum()
                potassium = (3.1, 4.4, 6.0)[label] + rng.normal(0, 0.12)
                rows.append(
                    {
                        "subject_id": subject_id,
                        "study_id": study_id,
                        "label_id": label,
                        "prediction": predicted,
                        "potassium": potassium,
                        **{f"prob_{idx}": probabilities[idx] for idx in range(3)},
                    }
                )
                study_id += 1
    return pd.DataFrame(rows)


def _metric_fn(sample: pd.DataFrame) -> dict:
    return classification_metrics(
        sample["label_id"].to_numpy(),
        sample["prediction"].to_numpy(),
        sample[[f"prob_{idx}" for idx in range(3)]].to_numpy(),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/synthetic_demo")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    for name in ("logs", "metrics", "reports", "figures"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    config = load_config(PROJECT_ROOT / "configs" / "mimic.yaml")
    config["project"]["output_dir"] = str(output_dir)
    config["evaluation"]["bootstrap_iterations"] = 100
    config["data"]["cohort_csv"] = str(output_dir / "synthetic_pairs.csv")
    config["data"]["split_csv"] = str(output_dir / "synthetic_pairs_split.csv")

    predictions = _predictions(config["project"]["seed"])
    metrics = _metric_fn(predictions)
    intervals = bootstrap_confidence_intervals(
        predictions,
        _metric_fn,
        iterations=config["evaluation"]["bootstrap_iterations"],
        seed=config["project"]["seed"] + 2,
    )
    metrics["target"] = {
        "recall": config["calibration"]["target_recall"],
        "specificity": config["calibration"]["target_specificity"],
        "met": target_is_met(
            metrics,
            config["calibration"]["target_recall"],
            config["calibration"]["target_specificity"],
        ),
    }

    epochs = np.arange(1, 16)
    history = pd.DataFrame(
        {
            "epoch": epochs,
            "train_loss": 1.2 * np.exp(-epochs / 5) + 0.16,
            "val_loss": 1.1 * np.exp(-epochs / 5.5) + 0.22,
            "train_accuracy": np.minimum(0.96, 0.48 + 0.04 * epochs),
            "val_accuracy": np.minimum(0.92, 0.44 + 0.038 * epochs),
            "val_macro_auroc_ovr": np.minimum(0.95, 0.62 + 0.025 * epochs),
        }
    )
    history.to_csv(output_dir / "logs" / "training_history.csv", index=False)
    write_json(
        output_dir / "logs" / "training_summary.json",
        {
            "best_epoch": 15,
            "elapsed_seconds": 312.4,
            "total_parameters": "Not computed (PyTorch unavailable)",
            "trainable_parameters": "Not computed (PyTorch unavailable)",
            "synthetic": True,
        },
    )
    class_counts = {
        name: int((predictions["label_id"] == idx).sum())
        for idx, name in enumerate(("HypoK", "NK", "HyperK"))
    }
    cohort_summary = {
        "records": len(predictions),
        "subjects": predictions["subject_id"].nunique(),
        "class_counts": class_counts,
        "potassium_itemids": [50971],
        "synthetic": True,
    }
    write_json(Path(config["data"]["cohort_csv"]).with_suffix(".summary.json"), cohort_summary)
    split_counts = {}
    subject_allocation = {"train": 95, "validation": 20, "test": 20}
    class_allocations = {}
    for key, value in class_counts.items():
        train = int(value * 0.70)
        validation = int(value * 0.15)
        class_allocations[key] = {
            "train": train,
            "validation": validation,
            "test": value - train - validation,
        }
    for name in ("train", "validation", "test"):
        split_class_counts = {
            key: allocation[name] for key, allocation in class_allocations.items()
        }
        split_counts[name] = {
            "subjects": subject_allocation[name],
            "records": sum(split_class_counts.values()),
            "class_counts": split_class_counts,
        }
    write_json(
        Path(config["data"]["split_csv"]).with_suffix(".summary.json"),
        {"splits": split_counts, "synthetic": True},
    )
    calibration = {
        "selected_head": "classification",
        "temperature": 1.0,
        "low_threshold": 0.65,
        "high_threshold": 1.35,
        "target_met_on_validation": True,
        "synthetic": True,
    }
    write_json(output_dir / "metrics" / "test_metrics.json", metrics)
    write_json(output_dir / "metrics" / "test_confidence_intervals.json", intervals)
    predictions.to_csv(output_dir / "metrics" / "test_predictions.csv", index=False)
    report = create_validation_report(
        config,
        metrics,
        intervals,
        calibration,
        predictions,
        output_dir,
        synthetic=True,
    )
    print(
        json.dumps(
            {
                "report": str(report),
                "target_met_in_synthetic_example": metrics["target"]["met"],
                "warning": "Synthetic demonstration only; not a trained model result.",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
