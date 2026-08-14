from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import apply_calibration
from .config import ensure_output_dirs
from .dataset import load_split_datasets
from .metrics import (
    bootstrap_confidence_intervals,
    classification_metrics,
    target_is_met,
)
from .model import build_model
from .reporting import create_validation_report
from .training import _make_loaders, choose_device, collect_predictions
from .utils import read_json, write_json


def evaluate_model(config: dict) -> dict:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to evaluate the model") from exc

    output_dir = ensure_output_dirs(config)
    checkpoint_path = output_dir / "checkpoints" / "best.pt"
    calibration_path = output_dir / "metrics" / "calibration.json"
    if not checkpoint_path.exists() or not calibration_path.exists():
        raise FileNotFoundError("Train the model before running final evaluation")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = choose_device(config["training"]["device"])
    model.to(device)
    datasets = load_split_datasets(config)
    loaders = _make_loaders(config, datasets)
    predictions = collect_predictions(model, loaders["test"], device)
    calibration = read_json(calibration_path)
    logits = predictions[[f"logit_{idx}" for idx in range(3)]].to_numpy()
    ordinal = predictions[[f"ordinal_logit_{idx}" for idx in range(2)]].to_numpy()
    y_pred, probabilities = apply_calibration(
        logits,
        ordinal,
        predictions["predicted_potassium"].to_numpy(),
        calibration,
        binary_logits=(
            predictions[["hypok_binary_logit", "hyperk_binary_logit"]].to_numpy()
            if "hypok_binary_logit" in predictions
            else None
        ),
    )
    predictions["prediction"] = y_pred
    for idx in range(3):
        predictions[f"prob_{idx}"] = probabilities[:, idx]
    y_true = predictions["label_id"].to_numpy()
    metrics = classification_metrics(y_true, y_pred, probabilities)

    def metric_fn(sample: pd.DataFrame) -> dict:
        return classification_metrics(
            sample["label_id"].to_numpy(),
            sample["prediction"].to_numpy(),
            sample[[f"prob_{idx}" for idx in range(3)]].to_numpy(),
        )

    section = config["evaluation"]
    intervals = bootstrap_confidence_intervals(
        predictions,
        metric_fn,
        group_column=section["bootstrap_unit"],
        iterations=int(section["bootstrap_iterations"]),
        confidence_level=float(section["bootstrap_confidence_level"]),
        seed=int(config["project"]["seed"]) + 2,
    )
    metrics["target"] = {
        "recall": float(config["calibration"]["target_recall"]),
        "specificity": float(config["calibration"]["target_specificity"]),
        "met": target_is_met(
            metrics,
            float(config["calibration"]["target_recall"]),
            float(config["calibration"]["target_specificity"]),
        ),
    }
    write_json(output_dir / "metrics" / "test_metrics.json", metrics)
    write_json(output_dir / "metrics" / "test_confidence_intervals.json", intervals)
    if section.get("save_predictions", True):
        predictions.to_csv(output_dir / "metrics" / "test_predictions.csv", index=False)
    report_path = create_validation_report(
        config,
        metrics,
        intervals,
        calibration,
        predictions,
        output_dir,
    )
    return {"metrics": metrics, "confidence_intervals": intervals, "report": str(report_path)}
