from __future__ import annotations

import numpy as np


def effective_number_weights(
    labels: np.ndarray | list[int],
    num_classes: int = 3,
    beta: float = 0.9999,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    if np.any(counts == 0):
        raise ValueError(f"Every training class must have samples; got {counts.tolist()}")
    weights = (1.0 - beta) / (1.0 - np.power(beta, counts))
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def build_multitask_loss(config: dict, class_weights: np.ndarray):
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to train the model") from exc

    training = config["training"]
    model_cfg = config["model"]
    weights = training["loss_weights"]
    class_weights_tensor = torch.as_tensor(class_weights, dtype=torch.float32)
    center = float(model_cfg["potassium_center"])
    scale = float(model_cfg["potassium_scale"])
    label_smoothing = float(training.get("label_smoothing", 0.0))

    def loss_fn(outputs: dict, batch: dict) -> tuple:
        device = outputs["logits"].device
        labels = batch["label"].to(device=device, dtype=torch.long)
        ordinal = batch["ordinal"].to(device=device, dtype=torch.float32)
        potassium = batch["potassium"].to(device=device, dtype=torch.float32)
        ce = F.cross_entropy(
            outputs["logits"],
            labels,
            weight=class_weights_tensor.to(device),
            label_smoothing=label_smoothing,
        )
        ordinal_loss = F.binary_cross_entropy_with_logits(
            outputs["ordinal_logits"], ordinal
        )
        target_z = (potassium - center) / scale
        regression = F.smooth_l1_loss(outputs["potassium_z"], target_z)
        total = (
            float(weights["classification"]) * ce
            + float(weights["ordinal"]) * ordinal_loss
            + float(weights["regression"]) * regression
        )
        components = {
            "classification_loss": ce.detach(),
            "ordinal_loss": ordinal_loss.detach(),
            "regression_loss": regression.detach(),
        }
        return total, components

    return loss_fn

