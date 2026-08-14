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
    binary_pos_weight = torch.as_tensor(
        training.get("binary_pos_weight", [1.0, 1.0]), dtype=torch.float32
    )
    if tuple(binary_pos_weight.shape) != (2,):
        raise ValueError("training.binary_pos_weight must contain [HypoK, HyperK]")

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
            float(weights.get("classification", 1.0)) * ce
            + float(weights.get("ordinal", 0.0)) * ordinal_loss
            + float(weights.get("regression", 0.0)) * regression
        )
        components = {
            "classification_loss": ce.detach(),
            "ordinal_loss": ordinal_loss.detach(),
            "regression_loss": regression.detach(),
        }
        if "binary_logits" in outputs:
            binary_targets = torch.stack(
                ((labels == 0).float(), (labels == 2).float()), dim=1
            )
            binary_loss = F.binary_cross_entropy_with_logits(
                outputs["binary_logits"],
                binary_targets,
                pos_weight=binary_pos_weight.to(device),
            )
            total = total + float(weights.get("dual_binary", 1.0)) * binary_loss
            components["dual_binary_loss"] = binary_loss.detach()
        if "per_lead_binary_logits" in outputs:
            lead_logits = outputs["per_lead_binary_logits"]
            lead_targets = binary_targets.unsqueeze(1).expand_as(lead_logits)
            per_lead_loss = F.binary_cross_entropy_with_logits(
                lead_logits,
                lead_targets,
                pos_weight=binary_pos_weight.to(device),
            )
            total = total + float(weights.get("per_lead", 0.0)) * per_lead_loss
            components["per_lead_loss"] = per_lead_loss.detach()
        return total, components

    return loss_fn
