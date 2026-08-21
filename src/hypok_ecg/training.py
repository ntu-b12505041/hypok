from __future__ import annotations

import math
import platform
import time

import numpy as np
import pandas as pd

from .calibration import (
    calibrate_predictions,
    dual_binary_probabilities,
    tune_dual_binary_thresholds,
)
from .config import ensure_output_dirs
from .dataset import load_split_datasets
from .losses import build_multitask_loss, effective_number_weights
from .metrics import classification_metrics
from .model import build_model
from .sampling import build_training_sampler
from .utils import seed_everything, write_json


def _torch():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install the project dependencies before training."
        ) from exc
    return torch, DataLoader


def choose_device(requested: str):
    torch, _ = _torch()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def _make_loaders(config: dict, datasets: dict):
    torch, DataLoader = _torch()
    section = config["training"]
    workers = int(section["num_workers"])
    common = {
        "batch_size": int(section["batch_size"]),
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
    }
    generator = torch.Generator().manual_seed(int(config["project"]["seed"]))
    train_sampler = build_training_sampler(config, datasets["train"])
    return {
        "train": DataLoader(
            datasets["train"],
            shuffle=train_sampler is None,
            sampler=train_sampler,
            generator=generator,
            drop_last=False,
            persistent_workers=False,
            **common,
        ),
        "validation": DataLoader(
            datasets["validation"],
            shuffle=False,
            drop_last=False,
            persistent_workers=workers > 0,
            **common,
        ),
        "test": DataLoader(
            datasets["test"],
            shuffle=False,
            drop_last=False,
            persistent_workers=workers > 0,
            **common,
        ),
    }


def _optimizer_and_scheduler(config: dict, model, steps_per_epoch: int):
    torch, _ = _torch()
    section = config["training"]
    lr = float(section["learning_rate"])
    weight_decay = float(section["weight_decay"])
    if section["optimizer"].lower() == "adamw":
        model_cfg = config["model"]
        if hasattr(model, "backbone") and model_cfg.get("backbone_learning_rate"):
            backbone_parameters = list(model.backbone.parameters())
            backbone_ids = {id(parameter) for parameter in backbone_parameters}
            head_parameters = [
                parameter
                for parameter in model.parameters()
                if id(parameter) not in backbone_ids
            ]
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": backbone_parameters,
                        "lr": float(model_cfg["backbone_learning_rate"]),
                        "name": "backbone",
                    },
                    {
                        "params": head_parameters,
                        "lr": float(model_cfg.get("head_learning_rate", lr)),
                        "name": "heads",
                    },
                ],
                weight_decay=weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
    else:
        raise ValueError(f"Unsupported optimizer: {section['optimizer']}")
    epochs = int(section["epochs"])
    warmup_steps = int(section.get("warmup_epochs", 0)) * steps_per_epoch
    total_steps = max(1, epochs * steps_per_epoch)

    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-6, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    return optimizer, scheduler


def _run_epoch(
    model,
    loader,
    loss_fn,
    device,
    optimizer=None,
    scheduler=None,
    scaler=None,
    gradient_clip_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    collect_calibration_inputs: bool = False,
) -> dict:
    torch, _ = _torch()
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    labels = []
    softmax_probabilities = []
    binary_logits_batches = []
    components = {}
    use_amp = scaler is not None and scaler.is_enabled()
    accumulation_steps = max(1, int(gradient_accumulation_steps))
    if training:
        optimizer.zero_grad(set_to_none=True)

    for batch_index, batch in enumerate(loader):
        ecg = batch["ecg"].to(device=device, dtype=torch.float32, non_blocking=True)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
            if use_amp
            else torch.autocast(device_type=device.type, enabled=False)
        )
        with context:
            outputs = model(ecg)
            loss, loss_components = loss_fn(outputs, batch)
        if training:
            backward_loss = loss / accumulation_steps
            should_step = (
                (batch_index + 1) % accumulation_steps == 0
                or batch_index + 1 == len(loader)
            )
            if use_amp:
                scaler.scale(backward_loss).backward()
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
            else:
                backward_loss.backward()
                if should_step:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip_norm
                    )
                    optimizer.step()
            if should_step:
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        batch_size = ecg.shape[0]
        total_examples += batch_size
        total_loss += float(loss.detach().cpu()) * batch_size
        for key, value in loss_components.items():
            components[key] = components.get(key, 0.0) + float(value.cpu()) * batch_size
        softmax_probabilities.append(
            torch.softmax(outputs["logits"].detach(), dim=1).cpu().numpy()
        )
        if "binary_logits" in outputs:
            binary_logits_batches.append(outputs["binary_logits"].detach().cpu().numpy())
        labels.append(batch["label"].cpu().numpy())

    y_true = np.concatenate(labels)
    binary_logits = (
        np.concatenate(binary_logits_batches) if binary_logits_batches else None
    )
    probabilities = (
        dual_binary_probabilities(binary_logits)
        if binary_logits is not None
        else np.concatenate(softmax_probabilities)
    )
    y_pred = probabilities.argmax(axis=1)
    metrics = classification_metrics(y_true, y_pred, probabilities)
    result = {
        "loss": total_loss / max(1, total_examples),
        **{key: value / max(1, total_examples) for key, value in components.items()},
        **{key: value for key, value in metrics.items() if key != "per_class"},
    }
    if collect_calibration_inputs:
        result["_calibration_inputs"] = {
            "y_true": y_true,
            "binary_logits": binary_logits,
        }
    return result


def collect_predictions(model, loader, device) -> pd.DataFrame:
    torch, _ = _torch()
    model.eval()
    rows = []
    with torch.inference_mode():
        for batch in loader:
            outputs = model(batch["ecg"].to(device=device, dtype=torch.float32))
            logits = outputs["logits"].cpu().numpy()
            ordinal = outputs["ordinal_logits"].cpu().numpy()
            binary = (
                outputs["binary_logits"].cpu().numpy()
                if "binary_logits" in outputs
                else None
            )
            potassium_pred = outputs["potassium"].cpu().numpy()
            batch_size = len(logits)
            for idx in range(batch_size):
                row = {
                    "subject_id": int(batch["subject_id"][idx]),
                    "study_id": int(batch["study_id"][idx]),
                    "label_id": int(batch["label"][idx]),
                    "potassium": float(batch["potassium"][idx]),
                    "predicted_potassium": float(potassium_pred[idx]),
                    **{f"logit_{j}": float(logits[idx, j]) for j in range(3)},
                    **{
                        f"ordinal_logit_{j}": float(ordinal[idx, j])
                        for j in range(2)
                    },
                }
                if binary is not None:
                    row.update(
                        {
                            "hypok_binary_logit": float(binary[idx, 0]),
                            "hyperk_binary_logit": float(binary[idx, 1]),
                        }
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def train_model(config: dict) -> dict:
    torch, _ = _torch()
    import scipy
    import sklearn
    import wfdb

    seed = int(config["project"]["seed"])
    seed_everything(seed)
    output_dir = ensure_output_dirs(config)
    datasets = load_split_datasets(config)
    loaders = _make_loaders(config, datasets)
    train_labels = datasets["train"].frame["label_id"].to_numpy()
    class_weights = effective_number_weights(
        train_labels,
        num_classes=int(config["model"]["num_classes"]),
        beta=float(config["training"]["effective_number_beta"]),
    )
    model = build_model(config)
    freeze_backbone_epochs = int(config["model"].get("freeze_backbone_epochs", 0))
    if freeze_backbone_epochs > 0:
        if not hasattr(model, "freeze_backbone"):
            raise ValueError(
                "freeze_backbone_epochs requires a model with freeze_backbone()"
            )
        model.freeze_backbone()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    initially_trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    device = choose_device(config["training"]["device"])
    model.to(device)
    loss_fn = build_multitask_loss(config, class_weights)
    accumulation_steps = int(
        config["training"].get("gradient_accumulation_steps", 1)
    )
    if accumulation_steps < 1:
        raise ValueError("training.gradient_accumulation_steps must be >= 1")
    optimizer_steps_per_epoch = math.ceil(
        len(loaders["train"]) / accumulation_steps
    )
    optimizer, scheduler = _optimizer_and_scheduler(
        config, model, optimizer_steps_per_epoch
    )
    use_amp = bool(config["training"]["mixed_precision"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_score = -np.inf
    best_rank = None
    best_epoch = -1
    epochs_without_improvement = 0
    patience = int(config["training"]["early_stopping_patience"])
    history = []
    sampling_audit = []
    checkpoint_path = output_dir / "checkpoints" / "best.pt"
    started = time.perf_counter()
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        if hasattr(datasets["train"], "set_epoch"):
            datasets["train"].set_epoch(epoch - 1)
        if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs + 1:
            model.unfreeze_backbone()
        train_stats = _run_epoch(
            model,
            loaders["train"],
            loss_fn,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
            gradient_accumulation_steps=accumulation_steps,
        )
        with torch.inference_mode():
            val_stats = _run_epoch(
                model,
                loaders["validation"],
                loss_fn,
                device,
                collect_calibration_inputs=True,
            )
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
        row.update(
            {f"train_{key}": value for key, value in train_stats.items() if np.isscalar(value)}
        )
        row.update(
            {f"val_{key}": value for key, value in val_stats.items() if np.isscalar(value)}
        )

        calibration_inputs = val_stats.get("_calibration_inputs", {})
        binary_logits = calibration_inputs.get("binary_logits")
        checkpoint_tuning = None
        if binary_logits is not None:
            checkpoint_tuning = tune_dual_binary_thresholds(
                calibration_inputs["y_true"],
                binary_logits,
                grid_size=int(config["training"].get("checkpoint_threshold_grid_size", 41)),
                target_recall=float(config["calibration"]["target_recall"]),
                target_specificity=float(config["calibration"]["target_specificity"]),
            )
            rank = tuple(checkpoint_tuning["rank"]) + (
                float(val_stats.get("macro_auroc_ovr", -np.inf)),
            )
            score = float(checkpoint_tuning["minimum_recall_specificity"])
            row.update(
                {
                    "val_minimum_six": score,
                    "val_target_met": int(checkpoint_tuning["target_met"]),
                    "val_hypo_threshold": checkpoint_tuning["hypo_threshold"],
                    "val_hyper_threshold": checkpoint_tuning["hyper_threshold"],
                    "val_conflict_rate": checkpoint_tuning["conflict_rate"],
                }
            )
        else:
            score = float(val_stats.get("macro_auroc_ovr", np.nan))
            if not np.isfinite(score):
                score = float(val_stats["balanced_accuracy"])
            rank = (
                0,
                score,
                float(val_stats["balanced_accuracy"]),
                float(val_stats["macro_f1"]),
            )

        history.append(row)
        sampler_audit = getattr(loaders["train"].sampler, "last_audit", None)
        if sampler_audit is not None:
            sampling_audit.append(dict(sampler_audit))
            pd.DataFrame(sampling_audit).to_csv(
                output_dir / "logs" / "sampling_audit.csv", index=False
            )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_score": score,
                    "validation_rank": list(rank),
                    "checkpoint_calibration": checkpoint_tuning,
                    "config": config,
                    "class_weights": class_weights.tolist(),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    elapsed = time.perf_counter() - started
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "logs" / "training_history.csv", index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_predictions = collect_predictions(model, loaders["validation"], device)
    logits = validation_predictions[[f"logit_{idx}" for idx in range(3)]].to_numpy()
    ordinal_logits = validation_predictions[
        [f"ordinal_logit_{idx}" for idx in range(2)]
    ].to_numpy()
    calibration = calibrate_predictions(
        validation_predictions["label_id"].to_numpy(),
        logits,
        ordinal_logits,
        validation_predictions["predicted_potassium"].to_numpy(),
        config,
        binary_logits=(
            validation_predictions[
                ["hypok_binary_logit", "hyperk_binary_logit"]
            ].to_numpy()
            if "hypok_binary_logit" in validation_predictions
            else None
        ),
    )
    write_json(output_dir / "metrics" / "calibration.json", calibration.to_dict())
    validation_predictions.to_csv(
        output_dir / "metrics" / "validation_predictions.csv", index=False
    )

    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "wfdb": wfdb.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "cuda_version": torch.version.cuda,
    }
    summary = {
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "best_validation_rank": list(best_rank) if best_rank is not None else None,
        "checkpoint_selection_metric": (
            "validation_minimum_six" if checkpoint.get("checkpoint_calibration") else "macro_auroc"
        ),
        "best_checkpoint_calibration": checkpoint.get("checkpoint_calibration"),
        "elapsed_seconds": elapsed,
        "elapsed_hours": elapsed / 3600.0,
        "epochs_completed": len(history),
        "checkpoint": str(checkpoint_path),
        "class_weights": class_weights.tolist(),
        "environment": environment,
        "validation_target_met": calibration.target_met_on_validation,
        "total_parameters": int(total_parameters),
        "initially_trainable_parameters": int(initially_trainable_parameters),
        "trainable_parameters": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        ),
        "freeze_backbone_epochs": freeze_backbone_epochs,
        "model_name": config["model"]["name"],
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": int(config["training"]["batch_size"])
        * accumulation_steps,
        "sampling": config.get("sampling", {"enabled": False}),
        "sampling_audit": (
            str(output_dir / "logs" / "sampling_audit.csv")
            if sampling_audit
            else None
        ),
    }
    write_json(output_dir / "logs" / "training_summary.json", summary)
    return summary
