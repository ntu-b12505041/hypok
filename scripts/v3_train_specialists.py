#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

CLASS_NAMES = ("HypoK", "NK", "HyperK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train three independent specialists on frozen ECG-CPC embeddings.")
    parser.add_argument("--embedding-dir", default="/content/drive/MyDrive/hypok_colab/v3/ecgcpc_embeddings")
    parser.add_argument("--output-dir", default="/content/drive/MyDrive/hypok_colab/v3/v3a_ecgcpc_specialists")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--threshold-grid", type=int, default=21)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--pos-weight-mode", choices=["none", "sqrt"], default="sqrt")
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict, np.ndarray]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    total = matrix.sum()
    result = {}
    for idx, name in enumerate(CLASS_NAMES):
        tp = matrix[idx, idx]
        fn = matrix[idx, :].sum() - tp
        fp = matrix[:, idx].sum() - tp
        tn = total - tp - fn - fp
        sens = tp / (tp + fn) if tp + fn else np.nan
        spec = tn / (tn + fp) if tn + fp else np.nan
        result[name] = {
            "sensitivity": float(sens),
            "specificity": float(spec),
            "support": int(tp + fn),
        }
    return result, matrix


def rank_prediction(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    per_class, _ = per_class_metrics(y_true, y_pred)
    six = []
    for name in CLASS_NAMES:
        six.extend([per_class[name]["sensitivity"], per_class[name]["specificity"]])
    minimum_six = float(np.nanmin(six))
    target_met = bool(np.all(np.asarray(six) > 0.85))
    return (
        int(target_met),
        minimum_six,
        float(balanced_accuracy_score(y_true, y_pred)),
        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    )


def safe_logit(p: np.ndarray | float) -> np.ndarray:
    value = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(value / (1 - value))


def joint_threshold_calibration(y_true: np.ndarray, probabilities: np.ndarray, grid_size: int) -> dict:
    # Class-specific thresholds are equivalent to additive biases in logit space.
    # We search them jointly on validation only and rank by the prespecified six metrics.
    quantiles = np.linspace(0.05, 0.95, max(9, grid_size))
    candidate_lists = []
    for idx in range(3):
        values = np.unique(np.concatenate([np.quantile(probabilities[:, idx], quantiles), [0.5]]))
        candidate_lists.append(values)

    raw_logits = safe_logit(probabilities)
    best = None
    for t0 in candidate_lists[0]:
        b0 = safe_logit(t0)
        for t1 in candidate_lists[1]:
            b1 = safe_logit(t1)
            left01 = raw_logits[:, :2] - np.asarray([b0, b1])
            for t2 in candidate_lists[2]:
                adjusted = np.column_stack([left01, raw_logits[:, 2] - safe_logit(t2)])
                pred = adjusted.argmax(axis=1)
                rank = rank_prediction(y_true, pred)
                if best is None or rank > best["rank"]:
                    best = {
                        "thresholds": [float(t0), float(t1), float(t2)],
                        "rank": rank,
                        "prediction": pred.copy(),
                    }
    if best is None:
        raise RuntimeError("Threshold search failed")
    prediction = best.pop("prediction")
    per_class, matrix = per_class_metrics(y_true, prediction)
    best["rank"] = list(best["rank"])
    best["minimum_six"] = float(best["rank"][1])
    best["target_met"] = bool(best["rank"][0])
    best["balanced_accuracy"] = float(best["rank"][2])
    best["macro_f1"] = float(best["rank"][3])
    best["per_class"] = per_class
    best["confusion_matrix"] = matrix.tolist()
    return best


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    embedding_dir = Path(args.embedding_dir)
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    checkpoints_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    logs_dir = output_dir / "logs"
    for path in (figures_dir, checkpoints_dir, metrics_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    x = np.load(embedding_dir / "embeddings.npy").astype(np.float32)
    meta = pd.read_csv(embedding_dir / "metadata.csv")
    if len(x) != len(meta):
        raise ValueError("Embedding rows and metadata rows differ")
    if set(meta["split"].unique()) - {"train", "validation"}:
        raise ValueError("V3-A must use train/validation only")

    train_mask = meta["split"].eq("train").to_numpy()
    val_mask = meta["split"].eq("validation").to_numpy()
    x_train_raw, x_val_raw = x[train_mask], x[val_mask]
    y_train = meta.loc[train_mask, "label_id"].to_numpy(dtype=np.int64)
    y_val = meta.loc[val_mask, "label_id"].to_numpy(dtype=np.int64)

    scaler = StandardScaler().fit(x_train_raw)
    x_train = scaler.transform(x_train_raw).astype(np.float32)
    x_val = scaler.transform(x_val_raw).astype(np.float32)
    np.save(metrics_dir / "scaler_mean.npy", scaler.mean_.astype(np.float32))
    np.save(metrics_dir / "scaler_scale.npy", scaler.scale_.astype(np.float32))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Use a Colab GPU runtime for V3-A")

    class SpecialistMLP(nn.Module):
        def __init__(self, input_dim: int, dropout: float):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        def forward(self, features):
            return self.net(features).squeeze(-1)

    x_train_tensor = torch.from_numpy(x_train)
    x_val_tensor = torch.from_numpy(x_val)
    histories = []
    train_logits_all = np.zeros((len(x_train), 3), dtype=np.float32)
    val_logits_all = np.zeros((len(x_val), 3), dtype=np.float32)
    specialist_summary = {}

    for class_id, class_name in enumerate(CLASS_NAMES):
        seed_all(args.seed + class_id)
        target_train = (y_train == class_id).astype(np.float32)
        target_val = (y_val == class_id).astype(np.float32)
        positives = float(target_train.sum())
        negatives = float(len(target_train) - positives)
        if args.pos_weight_mode == "sqrt":
            pos_weight_value = math.sqrt(negatives / max(1.0, positives))
        else:
            pos_weight_value = 1.0

        train_ds = TensorDataset(x_train_tensor, torch.from_numpy(target_train))
        val_ds = TensorDataset(x_val_tensor, torch.from_numpy(target_val))
        generator = torch.Generator().manual_seed(args.seed + class_id)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        model = SpecialistMLP(x_train.shape[1], args.dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))
        # Also track unweighted BCE on train and validation so loss curves remain
        # directly comparable even when training uses a moderate positive weight.
        report_criterion = nn.BCEWithLogitsLoss()

        best_auc = -np.inf
        best_state = None
        best_epoch = -1
        stale = 0
        class_history = []

        for epoch in range(1, args.epochs + 1):
            model.train()
            weighted_sum = 0.0
            report_sum = 0.0
            seen = 0
            for features, targets in train_loader:
                features = features.to(device)
                targets = targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(features)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                batch = len(targets)
                weighted_sum += float(loss.detach().cpu()) * batch
                report_sum += float(report_criterion(logits.detach(), targets).cpu()) * batch
                seen += batch
            scheduler.step()

            model.eval()
            val_logits_batches = []
            val_targets_batches = []
            val_report_sum = 0.0
            val_seen = 0
            with torch.inference_mode():
                for features, targets in val_loader:
                    features = features.to(device)
                    targets = targets.to(device)
                    logits = model(features)
                    batch = len(targets)
                    val_report_sum += float(report_criterion(logits, targets).cpu()) * batch
                    val_seen += batch
                    val_logits_batches.append(logits.cpu().numpy())
                    val_targets_batches.append(targets.cpu().numpy())
            val_logits = np.concatenate(val_logits_batches)
            val_targets = np.concatenate(val_targets_batches)
            val_auc = float(roc_auc_score(val_targets, val_logits))
            row = {
                "specialist": class_name,
                "epoch": epoch,
                "train_objective_loss": weighted_sum / max(1, seen),
                "train_bce_loss": report_sum / max(1, seen),
                "validation_bce_loss": val_report_sum / max(1, val_seen),
                "validation_auroc": val_auc,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            class_history.append(row)
            histories.append(row)

            if val_auc > best_auc + 1e-5:
                best_auc = val_auc
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
            if stale >= args.patience:
                break

        if best_state is None:
            raise RuntimeError(f"No checkpoint selected for {class_name}")
        model.load_state_dict(best_state)
        torch.save(
            {
                "model_state_dict": best_state,
                "input_dim": int(x_train.shape[1]),
                "class_id": class_id,
                "class_name": class_name,
                "best_epoch": best_epoch,
                "best_validation_auroc": best_auc,
                "pos_weight": pos_weight_value,
            },
            checkpoints_dir / f"{class_name.lower()}_specialist.pt",
        )

        model.eval()
        with torch.inference_mode():
            train_logits = []
            val_logits = []
            for start in range(0, len(x_train_tensor), args.batch_size):
                train_logits.append(model(x_train_tensor[start:start+args.batch_size].to(device)).cpu().numpy())
            for start in range(0, len(x_val_tensor), args.batch_size):
                val_logits.append(model(x_val_tensor[start:start+args.batch_size].to(device)).cpu().numpy())
        train_logits_all[:, class_id] = np.concatenate(train_logits)
        val_logits_all[:, class_id] = np.concatenate(val_logits)
        val_prob = 1.0 / (1.0 + np.exp(-val_logits_all[:, class_id]))
        try:
            pauc = float(roc_auc_score(target_val, val_prob, max_fpr=0.15))
        except ValueError:
            pauc = float("nan")
        specialist_summary[class_name] = {
            "best_epoch": int(best_epoch),
            "validation_auroc": float(best_auc),
            "validation_standardized_pauc_fpr_0_15": pauc,
            "train_positives": int(positives),
            "train_negatives": int(negatives),
            "pos_weight": float(pos_weight_value),
        }

    history = pd.DataFrame(histories)
    history.to_csv(logs_dir / "training_history.csv", index=False)

    val_probabilities = 1.0 / (1.0 + np.exp(-val_logits_all))
    train_probabilities = 1.0 / (1.0 + np.exp(-train_logits_all))
    calibration = joint_threshold_calibration(y_val, val_probabilities, args.threshold_grid)
    thresholds = np.asarray(calibration["thresholds"], dtype=float)
    val_prediction = (safe_logit(val_probabilities) - safe_logit(thresholds)).argmax(axis=1)
    train_prediction = (safe_logit(train_probabilities) - safe_logit(thresholds)).argmax(axis=1)

    val_per_class, val_matrix = per_class_metrics(y_val, val_prediction)
    train_per_class, train_matrix = per_class_metrics(y_train, train_prediction)
    val_aurocs = {
        CLASS_NAMES[idx]: float(roc_auc_score((y_val == idx).astype(int), val_probabilities[:, idx]))
        for idx in range(3)
    }

    val_rows = meta.loc[val_mask, ["subject_id", "study_id", "potassium", "label_id"]].reset_index(drop=True)
    for idx, name in enumerate(CLASS_NAMES):
        val_rows[f"prob_{name}"] = val_probabilities[:, idx]
    val_rows["prediction"] = val_prediction
    val_rows.to_csv(metrics_dir / "validation_predictions.csv", index=False)

    result = {
        "model": "frozen_ecgcpc_meanmax_3_specialist_mlp",
        "specialists": specialist_summary,
        "thresholds": {name: float(thresholds[idx]) for idx, name in enumerate(CLASS_NAMES)},
        "validation": {
            "minimum_six": float(calibration["minimum_six"]),
            "target_met": bool(calibration["target_met"]),
            "balanced_accuracy": float(calibration["balanced_accuracy"]),
            "macro_f1": float(calibration["macro_f1"]),
            "per_class": val_per_class,
            "auroc_ovr": val_aurocs,
            "confusion_matrix": val_matrix.tolist(),
        },
        "train_with_validation_thresholds": {
            "per_class": train_per_class,
            "confusion_matrix": train_matrix.tolist(),
        },
        "data": {
            "train_records": int(train_mask.sum()),
            "validation_records": int(val_mask.sum()),
            "feature_dim": int(x_train.shape[1]),
            "test_used": False,
        },
        "training": {
            "epochs_max": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "pos_weight_mode": args.pos_weight_mode,
            "seed": args.seed,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    (metrics_dir / "validation_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (metrics_dir / "thresholds.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    # Required training/validation loss visualization: use unweighted BCE for
    # both curves so train and validation are on the same scale.
    mean_history = history.groupby("epoch", as_index=False)[["train_bce_loss", "validation_bce_loss"]].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mean_history["epoch"], mean_history["train_bce_loss"], label="Train BCE")
    ax.plot(mean_history["epoch"], mean_history["validation_bce_loss"], label="Validation BCE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("V3-A ECG-CPC specialists: training vs validation loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "loss_curve.png", dpi=180)
    plt.close(fig)

    for class_name in CLASS_NAMES:
        sub = history[history["specialist"] == class_name]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sub["epoch"], sub["train_bce_loss"], label="Train BCE")
        ax.plot(sub["epoch"], sub["validation_bce_loss"], label="Validation BCE")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE loss")
        ax.set_title(f"{class_name} specialist loss")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / f"loss_curve_{class_name.lower()}.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for idx, name in enumerate(CLASS_NAMES):
        target = (y_val == idx).astype(int)
        fpr, tpr, _ = roc_curve(target, val_probabilities[:, idx])
        ax.plot(fpr, tpr, label=f"{name} AUROC={val_aurocs[name]:.3f}")
    ax.plot([0, 1], [0, 1], "--", linewidth=1)
    ax.axvline(0.15, linestyle=":", linewidth=1, label="FPR=0.15 target boundary")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("Sensitivity")
    ax.set_title("V3-A validation ROC")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "roc_curves.png", dpi=180)
    plt.close(fig)

    metric_rows = []
    for name in CLASS_NAMES:
        metric_rows.append({"class": name, "metric": "Sensitivity", "value": val_per_class[name]["sensitivity"]})
        metric_rows.append({"class": name, "metric": "Specificity", "value": val_per_class[name]["specificity"]})
    metric_frame = pd.DataFrame(metric_rows)
    x_positions = np.arange(3)
    width = 0.35
    sens = [val_per_class[name]["sensitivity"] for name in CLASS_NAMES]
    spec = [val_per_class[name]["specificity"] for name in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x_positions - width/2, sens, width, label="Sensitivity")
    ax.bar(x_positions + width/2, spec, width, label="Specificity")
    ax.axhline(0.85, linestyle="--", linewidth=1, label="Target 0.85")
    ax.set_xticks(x_positions, CLASS_NAMES)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"Validation sensitivity/specificity — minimum six={calibration['minimum_six']:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "sensitivity_specificity.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(val_matrix, interpolation="nearest")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(np.arange(3), CLASS_NAMES)
    ax.set_yticks(np.arange(3), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("V3-A validation confusion matrix")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(val_matrix[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(figures_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    print(json.dumps(result, indent=2))
    print("Required loss plot:", figures_dir / "loss_curve.png")
    print("Validation metrics:", metrics_dir / "validation_metrics.json")


if __name__ == "__main__":
    main()
