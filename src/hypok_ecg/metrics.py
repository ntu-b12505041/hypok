from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def classification_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    probabilities: np.ndarray | None = None,
    class_names: tuple[str, ...] = ("HypoK", "NK", "HyperK"),
) -> dict:
    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    labels = np.arange(len(class_names))
    matrix = confusion_matrix(true, pred, labels=labels)
    total = matrix.sum()
    per_class = {}
    for idx, name in enumerate(class_names):
        tp = matrix[idx, idx]
        fn = matrix[idx, :].sum() - tp
        fp = matrix[:, idx].sum() - tp
        tn = total - tp - fn - fp
        recall = tp / (tp + fn) if tp + fn else np.nan
        specificity = tn / (tn + fp) if tn + fp else np.nan
        precision = tp / (tp + fp) if tp + fp else np.nan
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall and np.isfinite(precision + recall)
            else np.nan
        )
        per_class[name] = {
            "recall": float(recall),
            "sensitivity": float(recall),
            "specificity": float(specificity),
            "precision": float(precision),
            "f1": float(f1),
            "support": int(tp + fn),
        }

    result = {
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, pred, average="weighted", zero_division=0)),
        "macro_precision": float(
            precision_score(true, pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(recall_score(true, pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(true, pred)),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(true, pred, weights="quadratic")
        ),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }
    if probabilities is not None:
        probs = np.asarray(probabilities, dtype=float)
        targets = label_binarize(true, classes=labels)
        try:
            result["macro_auroc_ovr"] = float(
                roc_auc_score(true, probs, labels=labels, multi_class="ovr", average="macro")
            )
        except ValueError:
            result["macro_auroc_ovr"] = float("nan")
        try:
            result["macro_auprc"] = float(
                average_precision_score(targets, probs, average="macro")
            )
        except ValueError:
            result["macro_auprc"] = float("nan")
    return result


def target_is_met(metrics: dict, recall: float = 0.85, specificity: float = 0.85) -> bool:
    return all(
        values["recall"] > recall and values["specificity"] > specificity
        for values in metrics["per_class"].values()
    )


def bootstrap_confidence_intervals(
    frame,
    metric_fn: Callable,
    group_column: str = "subject_id",
    iterations: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 20260723,
) -> dict:
    """Patient-cluster bootstrap for scalar overall and per-class metrics."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(frame[group_column].unique())
    samples: list[dict] = []
    grouped_indices = {
        group: frame.index[frame[group_column] == group].to_numpy() for group in groups
    }
    for _ in range(iterations):
        selected = rng.choice(groups, size=len(groups), replace=True)
        index = np.concatenate([grouped_indices[group] for group in selected])
        sample = frame.loc[index]
        samples.append(metric_fn(sample))

    alpha = 1.0 - confidence_level
    lower_q, upper_q = alpha / 2.0, 1.0 - alpha / 2.0
    keys = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_auroc_ovr",
        "macro_auprc",
    )
    intervals = {}
    for key in keys:
        values = np.asarray([item.get(key, np.nan) for item in samples], dtype=float)
        intervals[key] = {
            "lower": float(np.nanquantile(values, lower_q)),
            "upper": float(np.nanquantile(values, upper_q)),
        }
    class_names = next(iter(samples))["per_class"].keys()
    intervals["per_class"] = {}
    for name in class_names:
        intervals["per_class"][name] = {}
        for key in ("recall", "specificity", "precision", "f1"):
            values = np.asarray(
                [item["per_class"][name][key] for item in samples], dtype=float
            )
            intervals["per_class"][name][key] = {
                "lower": float(np.nanquantile(values, lower_q)),
                "upper": float(np.nanquantile(values, upper_q)),
            }
    return intervals
