from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, softmax

from .metrics import classification_metrics, target_is_met


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(y_true, dtype=int)

    def nll(log_temperature: float) -> float:
        temperature = np.exp(log_temperature)
        probs = softmax(logits / temperature, axis=1)
        selected = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
        return float(-np.log(selected).mean())

    result = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(result.x))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(np.asarray(logits, dtype=float) / float(temperature), axis=1)


def _predict_from_thresholds(score: np.ndarray, low: float, high: float) -> np.ndarray:
    prediction = np.ones(len(score), dtype=np.int64)
    prediction[score < low] = 0
    prediction[score >= high] = 2
    return prediction


def _minimum_recall_specificity(metrics: dict) -> float:
    values = []
    for item in metrics["per_class"].values():
        values.extend((item["recall"], item["specificity"]))
    return float(np.nanmin(values))


def _rank_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_recall: float,
    target_specificity: float,
) -> tuple[tuple, float]:
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    matrix = np.bincount(true * 3 + pred, minlength=9).reshape(3, 3)
    total = float(matrix.sum())
    recalls = []
    specificities = []
    f1s = []
    for idx in range(3):
        tp = float(matrix[idx, idx])
        fn = float(matrix[idx, :].sum() - matrix[idx, idx])
        fp = float(matrix[:, idx].sum() - matrix[idx, idx])
        tn = total - tp - fn - fp
        recall = tp / (tp + fn) if tp + fn else np.nan
        specificity = tn / (tn + fp) if tn + fp else np.nan
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        recalls.append(recall)
        specificities.append(specificity)
        f1s.append(f1)
    six = np.asarray(recalls + specificities, dtype=float)
    minimum = float(np.nanmin(six))
    feasible = bool(
        np.all(np.asarray(recalls) > target_recall)
        and np.all(np.asarray(specificities) > target_specificity)
    )
    balanced_accuracy = float(np.nanmean(recalls))
    macro_f1 = float(np.nanmean(f1s))
    return (
        int(feasible),
        minimum,
        balanced_accuracy,
        macro_f1,
    ), minimum


def tune_ordered_thresholds(
    y_true: np.ndarray,
    score: np.ndarray,
    grid_size: int = 101,
    target_recall: float = 0.85,
    target_specificity: float = 0.85,
) -> dict:
    true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    quantiles = np.linspace(0.01, 0.99, max(11, grid_size))
    candidates = np.unique(np.quantile(score, quantiles))
    best = None
    for low in candidates:
        for high in candidates[candidates > low]:
            prediction = _predict_from_thresholds(score, float(low), float(high))
            rank, minimum = _rank_from_predictions(
                true, prediction, target_recall, target_specificity
            )
            if best is None or rank > best["rank"]:
                best = {
                    "low_threshold": float(low),
                    "high_threshold": float(high),
                    "rank": rank,
                    "prediction": prediction,
                    "minimum_recall_specificity": minimum,
                }
    if best is None:
        raise ValueError("Could not find ordered thresholds")
    prediction = best.pop("prediction")
    metrics = classification_metrics(true, prediction)
    best["validation_metrics"] = metrics
    best["target_met"] = target_is_met(
        metrics, target_recall, target_specificity
    )
    best["rank"] = list(best["rank"])
    return best


def _safe_logit(probability: np.ndarray | float) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(value / (1.0 - value))


def dual_binary_probabilities(binary_logits: np.ndarray) -> np.ndarray:
    """Return three comparable OVR scores derived only from the two binary heads."""
    binary = expit(np.asarray(binary_logits, dtype=float))
    if binary.ndim != 2 or binary.shape[1] != 2:
        raise ValueError("binary_logits must have shape [records, 2]")
    hypo = binary[:, 0]
    hyper = binary[:, 1]
    nk = (1.0 - hypo) * (1.0 - hyper)
    scores = np.stack((hypo, nk, hyper), axis=1)
    denominator = np.clip(scores.sum(axis=1, keepdims=True), 1e-12, None)
    return scores / denominator


def predict_dual_binary(
    binary_logits: np.ndarray,
    hypo_threshold: float,
    hyper_threshold: float,
) -> tuple[np.ndarray, float]:
    binary = expit(np.asarray(binary_logits, dtype=float))
    if binary.ndim != 2 or binary.shape[1] != 2:
        raise ValueError("binary_logits must have shape [records, 2]")
    hypo_positive = binary[:, 0] >= float(hypo_threshold)
    hyper_positive = binary[:, 1] >= float(hyper_threshold)
    prediction = np.ones(len(binary), dtype=np.int64)
    prediction[hypo_positive & ~hyper_positive] = 0
    prediction[hyper_positive & ~hypo_positive] = 2
    conflict = hypo_positive & hyper_positive
    if np.any(conflict):
        hypo_margin = _safe_logit(binary[conflict, 0]) - _safe_logit(hypo_threshold)
        hyper_margin = _safe_logit(binary[conflict, 1]) - _safe_logit(hyper_threshold)
        prediction[conflict] = np.where(hypo_margin >= hyper_margin, 0, 2)
    return prediction, float(conflict.mean())


def tune_dual_binary_thresholds(
    y_true: np.ndarray,
    binary_logits: np.ndarray,
    grid_size: int = 41,
    target_recall: float = 0.85,
    target_specificity: float = 0.85,
) -> dict:
    true = np.asarray(y_true, dtype=int)
    binary = expit(np.asarray(binary_logits, dtype=float))
    if binary.ndim != 2 or binary.shape[1] != 2:
        raise ValueError("binary_logits must have shape [records, 2]")
    quantiles = np.linspace(0.01, 0.99, max(11, int(grid_size)))
    hypo_candidates = np.unique(
        np.concatenate((np.quantile(binary[:, 0], quantiles), [0.5]))
    )
    hyper_candidates = np.unique(
        np.concatenate((np.quantile(binary[:, 1], quantiles), [0.5]))
    )
    best = None
    for hypo_threshold in hypo_candidates:
        for hyper_threshold in hyper_candidates:
            prediction, conflict_rate = predict_dual_binary(
                binary_logits, float(hypo_threshold), float(hyper_threshold)
            )
            rank, minimum = _rank_from_predictions(
                true, prediction, target_recall, target_specificity
            )
            if best is None or rank > best["rank"]:
                best = {
                    "hypo_threshold": float(hypo_threshold),
                    "hyper_threshold": float(hyper_threshold),
                    "conflict_rate": float(conflict_rate),
                    "rank": rank,
                    "prediction": prediction,
                    "minimum_recall_specificity": minimum,
                }
    if best is None:
        raise ValueError("Could not find dual-binary thresholds")
    prediction = best.pop("prediction")
    probabilities = dual_binary_probabilities(binary_logits)
    metrics = classification_metrics(true, prediction, probabilities)
    best["validation_metrics"] = metrics
    best["target_met"] = target_is_met(
        metrics, target_recall, target_specificity
    )
    best["rank"] = list(best["rank"])
    return best


@dataclass
class CalibrationResult:
    temperature: float
    selected_head: str
    low_threshold: float | None
    high_threshold: float | None
    hypo_threshold: float | None
    hyper_threshold: float | None
    conflict_rate: float | None
    target_met_on_validation: bool
    validation_metrics: dict
    candidate_results: dict

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "selected_head": self.selected_head,
            "low_threshold": self.low_threshold,
            "high_threshold": self.high_threshold,
            "hypo_threshold": self.hypo_threshold,
            "hyper_threshold": self.hyper_threshold,
            "conflict_rate": self.conflict_rate,
            "target_met_on_validation": self.target_met_on_validation,
            "validation_metrics": self.validation_metrics,
            "candidate_results": self.candidate_results,
        }


def calibrate_predictions(
    y_true: np.ndarray,
    logits: np.ndarray,
    ordinal_logits: np.ndarray,
    potassium_prediction: np.ndarray,
    config: dict,
    binary_logits: np.ndarray | None = None,
) -> CalibrationResult:
    section = config["calibration"]
    temperature = (
        fit_temperature(logits, y_true) if section.get("temperature_scaling", True) else 1.0
    )
    probabilities = apply_temperature(logits, temperature)
    candidates = {
        "classification": probabilities @ np.arange(probabilities.shape[1]),
        "ordinal": expit(np.asarray(ordinal_logits)).sum(axis=1),
        "regression": np.asarray(potassium_prediction, dtype=float),
    }
    results = {}
    for name, score in candidates.items():
        results[name] = tune_ordered_thresholds(
            y_true,
            score,
            grid_size=int(section["threshold_grid_size"]),
            target_recall=float(section["target_recall"]),
            target_specificity=float(section["target_specificity"]),
        )
    if binary_logits is not None:
        binary_probabilities = expit(np.asarray(binary_logits, dtype=float))
        results["dual_binary"] = tune_ordered_thresholds(
            y_true,
            binary_probabilities[:, 1] - binary_probabilities[:, 0],
            grid_size=int(section["threshold_grid_size"]),
            target_recall=float(section["target_recall"]),
            target_specificity=float(section["target_specificity"]),
        )
        results["dual_binary_independent"] = tune_dual_binary_thresholds(
            y_true,
            binary_logits,
            grid_size=int(section["threshold_grid_size"]),
            target_recall=float(section["target_recall"]),
            target_specificity=float(section["target_specificity"]),
        )
    primary_head = section.get("primary_head")
    if primary_head is not None:
        if primary_head not in results:
            raise ValueError(f"Requested calibration.primary_head unavailable: {primary_head}")
        best_name = str(primary_head)
    else:
        best_name = max(results, key=lambda name: tuple(results[name]["rank"]))
    selected = results[best_name]
    independent = best_name == "dual_binary_independent"
    return CalibrationResult(
        temperature=temperature,
        selected_head=best_name,
        low_threshold=None if independent else float(selected["low_threshold"]),
        high_threshold=None if independent else float(selected["high_threshold"]),
        hypo_threshold=(float(selected["hypo_threshold"]) if independent else None),
        hyper_threshold=(float(selected["hyper_threshold"]) if independent else None),
        conflict_rate=(float(selected["conflict_rate"]) if independent else None),
        target_met_on_validation=bool(selected["target_met"]),
        validation_metrics=selected["validation_metrics"],
        candidate_results=results,
    )


def apply_calibration(
    logits: np.ndarray,
    ordinal_logits: np.ndarray,
    potassium_prediction: np.ndarray,
    calibration: dict,
    binary_logits: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = apply_temperature(logits, float(calibration["temperature"]))
    head = calibration["selected_head"]
    if head == "dual_binary_independent":
        if binary_logits is None:
            raise ValueError("dual_binary_independent calibration requires binary_logits")
        prediction, _ = predict_dual_binary(
            binary_logits,
            float(calibration["hypo_threshold"]),
            float(calibration["hyper_threshold"]),
        )
        return prediction, dual_binary_probabilities(binary_logits)
    if head == "classification":
        score = probabilities @ np.arange(probabilities.shape[1])
    elif head == "ordinal":
        score = expit(np.asarray(ordinal_logits)).sum(axis=1)
    elif head == "regression":
        score = np.asarray(potassium_prediction, dtype=float)
    elif head == "dual_binary":
        if binary_logits is None:
            raise ValueError("dual_binary calibration requires binary_logits")
        binary_probabilities = expit(np.asarray(binary_logits, dtype=float))
        score = binary_probabilities[:, 1] - binary_probabilities[:, 0]
    else:
        raise ValueError(f"Unknown calibrated head: {head}")
    prediction = _predict_from_thresholds(
        score,
        float(calibration["low_threshold"]),
        float(calibration["high_threshold"]),
    )
    return prediction, probabilities
