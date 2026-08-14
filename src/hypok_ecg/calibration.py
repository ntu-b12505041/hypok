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
            metrics = classification_metrics(true, prediction)
            minimum = _minimum_recall_specificity(metrics)
            feasible = target_is_met(metrics, target_recall, target_specificity)
            rank = (
                int(feasible),
                minimum,
                metrics["balanced_accuracy"],
                metrics["macro_f1"],
            )
            if best is None or rank > best["rank"]:
                best = {
                    "low_threshold": float(low),
                    "high_threshold": float(high),
                    "rank": rank,
                    "validation_metrics": metrics,
                    "target_met": bool(feasible),
                    "minimum_recall_specificity": minimum,
                }
    if best is None:
        raise ValueError("Could not find ordered thresholds")
    best["rank"] = list(best["rank"])
    return best


@dataclass
class CalibrationResult:
    temperature: float
    selected_head: str
    low_threshold: float
    high_threshold: float
    target_met_on_validation: bool
    validation_metrics: dict
    candidate_results: dict

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "selected_head": self.selected_head,
            "low_threshold": self.low_threshold,
            "high_threshold": self.high_threshold,
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
    if binary_logits is not None:
        binary_probabilities = expit(np.asarray(binary_logits, dtype=float))
        if binary_probabilities.ndim != 2 or binary_probabilities.shape[1] != 2:
            raise ValueError("binary_logits must have shape [records, 2]")
        # Ordered evidence: negative means stronger HypoK evidence; positive means
        # stronger HyperK evidence. Two validation thresholds leave the centre as NK.
        candidates["dual_binary"] = (
            binary_probabilities[:, 1] - binary_probabilities[:, 0]
        )
    results = {}
    best_name = None
    best_rank = None
    for name, score in candidates.items():
        result = tune_ordered_thresholds(
            y_true,
            score,
            grid_size=int(section["threshold_grid_size"]),
            target_recall=float(section["target_recall"]),
            target_specificity=float(section["target_specificity"]),
        )
        results[name] = result
        rank = tuple(result["rank"])
        if best_rank is None or rank > best_rank:
            best_name, best_rank = name, rank
    selected = results[best_name]
    return CalibrationResult(
        temperature=temperature,
        selected_head=str(best_name),
        low_threshold=float(selected["low_threshold"]),
        high_threshold=float(selected["high_threshold"]),
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
