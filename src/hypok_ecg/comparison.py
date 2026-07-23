from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .config import load_config
from .utils import read_json


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_runs(
    baseline_config_path: str,
    finetune_config_path: str,
    output_dir: str = "outputs/model_comparison",
) -> Path:
    baseline = load_config(baseline_config_path)
    finetune = load_config(finetune_config_path)
    baseline_split = Path(baseline["data"]["split_csv"]).expanduser().resolve()
    finetune_split = Path(finetune["data"]["split_csv"]).expanduser().resolve()
    if baseline_split != finetune_split:
        raise ValueError(
            "The models do not reference the same split CSV; a paired comparison "
            "requires identical patients and records."
        )
    if not baseline_split.exists():
        raise FileNotFoundError(f"Missing shared split file: {baseline_split}")
    split_sha256 = _digest(baseline_split)

    runs = [
        ("SE-ResNet multitask", baseline),
        ("ECGFounder fine-tuned", finetune),
    ]
    loaded = []
    for display_name, config in runs:
        root = Path(config["project"]["output_dir"]).expanduser().resolve()
        metrics_path = root / "metrics" / "test_metrics.json"
        summary_path = root / "logs" / "training_summary.json"
        predictions_path = root / "metrics" / "test_predictions.csv"
        if (
            not metrics_path.exists()
            or not summary_path.exists()
            or not predictions_path.exists()
        ):
            raise FileNotFoundError(
                "Complete train/evaluate before comparison. Missing one or more of: "
                f"{metrics_path}, {summary_path}, {predictions_path}"
            )
        loaded.append(
            (
                display_name,
                config,
                read_json(metrics_path),
                read_json(summary_path),
                pd.read_csv(predictions_path),
            )
        )

    required_prediction_columns = {"study_id", "label_id", "prediction"}
    for name, _, _, _, predictions in loaded:
        missing = required_prediction_columns.difference(predictions.columns)
        if missing:
            raise ValueError(
                f"{name} predictions are missing columns: {sorted(missing)}"
            )
        if predictions["study_id"].duplicated().any():
            raise ValueError(f"{name} predictions contain duplicate study_id values")

    baseline_pairs = loaded[0][4][["study_id", "label_id"]].sort_values("study_id")
    finetune_pairs = loaded[1][4][["study_id", "label_id"]].sort_values("study_id")
    if not baseline_pairs.reset_index(drop=True).equals(
        finetune_pairs.reset_index(drop=True)
    ):
        raise ValueError(
            "The runs do not contain identical test study IDs and reference labels; "
            "a paired model comparison would be invalid."
        )

    rows = []
    for name, config, metrics, summary, _ in loaded:
        base = {
            "model": name,
            "model_key": config["model"]["name"],
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_auroc_ovr": metrics.get("macro_auroc_ovr"),
            "macro_auprc": metrics.get("macro_auprc"),
            "training_seconds": summary.get("elapsed_seconds"),
            "target_met": metrics.get("target", {}).get("met", False),
        }
        for class_name, values in metrics["per_class"].items():
            base[f"{class_name}_recall"] = values["recall"]
            base[f"{class_name}_specificity"] = values["specificity"]
            base[f"{class_name}_f1"] = values["f1"]
        rows.append(base)
    frame = pd.DataFrame(rows)
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target / "model_comparison.csv", index=False)

    columns = [
        "model",
        "balanced_accuracy",
        "macro_f1",
        "macro_auroc_ovr",
        "macro_auprc",
        "HypoK_recall",
        "HypoK_specificity",
        "NK_recall",
        "NK_specificity",
        "HyperK_recall",
        "HyperK_specificity",
        "training_seconds",
        "target_met",
    ]
    shown = frame[columns].copy()
    for column in columns[1:-2]:
        shown[column] = shown[column].map(
            lambda value: "—" if pd.isna(value) else f"{value:.3f}"
        )
    markdown = f"""# Paired Model Comparison

Both models use the exact same patient-level split.

- Split: `{baseline_split}`
- Split SHA-256: `{split_sha256}`
- Acceptance rule: every class recall > 0.85 and specificity > 0.85
- Test predictions are paired by study and patient; do not select a winner from
  test results and then continue tuning on this same test set.

{shown.to_markdown(index=False)}

## Interpretation

ECGFounder is the pretrained control and SE-ResNet is the task-specific model.
Compare class-level recall and specificity first, then macro AUPRC and confidence
intervals. A higher AUROC alone does not satisfy the prespecified acceptance rule.
"""
    report = target / "model_comparison.md"
    report.write_text(markdown, encoding="utf-8")
    return report
