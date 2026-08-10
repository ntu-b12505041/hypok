from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hypok-ecg-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize

from .metrics import target_is_met


CLASS_NAMES = ("HypoK", "NK", "HyperK")


def _stamp_synthetic(fig, synthetic: bool) -> None:
    if synthetic:
        fig.text(
            0.5,
            0.5,
            "SYNTHETIC DEMO — NOT MODEL PERFORMANCE",
            ha="center",
            va="center",
            rotation=25,
            fontsize=18,
            color="crimson",
            alpha=0.16,
            weight="bold",
        )


def plot_training_history(
    history: pd.DataFrame, output_path: str | Path, synthetic: bool = False
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(history["epoch"], history["train_loss"], label="Train")
    axes[0].plot(history["epoch"], history["val_loss"], label="Validation")
    axes[0].set(title="Training and Validation Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(history["epoch"], history["train_accuracy"], label="Train accuracy")
    axes[1].plot(history["epoch"], history["val_accuracy"], label="Validation accuracy")
    if "val_macro_auroc_ovr" in history:
        axes[1].plot(history["epoch"], history["val_macro_auroc_ovr"], label="Val macro AUROC")
    axes[1].set(
        title="Accuracy and Validation AUROC",
        xlabel="Epoch",
        ylabel="Score",
        ylim=(0, 1.02),
    )
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    _stamp_synthetic(fig, synthetic)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    matrix: np.ndarray, output_path: str | Path, synthetic: bool = False
) -> None:
    matrix = np.asarray(matrix, dtype=int)
    row_sum = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sum, where=row_sum != 0)
    annotation = np.empty(matrix.shape, dtype=object)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            annotation[i, j] = f"{matrix[i, j]}\n{normalized[i, j]:.1%}"
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        normalized,
        annot=annotation,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={"label": "Row-normalized proportion"},
        ax=ax,
    )
    ax.set(xlabel="Predicted class", ylabel="True class", title="Test Confusion Matrix")
    _stamp_synthetic(fig, synthetic)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_class_metrics(
    metrics: dict,
    output_path: str | Path,
    target: float = 0.85,
    synthetic: bool = False,
) -> None:
    rows = []
    for name, values in metrics["per_class"].items():
        for metric in ("recall", "specificity", "precision", "f1"):
            rows.append({"class": name, "metric": metric, "value": values[metric]})
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=frame, x="class", y="value", hue="metric", ax=ax)
    ax.axhline(target, color="red", linestyle="--", label=f"Target {target:.2f}")
    ax.set(title="Per-Class Test Metrics", xlabel="", ylabel="Score", ylim=(0, 1.03))
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    _stamp_synthetic(fig, synthetic)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    output_path: str | Path,
    synthetic: bool = False,
) -> None:
    targets = label_binarize(y_true, classes=np.arange(3))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    for idx, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(targets[:, idx], probabilities[:, idx])
        precision, recall, _ = precision_recall_curve(targets[:, idx], probabilities[:, idx])
        axes[0].plot(fpr, tpr, label=name)
        axes[1].plot(recall, precision, label=name)
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set(title="One-vs-Rest ROC Curves", xlabel="False positive rate", ylabel="Recall")
    axes[1].set(title="One-vs-Rest Precision–Recall Curves", xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.25)
    _stamp_synthetic(fig, synthetic)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _load_optional_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "Not available"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining = seconds % 60
    return f"{hours} h {minutes} min {remaining:.1f} s"


def create_validation_report(
    config: dict,
    metrics: dict,
    confidence_intervals: dict,
    calibration: dict,
    predictions: pd.DataFrame,
    output_dir: str | Path,
    synthetic: bool = False,
) -> Path:
    root = Path(output_dir)
    figures = root / "figures"
    reports = root / "reports"
    figures.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    history_path = root / "logs" / "training_history.csv"
    if history_path.exists():
        plot_training_history(
            pd.read_csv(history_path),
            figures / "training_curves.png",
            synthetic=synthetic,
        )
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        figures / "confusion_matrix.png",
        synthetic=synthetic,
    )
    plot_class_metrics(
        metrics,
        figures / "per_class_metrics.png",
        target=float(config["calibration"]["target_recall"]),
        synthetic=synthetic,
    )
    probs = predictions[[f"prob_{idx}" for idx in range(3)]].to_numpy()
    plot_roc_pr(
        predictions["label_id"].to_numpy(),
        probs,
        figures / "roc_pr_curves.png",
        synthetic=synthetic,
    )

    training_summary = _load_optional_json(root / "logs" / "training_summary.json")
    cohort_summary = _load_optional_json(
        Path(config["data"]["cohort_csv"]).with_suffix(".summary.json")
    )
    split_summary = _load_optional_json(
        Path(config["data"]["split_csv"]).with_suffix(".summary.json")
    )
    target_recall = float(config["calibration"]["target_recall"])
    target_specificity = float(config["calibration"]["target_specificity"])
    passed = target_is_met(metrics, target_recall, target_specificity)
    if synthetic:
        status = "PIPELINE DEMO PASS — NOT A MODEL RESULT"
    else:
        status = "PASS" if passed else "NOT MET"
    title_prefix = "SYNTHETIC DEMONSTRATION — " if synthetic else ""
    model_cfg = config["model"]
    if model_cfg["name"] == "ecgfounder_multitask":
        model_description = f"""- Architecture: ECGFounder 12-lead pretrained backbone + dyskalemia multitask heads
- Pretraining source: Harvard–Emory ECG Database (HEEDB)
- Backbone family: RegNet/Net1D with squeeze-and-excitation
- Input: 12 leads × {int(config['preprocess']['target_sampling_rate'] * config['preprocess']['duration_seconds'])} samples
- Pretrained checkpoint: `{model_cfg['checkpoint_path']}`
- Checkpoint SHA-256: `{model_cfg.get('checkpoint_sha256') or 'Not pinned'}`
- Backbone frozen epochs: {model_cfg.get('freeze_backbone_epochs', 0)}
- Backbone learning rate: {model_cfg.get('backbone_learning_rate')}
- Head learning rate: {model_cfg.get('head_learning_rate')}
- Dropout: {model_cfg['dropout']}"""
    else:
        model_description = f"""- Architecture: SE-ResNet1D multitask network trained from scratch
- Input: {model_cfg['input_leads']} leads
- Base channels: {model_cfg['base_channels']}
- Residual blocks per stage: `{model_cfg['stage_blocks']}`
- Kernel size: {model_cfg['kernel_size']}
- Dropout: {model_cfg['dropout']}"""
    normalization = config["preprocess"].get("normalization", "none")
    normalization_description = (
        "Apply one global z-score across all leads and samples, matching the "
        "official ECGFounder preprocessing."
        if normalization == "global_zscore"
        else "Do not normalize each ECG by its own standard deviation, preserving "
        "clinically relevant T-wave amplitude."
    )
    preprocess_cfg = config["preprocess"]
    filter_description = (
        "Apply the official ECGFounder sequence: 50 Hz notch, "
        "0.67–40 Hz band-pass, and 0.4-second median-filter baseline removal."
        if preprocess_cfg.get("profile") == "ecgfounder_official"
        else f"Apply {preprocess_cfg['bandpass_low_hz']}–"
        f"{preprocess_cfg['bandpass_high_hz']} Hz band-pass filtering."
    )
    clip_description = (
        "Do not amplitude-clip the checkpoint-compatible ECGFounder input."
        if preprocess_cfg.get("clip_millivolts") is None
        else f"Clip only implausible extremes at "
        f"±{preprocess_cfg['clip_millivolts']} mV."
    )

    per_class_rows = []
    for name, values in metrics["per_class"].items():
        ci = confidence_intervals.get("per_class", {}).get(name, {})
        per_class_rows.append(
            "| {name} | {support} | {recall:.3f} ({rlo:.3f}–{rhi:.3f}) | "
            "{specificity:.3f} ({slo:.3f}–{shi:.3f}) | {precision:.3f} | {f1:.3f} |".format(
                name=name,
                support=values["support"],
                recall=values["recall"],
                rlo=ci.get("recall", {}).get("lower", float("nan")),
                rhi=ci.get("recall", {}).get("upper", float("nan")),
                specificity=values["specificity"],
                slo=ci.get("specificity", {}).get("lower", float("nan")),
                shi=ci.get("specificity", {}).get("upper", float("nan")),
                precision=values["precision"],
                f1=values["f1"],
            )
        )
    split_lines = []
    for name, values in split_summary.get("splits", {}).items():
        split_lines.append(
            f"| {name} | {values['subjects']:,} | {values['records']:,} | "
            f"{values.get('class_counts', {})} |"
        )
    overall_keys = (
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("macro_f1", "Macro F1"),
        ("weighted_f1", "Weighted F1"),
        ("macro_auroc_ovr", "Macro AUROC (OvR)"),
        ("macro_auprc", "Macro AUPRC"),
        ("mcc", "Matthews correlation coefficient"),
        ("quadratic_weighted_kappa", "Quadratic weighted kappa"),
    )
    overall_rows = []
    for key, label in overall_keys:
        value = metrics.get(key, float("nan"))
        ci = confidence_intervals.get(key)
        interval = f"{ci['lower']:.3f}–{ci['upper']:.3f}" if ci else "—"
        overall_rows.append(f"| {label} | {value:.3f} | {interval} |")

    cohort_source = cohort_summary.get(
        "cohort_source", config["data"].get("cohort_source", "clinical")
    )
    if cohort_source == "precomputed":
        dataset_details = f"""- Cohort source: externally precomputed ECG–potassium pairs
- Upstream matching independently verified in this project: **No**
- Assumed upstream matching rule: `{cohort_summary.get('matching_assumption', config['data'].get('precomputed_matching_assumption', {}))}`
- Local eligibility checks: WFDB files readable, ECG time agrees with header, and all 12 standard leads are present
- Source records before local eligibility checks: {cohort_summary.get('source_records', 'Not available')}
- Excluded unreadable/missing waveforms: {cohort_summary.get('header_errors', 'Not available')}
- Excluded ECG-time mismatches: {cohort_summary.get('ecg_time_mismatches', 'Not available')}
- Excluded incomplete/nonstandard leads: {cohort_summary.get('incomplete_standard_leads', 'Not available')}"""
    else:
        dataset_details = f"""- ECG–laboratory matching window: ±{config['data']['lab_window_minutes']} minutes
- Potassium item IDs: `{cohort_summary.get('potassium_itemids', config['data']['potassium_itemids'])}`"""

    report = f"""# {title_prefix}ECG Dyskalemia Validation Report

## Executive summary

This report evaluates a three-class ECG model for hypokalemia (HypoK),
normokalemia (NK), and hyperkalemia (HyperK). The prespecified acceptance
criterion is recall > {target_recall:.2f} and specificity > {target_specificity:.2f}
for every class. Final test status: **{status}**.

{"This report uses generated synthetic predictions only. It verifies the reporting pipeline and must not be interpreted as clinical or MIMIC performance." if synthetic else "The test split was held out from model selection and decision-threshold calibration."}

## Dataset and versions

- MIMIC-IV-ECG version: `{"synthetic schema; target " if synthetic else ""}{config['data']['mimic_ecg_version']}`
- MIMIC-IV Clinical version: `{"synthetic schema; target " if synthetic else ""}{config['data']['mimic_clinical_version']}`
{dataset_details}
- Paired records: {cohort_summary.get('records', 'Not available')}
- Unique patients: {cohort_summary.get('subjects', 'Not available')}
- Class counts: `{cohort_summary.get('class_counts', {})}`

## Label definition

- HypoK: K⁺ < {config['labels']['hypokalemia_upper']} mmol/L
- NK: {config['labels']['hypokalemia_upper']} ≤ K⁺ < {config['labels']['hyperkalemia_lower']} mmol/L
- HyperK: K⁺ ≥ {config['labels']['hyperkalemia_lower']} mmol/L

## Data split

All ECGs from a patient are assigned to exactly one split. Stratification is
performed at patient level using whether the patient ever has HyperK, otherwise
HypoK, otherwise NK only.

| Split | Patients | ECG–K⁺ pairs | Class counts |
|---|---:|---:|---|
{chr(10).join(split_lines) if split_lines else "| Not available | — | — | — |"}

## Preprocessing

1. Load calibrated physical WFDB signals in millivolts.
2. Reorder to the standard 12 leads.
3. {filter_description}
4. Resample to {config['preprocess']['target_sampling_rate']} Hz.
5. Center-crop or zero-pad to {config['preprocess']['duration_seconds']} seconds.
6. {clip_description}
7. {normalization_description}

## Model architecture and parameters

{model_description}
- Heads: three-class softmax, monotonic ordinal, continuous K⁺ regression
- Loss weights: `{config['training']['loss_weights']}`
- Optimizer: {config['training']['optimizer']}
- Initial learning rate: {config['training']['learning_rate']}
- Weight decay: {config['training']['weight_decay']}
- Batch size: {config['training']['batch_size']}
- Total parameters: {training_summary.get('total_parameters', 'Not available')}
- Trainable parameters: {training_summary.get('trainable_parameters', 'Not available')}
- Best epoch: {training_summary.get('best_epoch', 'Not available')}

## Validation method

- Model selection: validation macro one-vs-rest AUROC.
- Probability calibration: temperature scaling fitted on validation only.
- Decision rule: ordered low/high thresholds selected on validation only.
- Selected prediction head: `{calibration.get('selected_head', 'Not available')}`
- Selected thresholds: `{calibration.get('low_threshold', '—')}`, `{calibration.get('high_threshold', '—')}`
- Confidence intervals: patient-cluster bootstrap with
  {config['evaluation']['bootstrap_iterations']} iterations.
- The test set is not used for model selection or threshold tuning.

## Overall test metrics

| Metric | Estimate | 95% CI |
|---|---:|---:|
{chr(10).join(overall_rows)}

## Per-class test metrics

| Class | Support | Recall (95% CI) | Specificity (95% CI) | Precision | F1 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(per_class_rows)}

## Confusion matrix

![Confusion matrix](../figures/confusion_matrix.png)

## Training process

Training duration: **{_format_seconds(training_summary.get('elapsed_seconds'))}**{" (synthetic placeholder; no model was trained)" if synthetic else ""}

- Device: `{training_summary.get('environment', {}).get('device', 'Not available')}`
- GPU: `{training_summary.get('environment', {}).get('gpu_name', 'Not available')}`
- PyTorch: `{training_summary.get('environment', {}).get('torch', 'Not available')}`

![Training curves](../figures/training_curves.png)

## Additional validation figures

![Per-class metrics](../figures/per_class_metrics.png)

![ROC and precision–recall curves](../figures/roc_pr_curves.png)

## Interpretation and limitations

- Meeting the target on the validation set does not imply the target is met on
  the held-out test set; the report uses the held-out result for the final status.
- MIMIC-IV primarily represents emergency and critical-care populations.
- ECG device clocks may not be perfectly synchronized with the clinical system;
  matching-window sensitivity analyses are required.
- No external-hospital test set is included in MIMIC-IV. Reported performance is
  therefore internal temporal-domain performance until an external cohort is added.
- HyperK is expected to be the rarest class. AUROC alone is insufficient, so
  AUPRC, class recall, specificity, confusion matrix, and bootstrap intervals are
  reported.
- This model is for research and is not a clinical diagnostic device.
"""
    output = reports / "validation_report.md"
    output.write_text(report, encoding="utf-8")
    return output
