from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration and resolve project-relative paths at runtime."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    config = deepcopy(config)
    config["_meta"] = {"config_path": str(config_path)}
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = ("project", "data", "labels", "split", "preprocess", "model", "training")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")

    ratios = [
        float(config["split"]["train_ratio"]),
        float(config["split"]["validation_ratio"]),
        float(config["split"]["test_ratio"]),
    ]
    if abs(sum(ratios) - 1.0) > 1e-8 or any(r <= 0 for r in ratios):
        raise ValueError(f"Split ratios must be positive and sum to 1; got {ratios}")

    low = float(config["labels"]["hypokalemia_upper"])
    high = float(config["labels"]["hyperkalemia_lower"])
    if low >= high:
        raise ValueError("Hypokalemia threshold must be below hyperkalemia threshold")

    cohort_source = str(config["data"].get("cohort_source", "clinical")).lower()
    if cohort_source not in {"clinical", "precomputed"}:
        raise ValueError("data.cohort_source must be 'clinical' or 'precomputed'")
    if cohort_source == "precomputed" and not config["data"].get(
        "precomputed_cohort_csv"
    ):
        raise ValueError(
            "data.precomputed_cohort_csv is required when cohort_source=precomputed"
        )

    sampling = config.get("sampling", {})
    if bool(sampling.get("enabled", False)):
        strategy = str(sampling.get("strategy", "")).lower()
        if strategy != "rotating_nk_subsampling":
            raise ValueError(
                "sampling.strategy must be 'rotating_nk_subsampling' when enabled"
            )
        majority_class_id = int(sampling.get("majority_class_id", 1))
        num_classes = int(config["model"]["num_classes"])
        if majority_class_id < 0 or majority_class_id >= num_classes:
            raise ValueError("sampling.majority_class_id is outside model classes")
        ratio = float(sampling.get("majority_to_minority_total_ratio", 1.5))
        if ratio <= 0:
            raise ValueError(
                "sampling.majority_to_minority_total_ratio must be positive"
            )

    model_name = config["model"]["name"]
    supported_models = {
        "se_resnet1d_multitask",
        "se_resnet1d_dual_binary",
        "k_morphnet_v2",
        "ecgfounder_multitask",
    }
    if model_name not in supported_models:
        raise ValueError(f"Unsupported model.name: {model_name}")
    if model_name != "ecgfounder_multitask":
        expected_leads = len(config["data"]["lead_order"])
        if int(config["model"].get("input_leads", expected_leads)) != expected_leads:
            raise ValueError("model.input_leads must match data.lead_order")
    if model_name == "k_morphnet_v2":
        kernels = [int(value) for value in config["model"]["stem_kernel_sizes"]]
        if not kernels or any(value <= 0 or value % 2 == 0 for value in kernels):
            raise ValueError("model.stem_kernel_sizes must contain positive odd values")
        if int(config["model"]["embedding_dim"]) % int(
            config["model"]["attention_heads"]
        ):
            raise ValueError("model.embedding_dim must be divisible by attention_heads")
    accumulation = int(config["training"].get("gradient_accumulation_steps", 1))
    if accumulation < 1:
        raise ValueError("training.gradient_accumulation_steps must be >= 1")
    if model_name == "ecgfounder_multitask":
        preprocess = config["preprocess"]
        if int(preprocess["target_sampling_rate"]) != 500:
            raise ValueError("ECGFounder requires target_sampling_rate=500")
        if float(preprocess["duration_seconds"]) != 10.0:
            raise ValueError("ECGFounder requires duration_seconds=10")
        if preprocess.get("normalization") != "global_zscore":
            raise ValueError("ECGFounder requires normalization=global_zscore")
        if preprocess.get("profile") != "ecgfounder_official":
            raise ValueError("ECGFounder requires profile=ecgfounder_official")
        if len(config["data"]["lead_order"]) != 12:
            raise ValueError("ECGFounder control requires all 12 standard leads")
        if not config["model"].get("checkpoint_path"):
            raise ValueError("ECGFounder requires model.checkpoint_path")


def ensure_output_dirs(config: dict[str, Any]) -> Path:
    output_dir = Path(config["project"]["output_dir"]).expanduser().resolve()
    for name in ("checkpoints", "figures", "metrics", "reports", "logs"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    return output_dir
