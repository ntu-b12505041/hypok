from __future__ import annotations

import argparse

import numpy as np

from hypok_ecg.config import load_config
from hypok_ecg.dataset import load_split_datasets
from hypok_ecg.losses import build_multitask_loss, effective_number_weights
from hypok_ecg.model import build_model
from hypok_ecg.training import _make_loaders, choose_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/mimic_v2a_corrected_v1.yaml",
    )
    args = parser.parse_args()

    import torch

    config = load_config(args.config)
    datasets = load_split_datasets(config)
    datasets["train"].set_epoch(0)
    loaders = _make_loaders(config, datasets)
    labels = datasets["train"].frame["label_id"].to_numpy()
    class_weights = effective_number_weights(
        labels,
        num_classes=int(config["model"]["num_classes"]),
        beta=float(config["training"]["effective_number_beta"]),
    )
    model = build_model(config)
    device = choose_device(config["training"]["device"])
    model.to(device)
    loss_fn = build_multitask_loss(config, class_weights)

    batch = next(iter(loaders["train"]))
    ecg = batch["ecg"].to(device=device, dtype=torch.float32)
    outputs = model(ecg)
    loss, components = loss_fn(outputs, batch)
    loss.backward()

    if "binary_logits" not in outputs or tuple(outputs["binary_logits"].shape) != (
        ecg.shape[0],
        2,
    ):
        raise RuntimeError("Corrected V2-A must emit [batch, 2] binary_logits")
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite loss in smoke test")
    if not torch.isfinite(outputs["binary_logits"]).all():
        raise RuntimeError("Non-finite binary logits in smoke test")

    print("SMOKE PASS")
    print("device:", device)
    print("torch:", torch.__version__)
    print("cuda:", torch.version.cuda)
    print("gpu:", torch.cuda.get_device_name(0) if device.type == "cuda" else "NONE")
    print("batch_shape:", tuple(ecg.shape))
    print("binary_logits_shape:", tuple(outputs["binary_logits"].shape))
    print("loss:", float(loss.detach().cpu()))
    print(
        "loss_components:",
        {key: float(value.cpu()) for key, value in components.items()},
    )
    print("class_weights:", np.asarray(class_weights).tolist())
    print("train_records:", len(datasets["train"]))
    print("validation_records:", len(datasets["validation"]))
    print("test_records_not_loaded_for_inference:", len(datasets["test"]))


if __name__ == "__main__":
    main()
