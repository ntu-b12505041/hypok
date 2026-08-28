#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf


@dataclass
class LoadReport:
    checkpoint: str
    loaded_parameter_tensors: int
    total_parameter_tensors: int
    loaded_parameter_numel: int
    total_parameter_numel: int
    parameter_coverage: float
    missing_parameter_names: list[str]
    unexpected_keys: list[str]


class BareECGCPC(torch.nn.Module):
    """Dataset-free ECG-CPC encoder + S4 predictor.

    This recreates exactly the ts encoder described by the released composed
    config, then loads only the pretrained `ts_encoder.encoder.*` and
    `ts_encoder.predictor.*` tensors. It deliberately does not construct the
    benchmark ECG task class, dataloaders, metrics, or PTB-XL metadata.
    """

    def __init__(self, ts_encoder: torch.nn.Module):
        super().__init__()
        self.ts_encoder = ts_encoder

    def forward(self, seq: torch.Tensor) -> dict[str, torch.Tensor]:
        seq = torch.nan_to_num(seq)
        return self.ts_encoder(seq=seq)


def _load_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    # This is an official ECG-CPC release checkpoint. Try the safer loader
    # first; fall back because Lightning checkpoints can contain OmegaConf
    # objects outside the tensor state_dict.
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "state_dict" not in obj:
        raise RuntimeError(f"Unexpected ECG-CPC checkpoint structure: {path}")
    state = obj["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint['state_dict'] is not a mapping")
    return state


def _cfg_int(node: Any, key: str, default: int = 0) -> int:
    """Read an optional integer from an OmegaConf node.

    The released checkpoint YAML is not guaranteed to contain every field that
    appears after Hydra structured-default composition. Static ECG features are
    not used by ECG-CPC, so absent static/frequency dimensions must safely be 0.
    """
    value = node.get(key, default) if hasattr(node, "get") else getattr(node, key, default)
    if value is None:
        value = default
    return int(value)


def load_bare_ecgcpc(config_path: str | Path) -> tuple[BareECGCPC, Any, LoadReport]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    cfg = OmegaConf.load(config_path)
    checkpoint = Path(str(cfg.trainer.pretrained))
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    # Import only model-building primitives. In contrast to the official
    # load_model_from_config(), this never instantiates clinical_ts.task.ecg,
    # so no PTB-XL df_memmap.pkl is required.
    from clinical_ts.template_modules import ShapeConfig, TimeSeriesEncoder

    fs = float(cfg.base.fs)
    input_seconds = float(cfg.base.input_size)
    input_channels = _cfg_int(cfg.base, "input_channels", 12)
    input_length = int(round(input_seconds * fs))

    if input_channels != 12:
        raise RuntimeError(f"Unexpected ECG-CPC input_channels={input_channels}; expected 12")
    if input_length != 600:
        raise RuntimeError(
            f"Unexpected ECG-CPC input length {input_length} samples "
            f"({input_seconds}s @ {fs}Hz); expected 600"
        )

    input_shape = ShapeConfig(
        channels=input_channels,
        length=input_length,
        sequence_last=True,
        # These are structured-config defaults in the benchmark framework and
        # may be absent from the released raw YAML. ECG-CPC uses waveform only.
        static_dim=_cfg_int(cfg.base, "input_channels_cont", 0),
        channels2=_cfg_int(cfg.base, "freq_bins", 0),
        static_dim_cat=_cfg_int(cfg.base, "input_channels_cat", 0),
    )
    ts_encoder = TimeSeriesEncoder(
        cfg.ts,
        input_shape,
        static_stats_train=None,
        target_dim=None,
    )
    model = BareECGCPC(ts_encoder)

    released_state = _load_checkpoint_state(checkpoint)
    wanted_prefixes = ("ts_encoder.encoder.", "ts_encoder.predictor.")
    model_state = model.state_dict()

    loadable: dict[str, torch.Tensor] = {}
    source_keys = 0
    shape_mismatches: list[str] = []
    for source_name, tensor in released_state.items():
        if not source_name.startswith(wanted_prefixes):
            continue
        source_keys += 1
        # BareECGCPC retains a `ts_encoder.` prefix, so released names should
        # line up directly with the recreated module.
        if source_name not in model_state:
            continue
        if tuple(model_state[source_name].shape) != tuple(tensor.shape):
            shape_mismatches.append(
                f"{source_name}: model={tuple(model_state[source_name].shape)} checkpoint={tuple(tensor.shape)}"
            )
            continue
        loadable[source_name] = tensor

    if source_keys == 0:
        sample = list(released_state.keys())[:20]
        raise RuntimeError(
            "No ts_encoder encoder/predictor tensors were found in the official checkpoint. "
            f"First checkpoint keys: {sample}"
        )
    if shape_mismatches:
        raise RuntimeError("ECG-CPC checkpoint shape mismatch:\n" + "\n".join(shape_mismatches[:20]))

    incompatible = model.load_state_dict(loadable, strict=False)

    parameter_names = {name for name, _ in model.named_parameters()}
    loaded_parameter_names = parameter_names.intersection(loadable.keys())
    missing_parameter_names = sorted(parameter_names - loaded_parameter_names)
    total_numel = sum(p.numel() for _, p in model.named_parameters())
    loaded_numel = sum(p.numel() for name, p in model.named_parameters() if name in loaded_parameter_names)
    coverage = loaded_numel / max(total_numel, 1)

    # Missing non-parameter buffers can be generated by the S4 implementation;
    # parameters themselves must be essentially fully covered.
    if coverage < 0.995:
        raise RuntimeError(
            f"ECG-CPC parameter coverage too low: {coverage:.4%}. "
            f"Missing parameters (first 30): {missing_parameter_names[:30]}"
        )

    report = LoadReport(
        checkpoint=str(checkpoint),
        loaded_parameter_tensors=len(loaded_parameter_names),
        total_parameter_tensors=len(parameter_names),
        loaded_parameter_numel=int(loaded_numel),
        total_parameter_numel=int(total_numel),
        parameter_coverage=float(coverage),
        missing_parameter_names=missing_parameter_names,
        unexpected_keys=list(incompatible.unexpected_keys),
    )

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model, cfg, report
