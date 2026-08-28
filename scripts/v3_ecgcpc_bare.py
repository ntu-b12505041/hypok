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
    """Dataset-free ECG-CPC encoder + S4 predictor."""

    def __init__(self, ts_encoder: torch.nn.Module):
        super().__init__()
        self.ts_encoder = ts_encoder

    def forward(self, seq: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.ts_encoder(seq=torch.nan_to_num(seq))


def _compose_official_config(config_path: Path):
    """Compose the release YAML with the benchmark's own Hydra defaults.

    The release YAML is sparse. The official loader first registers structured
    defaults and calls Hydra compose(), which fills fields such as ts.pass_static,
    ts.pre/post, ts.mask, ts.loss, and optional base fields. We reproduce that
    exact config step but stop before ECGModel construction, so no PTB-XL data
    are touched.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from clinical_ts.config import create_default_config
    import clinical_ts.models.ecg_foundation_models.ecg_cpc.basic_io as official_basic_io

    raw = OmegaConf.load(config_path)
    if not hasattr(raw, "get") or raw.get("defaults") is None:
        raise RuntimeError(
            "Released ECG-CPC YAML has no Hydra defaults; cannot safely reconstruct architecture."
        )

    create_default_config()
    gh = GlobalHydra.instance()
    if gh.is_initialized():
        gh.clear()

    # Match the official loader's searchpath logic exactly.
    project_conf_root = (Path(official_basic_io.__file__).resolve().parents[2] / "conf").as_posix()
    overrides = [f"hydra.searchpath=[{project_conf_root}]"]

    try:
        with initialize_config_dir(
            version_base=None,
            config_dir=str(config_path.parent.resolve()),
        ):
            cfg = compose(config_name=config_path.stem, overrides=overrides)
    finally:
        if gh.is_initialized():
            gh.clear()

    required = {
        "ts.pass_static": OmegaConf.select(cfg, "ts.pass_static"),
        "ts.enc._target_": OmegaConf.select(cfg, "ts.enc._target_"),
        "ts.pred._target_": OmegaConf.select(cfg, "ts.pred._target_"),
        "ts.pre._target_": OmegaConf.select(cfg, "ts.pre._target_"),
        "ts.post._target_": OmegaConf.select(cfg, "ts.post._target_"),
        "ts.head._target_": OmegaConf.select(cfg, "ts.head._target_"),
        "ts.head_ssl._target_": OmegaConf.select(cfg, "ts.head_ssl._target_"),
        "ts.mask._target_": OmegaConf.select(cfg, "ts.mask._target_"),
        "ts.loss._target_": OmegaConf.select(cfg, "ts.loss._target_"),
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise RuntimeError(f"Hydra composition incomplete; missing fields: {missing}")

    return cfg


def _load_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # Official release checkpoint; fallback handles Lightning/OmegaConf metadata.
        obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "state_dict" not in obj:
        raise RuntimeError(f"Unexpected ECG-CPC checkpoint structure: {path}")
    state = obj["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint['state_dict'] is not a mapping")
    return state


def load_bare_ecgcpc(config_path: str | Path) -> tuple[BareECGCPC, Any, LoadReport]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    # Important: never build from the raw sparse YAML.
    cfg = _compose_official_config(config_path)

    checkpoint = Path(str(cfg.trainer.pretrained))
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    from clinical_ts.template_modules import ShapeConfig, TimeSeriesEncoder

    fs = float(cfg.base.fs)
    input_seconds = float(cfg.base.input_size)
    input_channels = int(cfg.base.input_channels)
    input_length = int(round(input_seconds * fs))

    if input_channels != 12:
        raise RuntimeError(f"Unexpected ECG-CPC input_channels={input_channels}; expected 12")
    if abs(fs - 240.0) > 1e-6 or abs(input_seconds - 2.5) > 1e-6 or input_length != 600:
        raise RuntimeError(
            "Unexpected ECG-CPC input contract: "
            f"{input_seconds}s @ {fs}Hz = {input_length}; expected 2.5s @ 240Hz = 600"
        )

    input_shape = ShapeConfig(
        channels=input_channels,
        length=input_length,
        sequence_last=True,
        static_dim=int(cfg.base.input_channels_cont),
        channels2=int(cfg.base.freq_bins),
        static_dim_cat=int(cfg.base.input_channels_cat),
    )

    # Build only the waveform backbone. No ECGModel, metrics, dataloaders,
    # or dataset preprocessing are instantiated here.
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
    unknown_source_keys: list[str] = []
    shape_mismatches: list[str] = []

    for source_name, tensor in released_state.items():
        if not source_name.startswith(wanted_prefixes):
            continue
        source_keys += 1
        if source_name not in model_state:
            unknown_source_keys.append(source_name)
            continue
        if tuple(model_state[source_name].shape) != tuple(tensor.shape):
            shape_mismatches.append(
                f"{source_name}: model={tuple(model_state[source_name].shape)} checkpoint={tuple(tensor.shape)}"
            )
            continue
        loadable[source_name] = tensor

    if source_keys == 0:
        raise RuntimeError(
            "No ts_encoder.encoder/predictor tensors found in official checkpoint. "
            f"First keys: {list(released_state.keys())[:20]}"
        )
    if shape_mismatches:
        raise RuntimeError("ECG-CPC checkpoint shape mismatch:\n" + "\n".join(shape_mismatches[:20]))

    incompatible = model.load_state_dict(loadable, strict=False)

    parameter_names = {name for name, _ in model.named_parameters()}
    loaded_parameter_names = parameter_names.intersection(loadable.keys())
    missing_parameter_names = sorted(parameter_names - loaded_parameter_names)
    total_numel = sum(p.numel() for _, p in model.named_parameters())
    loaded_numel = sum(
        p.numel() for name, p in model.named_parameters() if name in loaded_parameter_names
    )
    coverage = loaded_numel / max(total_numel, 1)

    # A partially initialized frozen foundation model would invalidate V3-A.
    if coverage < 0.995:
        raise RuntimeError(
            f"ECG-CPC parameter coverage too low: {coverage:.4%}. "
            f"Missing parameters (first 30): {missing_parameter_names[:30]}"
        )
    if unknown_source_keys:
        raise RuntimeError(
            "Official backbone contains keys absent from recreated model. "
            f"First 30: {unknown_source_keys[:30]}"
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
