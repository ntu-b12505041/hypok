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
    resized_s4_cache_buffers: list[str]
    s4_cache_lengths: dict[str, int]
    loaded_buffer_tensors: int
    verified_backbone_tensors: int


class BareECGCPC(torch.nn.Module):
    """Dataset-free ECG-CPC encoder + S4 predictor."""

    def __init__(self, ts_encoder: torch.nn.Module):
        super().__init__()
        self.ts_encoder = ts_encoder

    def forward(self, seq: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.ts_encoder(seq=torch.nan_to_num(seq))


def _compose_official_config(config_path: Path):
    """Compose the release YAML with the benchmark's own Hydra defaults."""
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
        obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "state_dict" not in obj:
        raise RuntimeError(f"Unexpected ECG-CPC checkpoint structure: {path}")
    state = obj["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint['state_dict'] is not a mapping")
    return state


def _is_s4_fft_cache(name: str) -> bool:
    return name.endswith(".kernel.kernel.omega") or name.endswith(".kernel.kernel.z")


def _replace_registered_buffer(
    model: torch.nn.Module,
    name: str,
    source: torch.Tensor,
) -> None:
    """Replace buffer storage instead of copy_ into a possibly overlapping view.

    ECG-CPC's S4 implementation creates B/P/w through repeat/expand. On modern
    PyTorch those registered buffers may have overlapping storage, so the normal
    load_state_dict copy_ path raises even when shapes are identical. Replacing
    the registered buffer tensor is numerically equivalent for frozen inference
    and matches the checkpoint values exactly.
    """
    module_name, leaf = name.rsplit(".", 1)
    modules = dict(model.named_modules())
    module = modules.get(module_name)
    if module is None:
        raise RuntimeError(f"Could not find buffer owner module for {name}")
    if leaf not in module._buffers:
        raise RuntimeError(f"{name} is not a registered buffer on recreated model")
    target = module._buffers[leaf]
    if target is None:
        module._buffers[leaf] = source.detach().clone()
    else:
        module._buffers[leaf] = source.detach().to(
            device=target.device, dtype=target.dtype
        ).clone()


def _restore_s4_fft_caches(
    model: torch.nn.Module,
    cache_tensors: dict[str, torch.Tensor],
) -> tuple[list[str], dict[str, int]]:
    """Restore length-dependent omega/z buffers and synchronize kernel.L."""
    before = dict(model.named_buffers())
    named_modules = dict(model.named_modules())
    resized: list[str] = []
    cache_lengths: dict[str, int] = {}
    module_lengths: dict[str, set[int]] = {}

    for name, source in cache_tensors.items():
        if not _is_s4_fft_cache(name):
            raise RuntimeError(f"Refusing to resize an unrecognized buffer: {name}")
        if name not in before:
            raise RuntimeError(f"Checkpoint S4 cache buffer absent from recreated model: {name}")
        if source.ndim != 2 or source.shape[-1] != 2 or source.shape[0] < 2:
            raise RuntimeError(f"Unexpected S4 cache tensor shape for {name}: {tuple(source.shape)}")

        inferred_L = 2 * (int(source.shape[0]) - 1)
        module_name = name.rsplit(".", 1)[0]
        module_lengths.setdefault(module_name, set()).add(inferred_L)
        if tuple(before[name].shape) != tuple(source.shape):
            resized.append(name)
        _replace_registered_buffer(model, name, source)

    for module_name, lengths in module_lengths.items():
        if len(lengths) != 1:
            raise RuntimeError(
                f"Inconsistent omega/z cache lengths for {module_name}: {sorted(lengths)}"
            )
        inferred_L = next(iter(lengths))
        module = named_modules.get(module_name)
        if module is None or not hasattr(module, "L"):
            raise RuntimeError(f"Could not synchronize S4 cache owner: {module_name}")
        module.L = inferred_L
        cache_lengths[module_name] = inferred_L

    return sorted(resized), cache_lengths


def _verify_exact_checkpoint_values(
    model: torch.nn.Module,
    tensors: dict[str, torch.Tensor],
) -> int:
    """Fail if any loaded backbone tensor differs from the released checkpoint."""
    state = model.state_dict()
    verified = 0
    for name, source in tensors.items():
        if name not in state:
            raise RuntimeError(f"Loaded tensor disappeared from model state: {name}")
        target = state[name].detach().cpu()
        expected = source.detach().to(dtype=target.dtype, device="cpu")
        if tuple(target.shape) != tuple(expected.shape):
            raise RuntimeError(
                f"Post-load shape mismatch for {name}: model={tuple(target.shape)} checkpoint={tuple(expected.shape)}"
            )
        if not torch.equal(target, expected):
            if target.is_floating_point():
                diff = float((target - expected).abs().max().item())
                detail = f"max_abs_diff={diff}"
            else:
                detail = "non-floating tensor differs"
            raise RuntimeError(f"Post-load checkpoint verification failed for {name}: {detail}")
        verified += 1
    return verified


def load_bare_ecgcpc(config_path: str | Path) -> tuple[BareECGCPC, Any, LoadReport]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

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
    named_parameters = dict(model.named_parameters())
    named_buffers = dict(model.named_buffers())
    parameter_names = set(named_parameters)
    buffer_names = set(named_buffers)

    parameter_tensors: dict[str, torch.Tensor] = {}
    buffer_tensors: dict[str, torch.Tensor] = {}
    s4_cache_tensors: dict[str, torch.Tensor] = {}
    source_keys = 0
    unknown_source_keys: list[str] = []
    parameter_shape_mismatches: list[str] = []
    other_buffer_shape_mismatches: list[str] = []

    for source_name, tensor in released_state.items():
        if not source_name.startswith(wanted_prefixes):
            continue
        source_keys += 1

        if source_name not in model_state:
            unknown_source_keys.append(source_name)
            continue

        target_shape = tuple(model_state[source_name].shape)
        source_shape = tuple(tensor.shape)

        # Length-dependent S4 caches are always handled separately, even if a
        # future checkpoint happens to have the same downstream cache length.
        if _is_s4_fft_cache(source_name):
            if source_name not in buffer_names:
                other_buffer_shape_mismatches.append(
                    f"Known S4 cache is not a registered buffer: {source_name}"
                )
            else:
                s4_cache_tensors[source_name] = tensor
            continue

        if source_name in parameter_names:
            if target_shape != source_shape:
                parameter_shape_mismatches.append(
                    f"{source_name}: model={target_shape} checkpoint={source_shape}"
                )
            else:
                parameter_tensors[source_name] = tensor
            continue

        if source_name in buffer_names:
            if target_shape != source_shape:
                other_buffer_shape_mismatches.append(
                    f"{source_name}: model={target_shape} checkpoint={source_shape}"
                )
            else:
                buffer_tensors[source_name] = tensor
            continue

        unknown_source_keys.append(source_name)

    if source_keys == 0:
        raise RuntimeError(
            "No ts_encoder.encoder/predictor tensors found in official checkpoint. "
            f"First keys: {list(released_state.keys())[:20]}"
        )
    if parameter_shape_mismatches:
        raise RuntimeError(
            "Trainable ECG-CPC parameter shape mismatch:\n"
            + "\n".join(parameter_shape_mismatches[:30])
        )
    if other_buffer_shape_mismatches:
        raise RuntimeError(
            "Unexpected ECG-CPC buffer mismatch:\n"
            + "\n".join(other_buffer_shape_mismatches[:30])
        )
    if unknown_source_keys:
        raise RuntimeError(
            "Official backbone contains keys absent from recreated model. "
            f"First 30: {unknown_source_keys[:30]}"
        )

    # Do NOT call load_state_dict here. S4 B/P/w buffers can be expanded views
    # with overlapping storage, and PyTorch's copy_ based loader rejects them.
    # Parameters and buffers are installed explicitly from cloned checkpoint
    # tensors instead.
    with torch.no_grad():
        for name, source in parameter_tensors.items():
            target = named_parameters[name]
            target.data = source.detach().to(
                device=target.device, dtype=target.dtype
            ).clone()

    for name, source in buffer_tensors.items():
        _replace_registered_buffer(model, name, source)

    resized_s4_caches, s4_cache_lengths = _restore_s4_fft_caches(
        model, s4_cache_tensors
    )

    loaded_parameter_names = set(parameter_tensors)
    missing_parameter_names = sorted(parameter_names - loaded_parameter_names)
    total_numel = sum(p.numel() for _, p in model.named_parameters())
    loaded_numel = sum(
        p.numel() for name, p in model.named_parameters() if name in loaded_parameter_names
    )
    coverage = loaded_numel / max(total_numel, 1)

    if coverage < 0.999999 or missing_parameter_names:
        raise RuntimeError(
            f"ECG-CPC trainable parameter coverage is not 100%: {coverage:.6%}. "
            f"Missing parameters (first 30): {missing_parameter_names[:30]}"
        )

    if len(s4_cache_lengths) != int(cfg.ts.pred.layers):
        raise RuntimeError(
            f"Expected S4 cache state for {int(cfg.ts.pred.layers)} layers, "
            f"found {len(s4_cache_lengths)}: {s4_cache_lengths}"
        )

    # Numerical audit after installation. This catches silent assignment bugs,
    # including overlapping-buffer and cache restoration errors.
    installed = {}
    installed.update(parameter_tensors)
    installed.update(buffer_tensors)
    installed.update(s4_cache_tensors)
    verified_count = _verify_exact_checkpoint_values(model, installed)

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    return model, cfg, LoadReport(
        checkpoint=str(checkpoint),
        loaded_parameter_tensors=len(loaded_parameter_names),
        total_parameter_tensors=len(parameter_names),
        loaded_parameter_numel=int(loaded_numel),
        total_parameter_numel=int(total_numel),
        parameter_coverage=float(coverage),
        missing_parameter_names=missing_parameter_names,
        unexpected_keys=[],
        resized_s4_cache_buffers=resized_s4_caches,
        s4_cache_lengths=s4_cache_lengths,
        loaded_buffer_tensors=len(buffer_tensors) + len(s4_cache_tensors),
        verified_backbone_tensors=verified_count,
    )
