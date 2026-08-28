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
    """Return True only for the known length-dependent S4 FFT cache buffers."""
    return name.endswith(".kernel.kernel.omega") or name.endswith(".kernel.kernel.z")


def _restore_s4_fft_caches(
    model: torch.nn.Module,
    cache_tensors: dict[str, torch.Tensor],
) -> tuple[list[str], dict[str, int]]:
    """Restore checkpoint S4 FFT cache buffers and their internal cache length.

    In S4, omega/z are registered buffers, not trainable weights. Their first
    dimension is L//2+1 and therefore legitimately changes when downstream
    sequence length differs from pretraining. The benchmark's own custom
    checkpoint loader replaces buffer.data directly, so shape equality is not
    required. We do the same, but only for these two explicitly verified cache
    buffers and also keep the kernel's Python `L` attribute consistent.
    """
    named_buffers = dict(model.named_buffers())
    named_modules = dict(model.named_modules())
    resized: list[str] = []
    cache_lengths: dict[str, int] = {}
    module_lengths: dict[str, set[int]] = {}

    for name, source in cache_tensors.items():
        if not _is_s4_fft_cache(name):
            raise RuntimeError(f"Refusing to resize an unrecognized buffer: {name}")
        if name not in named_buffers:
            raise RuntimeError(f"Checkpoint S4 cache buffer absent from recreated model: {name}")
        if source.ndim != 2 or source.shape[-1] != 2 or source.shape[0] < 2:
            raise RuntimeError(f"Unexpected S4 cache tensor shape for {name}: {tuple(source.shape)}")

        inferred_L = 2 * (int(source.shape[0]) - 1)
        module_name = name.rsplit(".", 1)[0]
        module_lengths.setdefault(module_name, set()).add(inferred_L)

        target = named_buffers[name]
        if tuple(target.shape) != tuple(source.shape):
            resized.append(name)
        # This intentionally mirrors the official benchmark's custom buffer
        # loading semantics, which permits a buffer to change shape.
        target.data = source.detach().to(device=target.device, dtype=target.dtype).clone()

    for module_name, lengths in module_lengths.items():
        if len(lengths) != 1:
            raise RuntimeError(
                f"Inconsistent omega/z cache lengths for {module_name}: {sorted(lengths)}"
            )
        inferred_L = next(iter(lengths))
        module = named_modules.get(module_name)
        if module is None:
            raise RuntimeError(f"Could not find S4 kernel module {module_name}")
        if not hasattr(module, "L"):
            raise RuntimeError(f"S4 cache owner has no L attribute: {module_name}")
        module.L = inferred_L
        cache_lengths[module_name] = inferred_L

    return sorted(resized), cache_lengths


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
    parameter_names = {name for name, _ in model.named_parameters()}
    buffer_names = {name for name, _ in model.named_buffers()}

    loadable: dict[str, torch.Tensor] = {}
    resizeable_s4_caches: dict[str, torch.Tensor] = {}
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
        if target_shape == source_shape:
            loadable[source_name] = tensor
            continue

        if source_name in parameter_names:
            parameter_shape_mismatches.append(
                f"{source_name}: model={target_shape} checkpoint={source_shape}"
            )
        elif source_name in buffer_names and _is_s4_fft_cache(source_name):
            resizeable_s4_caches[source_name] = tensor
        else:
            other_buffer_shape_mismatches.append(
                f"{source_name}: model={target_shape} checkpoint={source_shape}"
            )

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
            "Unexpected non-cache ECG-CPC buffer shape mismatch:\n"
            + "\n".join(other_buffer_shape_mismatches[:30])
        )
    if unknown_source_keys:
        raise RuntimeError(
            "Official backbone contains keys absent from recreated model. "
            f"First 30: {unknown_source_keys[:30]}"
        )

    # Load every same-shaped tensor first. The known S4 length caches are
    # restored separately because torch.nn.Module.load_state_dict rejects their
    # legitimate checkpoint-vs-downstream shape difference.
    incompatible = model.load_state_dict(loadable, strict=False)
    resized_s4_caches, s4_cache_lengths = _restore_s4_fft_caches(
        model, resizeable_s4_caches
    )

    loaded_parameter_names = parameter_names.intersection(loadable.keys())
    missing_parameter_names = sorted(parameter_names - loaded_parameter_names)
    total_numel = sum(p.numel() for _, p in model.named_parameters())
    loaded_numel = sum(
        p.numel() for name, p in model.named_parameters() if name in loaded_parameter_names
    )
    coverage = loaded_numel / max(total_numel, 1)

    # Frozen V3-A is valid only if all trainable foundation-model parameters
    # are represented by the official checkpoint. Buffers are audited above
    # separately and do not enter parameter coverage.
    if coverage < 0.999999 or missing_parameter_names:
        raise RuntimeError(
            f"ECG-CPC trainable parameter coverage is not 100%: {coverage:.6%}. "
            f"Missing parameters (first 30): {missing_parameter_names[:30]}"
        )

    # Each of the four S4 layers should have checkpoint omega/z caches. If the
    # release architecture changes, fail rather than silently proceeding.
    if len(s4_cache_lengths) != int(cfg.ts.pred.layers):
        raise RuntimeError(
            f"Expected S4 cache state for {int(cfg.ts.pred.layers)} layers, "
            f"found {len(s4_cache_lengths)}: {s4_cache_lengths}"
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
        resized_s4_cache_buffers=resized_s4_caches,
        s4_cache_lengths=s4_cache_lengths,
    )

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model, cfg, report
