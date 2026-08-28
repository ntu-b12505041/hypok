#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from v3_ecgcpc_bare import load_bare_ecgcpc

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frozen ECG-CPC embeddings from the Colab pilot ECGs."
    )
    parser.add_argument("--split-csv", default="/content/colab_pilot/pilot_split.csv")
    parser.add_argument("--ecg-root", default="/content/colab_pilot/ecg")
    parser.add_argument("--benchmark-dir", default="/content/ecg-fm-benchmarking")
    parser.add_argument("--cpc-config", default="/content/ecgcpc_checkpoint/ecgcpc_colab_patched.yaml")
    parser.add_argument("--output-dir", default="/content/drive/MyDrive/hypok_colab/v3/ecgcpc_embeddings")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--source-seconds", type=float, default=10.0)
    parser.add_argument("--target-fs", type=int, default=240)
    parser.add_argument("--crop-seconds", type=float, default=2.5)
    parser.add_argument(
        "--normalization",
        choices=["none", "per_lead_zscore"],
        default="none",
        help="Start amplitude-preserving; z-score is an explicit ablation, not silently applied.",
    )
    return parser.parse_args()


def _prepare_signal(record_path: Path, target_fs: int, source_seconds: float, normalization: str) -> np.ndarray:
    import wfdb

    record = wfdb.rdrecord(str(record_path))
    if record.p_signal is None:
        raise ValueError(f"No physical signal: {record_path}")
    names = list(record.sig_name)
    missing = [lead for lead in LEADS if lead not in names]
    if missing:
        raise ValueError(f"Missing leads {missing}: {record_path}")
    x = np.asarray(record.p_signal[:, [names.index(lead) for lead in LEADS]], dtype=np.float32).T
    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite signal: {record_path}")

    source_fs = int(round(float(record.fs)))
    wanted_source = int(round(source_fs * source_seconds))
    if x.shape[1] < wanted_source:
        x = np.pad(x, ((0, 0), (0, wanted_source - x.shape[1])), mode="edge")
    else:
        x = x[:, :wanted_source]

    # Foundation-model probe: match the released downstream sample rate and
    # preserve physical ECG amplitudes by default. No label-dependent transform.
    if source_fs != target_fs:
        gcd = np.gcd(source_fs, target_fs)
        x = resample_poly(x, target_fs // gcd, source_fs // gcd, axis=1).astype(np.float32)
    x = np.clip(x, -5.0, 5.0)

    wanted_target = int(round(target_fs * source_seconds))
    if x.shape[1] < wanted_target:
        x = np.pad(x, ((0, 0), (0, wanted_target - x.shape[1])), mode="edge")
    else:
        x = x[:, :wanted_target]

    if normalization == "per_lead_zscore":
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)
        x = (x - mean) / np.maximum(std, 1e-6)
    return x.astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError("PyTorch is required") from exc

    split_path = Path(args.split_csv)
    ecg_root = Path(args.ecg_root)
    benchmark = Path(args.benchmark_dir)
    config_path = Path(args.cpc_config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if not split_path.exists():
        raise FileNotFoundError(split_path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not (benchmark / "code").exists():
        raise FileNotFoundError(f"ECG benchmark code not found: {benchmark / 'code'}")

    sys.path.insert(0, str((benchmark / "code").resolve()))

    frame = pd.read_csv(split_path).reset_index(drop=True)
    required = {"subject_id", "study_id", "record_path", "label_id", "potassium", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing pilot columns: {sorted(missing)}")
    if set(frame["split"].unique()) - {"train", "validation"}:
        raise ValueError("V3 pilot extractor accepts train/validation only; do not expose test here")

    target_len = int(round(args.target_fs * args.source_seconds))
    crop_len = int(round(args.target_fs * args.crop_seconds))
    if target_len % crop_len:
        raise ValueError("source_seconds must divide exactly into crop_seconds")
    crops_per_ecg = target_len // crop_len

    class PilotDataset(Dataset):
        def __len__(self):
            return len(frame)

        def __getitem__(self, index):
            row = frame.iloc[index]
            x = _prepare_signal(
                ecg_root / str(row["record_path"]),
                args.target_fs,
                args.source_seconds,
                args.normalization,
            )
            crops = np.stack(
                [x[:, i * crop_len : (i + 1) * crop_len] for i in range(crops_per_ecg)],
                axis=0,
            )
            return torch.from_numpy(crops), index

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("V3 embedding extraction expects a Colab GPU runtime")

    print("Loading bare official ECG-CPC backbone:", config_path)
    model, cfg, report = load_bare_ecgcpc(config_path)
    model.to(device)
    print("Parameter coverage:", f"{report.parameter_coverage:.4%}")
    print("Official input:", float(cfg.base.input_size), "sec @", float(cfg.base.fs), "Hz")

    loader = DataLoader(
        PilotDataset(),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    embeddings = None
    feature_dim = None
    started = time.perf_counter()
    completed = 0

    with torch.inference_mode():
        for crops, indices in loader:
            batch, n_crops, channels, samples = crops.shape
            if samples != crop_len or channels != 12:
                raise RuntimeError(f"Unexpected crop shape: {tuple(crops.shape)}")
            x = crops.reshape(batch * n_crops, channels, samples).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            output_dict = model(seq=x)
            if "seq" not in output_dict:
                raise RuntimeError(f"ECG-CPC output does not contain 'seq': {output_dict.keys()}")
            seq = output_dict["seq"]
            if seq.ndim != 3:
                raise RuntimeError(f"Expected rank-3 CPC sequence features, got {tuple(seq.shape)}")
            if seq.shape[-1] == 512:
                pooled = seq.mean(dim=1)
            elif seq.shape[1] == 512:
                pooled = seq.mean(dim=2)
            else:
                raise RuntimeError(f"Could not locate the documented 512-d feature axis: {tuple(seq.shape)}")

            pooled = pooled.reshape(batch, n_crops, 512)
            representation = torch.cat(
                [pooled.mean(dim=1), pooled.max(dim=1).values], dim=1
            ).cpu().numpy().astype(np.float32)

            if embeddings is None:
                feature_dim = representation.shape[1]
                embeddings = np.empty((len(frame), feature_dim), dtype=np.float32)
            embeddings[np.asarray(indices)] = representation
            completed += batch
            if completed % 1024 < batch:
                elapsed = time.perf_counter() - started
                print(f"Embedded {completed}/{len(frame)} ECGs ({elapsed/60:.1f} min)")

    if embeddings is None or not np.isfinite(embeddings).all():
        raise RuntimeError("Embedding extraction failed or produced non-finite values")

    np.save(output / "embeddings.npy", embeddings.astype(np.float16))
    frame.to_csv(output / "metadata.csv", index=False)
    summary = {
        "records": int(len(frame)),
        "train_records": int((frame["split"] == "train").sum()),
        "validation_records": int((frame["split"] == "validation").sum()),
        "feature_dim": int(feature_dim),
        "base_feature_dim": 512,
        "crop_seconds": args.crop_seconds,
        "crops_per_ecg": int(crops_per_ecg),
        "target_fs": int(args.target_fs),
        "official_model_fs": float(cfg.base.fs),
        "official_model_input_seconds": float(cfg.base.input_size),
        "normalization": args.normalization,
        "aggregation": "crop_mean_concat_crop_max",
        "parameter_coverage": report.parameter_coverage,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "elapsed_seconds": time.perf_counter() - started,
        "test_used": False,
    }
    (output / "embedding_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Saved:", output / "embeddings.npy")
    print("Saved:", output / "metadata.csv")


if __name__ == "__main__":
    main()
