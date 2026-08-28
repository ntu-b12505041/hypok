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
FEATURE_DIM = 1024  # crop-mean 512 + crop-max 512


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
    parser.add_argument("--flush-every", type=int, default=512)
    parser.add_argument("--restart", action="store_true", help="Discard any partial extraction and start from row 0.")
    parser.add_argument(
        "--normalization",
        choices=["none", "per_lead_zscore"],
        default="none",
        help="Start amplitude-preserving; z-score is an explicit ablation, not silently applied.",
    )
    return parser.parse_args()


def _prepare_signal(
    record_path: Path,
    target_fs: int,
    source_seconds: float,
    normalization: str,
) -> np.ndarray:
    import wfdb

    record = wfdb.rdrecord(str(record_path))
    if record.p_signal is None:
        raise ValueError(f"No physical signal: {record_path}")
    names = list(record.sig_name)
    missing = [lead for lead in LEADS if lead not in names]
    if missing:
        raise ValueError(f"Missing leads {missing}: {record_path}")

    x = np.asarray(
        record.p_signal[:, [names.index(lead) for lead in LEADS]], dtype=np.float32
    ).T
    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite signal: {record_path}")

    source_fs = int(round(float(record.fs)))
    wanted_source = int(round(source_fs * source_seconds))
    if x.shape[1] < wanted_source:
        x = np.pad(x, ((0, 0), (0, wanted_source - x.shape[1])), mode="edge")
    else:
        x = x[:, :wanted_source]

    # Match the released downstream sample rate. The official config has
    # normalize=false, so the primary V3-A probe preserves physical amplitude.
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


def _write_progress(path: Path, *, completed: int, total: int, started_at: float, frame: pd.DataFrame) -> None:
    payload = {
        "completed": int(completed),
        "total": int(total),
        "elapsed_seconds_current_process": float(time.perf_counter() - started_at),
        "first_study_id": int(frame.iloc[0]["study_id"]),
        "last_study_id": int(frame.iloc[-1]["study_id"]),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader, Dataset

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
    if crop_len != 600 or crops_per_ecg != 4:
        raise ValueError(
            f"Primary V3-A expects four 2.5 s crops at 240 Hz; got crop_len={crop_len}, crops={crops_per_ecg}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("V3 embedding extraction expects a Colab GPU runtime")

    print("Loading bare official ECG-CPC backbone:", config_path)
    model, cfg, report = load_bare_ecgcpc(config_path)
    if report.parameter_coverage < 0.999999:
        raise RuntimeError(f"ECG-CPC pretrained parameter coverage is not 100%: {report.parameter_coverage:.6%}")
    model.to(device)
    model.eval()
    print("Parameter coverage:", f"{report.parameter_coverage:.6%}")
    print("Official input:", float(cfg.base.input_size), "sec @", float(cfg.base.fs), "Hz")
    print("GPU:", torch.cuda.get_device_name(0))

    embeddings_path = output / "embeddings.npy"
    metadata_path = output / "metadata.csv"
    progress_path = output / "extraction_progress.json"

    if args.restart:
        for path in (embeddings_path, metadata_path, progress_path, output / "embedding_summary.json"):
            if path.exists():
                path.unlink()

    completed = 0
    if embeddings_path.exists() or progress_path.exists():
        if not (embeddings_path.exists() and progress_path.exists() and metadata_path.exists()):
            raise RuntimeError(
                "Partial extraction files are inconsistent. Use --restart to discard them safely."
            )
        old_meta = pd.read_csv(metadata_path)
        if len(old_meta) != len(frame) or not np.array_equal(
            old_meta["study_id"].to_numpy(), frame["study_id"].to_numpy()
        ):
            raise RuntimeError("Existing embedding metadata does not match current pilot split. Use --restart.")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        completed = int(progress.get("completed", 0))
        if not 0 <= completed <= len(frame):
            raise RuntimeError(f"Invalid extraction progress: {completed}/{len(frame)}")
        embeddings = np.lib.format.open_memmap(embeddings_path, mode="r+")
        if embeddings.shape != (len(frame), FEATURE_DIM) or embeddings.dtype != np.float32:
            raise RuntimeError(
                f"Existing embeddings have shape/dtype {embeddings.shape}/{embeddings.dtype}; expected {(len(frame), FEATURE_DIM)}/float32"
            )
        print(f"Resuming extraction at row {completed}/{len(frame)}")
    else:
        frame.to_csv(metadata_path, index=False)
        embeddings = np.lib.format.open_memmap(
            embeddings_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(frame), FEATURE_DIM),
        )
        _write_progress(
            progress_path,
            completed=0,
            total=len(frame),
            started_at=time.perf_counter(),
            frame=frame,
        )
        print(f"Starting fresh extraction for {len(frame)} ECGs")

    if completed == len(frame):
        print("Embedding extraction already complete; verifying existing output.")
    else:
        start_index = completed

        class PilotDataset(Dataset):
            def __len__(self):
                return len(frame) - start_index

            def __getitem__(self, local_index):
                global_index = start_index + local_index
                row = frame.iloc[global_index]
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
                return torch.from_numpy(crops), global_index

        loader = DataLoader(
            PilotDataset(),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

        started = time.perf_counter()
        last_flush = completed

        with torch.inference_mode():
            for crops, indices in loader:
                batch, n_crops, channels, samples = crops.shape
                if samples != crop_len or channels != 12 or n_crops != crops_per_ecg:
                    raise RuntimeError(f"Unexpected crop shape: {tuple(crops.shape)}")

                x = crops.reshape(batch * n_crops, channels, samples).to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                output_dict = model(seq=x)
                if "seq" not in output_dict:
                    raise RuntimeError(f"ECG-CPC output does not contain 'seq': {output_dict.keys()}")
                seq = output_dict["seq"]
                if seq.ndim != 3 or seq.shape[-1] != 512:
                    raise RuntimeError(f"Unexpected CPC sequence features: {tuple(seq.shape)}")
                if not torch.isfinite(seq).all():
                    raise RuntimeError("ECG-CPC produced non-finite features")

                pooled = seq.mean(dim=1).reshape(batch, n_crops, 512)
                representation = torch.cat(
                    [pooled.mean(dim=1), pooled.max(dim=1).values], dim=1
                ).cpu().numpy().astype(np.float32, copy=False)
                if representation.shape != (batch, FEATURE_DIM):
                    raise RuntimeError(f"Unexpected representation shape: {representation.shape}")
                if not np.isfinite(representation).all():
                    raise RuntimeError("Non-finite pooled ECG-CPC representation")

                global_indices = np.asarray(indices, dtype=np.int64)
                embeddings[global_indices] = representation
                completed = int(global_indices[-1]) + 1

                if completed - last_flush >= args.flush_every or completed == len(frame):
                    embeddings.flush()
                    _write_progress(
                        progress_path,
                        completed=completed,
                        total=len(frame),
                        started_at=started,
                        frame=frame,
                    )
                    elapsed = time.perf_counter() - started
                    processed_this_run = completed - start_index
                    rate = processed_this_run / max(elapsed, 1e-6)
                    remaining = len(frame) - completed
                    eta_min = remaining / max(rate, 1e-6) / 60.0
                    print(
                        f"Embedded {completed}/{len(frame)} ECGs | "
                        f"{rate:.2f} ECG/s | ETA {eta_min:.1f} min"
                    )
                    last_flush = completed

    embeddings.flush()
    final = np.load(embeddings_path, mmap_mode="r")
    if final.shape != (len(frame), FEATURE_DIM):
        raise RuntimeError(f"Final embedding shape mismatch: {final.shape}")
    # 17k x 1024 float32 is small enough for a definitive finite-value audit.
    if not np.isfinite(np.asarray(final)).all():
        raise RuntimeError("Final embeddings contain non-finite values")

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if int(progress.get("completed", 0)) != len(frame):
        _write_progress(
            progress_path,
            completed=len(frame),
            total=len(frame),
            started_at=time.perf_counter(),
            frame=frame,
        )

    summary = {
        "records": int(len(frame)),
        "train_records": int((frame["split"] == "train").sum()),
        "validation_records": int((frame["split"] == "validation").sum()),
        "feature_dim": FEATURE_DIM,
        "base_feature_dim": 512,
        "embedding_dtype": "float32",
        "crop_seconds": float(args.crop_seconds),
        "crops_per_ecg": int(crops_per_ecg),
        "target_fs": int(args.target_fs),
        "official_model_fs": float(cfg.base.fs),
        "official_model_input_seconds": float(cfg.base.input_size),
        "normalization": args.normalization,
        "aggregation": "per_crop_sequence_mean_then_crop_mean_concat_crop_max",
        "parameter_coverage": float(report.parameter_coverage),
        "loaded_parameter_tensors": int(report.loaded_parameter_tensors),
        "loaded_buffer_tensors": int(report.loaded_buffer_tensors),
        "verified_backbone_tensors": int(report.verified_backbone_tensors),
        "s4_cache_lengths": report.s4_cache_lengths,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "test_used": False,
        "completed": True,
    }
    (output / "embedding_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print("Saved:", embeddings_path)
    print("Saved:", metadata_path)
    print("V3 ECG-CPC EMBEDDING EXTRACTION PASS")


if __name__ == "__main__":
    main()
