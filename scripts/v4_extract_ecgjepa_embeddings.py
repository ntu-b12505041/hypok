#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from v4_smoke_ecgjepa import prepare_signal

FEATURE_DIM = 768


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen ECG-JEPA embeddings from pilot ECGs.")
    parser.add_argument("--split-csv", default="/content/colab_pilot/pilot_split.csv")
    parser.add_argument("--ecg-root", default="/content/colab_pilot/ecg")
    parser.add_argument("--benchmark-dir", default="/content/ecg-fm-benchmarking")
    parser.add_argument("--checkpoint", default="/content/ecgjepa_checkpoint/multiblock_epoch100.pth")
    parser.add_argument("--output-dir", default="/content/drive/MyDrive/hypok_colab/v4/ecgjepa_multiblock_embeddings")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--flush-every", type=int, default=512)
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def write_progress(path: Path, completed: int, total: int, started: float) -> None:
    payload = {
        "completed": int(completed),
        "total": int(total),
        "elapsed_seconds_current_process": float(time.perf_counter() - started),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader, Dataset

    frame = pd.read_csv(args.split_csv).reset_index(drop=True)
    required = {"subject_id", "study_id", "record_path", "label_id", "potassium", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing pilot columns: {sorted(missing)}")
    if set(frame["split"].unique()) - {"train", "validation"}:
        raise ValueError("V4 extractor accepts train/validation only; test exposure is forbidden")
    if not {"train", "validation"}.issubset(set(frame["split"].unique())):
        raise ValueError("Both train and validation splits are required")

    benchmark_code = Path(args.benchmark_dir) / "code"
    if not benchmark_code.exists():
        raise FileNotFoundError(benchmark_code)
    sys.path.insert(0, str(benchmark_code.resolve()))
    from clinical_ts.models.ecg_foundation_models.ecg_jepa.ecg_jepa_utils import load_encoder

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    encoder, feature_dim = load_encoder(str(checkpoint))
    if feature_dim != FEATURE_DIM:
        raise RuntimeError(f"Unexpected ECG-JEPA feature dim {feature_dim}")
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Use a Colab GPU runtime")
    encoder.to(device)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    embeddings_path = output / "embeddings.npy"
    metadata_path = output / "metadata.csv"
    progress_path = output / "extraction_progress.json"
    summary_path = output / "embedding_summary.json"

    if args.restart:
        for p in (embeddings_path, metadata_path, progress_path, summary_path):
            if p.exists():
                p.unlink()

    completed = 0
    if embeddings_path.exists() or progress_path.exists():
        if not (embeddings_path.exists() and progress_path.exists() and metadata_path.exists()):
            raise RuntimeError("Partial V4 files are inconsistent; use --restart to reset safely")
        old_meta = pd.read_csv(metadata_path)
        if len(old_meta) != len(frame) or not np.array_equal(old_meta["study_id"].to_numpy(), frame["study_id"].to_numpy()):
            raise RuntimeError("Existing V4 metadata does not match current pilot split")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        completed = int(progress.get("completed", 0))
        embeddings = np.lib.format.open_memmap(embeddings_path, mode="r+")
        if embeddings.shape != (len(frame), FEATURE_DIM) or embeddings.dtype != np.float32:
            raise RuntimeError(f"Existing V4 embeddings invalid: {embeddings.shape}/{embeddings.dtype}")
        print(f"Resuming ECG-JEPA extraction at {completed}/{len(frame)}")
    else:
        frame.to_csv(metadata_path, index=False)
        embeddings = np.lib.format.open_memmap(
            embeddings_path, mode="w+", dtype=np.float32, shape=(len(frame), FEATURE_DIM)
        )
        write_progress(progress_path, 0, len(frame), time.perf_counter())
        print(f"Starting fresh ECG-JEPA extraction for {len(frame)} ECGs")

    start_index = completed

    class PilotDataset(Dataset):
        def __len__(self):
            return len(frame) - start_index

        def __getitem__(self, local_index):
            global_index = start_index + local_index
            row = frame.iloc[global_index]
            x = prepare_signal(Path(args.ecg_root) / str(row["record_path"]))
            return torch.from_numpy(x), global_index

    if completed < len(frame):
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
            for x, indices in loader:
                if x.ndim != 3 or x.shape[1:] != (8, 2500):
                    raise RuntimeError(f"Unexpected ECG-JEPA batch shape: {tuple(x.shape)}")
                x = x.to(device=device, dtype=torch.float32, non_blocking=True)
                z = encoder.representation(x)
                if z.ndim != 2 or z.shape[1] != FEATURE_DIM:
                    raise RuntimeError(f"Unexpected ECG-JEPA representation shape: {tuple(z.shape)}")
                if not torch.isfinite(z).all():
                    raise RuntimeError("ECG-JEPA produced non-finite representation")

                rep = z.cpu().numpy().astype(np.float32, copy=False)
                global_indices = np.asarray(indices, dtype=np.int64)
                embeddings[global_indices] = rep
                completed = int(global_indices[-1]) + 1

                if completed - last_flush >= args.flush_every or completed == len(frame):
                    embeddings.flush()
                    write_progress(progress_path, completed, len(frame), started)
                    elapsed = time.perf_counter() - started
                    done_this_run = completed - start_index
                    rate = done_this_run / max(elapsed, 1e-6)
                    eta = (len(frame) - completed) / max(rate, 1e-6) / 60.0
                    print(f"Embedded {completed}/{len(frame)} ECGs | {rate:.2f} ECG/s | ETA {eta:.1f} min")
                    last_flush = completed

    embeddings.flush()
    final = np.load(embeddings_path, mmap_mode="r")
    if final.shape != (len(frame), FEATURE_DIM) or not np.isfinite(np.asarray(final)).all():
        raise RuntimeError("Final ECG-JEPA embedding audit failed")

    write_progress(progress_path, len(frame), len(frame), time.perf_counter())
    summary = {
        "model": "ECG-JEPA multi-block",
        "checkpoint": str(checkpoint),
        "records": int(len(frame)),
        "train_records": int((frame["split"] == "train").sum()),
        "validation_records": int((frame["split"] == "validation").sum()),
        "feature_dim": FEATURE_DIM,
        "embedding_dtype": "float32",
        "input_seconds": 10.0,
        "target_fs": 250,
        "input_samples": 2500,
        "lead_order": ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"],
        "normalization": "none",
        "aggregation": "official_encoder_representation_global_mean_pool",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "test_used": False,
        "completed": True,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Saved:", embeddings_path)
    print("Saved:", metadata_path)
    print("V4 ECG-JEPA EMBEDDING EXTRACTION PASS")


if __name__ == "__main__":
    main()
