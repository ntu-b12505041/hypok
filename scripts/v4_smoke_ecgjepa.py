#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
JEPA_LEADS = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]


def prepare_signal(record_path: Path) -> np.ndarray:
    import wfdb

    record = wfdb.rdrecord(str(record_path))
    if record.p_signal is None:
        raise ValueError(f"No physical signal: {record_path}")
    names = list(record.sig_name)
    missing = [lead for lead in LEADS_12 if lead not in names]
    if missing:
        raise ValueError(f"Missing leads {missing}: {record_path}")

    x = np.asarray(record.p_signal[:, [names.index(lead) for lead in JEPA_LEADS]], dtype=np.float32).T
    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite ECG: {record_path}")

    source_fs = int(round(float(record.fs)))
    wanted = int(round(source_fs * 10.0))
    if x.shape[1] < wanted:
        x = np.pad(x, ((0, 0), (0, wanted - x.shape[1])), mode="edge")
    else:
        x = x[:, :wanted]

    if source_fs != 250:
        gcd = int(np.gcd(source_fs, 250))
        x = resample_poly(x, 250 // gcd, source_fs // gcd, axis=1).astype(np.float32)

    if x.shape[1] < 2500:
        x = np.pad(x, ((0, 0), (0, 2500 - x.shape[1])), mode="edge")
    else:
        x = x[:, :2500]

    # Match the official ECG-JEPA downstream pipeline: no per-record z-score.
    return np.clip(x, -5.0, 5.0).astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-csv", default="/content/colab_pilot/pilot_split.csv")
    parser.add_argument("--ecg-root", default="/content/colab_pilot/ecg")
    parser.add_argument("--benchmark-dir", default="/content/ecg-fm-benchmarking")
    parser.add_argument("--checkpoint", default="/content/ecgjepa_checkpoint/multiblock_epoch100.pth")
    args = parser.parse_args()

    import torch

    frame = pd.read_csv(args.split_csv)
    if set(frame["split"].unique()) - {"train", "validation"}:
        raise RuntimeError("Smoke test must not expose test")

    row = frame.iloc[0]
    x = prepare_signal(Path(args.ecg_root) / str(row["record_path"]))
    if x.shape != (8, 2500):
        raise RuntimeError(f"Unexpected ECG-JEPA input shape: {x.shape}")

    sys.path.insert(0, str((Path(args.benchmark_dir) / "code").resolve()))
    from clinical_ts.models.ecg_foundation_models.ecg_jepa.ecg_jepa_utils import load_encoder

    encoder, feature_dim = load_encoder(args.checkpoint)
    if feature_dim != 768:
        raise RuntimeError(f"Unexpected feature_dim={feature_dim}")
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Select a Colab GPU runtime")
    encoder.to(device)

    with torch.inference_mode():
        z = encoder.representation(torch.from_numpy(x).unsqueeze(0).to(device))
    if tuple(z.shape) != (1, 768):
        raise RuntimeError(f"Unexpected representation shape: {tuple(z.shape)}")
    if not torch.isfinite(z).all():
        raise RuntimeError("Non-finite ECG-JEPA features")

    print("study_id:", int(row["study_id"]))
    print("input shape:", x.shape)
    print("lead order:", JEPA_LEADS)
    print("representation shape:", tuple(z.shape))
    print("feature finite:", bool(torch.isfinite(z).all().item()))
    print("GPU:", torch.cuda.get_device_name(0))
    print("V4 ECG-JEPA REAL-ECG SMOKE PASS")


if __name__ == "__main__":
    main()
