#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from v3_extract_ecgcpc_embeddings import _prepare_signal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-csv", default="/content/colab_pilot/pilot_split.csv")
    parser.add_argument("--ecg-root", default="/content/colab_pilot/ecg")
    parser.add_argument("--benchmark-dir", default="/content/ecg-fm-benchmarking")
    parser.add_argument("--cpc-config", default="/content/ecgcpc_checkpoint/ecgcpc_colab_patched.yaml")
    args = parser.parse_args()

    import torch

    frame = pd.read_csv(args.split_csv)
    if set(frame["split"].unique()) - {"train", "validation"}:
        raise RuntimeError("Smoke test must not expose a test split")
    row = frame.iloc[0]
    signal = _prepare_signal(
        Path(args.ecg_root) / str(row["record_path"]),
        target_fs=240,
        source_seconds=10.0,
        normalization="none",
    )
    if signal.shape != (12, 2400):
        raise RuntimeError(f"Unexpected resampled ECG shape: {signal.shape}")
    crops = np.stack([signal[:, i * 600 : (i + 1) * 600] for i in range(4)], axis=0)

    benchmark_code = Path(args.benchmark_dir) / "code"
    sys.path.insert(0, str(benchmark_code.resolve()))
    from clinical_ts.models.ecg_foundation_models.ecg_cpc.basic_io import load_model_from_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Select a Colab GPU runtime")
    model, _ = load_model_from_config(args.cpc_config)
    model.eval().to(device)
    with torch.inference_mode():
        output = model(seq=torch.from_numpy(crops).to(device=device, dtype=torch.float32))
    if "seq" not in output:
        raise RuntimeError(f"ECG-CPC output keys: {output.keys()}")
    seq = output["seq"]
    if seq.ndim != 3 or 512 not in seq.shape:
        raise RuntimeError(f"Unexpected ECG-CPC sequence feature shape: {tuple(seq.shape)}")
    print("study_id:", int(row["study_id"]))
    print("input crops:", crops.shape)
    print("CPC sequence features:", tuple(seq.shape))
    print("GPU:", torch.cuda.get_device_name(0))
    print("V3 ECG-CPC REAL-ECG SMOKE PASS")


if __name__ == "__main__":
    main()
