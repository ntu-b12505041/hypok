#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def main() -> None:
    source_path = Path(__file__).with_name("v3_train_specialists.py")
    text = source_path.read_text(encoding="utf-8")

    replacements = {
        "Train three independent specialists on frozen ECG-CPC embeddings.":
            "Train three independent specialists on frozen ECG-JEPA embeddings.",
        "/content/drive/MyDrive/hypok_colab/v3/ecgcpc_embeddings":
            "/content/drive/MyDrive/hypok_colab/v4/ecgjepa_multiblock_embeddings",
        "/content/drive/MyDrive/hypok_colab/v3/v3a_ecgcpc_specialists":
            "/content/drive/MyDrive/hypok_colab/v4/v4a_ecgjepa_multiblock_specialists",
        "V3-A must use train/validation only": "V4-A must use train/validation only",
        "Use a Colab GPU runtime for V3-A": "Use a Colab GPU runtime for V4-A",
        "===== V3-A FROZEN ECG-CPC SPECIALISTS =====":
            "===== V4-A FROZEN ECG-JEPA MULTI-BLOCK SPECIALISTS =====",
        "frozen_ecgcpc_meanmax_3_specialist_mlp":
            "frozen_ecgjepa_multiblock_3_specialist_mlp",
        "V3-A ECG-CPC specialists: training vs validation loss":
            "V4-A ECG-JEPA specialists: training vs validation loss",
        "V3-A validation ROC": "V4-A ECG-JEPA validation ROC",
        "V3-A validation confusion matrix": "V4-A ECG-JEPA validation confusion matrix",
        "V3-A SPECIALIST TRAINING PASS": "V4-A ECG-JEPA SPECIALIST TRAINING PASS",
        'parser.add_argument("--pos-weight-mode", choices=["none", "sqrt"], default="sqrt")':
            'parser.add_argument("--pos-weight-mode", choices=["none", "sqrt", "clipped_sqrt"], default="clipped_sqrt")',
        'if args.pos_weight_mode == "sqrt":\n            pos_weight_value = math.sqrt(negatives / max(1.0, positives))\n        else:\n            pos_weight_value = 1.0':
            'if args.pos_weight_mode == "sqrt":\n            pos_weight_value = math.sqrt(negatives / max(1.0, positives))\n        elif args.pos_weight_mode == "clipped_sqrt":\n            pos_weight_value = max(1.0, math.sqrt(negatives / max(1.0, positives)))\n        else:\n            pos_weight_value = 1.0',
    }
    for old, new in replacements.items():
        if old not in text:
            raise RuntimeError(f"Expected V3 trainer marker not found: {old}")
        text = text.replace(old, new)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_v4_train.py", delete=False, encoding="utf-8"
    ) as f:
        f.write(text)
        temp_path = f.name

    try:
        os.execv(sys.executable, [sys.executable, temp_path, *sys.argv[1:]])
    finally:
        Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
