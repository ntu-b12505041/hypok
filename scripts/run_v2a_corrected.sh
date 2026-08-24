#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/experiments/mimic_v2a_corrected_v1.yaml}"
LOG_DIR="run_logs"
mkdir -p "$LOG_DIR"

printf '===== CONFIG VALIDATION =====\n'
hypok-ecg validate-config --config "$CONFIG"

printf '===== UNIT TESTS =====\n'
python -m unittest tests.test_v2a_corrected

printf '===== GPU STATUS =====\n'
nvidia-smi

printf '===== CUDA COMPUTE SMOKE =====\n'
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")
print("GPU:", torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("CUDA compute smoke PASS:", float(y.mean()))
PY

printf '===== MODEL + REAL ECG BATCH SMOKE =====\n'
python scripts/smoke_v2a_corrected.py --config "$CONFIG"

printf '===== START FORMAL TRAINING =====\n'
hypok-ecg train --config "$CONFIG" 2>&1 | tee "$LOG_DIR/train_v2a_corrected_$(date +%Y%m%d_%H%M%S).log"
