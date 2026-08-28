#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCHMARK_DIR="${1:-/content/ecg-fm-benchmarking}"
CHECKPOINT_DIR="${2:-/content/ecgjepa_checkpoint}"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/multiblock_epoch100.pth"
GDRIVE_ID="1gMOT4xjQQg0GZkY1iE6NuDzua4ALw00l"

printf '\n===== V4 ECG-JEPA COLAB SETUP =====\n'
python --version
nvidia-smi || true

if [ ! -d "${BENCHMARK_DIR}/.git" ]; then
  git clone --depth 1 https://github.com/AI4HealthUOL/ecg-fm-benchmarking.git "${BENCHMARK_DIR}"
else
  git -C "${BENCHMARK_DIR}" pull --ff-only
fi

python -m pip install -q "gdown==5.2.0" "timm>=1.0.15" wfdb scipy scikit-learn matplotlib
mkdir -p "${CHECKPOINT_DIR}"

if [ ! -s "${CHECKPOINT_PATH}" ]; then
  echo "Downloading official ECG-JEPA multi-block checkpoint..."
  rm -f "${CHECKPOINT_PATH}"
  python -m gdown "https://drive.google.com/uc?id=${GDRIVE_ID}" -O "${CHECKPOINT_PATH}"
fi

python - <<PY
from pathlib import Path
p = Path("${CHECKPOINT_PATH}")
if not p.exists() or p.stat().st_size < 10_000_000:
    raise RuntimeError(f"ECG-JEPA checkpoint missing or suspiciously small: {p} ({p.stat().st_size if p.exists() else 0} bytes)")
print("Checkpoint:", p)
print("Checkpoint GiB:", round(p.stat().st_size/1024**3, 3))
PY

printf '\n===== ECG-JEPA BACKBONE SMOKE TEST =====\n'
PYTHONPATH="${BENCHMARK_DIR}/code:${PYTHONPATH:-}" python - <<PY
from pathlib import Path
import torch
from clinical_ts.models.ecg_foundation_models.ecg_jepa.ecg_jepa_utils import load_encoder

ckpt = Path("${CHECKPOINT_PATH}")
encoder, feature_dim = load_encoder(str(ckpt))
if feature_dim != 768:
    raise RuntimeError(f"Unexpected ECG-JEPA feature_dim={feature_dim}")
if len(encoder.leads) != 8 or int(encoder.p) != 50 or int(encoder.t) != 50:
    raise RuntimeError(f"Unexpected ECG-JEPA input contract: leads={encoder.leads}, p={encoder.p}, t={encoder.t}")

for p in encoder.parameters():
    p.requires_grad_(False)
encoder.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("Select a Colab GPU runtime")
encoder.to(device)

x = torch.zeros(2, 8, 2500, dtype=torch.float32, device=device)
with torch.inference_mode():
    z = encoder.representation(x)
if tuple(z.shape) != (2, 768):
    raise RuntimeError(f"Unexpected ECG-JEPA representation shape: {tuple(z.shape)}")
if not torch.isfinite(z).all():
    raise RuntimeError("ECG-JEPA produced non-finite features")

params = sum(p.numel() for p in encoder.parameters())
print("Input contract: 8 leads x 2500 samples (10 sec @ 250 Hz)")
print("Feature dim:", feature_dim)
print("Encoder parameters:", params)
print("Output shape:", tuple(z.shape))
print("GPU:", torch.cuda.get_device_name(0))
print("V4 ECG-JEPA SETUP PASS")
PY
