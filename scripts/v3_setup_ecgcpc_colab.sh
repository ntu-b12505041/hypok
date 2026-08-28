#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCHMARK_DIR="${1:-/content/ecg-fm-benchmarking}"
CHECKPOINT_DIR="${2:-/content/ecgcpc_checkpoint}"
ZIP_PATH="${3:-/content/ecgcpc_checkpoint.zip}"

printf '\n===== V3 ECG-CPC COLAB SETUP =====\n'
python --version
nvidia-smi || true

if [ ! -d "${BENCHMARK_DIR}/.git" ]; then
  git clone --depth 1 https://github.com/AI4HealthUOL/ecg-fm-benchmarking.git "${BENCHMARK_DIR}"
else
  git -C "${BENCHMARK_DIR}" pull --ff-only
fi

python -m pip install -q \
  "requests==2.32.4" \
  "lightning==2.5.2" \
  "pytorch-lightning==2.5.2" \
  "pykeops==2.3" \
  "keopscore==2.3" \
  "hydra-core==1.3.2" \
  "omegaconf==2.3.0" \
  "einops==0.8.1" \
  "resampy==0.4.3" \
  wfdb scipy scikit-learn matplotlib pyyaml

mkdir -p "${CHECKPOINT_DIR}"

if ! find "${CHECKPOINT_DIR}" -type f -name '*.yaml' -print -quit | grep -q .; then
  printf '\nDownloading the official ECG-CPC checkpoint archive through the Figshare API...\n'
  rm -f "${ZIP_PATH}"
  python - <<PY
from pathlib import Path
import json
import time
import urllib.request
import zipfile

ARTICLE_ID = 30192604
FILE_ID = 58173919
archive = Path("${ZIP_PATH}")
out = Path("${CHECKPOINT_DIR}")

api_url = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"
with urllib.request.urlopen(api_url, timeout=60) as r:
    article = json.load(r)
files = article.get("files", [])
entry = next((f for f in files if int(f.get("id", -1)) == FILE_ID), None)
if entry is None:
    raise RuntimeError(f"Figshare file id {FILE_ID} not found in article {ARTICLE_ID}")

url = entry.get("download_url")
expected = int(entry.get("size") or 0)
print("Figshare file:", entry.get("name"))
print("Expected bytes:", expected)
print("Download URL:", url)

for attempt in range(1, 4):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=180) as r, archive.open("wb") as f:
            status = getattr(r, "status", None)
            print(f"Attempt {attempt}: HTTP {status}")
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        size = archive.stat().st_size
        print("Downloaded bytes:", size)
        if expected and size != expected:
            raise RuntimeError(f"Size mismatch: got {size}, expected {expected}")
        if not zipfile.is_zipfile(archive):
            raise RuntimeError(f"Downloaded file is not a ZIP archive: {archive}")
        print("ZIP validation PASS")
        break
    except Exception:
        if archive.exists():
            archive.unlink()
        if attempt == 3:
            raise
        time.sleep(2 * attempt)

with zipfile.ZipFile(archive) as zf:
    zf.extractall(out)
print(f"Extracted ECG-CPC checkpoint archive to {out}")
PY
fi

python - <<PY
from pathlib import Path
from omegaconf import OmegaConf

root = Path("${CHECKPOINT_DIR}")
yamls = sorted(p for p in root.rglob("*.yaml") if p.name != "ecgcpc_colab_patched.yaml")
if not yamls:
    raise FileNotFoundError(f"No ECG-CPC YAML config found under {root}")
preferred = [p for p in yamls if "config_last_11597276_ckpt" in p.name]
config = preferred[0] if preferred else yamls[0]

weights = [p for pattern in ("*.ckpt", "*.pth", "*.pt") for p in root.rglob(pattern)]
weights = [p for p in weights if p.is_file()]
if not weights:
    raise FileNotFoundError(f"No ECG-CPC checkpoint weights found under {root}")
checkpoint = max(weights, key=lambda p: p.stat().st_size)

cfg = OmegaConf.load(config)
cfg.trainer.pretrained = str(checkpoint.resolve())
patched = root / "ecgcpc_colab_patched.yaml"
OmegaConf.save(cfg, patched)

print("Official config :", config)
print("Checkpoint      :", checkpoint)
print("Patched config  :", patched)
print("Checkpoint GiB  :", round(checkpoint.stat().st_size / 1024**3, 3))
PY

printf '\n===== DEPENDENCY CHECK =====\n'
python - <<'PY'
import requests
import lightning
import pykeops
import resampy
print("requests:", requests.__version__)
print("lightning:", lightning.__version__)
print("pykeops:", pykeops.__version__)
print("resampy:", resampy.__version__)
print("DEPENDENCIES PASS")
PY

printf '\n===== BARE BACKBONE SMOKE TEST =====\n'
PYTHONPATH="${REPO_DIR}/scripts:${BENCHMARK_DIR}/code:${PYTHONPATH:-}" python - <<PY
from pathlib import Path
import torch
from v3_ecgcpc_bare import load_bare_ecgcpc

if not torch.cuda.is_available():
    raise RuntimeError("ECG-CPC setup smoke requires a Colab GPU runtime")
device = torch.device("cuda")

config = Path("${CHECKPOINT_DIR}") / "ecgcpc_colab_patched.yaml"
model, cfg, report = load_bare_ecgcpc(config)
print("ECG-CPC backbone:", type(model.ts_encoder).__name__)
print("Official input  :", float(cfg.base.input_size), "sec @", float(cfg.base.fs), "Hz")
print("Parameter coverage:", f"{report.parameter_coverage:.6%}")
print("Loaded parameter tensors:", f"{report.loaded_parameter_tensors}/{report.total_parameter_tensors}")
print("Loaded buffer tensors:", report.loaded_buffer_tensors)
print("Exactly verified backbone tensors:", report.verified_backbone_tensors)
print("Restored S4 cache lengths:", report.s4_cache_lengths)

model.to(device)
x = torch.zeros(
    2,
    int(cfg.base.input_channels),
    int(round(float(cfg.base.input_size) * float(cfg.base.fs))),
    device=device,
)
with torch.inference_mode():
    out = model(seq=x)
seq = out["seq"]
print("Output seq shape:", tuple(seq.shape))
print("Output finite:", bool(torch.isfinite(seq).all().item()))
if seq.ndim != 3 or seq.shape[-1] != 512:
    raise RuntimeError(f"Unexpected CPC output shape: {tuple(seq.shape)}")
if not torch.isfinite(seq).all():
    raise RuntimeError("ECG-CPC setup smoke produced non-finite values")
print("SETUP PASS")
PY
