#!/usr/bin/env bash
set -euo pipefail

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

# Match the lightweight runtime pieces used by the official ICLR 2026
# ECG-FM benchmarking environment, while deliberately leaving Colab's torch
# installation untouched. Pin requests back to Colab's required version.
python -m pip install -q \
  "requests==2.32.4" \
  "lightning==2.5.2" \
  "pytorch-lightning==2.5.2" \
  "pykeops==2.3" \
  "keopscore==2.3" \
  "hydra-core==1.3.2" \
  "omegaconf==2.3.0" \
  "einops==0.8.1" \
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

last_error = None
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
    except Exception as e:
        last_error = e
        if archive.exists():
            archive.unlink()
        if attempt == 3:
            raise
        print("Download failed, retrying:", repr(e))
        time.sleep(2 * attempt)

with zipfile.ZipFile(archive) as zf:
    zf.extractall(out)
print(f"Extracted ECG-CPC checkpoint archive to {out}")
PY
fi

# The released YAML can retain the authors' absolute checkpoint location.
# Patch only trainer.pretrained to the local largest model checkpoint while
# preserving the rest of the official configuration.
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
if not hasattr(cfg, "trainer"):
    raise RuntimeError(f"Unexpected ECG-CPC config: trainer section missing in {config}")
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
import lightning.pytorch
import pykeops
print("requests:", requests.__version__)
print("lightning:", lightning.__version__)
print("pykeops:", pykeops.__version__)
print("DEPENDENCIES PASS")
PY

printf '\n===== IMPORT SMOKE TEST =====\n'
PYTHONPATH="${BENCHMARK_DIR}/code:${PYTHONPATH:-}" python - <<PY
from pathlib import Path
from clinical_ts.models.ecg_foundation_models.ecg_cpc.basic_io import load_model_from_config

config = Path("${CHECKPOINT_DIR}") / "ecgcpc_colab_patched.yaml"
model, cfg = load_model_from_config(str(config))
print("ECG-CPC class:", type(model).__name__)
print("Patched config:", config)
print("SETUP PASS")
PY
