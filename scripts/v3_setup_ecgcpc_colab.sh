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

# Do not replace Colab's existing torch install. These are the lightweight
# runtime dependencies needed by the official ECG-CPC config/model loader.
python -m pip install -q --upgrade \
  hydra-core omegaconf einops pytorch-lightning wfdb scipy scikit-learn matplotlib pyyaml

mkdir -p "${CHECKPOINT_DIR}"

if ! find "${CHECKPOINT_DIR}" -type f -name '*.yaml' -print -quit | grep -q .; then
  printf '\nDownloading the official ECG-CPC checkpoint archive from the Figshare file linked by the ICLR 2026 benchmark...\n'
  if command -v wget >/dev/null 2>&1; then
    wget -O "${ZIP_PATH}" "https://figshare.com/ndownloader/files/58173919"
  else
    curl -L "https://figshare.com/ndownloader/files/58173919" -o "${ZIP_PATH}"
  fi
  python - <<PY
from pathlib import Path
import shutil
import zipfile

archive = Path("${ZIP_PATH}")
out = Path("${CHECKPOINT_DIR}")
if not zipfile.is_zipfile(archive):
    raise RuntimeError(f"Downloaded file is not a ZIP archive: {archive}")
with zipfile.ZipFile(archive) as zf:
    zf.extractall(out)
print(f"Extracted ECG-CPC checkpoint archive to {out}")
PY
fi

# The released YAML can retain the authors' absolute checkpoint location.
# Patch only trainer.pretrained to the local, largest model checkpoint while
# preserving the rest of the official configuration verbatim.
python - <<PY
from pathlib import Path
from omegaconf import OmegaConf

root = Path("${CHECKPOINT_DIR}")
yamls = sorted(root.rglob("*.yaml"))
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
