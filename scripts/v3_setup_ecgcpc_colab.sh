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
  hydra-core omegaconf einops pytorch-lightning wfdb scipy scikit-learn matplotlib pyyaml requests

mkdir -p "${CHECKPOINT_DIR}"

if ! find "${CHECKPOINT_DIR}" -type f -name '*.yaml' -print -quit | grep -q .; then
  printf '\nDownloading the official ECG-CPC checkpoint archive through the Figshare API...\n'
  rm -f "${ZIP_PATH}"

  python - <<PY
from pathlib import Path
import time
import zipfile
import requests

article_id = 30192604
file_id = 58173919
archive = Path("${ZIP_PATH}")

session = requests.Session()
session.headers.update({"User-Agent": "hypok-v3-colab/1.0"})

meta_url = f"https://api.figshare.com/v2/articles/{article_id}"
resp = session.get(meta_url, timeout=60)
resp.raise_for_status()
meta = resp.json()
files = meta.get("files", [])
match = next((f for f in files if int(f.get("id", -1)) == file_id), None)
if match is None:
    available = [(f.get("id"), f.get("name")) for f in files]
    raise RuntimeError(
        f"Figshare article {article_id} does not contain file {file_id}. "
        f"Available files: {available}"
    )

download_url = match.get("download_url") or f"https://ndownloader.figshare.com/files/{file_id}"
print("Figshare file:", match.get("name"))
print("Expected bytes:", match.get("size"))
print("Download URL:", download_url)

last_error = None
for attempt in range(1, 6):
    try:
        with session.get(download_url, stream=True, allow_redirects=True, timeout=(30, 300)) as r:
            print(f"Attempt {attempt}: HTTP {r.status_code}")
            if r.status_code == 202:
                last_error = RuntimeError("Figshare returned HTTP 202; retrying")
                time.sleep(5 * attempt)
                continue
            r.raise_for_status()
            with archive.open("wb") as handle:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        if archive.exists() and archive.stat().st_size > 0:
            break
    except Exception as exc:
        last_error = exc
        if archive.exists():
            archive.unlink()
        time.sleep(5 * attempt)
else:
    raise RuntimeError(f"Failed to download ECG-CPC checkpoint: {last_error}")

print("Downloaded bytes:", archive.stat().st_size)
expected = match.get("size")
if expected and archive.stat().st_size != int(expected):
    print(
        "WARNING: downloaded size differs from Figshare metadata:",
        archive.stat().st_size,
        "vs",
        expected,
    )

if not zipfile.is_zipfile(archive):
    # Keep a short diagnostic without dumping binary content.
    prefix = archive.read_bytes()[:200]
    raise RuntimeError(
        f"Downloaded file is not a ZIP archive: {archive} "
        f"({archive.stat().st_size} bytes). First bytes={prefix!r}"
    )

print("ZIP validation PASS")
PY

  python - <<PY
from pathlib import Path
import zipfile

archive = Path("${ZIP_PATH}")
out = Path("${CHECKPOINT_DIR}")
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
