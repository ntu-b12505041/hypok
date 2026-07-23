#!/usr/bin/env python3
"""Download the official ECGFounder 12-lead checkpoint and print its digest."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="checkpoints/ecgfounder/12_lead_ECGFounder.pth",
    )
    args = parser.parse_args()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            'Install the foundation-model extras first: pip install -e ".[foundation]"'
        ) from exc

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    cached = Path(
        hf_hub_download(
            repo_id="PKUDigitalHealth/ECGFounder",
            filename="12_lead_ECGFounder.pth",
        )
    )
    # A byte copy is intentional: the Hugging Face cache remains immutable.
    output.write_bytes(cached.read_bytes())
    digest = sha256(output)
    print(f"Saved: {output}")
    print(f"SHA-256: {digest}")
    print(
        "Set model.checkpoint_sha256 in configs/ecgfounder_finetune.yaml "
        f"to: {digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
