from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .labels import PotassiumLabeler
from .preprocess import ECGAugmenter, ECGPreprocessor


def augmentation_rng(seed: int, epoch: int, index: int) -> np.random.Generator:
    """Create a deterministic RNG that changes across epochs for one record."""
    sequence = np.random.SeedSequence([int(seed), int(epoch), int(index)])
    return np.random.default_rng(sequence)


class MIMICECGPotassiumDataset:
    """PyTorch-compatible lazy WFDB dataset."""

    def __init__(
        self,
        frame: pd.DataFrame,
        ecg_root: str | Path,
        lead_order: list[str],
        preprocessor: ECGPreprocessor,
        labeler: PotassiumLabeler,
        augmenter: ECGAugmenter | None = None,
        seed: int = 20260723,
    ) -> None:
        try:
            from torch.utils.data import Dataset
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for model training") from exc
        _ = Dataset
        self.frame = frame.reset_index(drop=True).copy()
        self.ecg_root = Path(ecg_root).expanduser().resolve()
        self.lead_order = list(lead_order)
        self.preprocessor = preprocessor
        self.labeler = labeler
        self.augmenter = augmenter
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.frame)

    def _load(self, record_path: str) -> tuple[np.ndarray, float, list[str]]:
        try:
            import wfdb
        except ImportError as exc:
            raise RuntimeError("wfdb is required to read MIMIC-IV-ECG") from exc
        path = self.ecg_root / record_path
        record = wfdb.rdrecord(str(path))
        if record.p_signal is None:
            raise ValueError(f"Physical signal unavailable: {path}")
        return record.p_signal, float(record.fs), list(record.sig_name)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        signal, fs, names = self._load(str(row["record_path"]))
        missing = [lead for lead in self.lead_order if lead not in names]
        if missing:
            raise ValueError(f"Missing leads {missing} in study {row['study_id']}")
        indices = [names.index(lead) for lead in self.lead_order]
        signal = signal[:, indices]
        signal = self.preprocessor(signal, fs)
        if self.augmenter is not None:
            signal = self.augmenter(signal, augmentation_rng(self.seed, self.epoch, index))
        label_id = int(row["label_id"])
        ordinal = self.labeler.ordinal_targets([label_id])[0]
        return {
            "ecg": signal,
            "label": np.int64(label_id),
            "ordinal": ordinal,
            "potassium": np.float32(row["potassium"]),
            "subject_id": np.int64(row["subject_id"]),
            "study_id": np.int64(row["study_id"]),
        }


def load_split_datasets(config: dict) -> dict[str, MIMICECGPotassiumDataset]:
    frame = pd.read_csv(config["data"]["split_csv"])
    labeler = PotassiumLabeler.from_config(config)
    preprocessor = ECGPreprocessor.from_config(config)
    augmenter = (
        ECGAugmenter.from_config(config) if config["augmentation"].get("enabled") else None
    )
    common = {
        "ecg_root": config["data"]["ecg_root"],
        "lead_order": config["data"]["lead_order"],
        "preprocessor": preprocessor,
        "labeler": labeler,
        "seed": int(config["project"]["seed"]),
    }
    return {
        "train": MIMICECGPotassiumDataset(
            frame[frame["split"] == "train"], augmenter=augmenter, **common
        ),
        "validation": MIMICECGPotassiumDataset(
            frame[frame["split"] == "validation"], augmenter=None, **common
        ),
        "test": MIMICECGPotassiumDataset(
            frame[frame["split"] == "test"], augmenter=None, **common
        ),
    }
