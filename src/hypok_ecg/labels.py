from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PotassiumLabeler:
    hypokalemia_upper: float = 3.5
    hyperkalemia_lower: float = 5.5
    hypokalemia_inclusive: bool = False
    hyperkalemia_inclusive: bool = True
    names: tuple[str, str, str] = ("HypoK", "NK", "HyperK")

    @classmethod
    def from_config(cls, config: dict) -> "PotassiumLabeler":
        section = config["labels"]
        return cls(
            hypokalemia_upper=float(section["hypokalemia_upper"]),
            hyperkalemia_lower=float(section["hyperkalemia_lower"]),
            hypokalemia_inclusive=bool(section.get("hypokalemia_inclusive", False)),
            hyperkalemia_inclusive=bool(section.get("hyperkalemia_inclusive", True)),
            names=tuple(section.get("names", ("HypoK", "NK", "HyperK"))),
        )

    def transform(self, potassium: np.ndarray | list[float]) -> np.ndarray:
        values = np.asarray(potassium, dtype=float)
        if self.hypokalemia_inclusive:
            low = values <= self.hypokalemia_upper
        else:
            low = values < self.hypokalemia_upper
        if self.hyperkalemia_inclusive:
            high = values >= self.hyperkalemia_lower
        else:
            high = values > self.hyperkalemia_lower
        labels = np.ones(values.shape, dtype=np.int64)
        labels[low] = 0
        labels[high] = 2
        labels[~np.isfinite(values)] = -1
        return labels

    def label_names(self, label_ids: np.ndarray | list[int]) -> np.ndarray:
        ids = np.asarray(label_ids, dtype=int)
        result = np.full(ids.shape, "Unknown", dtype=object)
        for idx, name in enumerate(self.names):
            result[ids == idx] = name
        return result

    def ordinal_targets(self, label_ids: np.ndarray | list[int]) -> np.ndarray:
        """CORAL-style targets: [label >= NK, label >= HyperK]."""
        labels = np.asarray(label_ids, dtype=int)
        return np.stack((labels >= 1, labels >= 2), axis=-1).astype(np.float32)

