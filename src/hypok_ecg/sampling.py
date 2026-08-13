from __future__ import annotations

from collections.abc import Iterator

import numpy as np


class RotatingMajoritySampler:
    """Keep every minority example and rotate through a majority-class pool.

    The majority order is shuffled once with a fixed seed. Each epoch selects a
    contiguous circular window from that order, so every majority example is
    visited before the rotation repeats. The combined epoch indices are shuffled
    independently, also deterministically, before being handed to DataLoader.
    """

    def __init__(
        self,
        labels: np.ndarray | list[int],
        subject_ids: np.ndarray | list[int],
        majority_class_id: int = 1,
        majority_to_minority_total_ratio: float = 1.5,
        seed: int = 20260723,
        rotate_each_epoch: bool = True,
        class_names: list[str] | None = None,
    ) -> None:
        self.labels = np.asarray(labels, dtype=int)
        self.subject_ids = np.asarray(subject_ids)
        if self.labels.ndim != 1:
            raise ValueError("labels must be one-dimensional")
        if len(self.labels) != len(self.subject_ids):
            raise ValueError("labels and subject_ids must have the same length")
        if len(self.labels) == 0:
            raise ValueError("Cannot sample an empty training set")
        if majority_to_minority_total_ratio <= 0:
            raise ValueError("majority_to_minority_total_ratio must be positive")

        self.majority_class_id = int(majority_class_id)
        self.ratio = float(majority_to_minority_total_ratio)
        self.seed = int(seed)
        self.rotate_each_epoch = bool(rotate_each_epoch)
        self.class_names = class_names or ["HypoK", "NK", "HyperK"]
        self.epoch = 0
        self.last_audit: dict | None = None

        self.majority_indices = np.flatnonzero(
            self.labels == self.majority_class_id
        )
        self.minority_indices = np.flatnonzero(
            self.labels != self.majority_class_id
        )
        if len(self.majority_indices) == 0:
            raise ValueError(
                f"No examples found for majority class {self.majority_class_id}"
            )
        if len(self.minority_indices) == 0:
            raise ValueError("No minority examples found")

        requested = int(round(self.ratio * len(self.minority_indices)))
        self.majority_per_epoch = min(
            len(self.majority_indices), max(1, requested)
        )
        rng = np.random.default_rng(self.seed)
        self.majority_order = rng.permutation(self.majority_indices)

    def __len__(self) -> int:
        return len(self.minority_indices) + self.majority_per_epoch

    def _majority_for_epoch(self, epoch: int) -> np.ndarray:
        start = (
            epoch * self.majority_per_epoch
            if self.rotate_each_epoch
            else 0
        ) % len(self.majority_order)
        positions = (
            start + np.arange(self.majority_per_epoch)
        ) % len(self.majority_order)
        return self.majority_order[positions]

    def _build_audit(
        self, epoch: int, selected_indices: np.ndarray
    ) -> dict[str, int | float | str | bool]:
        selected_labels = self.labels[selected_indices]
        selected_subjects = self.subject_ids[selected_indices]
        audit: dict[str, int | float | str | bool] = {
            "epoch": int(epoch + 1),
            "strategy": "rotating_nk_subsampling",
            "records": int(len(selected_indices)),
            "subjects": int(np.unique(selected_subjects).size),
            "majority_class_id": self.majority_class_id,
            "majority_pool_records": int(len(self.majority_indices)),
            "majority_selected_records": int(self.majority_per_epoch),
            "majority_pool_fraction": float(
                self.majority_per_epoch / len(self.majority_indices)
            ),
            "rotate_each_epoch": self.rotate_each_epoch,
        }
        for class_id in sorted(np.unique(self.labels)):
            name = (
                self.class_names[class_id]
                if 0 <= class_id < len(self.class_names)
                else f"class_{class_id}"
            )
            mask = selected_labels == class_id
            audit[f"{name}_records"] = int(mask.sum())
            audit[f"{name}_subjects"] = int(
                np.unique(selected_subjects[mask]).size
            )
        return audit

    def __iter__(self) -> Iterator[int]:
        epoch = self.epoch
        selected_majority = self._majority_for_epoch(epoch)
        selected = np.concatenate([self.minority_indices, selected_majority])
        rng = np.random.default_rng(self.seed + 1 + epoch)
        rng.shuffle(selected)
        self.last_audit = self._build_audit(epoch, selected)
        self.epoch += 1
        return iter(selected.tolist())


def build_training_sampler(config: dict, dataset):
    section = config.get("sampling", {})
    if not bool(section.get("enabled", False)):
        return None
    strategy = str(section.get("strategy", "")).lower()
    if strategy != "rotating_nk_subsampling":
        raise ValueError(f"Unsupported sampling strategy: {strategy}")
    frame = dataset.frame
    return RotatingMajoritySampler(
        labels=frame["label_id"].to_numpy(),
        subject_ids=frame["subject_id"].to_numpy(),
        majority_class_id=int(section.get("majority_class_id", 1)),
        majority_to_minority_total_ratio=float(
            section.get("majority_to_minority_total_ratio", 1.5)
        ),
        seed=int(section.get("seed", config["project"]["seed"])),
        rotate_each_epoch=bool(section.get("rotate_each_epoch", True)),
        class_names=list(config["labels"]["names"]),
    )
