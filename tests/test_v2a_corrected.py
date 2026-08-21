from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from hypok_ecg.calibration import (
    apply_calibration,
    predict_dual_binary,
    tune_dual_binary_thresholds,
)
from hypok_ecg.config import load_config
from hypok_ecg.dataset import augmentation_rng


def _logit(probability):
    p = np.asarray(probability, dtype=float)
    return np.log(p / (1.0 - p))


class CorrectedDualBinaryTests(unittest.TestCase):
    def test_independent_thresholds_can_recover_separable_three_class_data(self):
        y_true = np.repeat(np.arange(3), 30)
        hypo_probability = np.concatenate(
            (np.full(30, 0.90), np.full(30, 0.08), np.full(30, 0.04))
        )
        hyper_probability = np.concatenate(
            (np.full(30, 0.04), np.full(30, 0.08), np.full(30, 0.90))
        )
        binary_logits = np.column_stack(
            (_logit(hypo_probability), _logit(hyper_probability))
        )
        result = tune_dual_binary_thresholds(
            y_true,
            binary_logits,
            grid_size=21,
            target_recall=0.85,
            target_specificity=0.85,
        )
        self.assertTrue(result["target_met"])
        self.assertGreater(result["minimum_recall_specificity"], 0.99)

    def test_conflict_rule_uses_threshold_normalized_logit_margin(self):
        binary_logits = np.asarray(
            [
                [_logit(0.90), _logit(0.70)],
                [_logit(0.70), _logit(0.90)],
                [_logit(0.10), _logit(0.10)],
            ],
            dtype=float,
        )
        prediction, conflict_rate = predict_dual_binary(
            binary_logits, hypo_threshold=0.60, hyper_threshold=0.60
        )
        np.testing.assert_array_equal(prediction, [0, 2, 1])
        self.assertAlmostEqual(conflict_rate, 2 / 3)

    def test_apply_calibration_returns_dual_binary_probabilities(self):
        binary_logits = np.asarray(
            [[_logit(0.8), _logit(0.1)], [_logit(0.1), _logit(0.8)]],
            dtype=float,
        )
        calibration = {
            "temperature": 1.0,
            "selected_head": "dual_binary_independent",
            "hypo_threshold": 0.5,
            "hyper_threshold": 0.5,
        }
        prediction, probabilities = apply_calibration(
            logits=np.zeros((2, 3)),
            ordinal_logits=np.zeros((2, 2)),
            potassium_prediction=np.zeros(2),
            calibration=calibration,
            binary_logits=binary_logits,
        )
        np.testing.assert_array_equal(prediction, [0, 2])
        self.assertEqual(probabilities.shape, (2, 3))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


class EpochAugmentationTests(unittest.TestCase):
    def test_rng_is_reproducible_within_epoch_and_changes_across_epochs(self):
        first = augmentation_rng(17, 3, 11).normal(size=8)
        repeated = augmentation_rng(17, 3, 11).normal(size=8)
        next_epoch = augmentation_rng(17, 4, 11).normal(size=8)
        np.testing.assert_allclose(first, repeated)
        self.assertFalse(np.allclose(first, next_epoch))


class CorrectedV2AConfigTests(unittest.TestCase):
    def test_corrected_config_keeps_controlled_ablation_settings(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(
            root / "configs" / "experiments" / "mimic_v2a_corrected_v1.yaml"
        )
        self.assertEqual(config["model"]["name"], "se_resnet1d_dual_binary")
        self.assertEqual(config["model"]["stage_blocks"], [2, 2, 2, 2])
        self.assertEqual(
            config["sampling"]["majority_to_minority_total_ratio"], 2.5
        )
        self.assertEqual(config["training"]["binary_pos_weight"], [2.0, 2.0])
        self.assertEqual(config["training"]["loss_weights"]["classification"], 0.0)
        self.assertEqual(
            config["calibration"]["primary_head"], "dual_binary_independent"
        )
        self.assertEqual(config["training"]["monitor"], "val_minimum_six")


if __name__ == "__main__":
    unittest.main()
