from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from hypok_ecg.calibration import tune_ordered_thresholds
from hypok_ecg.config import load_config
from hypok_ecg.labels import PotassiumLabeler
from hypok_ecg.metrics import classification_metrics, target_is_met
from hypok_ecg.mimic import build_precomputed_cohort, normalize_precomputed_cohort
from hypok_ecg.preprocess import ECGPreprocessor
from hypok_ecg.splits import make_patient_level_splits


class LabelTests(unittest.TestCase):
    def test_clinical_boundaries(self):
        labeler = PotassiumLabeler(hypokalemia_upper=3.5, hyperkalemia_lower=5.5)
        result = labeler.transform([2.9, 3.499, 3.5, 5.499, 5.5, 7.0])
        np.testing.assert_array_equal(result, [0, 0, 1, 1, 2, 2])


class SplitTests(unittest.TestCase):
    def test_patient_leakage_is_impossible(self):
        rows = []
        for patient in range(120):
            label = patient % 3
            for study in range(2):
                rows.append(
                    {
                        "subject_id": patient,
                        "study_id": patient * 10 + study,
                        "label_id": label,
                    }
                )
        split, _ = make_patient_level_splits(pd.DataFrame(rows), seed=7)
        counts = split.groupby("subject_id")["split"].nunique()
        self.assertTrue((counts == 1).all())
        self.assertEqual(set(split["split"]), {"train", "validation", "test"})


class PreprocessingTests(unittest.TestCase):
    def test_resample_shape_and_finite_values(self):
        rng = np.random.default_rng(3)
        signal = rng.normal(0, 0.1, size=(5000, 12))
        preprocessor = ECGPreprocessor(target_sampling_rate=250, duration_seconds=10)
        result = preprocessor(signal, 500)
        self.assertEqual(result.shape, (12, 2500))
        self.assertTrue(np.isfinite(result).all())

    def test_ecgfounder_profile_shape_and_global_zscore(self):
        rng = np.random.default_rng(4)
        signal = rng.normal(0, 0.1, size=(5000, 12))
        preprocessor = ECGPreprocessor(
            target_sampling_rate=500,
            duration_seconds=10,
            bandpass_low_hz=0.67,
            bandpass_high_hz=40,
            notch_hz=50,
            clip_millivolts=None,
            apply_notch=True,
            normalization="global_zscore",
            profile="ecgfounder_official",
        )
        result = preprocessor(signal, 500)
        self.assertEqual(result.shape, (12, 5000))
        self.assertAlmostEqual(float(result.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(result.std()), 1.0, places=5)


class ConfigurationTests(unittest.TestCase):
    def test_both_research_configs_are_valid(self):
        root = Path(__file__).resolve().parents[1]
        baseline = load_config(root / "configs" / "mimic.yaml")
        founder = load_config(root / "configs" / "ecgfounder_finetune.yaml")
        self.assertEqual(baseline["data"]["split_csv"], founder["data"]["split_csv"])
        self.assertEqual(founder["preprocess"]["target_sampling_rate"], 500)


class PrecomputedCohortTests(unittest.TestCase):
    def _config(self):
        root = Path(__file__).resolve().parents[1]
        return load_config(root / "configs" / "mimic.yaml")

    def test_schema_aliases_and_labels_are_normalized(self):
        frame = pd.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "study_id": [401, 402, 403],
                "ecg_time": ["2130-01-01 00:00:00"] * 3,
                "path": ["files/p1/401", "files/p2/402.hea", "files/p3/403"],
                "potassium_value": [3.4, 4.2, 5.5],
                "k_label": ["HypoK", "NK", "HyperK"],
            }
        )
        result = normalize_precomputed_cohort(frame, self._config())
        self.assertEqual(result["record_path"].tolist()[1], "files/p2/402")
        self.assertEqual(result["label_id"].tolist(), [0, 1, 2])

    def test_inconsistent_provided_label_is_rejected(self):
        frame = pd.DataFrame(
            {
                "subject_id": [1],
                "study_id": [401],
                "ecg_time": ["2130-01-01 00:00:00"],
                "path": ["files/p1/401"],
                "potassium_value": [5.8],
                "k_label": ["NK"],
            }
        )
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            normalize_precomputed_cohort(frame, self._config())

    def test_build_uses_only_selected_paths_and_keeps_complete_leads(self):
        with TemporaryDirectory() as temporary:
            temp = Path(temporary)
            source = temp / "provided.csv"
            output = temp / "processed" / "cohort.csv"
            ecg_root = temp / "ecg"
            ecg_root.mkdir()
            pd.DataFrame(
                {
                    "subject_id": [1, 2, 3],
                    "study_id": [401, 402, 403],
                    "ecg_time": ["2130-01-01 00:00:00"] * 3,
                    "path": ["files/p1/401", "files/p2/402", "files/p3/403"],
                    "potassium_value": [3.4, 4.2, 5.5],
                    "k_label": ["HypoK", "NK", "HyperK"],
                }
            ).to_csv(source, index=False)
            config = self._config()
            config["data"]["precomputed_cohort_csv"] = str(source)
            config["data"]["cohort_csv"] = str(output)
            config["data"]["ecg_root"] = str(ecg_root)
            leads = config["data"]["lead_order"]

            def fake_header(_root, row):
                return {
                    "subject_id": row["subject_id"],
                    "study_id": row["study_id"],
                    "record_path": row["record_path"],
                    "ecg_time": "2130-01-01 00:00:00",
                    "sampling_rate": 500.0,
                    "signal_length": 5000,
                    "n_sig": 12,
                    "lead_names": "|".join(leads),
                    "index_error": "",
                }

            with patch("hypok_ecg.mimic._read_header_row", side_effect=fake_header):
                cohort, summary = build_precomputed_cohort(config, workers=2)
            self.assertEqual(len(cohort), 3)
            self.assertEqual(summary["header_errors"], 0)
            self.assertEqual(set(cohort["label_id"]), {0, 1, 2})
            self.assertTrue(output.exists())


class MetricsTests(unittest.TestCase):
    def test_specificity_and_strict_target(self):
        y_true = np.repeat(np.arange(3), 20)
        y_pred = y_true.copy()
        metrics = classification_metrics(y_true, y_pred)
        self.assertTrue(target_is_met(metrics, 0.85, 0.85))
        for values in metrics["per_class"].values():
            values["recall"] = 0.85
        self.assertFalse(target_is_met(metrics, 0.85, 0.85))

    def test_ordered_threshold_tuning(self):
        y_true = np.repeat(np.arange(3), 25)
        scores = np.concatenate(
            [np.linspace(0.0, 0.4, 25), np.linspace(0.8, 1.2, 25), np.linspace(1.6, 2.0, 25)]
        )
        result = tune_ordered_thresholds(y_true, scores, grid_size=41)
        self.assertTrue(result["low_threshold"] < result["high_threshold"])
        self.assertTrue(result["target_met"])


if __name__ == "__main__":
    unittest.main()
