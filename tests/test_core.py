from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from hypok_ecg.calibration import tune_ordered_thresholds
from hypok_ecg.config import load_config, validate_config
from hypok_ecg.ecgfounder import _legacy_numpy_safe_globals
from hypok_ecg.labels import PotassiumLabeler
from hypok_ecg.metrics import classification_metrics, target_is_met
from hypok_ecg.model import KMorphNetV2, SEResNet1DDualBinary
from hypok_ecg.mimic import (
    _signal_value_quality,
    build_precomputed_cohort,
    normalize_precomputed_cohort,
)
from hypok_ecg.preprocess import ECGPreprocessor
from hypok_ecg.sampling import RotatingMajoritySampler
from hypok_ecg.splits import make_patient_level_splits

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


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
        self.assertTrue(baseline["sampling"]["enabled"])
        self.assertEqual(
            baseline["sampling"]["strategy"], "rotating_nk_subsampling"
        )

    def test_v2_experiment_configs_are_valid(self):
        root = Path(__file__).resolve().parents[1]
        v2a = load_config(
            root / "configs" / "experiments" / "mimic_v2a_dual_binary.yaml"
        )
        v2b = load_config(
            root / "configs" / "experiments" / "mimic_v2b_kmorphnet.yaml"
        )
        self.assertEqual(v2a["model"]["name"], "se_resnet1d_dual_binary")
        self.assertEqual(v2b["model"]["stem_kernel_sizes"], [7, 15, 31])
        self.assertEqual(v2b["training"]["gradient_accumulation_steps"], 4)
        self.assertEqual(v2a["data"]["split_csv"], v2b["data"]["split_csv"])

    def test_invalid_sampling_ratio_is_rejected(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "mimic.yaml")
        config["sampling"]["majority_to_minority_total_ratio"] = 0
        with self.assertRaisesRegex(ValueError, "must be positive"):
            validate_config(config)


class ECGFounderCheckpointTests(unittest.TestCase):
    def test_legacy_numpy_allowlist_is_narrow(self):
        safe_globals = _legacy_numpy_safe_globals()
        names = {
            item[1]
            if isinstance(item, tuple)
            else f"{item.__module__}.{item.__qualname__}"
            for item in safe_globals
        }
        self.assertIn("numpy.core.multiarray.scalar", names)
        self.assertIn("numpy.dtype", names)
        self.assertNotIn("builtins.eval", names)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class V2ModelTests(unittest.TestCase):
    def test_dual_binary_baseline_output_contract(self):
        model = SEResNet1DDualBinary(
            input_leads=12,
            base_channels=8,
            stage_blocks=(1, 1),
            kernel_size=7,
            dropout=0.0,
            se_reduction=4,
        )
        outputs = model(torch.randn(2, 12, 256))
        self.assertEqual(tuple(outputs["logits"].shape), (2, 3))
        self.assertEqual(tuple(outputs["binary_logits"].shape), (2, 2))
        expected_nk = -0.5 * outputs["binary_logits"].sum(dim=1)
        self.assertTrue(torch.allclose(outputs["logits"][:, 1], expected_nk))

    def test_k_morphnet_attention_and_output_contract(self):
        model = KMorphNetV2(
            input_leads=12,
            base_channels=8,
            stage_blocks=(1, 1),
            kernel_size=7,
            stem_kernel_sizes=(7, 15, 31),
            stem_branch_channels=4,
            embedding_dim=32,
            transformer_layers=1,
            attention_heads=4,
            transformer_ff_dim=64,
            temporal_attention_hidden=8,
            dropout=0.0,
            se_reduction=4,
        )
        outputs = model(torch.randn(2, 12, 256))
        self.assertEqual(tuple(outputs["logits"].shape), (2, 3))
        self.assertEqual(tuple(outputs["per_lead_binary_logits"].shape), (2, 12, 2))
        self.assertTrue(
            torch.allclose(
                outputs["hypok_lead_attention"].sum(dim=1),
                torch.ones(2),
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                outputs["hyperk_lead_attention"].sum(dim=1),
                torch.ones(2),
                atol=1e-5,
            )
        )


class SamplingTests(unittest.TestCase):
    def _sampler(self):
        labels = np.asarray([0] * 4 + [1] * 20 + [2] * 3)
        subject_ids = np.arange(len(labels)) // 2
        return RotatingMajoritySampler(
            labels,
            subject_ids,
            majority_class_id=1,
            majority_to_minority_total_ratio=1.0,
            seed=17,
        )

    def test_all_minority_examples_are_kept_and_majority_is_capped(self):
        sampler = self._sampler()
        selected = np.asarray(list(iter(sampler)))
        labels = sampler.labels[selected]
        self.assertEqual(len(selected), 14)
        self.assertEqual(int((labels == 0).sum()), 4)
        self.assertEqual(int((labels == 1).sum()), 7)
        self.assertEqual(int((labels == 2).sum()), 3)
        self.assertEqual(sampler.last_audit["NK_records"], 7)
        self.assertEqual(sampler.last_audit["records"], 14)

    def test_majority_rotation_covers_pool_without_changing_minorities(self):
        sampler = self._sampler()
        majority_seen = set()
        minority = set(np.flatnonzero(sampler.labels != 1))
        majority_selections = []
        for _ in range(3):
            selected = set(iter(sampler))
            self.assertTrue(minority.issubset(selected))
            majority = {index for index in selected if sampler.labels[index] == 1}
            majority_selections.append(majority)
            majority_seen.update(majority)
        self.assertNotEqual(majority_selections[0], majority_selections[1])
        self.assertEqual(majority_seen, set(np.flatnonzero(sampler.labels == 1)))

    def test_sampler_is_reproducible(self):
        first = self._sampler()
        second = self._sampler()
        for _ in range(4):
            self.assertEqual(list(iter(first)), list(iter(second)))


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

    def test_signal_quality_identifies_nonfinite_samples_and_leads(self):
        signal = np.asarray(
            [
                [0.1, np.nan, 0.3],
                [np.inf, 0.2, 0.4],
            ]
        )
        quality = _signal_value_quality(signal, ["I", "II", "III"])
        self.assertEqual(quality["signal_samples"], 6)
        self.assertEqual(quality["nonfinite_samples"], 2)
        self.assertAlmostEqual(quality["nonfinite_fraction"], 2 / 6)
        self.assertEqual(quality["nonfinite_leads"], "I|II")

    def test_build_excludes_nonfinite_waveform_before_split(self):
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
            config["data"]["precomputed_validate_signal_values"] = True
            leads = config["data"]["lead_order"]

            def fake_record(_root, row, validate_signal_values):
                nonfinite = int(row["study_id"] == 402)
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
                    "signal_samples": 60000,
                    "nonfinite_samples": nonfinite,
                    "nonfinite_fraction": nonfinite / 60000,
                    "nonfinite_leads": "II" if nonfinite else "",
                }

            with patch("hypok_ecg.mimic._read_precomputed_row", side_effect=fake_record):
                cohort, summary = build_precomputed_cohort(config, workers=2)
            self.assertEqual(len(cohort), 2)
            self.assertEqual(summary["header_errors"], 0)
            self.assertTrue(summary["signal_values_validated"])
            self.assertEqual(summary["nonfinite_waveforms"], 1)
            self.assertEqual(set(cohort["label_id"]), {0, 2})
            excluded = pd.read_csv(output.with_name("cohort.excluded.csv"))
            self.assertEqual(excluded["study_id"].tolist(), [402])
            self.assertEqual(excluded["exclusion_reason"].tolist(), ["nonfinite_signal"])
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
