from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.signal import (
    butter,
    filtfilt,
    iirnotch,
    medfilt,
    resample_poly,
    sosfiltfilt,
)


@dataclass
class ECGPreprocessor:
    target_sampling_rate: int = 250
    duration_seconds: float = 10.0
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 40.0
    notch_hz: float = 60.0
    notch_quality_factor: float = 30.0
    clip_millivolts: float | None = 5.0
    remove_baseline: bool = True
    apply_notch: bool = False
    normalization: str = "none"
    profile: str = "standard"

    @classmethod
    def from_config(cls, config: dict) -> "ECGPreprocessor":
        return cls(**config["preprocess"])

    @property
    def expected_samples(self) -> int:
        return int(round(self.target_sampling_rate * self.duration_seconds))

    def __call__(self, signal: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Return a lead-first, fixed-length ECG in calibrated millivolts."""
        array = np.asarray(signal, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(f"Expected samples x leads, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("ECG contains NaN or infinite values")
        nyquist = sampling_rate / 2.0
        high = min(self.bandpass_high_hz, nyquist * 0.95)
        if self.profile == "ecgfounder_official":
            if self.apply_notch and self.notch_hz < nyquist:
                b, a = iirnotch(
                    self.notch_hz,
                    self.notch_quality_factor,
                    fs=sampling_rate,
                )
                array = filtfilt(b, a, array, axis=0)
            b, a = butter(
                4,
                [self.bandpass_low_hz, high],
                btype="bandpass",
                fs=sampling_rate,
            )
            array = filtfilt(b, a, array, axis=0)
            if self.remove_baseline:
                kernel = int(0.4 * sampling_rate) + 1
                if kernel % 2 == 0:
                    kernel += 1
                baseline = np.column_stack(
                    [medfilt(array[:, lead], kernel_size=kernel) for lead in range(array.shape[1])]
                )
                array = array - baseline
        elif self.profile == "standard":
            if self.remove_baseline:
                sos = butter(
                    4,
                    [self.bandpass_low_hz / nyquist, high / nyquist],
                    btype="bandpass",
                    output="sos",
                )
                array = sosfiltfilt(sos, array, axis=0)
            elif high < nyquist:
                sos = butter(4, high / nyquist, btype="lowpass", output="sos")
                array = sosfiltfilt(sos, array, axis=0)
            if self.apply_notch and self.notch_hz < nyquist:
                b, a = iirnotch(
                    self.notch_hz / nyquist, self.notch_quality_factor
                )
                array = filtfilt(b, a, array, axis=0)
        else:
            raise ValueError(f"Unknown ECG preprocessing profile: {self.profile}")

        if int(round(sampling_rate)) != self.target_sampling_rate:
            ratio = Fraction(self.target_sampling_rate / sampling_rate).limit_denominator(1000)
            array = resample_poly(array, ratio.numerator, ratio.denominator, axis=0)

        target = self.expected_samples
        if array.shape[0] > target:
            start = (array.shape[0] - target) // 2
            array = array[start : start + target]
        elif array.shape[0] < target:
            total = target - array.shape[0]
            array = np.pad(array, ((total // 2, total - total // 2), (0, 0)))

        if self.clip_millivolts is not None:
            array = np.clip(array, -self.clip_millivolts, self.clip_millivolts)
        if self.normalization == "global_zscore":
            array = (array - array.mean()) / (array.std() + 1e-8)
        elif self.normalization != "none":
            raise ValueError(f"Unknown ECG normalization: {self.normalization}")
        return array.T.astype(np.float32, copy=False)


@dataclass
class ECGAugmenter:
    gaussian_noise_std_millivolts: float = 0.01
    max_time_shift_seconds: float = 0.08
    lead_dropout_probability: float = 0.05
    amplitude_scale_range: tuple[float, float] = (1.0, 1.0)
    sampling_rate: int = 250

    @classmethod
    def from_config(cls, config: dict) -> "ECGAugmenter":
        section = dict(config["augmentation"])
        section.pop("enabled", None)
        section["amplitude_scale_range"] = tuple(section["amplitude_scale_range"])
        section["sampling_rate"] = int(config["preprocess"]["target_sampling_rate"])
        return cls(**section)

    def __call__(self, signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        output = np.array(signal, dtype=np.float32, copy=True)
        if self.gaussian_noise_std_millivolts > 0:
            output += rng.normal(
                0.0, self.gaussian_noise_std_millivolts, size=output.shape
            ).astype(np.float32)
        max_shift = int(round(self.max_time_shift_seconds * self.sampling_rate))
        if max_shift > 0:
            shift = int(rng.integers(-max_shift, max_shift + 1))
            output = np.roll(output, shift, axis=-1)
        if self.lead_dropout_probability > 0:
            mask = rng.random(output.shape[0]) < self.lead_dropout_probability
            if mask.all():
                mask[int(rng.integers(0, output.shape[0]))] = False
            output[mask] = 0.0
        low, high = self.amplitude_scale_range
        if low != 1.0 or high != 1.0:
            output *= float(rng.uniform(low, high))
        return output
