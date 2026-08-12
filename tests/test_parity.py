from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("comfyui_h3_scheduled_audio_injection", ROOT / "__init__.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class AudioPreparationParityTests(unittest.TestCase):
    def test_mono_is_repeated_to_stereo(self):
        mono = torch.arange(6, dtype=torch.float32).reshape(1, 1, 6)

        stereo = MODULE._to_stereo(mono)

        self.assertEqual(tuple(stereo.shape), (1, 2, 6))
        torch.testing.assert_close(stereo[:, 0], mono[:, 0])
        torch.testing.assert_close(stereo[:, 1], mono[:, 0])

    def test_more_than_two_channels_is_rejected(self):
        waveform = torch.zeros((1, 3, 800), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "mono or stereo"):
            MODULE._to_stereo(waveform)

    def test_native_rate_crop_happens_before_resample(self):
        waveform = torch.arange(1_000, dtype=torch.float32).reshape(1, 1, 1_000)
        audio = {"waveform": waveform, "sample_rate": 1_000}
        captured = {}

        class RecordingResample:
            def __init__(self, source_rate, target_rate):
                self.source_rate = source_rate
                self.target_rate = target_rate

            def __call__(self, value):
                captured["shape"] = tuple(value.shape)
                return torch.zeros((1, 2, 800), dtype=torch.float32)

        with mock.patch.object(MODULE.torchaudio.transforms, "Resample", RecordingResample):
            result = MODULE._prepare_waveform(
                audio,
                target_latents=1,
                target_sample_rate=32_000,
            )

        self.assertEqual(captured["shape"], (1, 2, 25))
        self.assertEqual(tuple(result.shape), (1, 2, 800))


class FixedNoiseParityTests(unittest.TestCase):
    def test_noise_matches_diffusers_packed_row_order(self):
        clean = torch.zeros((1, 32, 2, 7), dtype=torch.float32)

        latent_noise = MODULE._fixed_noise_like(clean, noise_seed=123)
        packed_noise = latent_noise[0].permute(1, 2, 0).reshape(14, 32)
        expected = torch.randn(
            (14, 32),
            generator=torch.Generator(device="cpu").manual_seed(123),
            dtype=torch.float32,
            device="cpu",
        )

        torch.testing.assert_close(packed_noise, expected, rtol=0, atol=0)

    def test_invalid_audio_layout_is_rejected(self):
        clean = torch.zeros((1, 32, 1, 7), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "stereo 2"):
            MODULE._fixed_noise_like(clean, noise_seed=0)

    def test_state_keeps_blend_operands_in_float32(self):
        state = MODULE._ScheduledAudioState(
            clean_cpu=torch.zeros((1, 32, 2, 7), dtype=torch.float32),
            noise_cpu=torch.ones((1, 32, 2, 7), dtype=torch.float32),
        )
        reference = torch.zeros((1, 32, 2, 7), dtype=torch.bfloat16)

        clean, noise = state.tensors_like(reference)

        self.assertEqual(clean.dtype, torch.float32)
        self.assertEqual(noise.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
