from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Mapping

import torch
import torch.nn.functional as F
import torchaudio

import comfy.nested_tensor
import comfy.patcher_extension
from comfy.ldm.minimax.model import time_shift_sigma


_WRAPPER_KEY = "minimax_h3_scheduled_audio_injection"
_STATE_KEY = "minimax_h3_scheduled_audio_injection_state"
_H3_AUDIO_RATE = 32_000
_H3_AUDIO_LATENT_RATE = 40
_LOGGER = logging.getLogger(__name__)


def _validate_audio(audio: Mapping[str, Any], name: str) -> tuple[torch.Tensor, int]:
    if not isinstance(audio, Mapping):
        raise ValueError(f"{name} must be a connected AUDIO value")
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if not isinstance(waveform, torch.Tensor) or sample_rate is None:
        raise ValueError(f"{name} is missing waveform or sample_rate")
    if waveform.ndim != 3 or waveform.shape[0] != 1:
        raise ValueError(
            f"{name} must have shape [1, channels, samples], got {tuple(waveform.shape)}"
        )
    if waveform.shape[1] < 1 or waveform.shape[2] < 1:
        raise ValueError(f"{name} is empty")
    if not torch.isfinite(waveform).all():
        raise ValueError(f"{name} contains NaN or infinite samples")
    return waveform, int(sample_rate)


def _to_stereo(waveform: torch.Tensor) -> torch.Tensor:
    channels = int(waveform.shape[1])
    if channels == 2:
        return waveform
    if channels == 1:
        return waveform.expand(-1, 2, -1)
    return waveform.mean(dim=1, keepdim=True).expand(-1, 2, -1)


def _encode_for_template(audio_vae, audio: Mapping[str, Any], template: torch.Tensor) -> torch.Tensor:
    waveform, sample_rate = _validate_audio(audio, "drive_audio")
    vae_rate = int(getattr(audio_vae, "audio_sample_rate", _H3_AUDIO_RATE))
    if vae_rate != _H3_AUDIO_RATE:
        raise ValueError(f"MiniMax H3 audio VAE must run at 32000 Hz, got {vae_rate}")

    waveform = _to_stereo(waveform[:1]).to(dtype=torch.float32)
    if sample_rate != vae_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, vae_rate)

    samples_per_latent = vae_rate // _H3_AUDIO_LATENT_RATE
    target_samples = int(template.shape[-1]) * samples_per_latent
    if waveform.shape[-1] < target_samples:
        waveform = F.pad(waveform, (0, target_samples - waveform.shape[-1]))
    else:
        waveform = waveform[..., :target_samples]

    encoded = audio_vae.encode(waveform.movedim(1, -1).contiguous())
    if not isinstance(encoded, torch.Tensor) or encoded.ndim != 4:
        raise ValueError("The audio VAE did not return [B, 32, 2, T] latent data")
    if tuple(encoded.shape) != tuple(template.shape):
        raise ValueError(
            "Encoded audio does not match the H3 target grid: "
            f"encoded={tuple(encoded.shape)}, target={tuple(template.shape)}"
        )
    return encoded.to(device=template.device, dtype=template.dtype)


@dataclass
class _ScheduledAudioState:
    clean_cpu: torch.Tensor
    noise_cpu: torch.Tensor
    debug: bool = False
    calls: int = 0

    def tensors_like(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tuple(reference.shape) != tuple(self.clean_cpu.shape):
            raise RuntimeError(
                "Scheduled audio shape changed after graph preparation: "
                f"runtime={tuple(reference.shape)}, prepared={tuple(self.clean_cpu.shape)}"
            )
        clean = self.clean_cpu.to(device=reference.device, dtype=reference.dtype)
        noise = self.noise_cpu.to(device=reference.device, dtype=reference.dtype)
        return clean, noise


def _scheduled_audio_wrapper(
    executor,
    x,
    timestep,
    context,
    transformer_options=None,
    minimax_payload=None,
    **kwargs,
):
    options = transformer_options or {}
    state = options.get(_STATE_KEY)
    if not isinstance(state, _ScheduledAudioState):
        return executor(
            x,
            timestep,
            context,
            options,
            minimax_payload=minimax_payload,
            **kwargs,
        )
    if not isinstance(x, (list, tuple)) or len(x) != 2:
        raise RuntimeError("Scheduled audio injection requires a MiniMax H3 video/audio latent pair")

    video_x, audio_x = x
    inner = executor.class_obj
    if not hasattr(inner, "sigma_shift_video") or not hasattr(inner, "sigma_shift_audio"):
        raise RuntimeError("Scheduled audio injection was attached to a non-MiniMax-H3 model")

    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(0.0, 1.0)
    shift_v = float(options.get("minimax_h3_sigma_shift_video", inner.sigma_shift_video))
    shift_a = float(options.get("minimax_h3_sigma_shift_audio", inner.sigma_shift_audio))
    sigma_a = time_shift_sigma(sigma_v, shift_v, shift_a).clamp(0.0, 1.0)

    clean, fixed_noise = state.tensors_like(audio_x)
    scheduled_audio = (1.0 - sigma_a) * clean + sigma_a * fixed_noise
    if state.debug:
        state.calls += 1
        _LOGGER.warning(
            "[H3 scheduled injection] forward=%d sigma_video=%.6f sigma_audio=%.6f "
            "clean_shape=%s",
            state.calls,
            float(sigma_v),
            float(sigma_a),
            tuple(clean.shape),
        )
    return executor(
        [video_x, scheduled_audio],
        timestep,
        context,
        options,
        minimax_payload=minimax_payload,
        **kwargs,
    )


class MiniMaxH3ScheduledAudioInjection:
    """Drive H3 video with target audio at the audio schedule expected by each denoising step."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "av_latent": ("LATENT",),
                "audio_vae": ("VAE",),
                "drive_audio": ("AUDIO",),
                "noise_seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0x7FFFFFFFFFFFFFFF},
                ),
                "debug": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mux_audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("MODEL", "LATENT", "AUDIO")
    RETURN_NAMES = ("model", "av_latent", "mux_audio")
    FUNCTION = "inject"
    CATEGORY = "MiniMax H3/Audio"
    DESCRIPTION = (
        "Encodes supplied audio into H3 target-audio rows and replaces those rows at every "
        "denoising step with the same recording at H3's matching audio noise level. The optional "
        "mux_audio passes through unchanged for final delivery."
    )

    def inject(self, model, av_latent, audio_vae, drive_audio, noise_seed, debug=False, mux_audio=None):
        samples = av_latent.get("samples") if isinstance(av_latent, Mapping) else None
        if samples is None or not getattr(samples, "is_nested", False):
            raise ValueError("Scheduled audio injection requires a joint MiniMax H3 AV latent")
        parts = samples.unbind()
        if len(parts) != 2:
            raise ValueError(f"Expected exactly two H3 latent streams, got {len(parts)}")
        video, audio_template = parts
        if video.ndim != 5 or tuple(audio_template.shape[1:3]) != (32, 2):
            raise ValueError(
                "Unexpected H3 latent layout: "
                f"video={tuple(video.shape)}, audio={tuple(audio_template.shape)}"
            )
        if video.shape[0] != 1 or audio_template.shape[0] != 1:
            raise ValueError("MiniMax H3 scheduled audio injection supports batch size 1")

        clean = _encode_for_template(audio_vae, drive_audio, audio_template)
        clean_cpu = clean.detach().to(device="cpu", dtype=torch.float32).contiguous()
        generator = torch.Generator(device="cpu").manual_seed(int(noise_seed))
        fixed_noise = torch.randn(clean_cpu.shape, generator=generator, dtype=torch.float32)
        state = _ScheduledAudioState(
            clean_cpu=clean_cpu,
            noise_cpu=fixed_noise,
            debug=bool(debug),
        )
        if debug:
            _LOGGER.warning(
                "[H3 scheduled injection] prepared clean target shape=%s mean=%.6f std=%.6f",
                tuple(clean_cpu.shape),
                float(clean_cpu.mean()),
                float(clean_cpu.std()),
            )

        locked = dict(av_latent)
        locked["samples"] = comfy.nested_tensor.NestedTensor((video, clean))
        existing_mask = av_latent.get("noise_mask")
        if existing_mask is not None and getattr(existing_mask, "is_nested", False):
            mask_parts = existing_mask.unbind()
            video_mask = mask_parts[0] if mask_parts else torch.ones_like(video)
        else:
            video_mask = torch.ones_like(video)
        locked["noise_mask"] = comfy.nested_tensor.NestedTensor(
            (video_mask, torch.zeros_like(clean))
        )

        patched = model.clone()
        wrapper_type = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
        if patched.get_wrappers(wrapper_type, _WRAPPER_KEY):
            raise ValueError("Scheduled audio injection is already attached to this model")
        options = patched.model_options["transformer_options"] = (
            patched.model_options.get("transformer_options", {}).copy()
        )
        options[_STATE_KEY] = state
        patched.add_wrapper_with_key(wrapper_type, _WRAPPER_KEY, _scheduled_audio_wrapper)

        delivered_audio = mux_audio if mux_audio is not None else drive_audio
        _validate_audio(delivered_audio, "mux_audio")
        return patched, locked, delivered_audio


class MiniMaxH3DiffusersSchedule:
    """Build the exact shifted sigma grid used by the reference Diffusers H3 pipeline."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "num_inference_steps": (
                    "INT",
                    {"default": 20, "min": 2, "max": 1000},
                ),
            },
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "build"
    CATEGORY = "MiniMax H3/Sampling"
    DESCRIPTION = (
        "Matches MiniMaxH3Scheduler.set_timesteps(): num_inference_steps includes the terminal "
        "zero sigma, so 20 grid points produce 19 model evaluations."
    )

    def build(self, model, num_inference_steps):
        options = model.model_options.get("transformer_options", {})
        shift = options.get("minimax_h3_sigma_shift_video")
        if shift is None:
            model_sampling = model.get_model_object("model_sampling")
            shift = getattr(model_sampling, "shift", 12.0)
        shift = float(shift)
        if shift <= 0:
            raise ValueError(f"MiniMax H3 video sigma shift must be positive, got {shift}")

        base = torch.linspace(1.0, 0.0, int(num_inference_steps), dtype=torch.float32)
        sigmas = shift * base / (1.0 + (shift - 1.0) * base)
        return (torch.unique_consecutive(sigmas),)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3ScheduledAudioInjection": MiniMaxH3ScheduledAudioInjection,
    "MiniMaxH3DiffusersSchedule": MiniMaxH3DiffusersSchedule,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3ScheduledAudioInjection": "MiniMax H3 Scheduled Audio Injection",
    "MiniMaxH3DiffusersSchedule": "MiniMax H3 Diffusers Schedule",
}
