# ComfyUI H3 Scheduled Audio Injection

This custom node ports the `scheduled` replacement method from
`h3-audio-injection` to ComfyUI's native MiniMax H3 model wrapper API.

`MiniMaxH3ScheduledAudioInjection`:

- encodes `drive_audio` into the target H3 audio grid;
- pads short conditioning audio with waveform silence before VAE encoding;
- injects `x_t = (1 - sigma_audio) * x0 + sigma_audio * fixed_noise` before every H3 forward;
- uses a zero audio noise mask as the hard final-latent preservation layer;
- optionally passes a separate untouched `mux_audio` to the video output.

The separate mux input is useful when a vocal stem drives lip movement while the original mix is
the delivered soundtrack.

## Install

Copy this folder directly into `ComfyUI/custom_nodes/` and restart ComfyUI. Do not place it inside
another custom-node folder. It uses only PyTorch, torchaudio, and ComfyUI's built-in APIs.

## Scope

The node supports current native MiniMax H3 joint AV latents, batch size 1, and the 32 kHz stereo
H3 audio VAE. It intentionally implements scheduled injection only. It does not patch ComfyUI core.

Use 24 fps for H3 output. Render on H3's `17n + 5` frame grid, then trim decoded frames to the
requested soundtrack duration before the final mux.
