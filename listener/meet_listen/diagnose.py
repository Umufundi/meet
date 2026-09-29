"""One-shot checks the core runs before a meeting: `--probe`, `--prime`, `--mic-test`.

Each prints exactly one JSON object on stdout and exits. They exist so that
`meet setup` and `meet doctor` can ask the listener's own interpreter what it
has, instead of guessing from the outside: the core deliberately cannot import
torch, so only the listener can say whether torch imports.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from importlib import metadata

import numpy as np

PACKAGES = (
    "numpy",
    "torch",
    "torchaudio",
    "speechbrain",
    "faster-whisper",
    "ctranslate2",
    "silero-vad",
    "huggingface-hub",
    "sounddevice",
    "soundfile",
    "meet-listen",
)


def _versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _devices() -> tuple[list[dict], int | None, str | None]:
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        return [], None, f"audio backend unavailable: {exc}"
    try:
        found = sd.query_devices()
        from .audio import default_input

        default = default_input(sd)
    except Exception as exc:  # PortAudio raises its own error type
        return [], None, f"cannot enumerate audio devices: {exc}"
    devices = [
        {
            "index": index,
            "name": str(info["name"]),
            "channels": int(info["max_input_channels"]),
            "rate": int(info["default_samplerate"]),
        }
        for index, info in enumerate(found)
        if info["max_input_channels"] > 0
    ]
    return devices, default, None


def probe() -> dict:
    """Import the stack without loading weights and report what is there."""
    errors: list[str] = []
    for module in ("torch", "torchaudio", "speechbrain.inference.speaker", "faster_whisper", "silero_vad"):
        try:
            __import__(module)
        except Exception as exc:
            errors.append(f"import {module}: {exc}")
    devices, default, audio_error = _devices()
    from .models import EMBED_MODEL_ID

    return {
        "ok": not errors,
        "errors": errors,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "packages": _versions(),
        "embed_model": EMBED_MODEL_ID,
        "devices": devices,
        "default_device": default,
        "audio_error": audio_error,
    }


def _snapshots() -> dict[str, str]:
    """Which hub snapshot each cached model resolved to."""
    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir()
    except Exception:
        return {}
    out: dict[str, str] = {}
    for repo in info.repos:
        newest = max(repo.revisions, key=lambda r: r.last_modified, default=None)
        if newest is not None:
            out[repo.repo_id] = newest.commit_hash
    return out


def prime(asr_model: str, compute_type: str = "int8") -> dict:
    """Load all three models once, downloading as needed, and time each."""
    from .models import EMBED_MODEL_ID, EMBED_SOURCE, Embedder, Silero, Transcriber

    timings: dict[str, float] = {}
    try:
        t = time.monotonic()
        vad = Silero()
        vad(np.zeros(512, dtype=np.float32))
        timings["vad"] = round(time.monotonic() - t, 2)

        t = time.monotonic()
        transcriber = Transcriber(asr_model, compute_type)
        transcriber(np.zeros(16_000, dtype=np.float32))
        timings["asr"] = round(time.monotonic() - t, 2)

        t = time.monotonic()
        embedder = Embedder()
        # A deterministic tone: enough samples to embed, no speech needed.
        tone = 0.1 * np.sin(np.linspace(0, 2 * np.pi * 220, 16_000, dtype=np.float32))
        vector = embedder(tone)
        timings["embed"] = round(time.monotonic() - t, 2)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "timings": timings}

    return {
        "ok": vector is not None,
        "error": None if vector is not None else "embedding model produced no vector",
        "asr_model": asr_model,
        "embed_model": EMBED_MODEL_ID,
        "embed_source": EMBED_SOURCE,
        "embed_dim": int(vector.size) if vector is not None else 0,
        "snapshots": _snapshots(),
        "packages": _versions(),
        "timings": timings,
        "hf_home": os.environ.get("HF_HOME"),
    }


def mic_test(seconds: float, device: int | None) -> dict:
    """Record from the microphone and describe the signal.

    A stream that opens but delivers pure digital silence is the signature of
    Windows' microphone privacy switch (and of a hardware mute key), which is
    why "all zeros" is reported separately from "quiet".
    """
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        return {"ok": False, "error": f"audio backend unavailable: {exc}"}
    from .audio import FRAME, SAMPLE_RATE

    try:
        info = sd.query_devices(device, kind="input")
        pcm = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                     device=device, blocking=True)[:, 0]
    except Exception as exc:
        return {"ok": False, "error": f"cannot open microphone: {exc}"}

    frames = pcm[: pcm.size - pcm.size % FRAME].reshape(-1, FRAME)
    powers = np.mean(frames.astype(np.float64) ** 2, axis=1) if frames.size else np.zeros(1)
    floor = float(np.percentile(powers, 10))
    loud = float(np.percentile(powers, 90))
    return {
        "ok": True,
        "device": str(info["name"]),
        "seconds": seconds,
        "rms": float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2))) if pcm.size else 0.0,
        "peak": float(np.max(np.abs(pcm))) if pcm.size else 0.0,
        "clipping": float(np.mean(np.abs(pcm) >= 0.99)) if pcm.size else 0.0,
        "all_zero": bool(pcm.size and not np.any(pcm)),
        # Spread between the loud and quiet tenths of the recording: a rough
        # dynamic range, meaningful only if someone spoke during the test.
        "range_db": float(10 * np.log10(max(loud, 1e-12) / max(floor, 1e-12))),
        "noise_dbfs": float(10 * np.log10(max(floor, 1e-12))),
    }


def run(mode: str, args) -> int:
    import json

    if mode == "probe":
        result = probe()
    elif mode == "prime":
        result = prime(args.asr_model, args.compute_type)
    else:
        result = mic_test(args.mic_test, args.device)
    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()
    return 0 if result.get("ok") else 1
