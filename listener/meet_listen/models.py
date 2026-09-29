"""The two models the sidecar owns: what was said, and who it sounds like.

Both are loaded once and reused. The embedding model id travels with every
vector because embeddings from different models occupy different spaces and a
silent mix would corrupt every stored profile at once; the core refuses to
compare across ids, and this is where the id comes from.

Model choice is deliberately the ungated path. speechbrain/spkrec-ecapa-voxceleb
is Apache-2.0 and needs no Hugging Face terms acceptance, so the tool installs
and runs without an account. The gated pyannote segmentation models buy
overlap-aware diarization, which matters for cross-talk, and are worth adding
behind a flag later; they are not worth making a precondition for hearing
anything at all.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

EMBED_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
EMBED_MODEL_ID = "ecapa-voxceleb@sb1"
EMBED_DIM = 192

# Hugging Face revisions to load. None means "whatever the hub serves", which is
# only acceptable because `meet setup` records the snapshot it actually fetched
# in the runtime manifest and every later run is offline. Set a commit hash here
# to make a release load exactly one snapshot on every machine.
WHISPER_REVISION: str | None = os.environ.get("MEET_WHISPER_REVISION") or None
EMBED_REVISION: str | None = os.environ.get("MEET_EMBED_REVISION") or None


def offline() -> bool:
    """Models were primed by `meet setup`; never touch the network again."""
    return os.environ.get("MEET_MODELS_OFFLINE") == "1"


def _log(message: str) -> None:
    # stdout is the NDJSON channel and must stay clean.
    print(message, file=sys.stderr, flush=True)


@dataclass(frozen=True, slots=True)
class Transcription:
    text: str
    no_speech: float


class Transcriber:
    """faster-whisper over one settled segment at a time.

    Segments arrive already endpointed, so `vad_filter` is off: running a second
    VAD over audio that was cut at a silence only trims real speech. Beam size 1
    keeps latency predictable, which matters more here than the last point of
    word accuracy, because a late line is a line the human cannot correct in time.
    """

    def __init__(self, model_size: str = "small.en", compute_type: str = "int8") -> None:
        from faster_whisper import WhisperModel

        _log(f"loading whisper: {model_size} ({compute_type})")
        self.model_size = model_size
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type=compute_type,
            local_files_only=offline(),
            revision=WHISPER_REVISION,
        )

    def __call__(self, pcm: np.ndarray, *, prompt: str | None = None) -> Transcription:
        segments, _info = self.model.transcribe(
            pcm.astype(np.float32),
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=prompt,
        )
        parts: list[str] = []
        no_speech = 0.0
        for segment in segments:
            parts.append(segment.text)
            no_speech = max(no_speech, float(getattr(segment, "no_speech_prob", 0.0) or 0.0))
        return Transcription(text=" ".join(p.strip() for p in parts).strip(), no_speech=no_speech)


class Embedder:
    """ECAPA-TDNN speaker embeddings, L2-normalized at the boundary.

    Normalizing here rather than in the core means cosine similarity downstream
    is a dot product and cannot drift depending on who computed it.
    """

    def __init__(self) -> None:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        self.torch = torch
        cache = os.environ.get("MEET_MODEL_CACHE") or os.path.expanduser("~/.meet/models/ecapa")
        _log(f"loading speaker embedding: {EMBED_SOURCE}")
        # Copy rather than symlink out of the hub cache: creating symlinks on
        # Windows needs Developer Mode or admin, and fails with "a required
        # privilege is not held by the client" on an ordinary laptop.
        extra: dict = {}
        try:
            from speechbrain.utils.fetching import FetchConfig, LocalStrategy

            extra = {
                "local_strategy": LocalStrategy.COPY,
                "fetch_config": FetchConfig(revision=EMBED_REVISION, allow_network=not offline()),
            }
        except ImportError:  # speechbrain < 1.0
            pass
        self.model = EncoderClassifier.from_hparams(
            source=EMBED_SOURCE,
            savedir=cache,
            run_opts={"device": "cpu"},
            **extra,
        )
        self.model.eval()

    def __call__(self, pcm: np.ndarray) -> np.ndarray | None:
        if pcm.size < 8000:  # under half a second carries no reliable identity
            return None
        with self.torch.no_grad():
            wav = self.torch.from_numpy(pcm.astype(np.float32)).unsqueeze(0)
            vector = self.model.encode_batch(wav).squeeze().cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 1e-12:
            return None
        return vector / norm


class Silero:
    """Frame-level voice activity, 512 samples at a time."""

    def __init__(self, threshold: float = 0.5) -> None:
        import torch
        from silero_vad import load_silero_vad

        self.torch = torch
        self.threshold = threshold
        _log("loading silero vad")
        self.model = load_silero_vad()

    def __call__(self, frame: np.ndarray) -> bool:
        with self.torch.no_grad():
            probability = self.model(self.torch.from_numpy(frame.astype(np.float32)), 16_000).item()
        return probability >= self.threshold

    def reset(self) -> None:
        if hasattr(self.model, "reset_states"):
            self.model.reset_states()
