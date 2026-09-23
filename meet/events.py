"""The sidecar wire protocol: one JSON object per line, sidecar stdout -> core stdin.

Perception and judgment are split across a process boundary on purpose (see the
note in pyproject.toml). Everything that crosses it is described here, so the
listener can be swapped for a different ASR or embedding stack without the core
learning anything new. Unknown event kinds are dropped rather than raising: a
newer sidecar must never crash an older core mid-meeting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

SCHEMA = 1


@dataclass(frozen=True, slots=True)
class Quality:
    """Per-segment audio evidence, computed in the sidecar where the PCM lives.

    These decide whether a segment may become a durable voice sample. A poisoned
    voice profile is far more expensive than a missed enrollment, so every gate
    here fails closed.
    """

    speech_s: float
    snr_db: float
    clipping: float
    rms: float

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> "Quality":
        return cls(
            speech_s=float(raw.get("speech_s", 0.0)),
            snr_db=float(raw.get("snr_db", 0.0)),
            clipping=float(raw.get("clipping", 1.0)),
            rms=float(raw.get("rms", 0.0)),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "speech_s": self.speech_s,
            "snr_db": self.snr_db,
            "clipping": self.clipping,
            "rms": self.rms,
        }


@dataclass(frozen=True, slots=True)
class Ready:
    sample_rate: int
    device: str
    asr_model: str
    embed_model: str
    wav_path: str


@dataclass(frozen=True, slots=True)
class Level:
    """Input meter. Emitted even during silence so a dead microphone is visible."""

    rms: float
    peak: float


@dataclass(frozen=True, slots=True)
class Partial:
    """Provisional text for the tail of the buffer. Replaced, never accumulated."""

    text: str
    start_ms: int


@dataclass(frozen=True, slots=True)
class Utterance:
    """A settled span of speech with its voice vector.

    `embedding` is L2-normalized in the sidecar, so cosine similarity in the core
    is a plain dot product. `model_id` travels with it because embeddings from
    different models are not comparable and must never be silently mixed.
    """

    seq: int
    start_ms: int
    end_ms: int
    text: str
    no_speech: float
    embedding: list[float]
    dim: int
    model_id: str
    quality: Quality


@dataclass(frozen=True, slots=True)
class SidecarError:
    message: str
    fatal: bool = False


@dataclass(frozen=True, slots=True)
class Stopped:
    wav_path: str
    duration_ms: int


Event = Ready | Level | Partial | Utterance | SidecarError | Stopped


@dataclass(frozen=True, slots=True)
class Unknown:
    """A line the core does not understand. Kept so it can be logged, not raised."""

    raw: dict[str, Any] = field(default_factory=dict)


def parse_line(line: str) -> Event | Unknown | None:
    """Decode one NDJSON line. Returns None for blank lines and undecodable junk."""
    line = line.strip()
    if not line:
        return None
    try:
        raw = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    kind = raw.get("t")
    if kind == "ready":
        return Ready(
            sample_rate=int(raw.get("sample_rate", 16000)),
            device=str(raw.get("device", "")),
            asr_model=str(raw.get("asr_model", "")),
            embed_model=str(raw.get("embed_model", "")),
            wav_path=str(raw.get("wav_path", "")),
        )
    if kind == "level":
        return Level(rms=float(raw.get("rms", 0.0)), peak=float(raw.get("peak", 0.0)))
    if kind == "partial":
        return Partial(text=str(raw.get("text", "")), start_ms=int(raw.get("start_ms", 0)))
    if kind == "utterance":
        embedding = raw.get("embedding") or []
        return Utterance(
            seq=int(raw.get("seq", 0)),
            start_ms=int(raw.get("start_ms", 0)),
            end_ms=int(raw.get("end_ms", 0)),
            text=str(raw.get("text", "")),
            no_speech=float(raw.get("no_speech", 0.0)),
            embedding=[float(v) for v in embedding],
            dim=int(raw.get("dim", len(embedding))),
            model_id=str(raw.get("model_id", "")),
            quality=Quality.parse(raw.get("quality") or {}),
        )
    if kind == "error":
        return SidecarError(message=str(raw.get("message", "")), fatal=bool(raw.get("fatal", False)))
    if kind == "stopped":
        return Stopped(
            wav_path=str(raw.get("wav_path", "")),
            duration_ms=int(raw.get("duration_ms", 0)),
        )
    return Unknown(raw=raw)


def encode(obj: dict[str, Any]) -> str:
    """Core -> sidecar. Only `{"t": "stop"}` exists today; the frame is kept general."""
    return json.dumps(obj, separators=(",", ":")) + "\n"
