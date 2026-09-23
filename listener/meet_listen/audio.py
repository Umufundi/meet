"""Microphone capture, endpointing, and the per-segment quality evidence.

Capture never blocks on anything downstream. The PortAudio callback does one
thing, copy samples into a queue, and the WAV on disk is written from the raw
stream before any model sees it, so a crash in the ASR still leaves a complete
recording.

Endpointing is VAD-driven rather than fixed-window. Turn boundaries in a real
conversation are silences, so cutting there produces segments that are both
cheap to transcribe once (instead of re-transcribing a rolling buffer every
half second) and aligned with who is speaking, which is what the embedding
needs. A monologue is force-cut at `max_segment_s` so a long answer still
reaches the screen.
"""

from __future__ import annotations

import queue
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16_000
# Silero's contract at 16 kHz: exactly 512 samples per inference call.
FRAME = 512
FRAME_MS = FRAME / SAMPLE_RATE * 1000.0
CLIP_LEVEL = 0.99


@dataclass(frozen=True, slots=True)
class Segment:
    pcm: np.ndarray
    start_ms: int
    end_ms: int
    speech_s: float
    snr_db: float
    clipping: float
    rms: float


class NoiseFloor:
    """Rolling estimate of the room, taken only from frames the VAD calls silence.

    Estimating noise from everything, the usual shortcut, makes a loud talker
    look like a bad SNR and lets a quiet room look clean when the microphone is
    actually dead. Tracking silence alone keeps the number meaningful, which
    matters because SNR is one of the gates that decides whether a voice sample
    is allowed to become durable.
    """

    def __init__(self, initial: float = 1e-5, decay: float = 0.98) -> None:
        self.power = initial
        self.decay = decay

    def update(self, frame: np.ndarray) -> None:
        power = float(np.mean(frame.astype(np.float64) ** 2))
        self.power = self.decay * self.power + (1.0 - self.decay) * max(power, 1e-12)

    def snr_db(self, speech_power: float) -> float:
        if speech_power <= 0.0:
            return -99.0
        return float(10.0 * np.log10(speech_power / max(self.power, 1e-12)))


class Endpointer:
    """Accumulates VAD decisions into segments.

    `hangover_ms` is the silence that ends a turn. Too short and one sentence
    becomes three fragments, each too brief to embed reliably; too long and two
    speakers get glued into one segment, which is the worse error because the
    resulting vector belongs to nobody.
    """

    def __init__(
        self,
        *,
        hangover_ms: int = 700,
        min_segment_ms: int = 400,
        max_segment_s: float = 20.0,
        pre_roll_ms: int = 200,
    ) -> None:
        self.hangover_frames = max(1, int(hangover_ms / FRAME_MS))
        self.min_segment_samples = int(min_segment_ms / 1000.0 * SAMPLE_RATE)
        self.max_segment_samples = int(max_segment_s * SAMPLE_RATE)
        self.pre_roll_frames = max(1, int(pre_roll_ms / FRAME_MS))

        self.noise = NoiseFloor()
        self._pre_roll: list[np.ndarray] = []
        self._active: list[np.ndarray] = []
        self._speech_frames = 0
        self._silence_run = 0
        self._start_sample = 0
        self._cursor = 0

    def push(self, frame: np.ndarray, is_speech: bool) -> Segment | None:
        """Feed one 512-sample frame. Returns a segment when a turn ends."""
        self._cursor += frame.size

        if not self._active:
            if not is_speech:
                self.noise.update(frame)
                self._pre_roll.append(frame)
                del self._pre_roll[: -self.pre_roll_frames]
                return None
            # Speech started. The pre-roll recovers the onset the VAD clipped;
            # without it every segment loses its first consonant.
            self._active = [*self._pre_roll, frame]
            self._speech_frames = 1
            self._silence_run = 0
            self._start_sample = self._cursor - sum(f.size for f in self._active)
            self._pre_roll = []
            return None

        self._active.append(frame)
        if is_speech:
            self._speech_frames += 1
            self._silence_run = 0
        else:
            self._silence_run += 1
            self.noise.update(frame)

        active_samples = sum(f.size for f in self._active)
        ended = self._silence_run >= self.hangover_frames
        forced = active_samples >= self.max_segment_samples
        if not (ended or forced):
            return None
        return self._close(forced)

    def flush(self) -> Segment | None:
        """End of stream: emit whatever is still open."""
        return self._close(False) if self._active else None

    def _close(self, forced: bool) -> Segment | None:
        pcm = np.concatenate(self._active) if self._active else np.zeros(0, dtype=np.float32)
        start = self._start_sample
        speech_s = self._speech_frames * FRAME / SAMPLE_RATE

        self._active = []
        self._speech_frames = 0
        self._silence_run = 0
        # A forced cut is mid-sentence, so the tail doubles as the next segment's
        # pre-roll and no audio is lost across the boundary.
        self._pre_roll = [pcm[-self.pre_roll_frames * FRAME :]] if forced else []

        if pcm.size < self.min_segment_samples or speech_s <= 0.0:
            return None

        speech_power = float(np.mean(pcm.astype(np.float64) ** 2))
        return Segment(
            pcm=pcm,
            start_ms=int(start / SAMPLE_RATE * 1000),
            end_ms=int((start + pcm.size) / SAMPLE_RATE * 1000),
            speech_s=speech_s,
            snr_db=self.noise.snr_db(speech_power),
            clipping=float(np.mean(np.abs(pcm) >= CLIP_LEVEL)),
            rms=float(np.sqrt(speech_power)),
        )


class Microphone:
    """PortAudio input as a frame iterator, with an unbounded hand-off queue."""

    def __init__(self, device: int | None = None) -> None:
        import sounddevice as sd

        self._sd = sd
        self.device = device
        self.queue: queue.Queue[np.ndarray] = queue.Queue()
        self.overflows = 0
        self._stream: object | None = None

    def _callback(self, indata, _frames, _time, status) -> None:
        if status:
            self.overflows += 1
        # Copy: PortAudio reuses the buffer as soon as this returns.
        self.queue.put(indata[:, 0].copy())

    def __enter__(self) -> "Microphone":
        self._stream = self._sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME,
            device=self.device,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        return self

    def __exit__(self, *_exc) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    def frames(self, stop: "queue.Queue[bool]") -> object:
        while stop.empty():
            try:
                yield self.queue.get(timeout=0.25)
            except queue.Empty:
                continue


def describe_devices() -> str:
    import sounddevice as sd

    lines = ["index  ch  rate     name"]
    for index, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] < 1:
            continue
        lines.append(
            f"{index:>5}  {info['max_input_channels']:>2}  "
            f"{int(info['default_samplerate']):>6}  {info['name']}"
        )
    default = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else None
    lines.append(f"\ndefault input: {default}")
    return "\n".join(lines)


class FileSource:
    """Replay a recording through the identical path the microphone uses.

    Not a harness: a meeting that was recorded on a phone, or one whose live pass
    needs a second, unhurried run over the full audio, goes through here. Because
    it emits the same 512-sample frames as `Microphone`, every stage downstream
    (VAD, endpointing, quality, ASR, embedding) behaves exactly as it does live,
    so a discrepancy between the two paths is a real bug rather than a difference
    in plumbing.

    `realtime=True` paces playback to wall-clock so the terminal loop, questions
    included, can be exercised at the speed a human would actually experience.
    """

    def __init__(self, path: str, *, realtime: bool = False) -> None:
        self.path = path
        self.realtime = realtime

    def __enter__(self) -> "FileSource":
        return self

    def __exit__(self, *_exc) -> None:
        return None

    def frames(self, stop) -> object:
        import time

        import soundfile as sf

        with sf.SoundFile(self.path) as handle:
            if handle.samplerate != SAMPLE_RATE:
                raise ValueError(
                    f"{self.path} is {handle.samplerate} Hz; resample to {SAMPLE_RATE} Hz first"
                )
            period = FRAME / SAMPLE_RATE
            deadline = time.monotonic()
            while stop.empty():
                block = handle.read(FRAME, dtype="float32", always_2d=True)
                if block.shape[0] == 0:
                    return
                frame = block[:, 0]
                if frame.size < FRAME:
                    frame = np.pad(frame, (0, FRAME - frame.size))
                yield frame
                if self.realtime:
                    deadline += period
                    delay = deadline - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
