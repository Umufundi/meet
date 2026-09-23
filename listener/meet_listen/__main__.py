"""Sidecar entry point: microphone in, NDJSON out.

Threads, and why each exists:

  PortAudio callback  ->  capture queue        (never blocks, never computes)
  reader thread       ->  WAV on disk + VAD + endpointing
  worker thread       ->  transcribe + embed settled segments
  main thread         ->  stdin control, shutdown

Transcription runs off the capture path so a slow segment cannot drop audio: a
backlog delays lines on screen, which is recoverable, while a dropped frame is
gone from the recording forever. The WAV is written from the raw stream before
any model touches it, for the same reason.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time

import numpy as np

from .audio import FRAME, SAMPLE_RATE, Endpointer, FileSource, Microphone, Segment, describe_devices


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="meet_listen")
    parser.add_argument("--devices", action="store_true", help="list input devices and exit")
    parser.add_argument("--wav", help="path to write the raw recording")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--file", help="replay this 16 kHz mono WAV instead of the microphone")
    parser.add_argument("--realtime", action="store_true", help="pace --file to wall clock")
    parser.add_argument("--asr-model", default="small.en")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    args = parser.parse_args(argv)

    if args.devices:
        print(describe_devices())
        return 0
    if not args.wav:
        parser.error("--wav is required")

    import soundfile as sf

    from .models import EMBED_MODEL_ID, Embedder, Silero, Transcriber

    try:
        vad = Silero(threshold=args.vad_threshold)
        transcriber = Transcriber(args.asr_model, args.compute_type)
        embedder = Embedder()
    except Exception as exc:  # model load is the one failure worth reporting verbatim
        emit({"t": "error", "message": f"model load failed: {exc}", "fatal": True})
        return 1

    stop: queue.Queue[bool] = queue.Queue()
    work: queue.Queue[Segment | None] = queue.Queue()
    counter = {"seq": 0}
    started = time.monotonic()

    def worker() -> None:
        """Turn settled segments into utterance events. One at a time, in order."""
        while True:
            segment = work.get()
            if segment is None:
                return
            try:
                result = transcriber(segment.pcm)
                if not result.text:
                    continue
                vector = embedder(segment.pcm)
                counter["seq"] += 1
                emit(
                    {
                        "t": "utterance",
                        "seq": counter["seq"],
                        "start_ms": segment.start_ms,
                        "end_ms": segment.end_ms,
                        "text": result.text,
                        "no_speech": round(result.no_speech, 4),
                        "embedding": [round(float(v), 6) for v in vector] if vector is not None else [],
                        "dim": int(vector.size) if vector is not None else 0,
                        "model_id": EMBED_MODEL_ID,
                        "quality": {
                            "speech_s": round(segment.speech_s, 3),
                            "snr_db": round(segment.snr_db, 2),
                            "clipping": round(segment.clipping, 5),
                            "rms": round(segment.rms, 6),
                        },
                    }
                )
            except Exception as exc:
                # One bad segment must not end the meeting.
                emit({"t": "error", "message": f"segment failed: {exc}"})

    def reader(mic: Microphone, wav) -> None:
        endpointer = Endpointer()
        meter = 0.0
        last_meter = 0.0
        for frame in mic.frames(stop):
            if frame.size != FRAME:
                # PortAudio can hand back a short final block; pad so Silero's
                # fixed-size contract holds rather than dropping the audio.
                frame = np.pad(frame, (0, max(0, FRAME - frame.size)))[:FRAME]
            wav.write(frame)
            meter = max(meter, float(np.max(np.abs(frame))) if frame.size else 0.0)
            now = time.monotonic()
            if now - last_meter >= 0.25:
                emit(
                    {
                        "t": "level",
                        "rms": round(float(np.sqrt(np.mean(frame**2))), 5),
                        "peak": round(meter, 5),
                    }
                )
                meter = 0.0
                last_meter = now
            segment = endpointer.push(frame, vad(frame))
            if segment is not None:
                work.put(segment)
        tail = endpointer.flush()
        if tail is not None:
            work.put(tail)

    def control() -> None:
        """Watch stdin for the stop command.

        For live capture, EOF also stops: the parent owning that pipe has gone
        away, so the meeting is over whether or not it said so. For file replay
        the end of the file is the natural terminator, and EOF on an inherited
        stdin (a closed descriptor, a non-interactive shell) must not end the run
        before a single frame has been read.
        """
        for line in sys.stdin:
            try:
                if json.loads(line).get("t") == "stop":
                    stop.put(True)
                    return
            except (json.JSONDecodeError, AttributeError):
                continue
        if not args.file:
            stop.put(True)

    worker_thread = threading.Thread(target=worker, daemon=True, name="asr")
    worker_thread.start()
    threading.Thread(target=control, daemon=True, name="control").start()

    try:
        with sf.SoundFile(
            args.wav, mode="w", samplerate=SAMPLE_RATE, channels=1, subtype="PCM_16"
        ) as wav, (FileSource(args.file, realtime=args.realtime) if args.file else Microphone(args.device)) as mic:
            emit(
                {
                    "t": "ready",
                    "sample_rate": SAMPLE_RATE,
                    "device": args.file or str(args.device if args.device is not None else "default"),
                    "asr_model": args.asr_model,
                    "embed_model": EMBED_MODEL_ID,
                    "wav_path": args.wav,
                }
            )
            reader(mic, wav)
    except Exception as exc:
        emit({"t": "error", "message": f"capture failed: {exc}", "fatal": True})
        stop.put(True)

    work.put(None)
    # Drain the backlog: the last things said are usually the ones that matter.
    worker_thread.join(timeout=120)
    emit(
        {
            "t": "stopped",
            "wav_path": args.wav,
            "duration_ms": int((time.monotonic() - started) * 1000),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
