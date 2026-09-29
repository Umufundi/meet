"""A scripted stand-in for the real listener, for integration tests.

Speaks the same NDJSON protocol over the same process boundary, so the core's
supervisor, event pump, and shutdown path are exercised for real, without torch
or a microphone. The scenario is a JSON file named by FAKE_LISTENER_SCENARIO:
a list of lines to print verbatim (strings) or encode (objects).
"""

import argparse
import json
import os
import sys
import wave


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav")
    parser.add_argument("--file")
    parser.add_argument("--device")
    parser.add_argument("--asr-model")
    parser.add_argument("--realtime", action="store_true")
    args, _ = parser.parse_known_args()

    with wave.open(args.wav, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(b"\x00\x00" * 1600)

    print("loading pretend models", file=sys.stderr, flush=True)
    with open(os.environ["FAKE_LISTENER_SCENARIO"], encoding="utf-8") as handle:
        scenario = json.load(handle)
    for item in scenario:
        line = item if isinstance(item, str) else json.dumps(item)
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    if not args.file:
        for line in sys.stdin:  # live mode: run until told to stop
            if '"stop"' in line:
                break
    sys.stdout.write(json.dumps({"t": "stopped", "wav_path": args.wav, "duration_ms": 100}) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
