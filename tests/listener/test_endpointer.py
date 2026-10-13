"""The listener's pure-numpy stages: endpointing and the quality evidence
that gates learning. No torch, no microphone."""

import numpy as np
from meet_listen.audio import FRAME, SAMPLE_RATE, Endpointer, NoiseFloor

FRAMES_PER_S = SAMPLE_RATE // FRAME  # 31


def frames(seconds, level, rng):
    n = int(seconds * FRAMES_PER_S)
    return [(rng.standard_normal(FRAME) * level).astype(np.float32) for _ in range(n)]


def run(ep, schedule, rng):
    out = []
    for seconds, level, speech in schedule:
        for f in frames(seconds, level, rng):
            seg = ep.push(f, speech)
            if seg is not None:
                out.append(seg)
    return out


def test_one_turn_is_one_segment_with_honest_quality():
    rng = np.random.default_rng(0)
    ep = Endpointer()
    segs = run(ep, [(1.0, 0.001, False), (2.0, 0.1, True), (1.5, 0.001, False)], rng)
    assert len(segs) == 1
    seg = segs[0]
    assert 1.8 <= seg.speech_s <= 2.1
    # Starts with the pre-roll just before speech onset, not at zero.
    assert 700 <= seg.start_ms <= 1000
    assert seg.snr_db > 30
    assert seg.clipping == 0.0
    assert seg.end_ms > seg.start_ms


def test_a_blip_is_not_a_segment():
    rng = np.random.default_rng(1)
    segs = run(Endpointer(), [(0.5, 0.001, False), (0.05, 0.1, True), (1.0, 0.001, False)], rng)
    assert segs == []


def test_short_reply_survives_blip_filter():
    rng = np.random.default_rng(11)
    segs = run(Endpointer(), [(0.5, 0.001, False), (0.16, 0.1, True), (1.0, 0.001, False)], rng)
    assert len(segs) == 1
    assert 0.09 <= segs[0].speech_s <= 0.2


def test_monologue_is_force_cut_without_losing_audio():
    rng = np.random.default_rng(2)
    ep = Endpointer(max_segment_s=5.0)
    segs = run(ep, [(12.0, 0.1, True)], rng)
    tail = ep.flush()
    assert len(segs) == 2 and tail is not None
    assert all(s.end_ms - s.start_ms <= 5100 for s in segs)
    # Forced cuts overlap by the pre-roll: the next segment begins at or before
    # the previous one ended, so no audio falls between them.
    assert segs[1].start_ms <= segs[0].end_ms


def test_clipping_is_measured():
    rng = np.random.default_rng(3)
    loud = [np.clip(f * 50, -1, 1) for f in frames(2.0, 0.1, rng)]
    ep = Endpointer()
    for f in loud:
        ep.push(f, True)
    seg = ep.flush()
    assert seg.clipping > 0.05


def test_noise_floor_only_learns_from_silence():
    floor = NoiseFloor()
    for _ in range(200):
        floor.update(np.full(FRAME, 0.01, dtype=np.float32))
    assert abs(floor.snr_db(0.01**2)) < 0.5
    assert floor.snr_db(0.0) == -99.0
