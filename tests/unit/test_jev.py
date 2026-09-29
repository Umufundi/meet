import json

import httpx
import pytest

from meet.identity import credentials, jev
from meet.memory.people import Candidate

CANDS = [Candidate("marcus", "Marcus", 0.86, 0.85, 9, 3), Candidate("sarah", "Sarah", 0.84, 0.83, 7, 2)]


def ask(handler):
    return jev.ask(utterance="I'll send it", candidates=CANDS, cluster_key="3", cluster_similarity=0.8,
                   attendees=["Marcus", "Sarah"], recent=[("?", "hello")],
                   client=httpx.Client(transport=httpx.MockTransport(handler)))


def answer(choice="marcus", probs=None, confidence=0.95):
    probs = probs or {"marcus": 0.95, "sarah": 0.04, "UNKNOWN": 0.01}
    return httpx.Response(200, json={"answers": {"speaker": {
        "choice": choice, "probabilities": probs, "confidence": confidence}}, "latency_ms": 42})


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")


def test_no_key_means_unavailable_not_crash():
    assert jev.api_key() is None
    with pytest.raises(jev.JevUnavailable):
        ask(lambda r: answer())


def test_valid_answer(key):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers["authorization"]
        return answer()

    got = ask(handler)
    assert got.slug == "marcus" and got.latency_ms == 42
    assert seen["auth"] == "Bearer k"
    # UNKNOWN is always an option, and only voice-plausible people are offered.
    assert set(seen["body"]["questions"]["speaker"]["criteria"]) == {"marcus", "sarah", "UNKNOWN"}


def test_unknown_choice(key):
    got = ask(lambda r: answer("UNKNOWN", {"marcus": 0.2, "sarah": 0.2, "UNKNOWN": 0.6}, 0.6))
    assert got.slug is None


@pytest.mark.parametrize("response", [
    httpx.Response(500),
    httpx.Response(200, json={}),
    httpx.Response(200, json={"answers": {"speaker": {
        "choice": "james", "probabilities": {}, "confidence": 1}}}),
    answer(probs={"marcus": 0.9, "sarah": 0.9, "UNKNOWN": 0.9}),          # does not sum to 1
    answer("sarah"),                                                      # choice is not the max
    httpx.Response(200, content=b'{"answers": {"speaker": {"choice": "marcus", "confidence": NaN, '
                                b'"probabilities": {"marcus": 0.95, "sarah": 0.04, "UNKNOWN": 0.01}}}}'),
])
def test_garbage_is_rejected(key, response):
    with pytest.raises(jev.JevUnavailable):
        ask(lambda r: response)


def test_connection_failure_is_unavailable(key):
    def boom(request):
        raise httpx.ConnectError("no route to host")

    with pytest.raises(jev.JevUnavailable):
        ask(boom)


def test_key_from_os_store_is_used_once(monkeypatch):
    calls = []
    monkeypatch.setattr(credentials, "lookup", lambda service: calls.append(service) or "stored")
    jev._stored_key.cache_clear()
    assert jev.api_key() == "stored"
    assert jev.api_key() == "stored"
    assert calls == ["TypeSafe API Key"]
