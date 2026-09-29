from meet.config import POLICY
from meet.identity.policy import Action, decide, should_consult_jev
from meet.memory.people import Candidate, MatchResult


def cand(slug, sim):
    return Candidate(slug, slug.title(), sim, sim, 5, 2)


def result(*pairs):
    return MatchResult([cand(s, v) for s, v in pairs], "m")


def test_pinned_cluster_wins_over_everything():
    d = decide(result(("sarah", 0.99)), pinned_slug="marcus", pinned_name="Marcus")
    assert d.action is Action.AUTO and d.slug == "marcus" and d.source == "human"


def test_nothing_to_propose_asks():
    d = decide(result())
    assert d.action is Action.ASK and d.slug is None


def test_clear_match_with_lead_is_auto():
    d = decide(result(("marcus", 0.90), ("sarah", 0.70)))
    assert d.action is Action.AUTO and d.slug == "marcus"


def test_high_score_thin_margin_asks():
    """The confidently-wrong trap: two high scores are a question, not an answer."""
    d = decide(result(("marcus", 0.90), ("sarah", 0.87)))
    assert d.action is Action.ASK and d.slug is None and d.confidence == 0.90


def test_middle_band_is_provisional():
    d = decide(result(("marcus", 0.75)))
    assert d.action is Action.PROVISIONAL and d.slug == "marcus"


def test_below_provisional_asks():
    assert decide(result(("marcus", 0.68))).action is Action.ASK


def test_thresholds_are_on_the_cosine_scale():
    assert 0.5 < POLICY.match_min_similarity < POLICY.provisional_similarity < POLICY.auto_similarity < 0.9
    loose = POLICY.first_meeting()
    assert loose.auto_similarity < POLICY.auto_similarity


def test_jev_only_for_genuine_ties():
    tie = decide(result(("marcus", 0.90), ("sarah", 0.87)))
    assert should_consult_jev(tie)
    assert not should_consult_jev(decide(result()))
    assert not should_consult_jev(decide(result(("marcus", 0.90), ("sarah", 0.70))))
