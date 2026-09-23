"""When to label, when to hedge, and when to ask the human.

The product's success metric is questions-per-hour falling while confidently
wrong labels stay at zero. Those two pull in opposite directions, and this module
is where the tension is resolved. Three rules:

1. Score alone never decides. Sarah .84 against James .82 is a coin flip with
   a high number printed on it. A thin lead is a question.
2. A pinned cluster outranks the model. Once a human has said who a cluster is,
   later utterances in that cluster are labelled from the pin, not re-litigated.
3. Silence is not a label. When nothing clears the floor, the speaker renders as
   `?` and the question goes on screen. Guessing to avoid an empty cell is the
   one failure mode that destroys trust in the transcript.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..config import POLICY, Policy
from ..memory.people import Candidate, MatchResult


class Action(Enum):
    AUTO = "auto"            # label it, no human involvement
    PROVISIONAL = "provisional"  # label it, mark it soft, re-check at reconciliation
    ASK = "ask"              # put the question on screen, keep recording


@dataclass(frozen=True, slots=True)
class Decision:
    action: Action
    slug: str | None
    name: str | None
    confidence: float
    margin: float
    reason: str
    source: str = "auto"
    candidates: tuple[Candidate, ...] = ()

    @property
    def is_labelled(self) -> bool:
        return self.slug is not None and self.action is not Action.ASK


def decide(
    result: MatchResult,
    *,
    pinned_slug: str | None = None,
    pinned_name: str | None = None,
    policy: Policy = POLICY,
) -> Decision:
    """Turn ranked voice evidence into one of three outcomes."""
    candidates = tuple(result.candidates)

    if pinned_slug is not None:
        # Rule 2. The human already answered this question for this cluster.
        return Decision(
            action=Action.AUTO,
            slug=pinned_slug,
            name=pinned_name or pinned_slug,
            confidence=1.0,
            margin=1.0,
            reason="cluster was named by a human",
            source="human",
            candidates=candidates,
        )

    best = result.best
    if best is None:
        return Decision(
            action=Action.ASK,
            slug=None,
            name=None,
            confidence=0.0,
            margin=0.0,
            reason="no stored voice is close enough to propose",
            candidates=candidates,
        )

    margin = result.margin
    if best.similarity >= policy.auto_similarity and margin >= policy.min_margin:
        return Decision(
            Action.AUTO,
            best.slug,
            best.name,
            best.similarity,
            margin,
            "clear match with a clear lead",
            candidates=candidates,
        )
    if best.similarity >= policy.auto_similarity:
        # Rule 1: high score, no daylight. This is the case a naive threshold
        # gets wrong, and the one that produces a confidently wrong transcript.
        return Decision(
            Action.ASK,
            None,
            None,
            best.similarity,
            margin,
            f"{best.name} and the runner-up are {margin:.2f} apart",
            candidates=candidates,
        )
    if best.similarity >= policy.provisional_similarity and margin >= policy.min_margin:
        return Decision(
            Action.PROVISIONAL,
            best.slug,
            best.name,
            best.similarity,
            margin,
            "likely, pending a second look",
            candidates=candidates,
        )
    return Decision(
        Action.ASK,
        None,
        None,
        best.similarity,
        margin,
        "best match is below the asking threshold",
        candidates=candidates,
    )


def should_consult_jev(decision: Decision, policy: Policy = POLICY) -> bool:
    """Jev is worth a round trip only where voice evidence is genuinely split.

    Not when nothing matches (there is nothing to arbitrate, only a name to
    collect), and not when one candidate already leads clearly.
    """
    return decision.action is Action.ASK and len(decision.candidates) >= 2
