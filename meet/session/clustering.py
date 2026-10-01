"""Online speaker clustering inside one meeting.

A cluster is "this voice, whoever it turns out to be". It exists so the product
can show a coherent speaker before anyone has named them, and so that naming
them later costs one write. Clusters are deliberately cheap and wrong-able: an
over-split costs one `:merge`, while an over-merge silently attributes one
person's words to another, so the join threshold leans toward splitting.

The clustering itself is the incremental-centroid scheme from
KmingLyu/real-time-speaker-diarization-asr, kept because it is the right shape
for a live loop: no global re-clustering, no re-labelling of settled history,
O(clusters) per utterance. What is added here is the part that matters for this
product: a cluster a human has named is *pinned*, and a pinned cluster's centroid
stops drifting on automatic evidence alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import POLICY, Policy

# Where a line goes when its voice fits no cluster and may not start one. It is
# never a Cluster, never named, and never asked about; its lines render as `?`.
UNPLACED = "0"


@dataclass
class Cluster:
    key: str
    # Sum of member vectors; the centroid is this normalized. Kept as a sum so a
    # new member is one addition rather than a re-average over history.
    total: np.ndarray
    count: int = 0
    person_slug: str | None = None
    # True once a human said who this is. Automatic evidence may no longer move
    # the identity, and only human-trust vectors extend the centroid.
    pinned: bool = False
    # Meeting time (ms) until which this voice is not asked about, set when the
    # human skips its question. /undo reverses a skip; /name still works.
    skipped_until_ms: int = -1
    utterance_ids: list[int] = field(default_factory=list)

    @property
    def centroid(self) -> np.ndarray:
        norm = float(np.linalg.norm(self.total))
        return self.total / norm if norm > 1e-12 else self.total


@dataclass(frozen=True, slots=True)
class Assignment:
    key: str
    similarity: float
    created: bool
    runner_up: float


class OnlineClusterer:
    """Greedy nearest-centroid assignment with a new-cluster escape hatch."""

    def __init__(self, policy: Policy = POLICY, max_clusters: int = 24) -> None:
        self.policy = policy
        self.max_clusters = max_clusters
        self.clusters: dict[str, Cluster] = {}
        self._next = 1

    def _new_key(self) -> str:
        key = str(self._next)
        self._next += 1
        return key

    def assign(
        self, embedding: np.ndarray, *, trust: str = "auto", allow_create: bool = True
    ) -> Assignment:
        """Place one voice vector, creating a cluster when nothing is close enough.

        A vector that fits nowhere and may not start a cluster (too short to
        trust, or the cluster cap is reached) goes to UNPLACED rather than to
        the nearest stranger: a `?` is recoverable, a wrong name is not.
        """
        v = np.asarray(embedding, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(v))
        if norm <= 1e-12:
            raise ValueError("cannot cluster a zero-length embedding")
        v = v / norm

        live = [c for c in self.clusters.values() if c.count > 0]
        sims = sorted(
            ((float(np.dot(v, c.centroid)), c) for c in live), key=lambda pair: pair[0], reverse=True
        )
        runner_up = sims[1][0] if len(sims) > 1 else 0.0

        chosen: tuple[float, Cluster] | None = None
        if sims and sims[0][0] >= self.policy.cluster_join_similarity:
            chosen = sims[0]
        else:
            # A voice the human already named pulls a little harder than an
            # anonymous cluster, so naming someone once keeps them named.
            named = [(s, c) for s, c in sims if c.pinned]
            if named and named[0][0] >= self.policy.named_join_similarity:
                chosen = named[0]

            # Two named people nearly tied is not evidence for either.
            if len(named) > 1 and named[0][1].person_slug != named[1][1].person_slug:
                if named[0][0] - named[1][0] < self.policy.min_margin / 2:
                    chosen = None

        if chosen is None:
            if allow_create and len(self.clusters) < self.max_clusters:
                return self._create(v, trust)
            return Assignment(UNPLACED, sims[0][0] if sims else 0.0, created=False, runner_up=runner_up)

        best_sim, best = chosen
        # A pinned cluster represents a named human. Automatic evidence may join
        # it (so the transcript stays attributed) but must not drag its centroid,
        # or one bad segment slowly rewrites what that person sounds like. A
        # vector too short to start a cluster is too noisy to move one.
        if (not best.pinned or trust == "human") and allow_create:
            best.total = best.total + v
        best.count += 1
        return Assignment(best.key, best_sim, created=False, runner_up=runner_up)

    def _create(self, unit_vector: np.ndarray, trust: str) -> Assignment:
        key = self._new_key()
        self.clusters[key] = Cluster(key=key, total=unit_vector.copy(), count=1)
        return Assignment(key, 1.0, created=True, runner_up=0.0)

    def pin(self, key: str, person_slug: str) -> None:
        """A human named this cluster. Their answer is now authoritative for it."""
        cluster = self.clusters.get(key)
        if cluster is None:
            return
        cluster.person_slug = person_slug
        cluster.pinned = True

    def merge(self, source_key: str, target_key: str) -> bool:
        """Fold `source` into `target`. Returns False if either key is unknown.

        The source cluster is kept with count 0 rather than deleted, so a stale
        reference to it still resolves and the merge stays auditable.
        """
        source = self.clusters.get(source_key)
        target = self.clusters.get(target_key)
        if source is None or target is None or source is target:
            return False
        target.total = target.total + source.total
        target.count += source.count
        target.utterance_ids.extend(source.utterance_ids)
        if target.person_slug is None and source.person_slug is not None:
            target.person_slug = source.person_slug
            target.pinned = target.pinned or source.pinned
        elif target.person_slug == source.person_slug:
            # Same person: a human's pin on either side survives the merge.
            target.pinned = target.pinned or source.pinned
        source.total = np.zeros_like(source.total)
        source.count = 0
        source.utterance_ids = []
        source.person_slug = target.person_slug
        return True

    def resolve(self, key: str) -> str:
        """Follow a merged cluster to the cluster that absorbed it."""
        cluster = self.clusters.get(key)
        if cluster is None or cluster.count > 0:
            return key
        for other in self.clusters.values():
            if other.count > 0 and other.person_slug is not None:
                if other.person_slug == cluster.person_slug:
                    return other.key
        return key

    def named(self) -> dict[str, str]:
        return {k: c.person_slug for k, c in self.clusters.items() if c.person_slug}
