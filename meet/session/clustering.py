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

    def assign(self, embedding: np.ndarray, *, trust: str = "auto") -> Assignment:
        """Place one voice vector, creating a cluster when nothing is close enough."""
        v = np.asarray(embedding, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(v))
        if norm <= 1e-12:
            raise ValueError("cannot cluster a zero-length embedding")
        v = v / norm

        live = [c for c in self.clusters.values() if c.count > 0]
        if not live:
            return self._create(v, trust)

        sims = sorted(
            ((float(np.dot(v, c.centroid)), c) for c in live), key=lambda pair: pair[0], reverse=True
        )
        best_sim, best = sims[0]
        runner_up = sims[1][0] if len(sims) > 1 else 0.0

        if best_sim < self.policy.cluster_join_similarity and len(self.clusters) < self.max_clusters:
            return self._create(v, trust)

        # A pinned cluster represents a named human. Automatic evidence may join
        # it (so the transcript stays attributed) but must not drag its centroid,
        # or one bad segment slowly rewrites what that person sounds like.
        if not best.pinned or trust == "human":
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
