import numpy as np
import pytest

from meet.session.clustering import OnlineClusterer


def test_same_voice_joins_different_voice_splits(voices):
    c = OnlineClusterer()
    a1 = c.assign(voices.sample("a"))
    a2 = c.assign(voices.sample("a"))
    b1 = c.assign(voices.sample("b"))
    assert a1.created and not a2.created
    assert a1.key == a2.key
    assert b1.created and b1.key != a1.key


def test_zero_vector_is_rejected():
    with pytest.raises(ValueError):
        OnlineClusterer().assign(np.zeros(8))


def test_pinned_centroid_does_not_drift_on_auto_evidence(voices):
    c = OnlineClusterer()
    key = c.assign(voices.sample("a")).key
    c.pin(key, "marcus")
    before = c.clusters[key].centroid.copy()
    c.assign(voices.sample("a"))
    assert np.allclose(before, c.clusters[key].centroid)
    assert c.clusters[key].count == 2
    c.assign(voices.sample("a"), trust="human")
    assert not np.allclose(before, c.clusters[key].centroid)


def test_merge_folds_source_into_target(voices):
    c = OnlineClusterer()
    a = c.assign(voices.sample("a")).key
    b = c.assign(voices.sample("b")).key
    c.clusters[a].utterance_ids.append(1)
    c.clusters[b].utterance_ids.append(2)
    c.pin(b, "sarah")
    assert c.merge(b, a)
    assert c.clusters[a].count == 2
    assert c.clusters[a].person_slug == "sarah"
    assert sorted(c.clusters[a].utterance_ids) == [1, 2]
    assert c.clusters[b].count == 0
    assert not c.merge(a, a)
    assert not c.merge(a, "99")


def test_cluster_cap_forces_nearest(voices):
    c = OnlineClusterer(max_clusters=2)
    c.assign(voices.sample("a"))
    c.assign(voices.sample("b"))
    third = c.assign(voices.sample("c"))
    assert not third.created
    assert len(c.clusters) == 2
