"""The central loop: hear, separate, try to identify, ask, learn, ask less.

    HEAR -> SEPARATE -> TRY TO IDENTIFY -> KNOW? -> LABEL
                                            |
                                            NO
                                            v
                                        ASK HUMAN -> LEARN -> CONTINUE

Everything below exists to keep that loop intact while the engines around it stay
replaceable. Two properties are defended hard:

* Live results are provisional; human corrections are authoritative. Naming a
  cluster rewrites its whole past in one write and pins it against further
  automatic reinterpretation.
* One correction has three effects: it fixes the utterance, it fixes the cluster,
  and it improves the long-term person model. The third is the product. A tool
  that only fixes the transcript asks the same question forever.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from .. import events
from ..config import POLICY, Policy
from ..identity import jev as jev_client
from ..identity.policy import Action, Decision, decide, should_consult_jev
from ..memory import db, people
from .clustering import OnlineClusterer

UNKNOWN_LABEL = "?"


@dataclass(slots=True)
class Line:
    """One rendered transcript row. `cluster_key` is the only identity it holds."""

    utterance_id: int
    seq: int
    cluster_key: str
    start_ms: int
    end_ms: int
    text: str
    provisional: bool = False


@dataclass(slots=True)
class Question:
    """An open identity question. It never blocks capture."""

    id: int
    cluster_key: str
    utterance_id: int
    text: str
    options: list[tuple[str, str, float]]  # (slug, display name, probability)
    reason: str
    asked_at: float
    jev_note: str | None = None


@dataclass(slots=True)
class _Fold:
    """What an automatic same-person merge changed, so `undo` can split it."""

    source_key: str
    target_key: str
    moved: list[int]
    source_state: tuple
    target_state: tuple


@dataclass(slots=True)
class _Skip:
    question: Question
    previous_until_ms: int


@dataclass(slots=True)
class MeetingStats:
    utterances: int = 0
    questions_asked: int = 0
    auto_labelled: int = 0
    provisional: int = 0
    corrections: int = 0
    samples_learned: int = 0
    jev_calls: int = 0
    jev_failures: int = 0


class Meeting:
    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        title: str,
        attendees: list[str] | None = None,
        policy: Policy = POLICY,
        use_jev: bool = True,
    ) -> None:
        self.conn = conn
        self.policy = policy
        self.use_jev = use_jev
        self.id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        self.title = title
        self.clusterer = OnlineClusterer(policy=policy)
        self.lines: list[Line] = []
        self.questions: dict[int, Question] = {}
        self.stats = MeetingStats()
        self.model_id: str = ""
        self.notes: list[str] = []
        self._question_seq = 0
        # When the human last submitted a line, for the double-Enter guard.
        self.last_input_at = 0.0
        # (cluster_key, previous slug) per naming, a _Fold that rode on one, or
        # a _Skip.
        self._undo: list[tuple[str, str | None] | _Fold | _Skip] = []

        # An explicit attendee list narrows matching without ever inventing a
        # name: someone not on the list cannot be proposed, so an unexpected
        # voice becomes a question instead of the nearest wrong person.
        self.roster: dict[str, str] = {}
        for name in attendees or []:
            slug = db.upsert_person(conn, name)
            self.roster[slug] = name
        self.restrict = set(self.roster) if self.roster else None

        conn.execute(
            "INSERT INTO meeting(id, title, started_at) VALUES (?,?,?)",
            (self.id, title, db.now()),
        )

    # ── identity resolution ────────────────────────────────────────────────

    def label_for(self, cluster_key: str) -> str:
        cluster = self.clusterer.clusters.get(cluster_key)
        if cluster is None or cluster.person_slug is None:
            return UNKNOWN_LABEL
        return self._display(cluster.person_slug)

    def _display(self, slug: str) -> str:
        if slug in self.roster:
            return self.roster[slug]
        row = self.conn.execute("SELECT display_name FROM person WHERE slug=?", (slug,)).fetchone()
        return row["display_name"] if row else slug

    # ── the loop ───────────────────────────────────────────────────────────

    def on_utterance(self, event: events.Utterance) -> Line | None:
        """Handle one settled span of speech. Returns the rendered line, if any."""
        if not event.text.strip():
            return None
        if event.no_speech > self.policy.max_no_speech:
            return None
        if not event.embedding:
            return None

        self.model_id = event.model_id
        probe = np.asarray(event.embedding, dtype=np.float32)
        # A voice vector from under ~2 s of speech is mostly noise. Such a line
        # may join a voice already heard, but it never starts a new one and
        # never puts a question on screen: nobody should be asked who said "ok".
        short = event.quality.speech_s < self.policy.ask_min_speech_s
        assignment = self.clusterer.assign(probe, allow_create=not short)

        result = people.match(
            self.conn, probe, event.model_id, policy=self.policy, restrict_to=self.restrict
        )
        cluster = self.clusterer.clusters.get(assignment.key)
        pinned = cluster.person_slug if cluster is not None and cluster.pinned else None
        decision = decide(
            result,
            pinned_slug=pinned,
            pinned_name=self._display(pinned) if pinned else None,
            policy=self.policy,
        )

        if self.use_jev and should_consult_jev(decision, self.policy) and not short:
            decision = self._consult_jev(event, assignment, decision)

        key = assignment.key
        if cluster is None:
            # Unplaced, but the stored voices know who it is: file it with that
            # person's voice in this meeting, if they have spoken already.
            home = self._cluster_of(decision.slug) if decision.is_labelled else None
            if home is not None:
                key, cluster = home, self.clusterer.clusters[home]
                cluster.count += 1

        utterance_id = self._store_utterance(event, key)
        if cluster is not None:
            cluster.utterance_ids.append(utterance_id)
            skipped = event.start_ms < cluster.skipped_until_ms
            if short and not cluster.pinned:
                # Too little speech to name, rename, or ask about a voice.
                pass
            elif not ((short or skipped) and decision.action is Action.ASK):
                self._apply(decision, key, utterance_id, event, result)

        line = Line(
            utterance_id=utterance_id,
            seq=event.seq,
            cluster_key=key,
            start_ms=event.start_ms,
            end_ms=event.end_ms,
            text=event.text.strip(),
            provisional=decision.action is Action.PROVISIONAL and cluster is not None,
        )
        self.lines.append(line)
        self.stats.utterances += 1
        return line

    def _consult_jev(
        self, event: events.Utterance, assignment, decision: Decision
    ) -> Decision:
        """Ask Jev to break a genuine tie. Any failure falls through to the human."""
        recent = [(self.label_for(ln.cluster_key), ln.text) for ln in self.lines[-6:]]
        try:
            answer = jev_client.ask(
                utterance=event.text,
                candidates=list(decision.candidates),
                cluster_key=assignment.key,
                cluster_similarity=assignment.similarity,
                attendees=list(self.roster.values()),
                recent=recent,
            )
        except jev_client.JevUnavailable:
            self.stats.jev_failures += 1
            return decision
        self.stats.jev_calls += 1

        if answer.slug is None:
            return Decision(
                Action.ASK,
                None,
                None,
                answer.confidence,
                decision.margin,
                "Jev could not separate the candidates",
                source="jev",
                candidates=decision.candidates,
            )

        top = answer.probabilities.get(answer.slug, 0.0)
        others = [p for k, p in answer.probabilities.items() if k != answer.slug]
        margin = top - (max(others) if others else 0.0)
        # Jev's answer is held to the same two-part test as the voice model, but
        # on its own scale: these are probabilities, not cosine similarities.
        if answer.confidence >= self.policy.jev_auto_confidence and margin >= self.policy.jev_min_margin:
            action = Action.AUTO
        elif (
            answer.confidence >= self.policy.jev_provisional_confidence
            and margin >= self.policy.jev_min_margin
        ):
            action = Action.PROVISIONAL
        else:
            return Decision(
                Action.ASK,
                None,
                None,
                answer.confidence,
                margin,
                f"Jev leads {self._display(answer.slug)} by only {margin:.2f}",
                source="jev",
                candidates=decision.candidates,
            )
        return Decision(
            action,
            answer.slug,
            self._display(answer.slug),
            answer.confidence,
            margin,
            "Jev resolved the tie",
            source="jev",
            candidates=decision.candidates,
        )

    def _apply(
        self,
        decision: Decision,
        cluster_key: str,
        utterance_id: int,
        event: events.Utterance,
        result: people.MatchResult,
    ) -> None:
        cluster = self.clusterer.clusters[cluster_key]

        if decision.action is Action.ASK:
            self._open_question(cluster_key, utterance_id, event, decision)
            return

        if cluster.person_slug != decision.slug:
            cluster.person_slug = decision.slug
            self._write_cluster(cluster_key, decision.slug, decision.source)
            people.record_decision(
                self.conn,
                self.id,
                cluster_key,
                source="jev" if decision.source == "jev" else "auto",
                chosen_slug=decision.slug,
                candidates=list(decision.candidates),
                utterance_id=utterance_id,
                top_p=decision.confidence,
                margin=decision.margin,
                confidence=decision.confidence,
            )

        if decision.action is Action.AUTO:
            self.stats.auto_labelled += 1
            # Every auto-labelled utterance is a learning opportunity, including
            # the ones inherited from a pinned cluster. They go in as 'auto' and
            # face the consistency gate: the human vouched for the *cluster*, and
            # cluster membership was still decided by a machine, so a segment the
            # clusterer misfiled must not quietly become part of someone's voice.
            # Only a sure automatic label teaches; a borderline one must not
            # feed the very profile that produced it.
            sure = decision.source == "human" or decision.confidence >= self.policy.auto_learn_similarity
            if decision.slug and sure:
                outcome = people.learn(
                    self.conn,
                    decision.slug,
                    np.asarray(event.embedding, dtype=np.float32),
                    event.model_id,
                    trust="auto",
                    speech_s=event.quality.speech_s,
                    snr_db=event.quality.snr_db,
                    clipping=event.quality.clipping,
                    rms=event.quality.rms,
                    meeting_id=self.id,
                    cluster_key=cluster_key,
                    utterance_id=utterance_id,
                    similarity=decision.confidence,
                    margin=decision.margin,
                    policy=self.policy,
                )
                if outcome.stored:
                    self.stats.samples_learned += 1
        else:
            self.stats.provisional += 1

    def _open_question(
        self, cluster_key: str, utterance_id: int, event: events.Utterance, decision: Decision
    ) -> Question:
        """Put the question on screen. Capture is unaffected; it keeps running."""
        # One open question per cluster. Asking twice about the same voice before
        # the first answer arrives is the fastest way to make the tool feel broken.
        for question in self.questions.values():
            if question.cluster_key == cluster_key:
                return question

        self._question_seq += 1
        options = [(c.slug, c.name, c.similarity) for c in decision.candidates[:4]]
        if len(options) < 4:
            # Offer the people already named in this meeting, closest voice
            # first, so the likely answer is one keystroke even when the stored
            # voices were not sure enough to propose anyone.
            offered = {slug for slug, _, _ in options}
            options += self._meeting_people(event.embedding, exclude=offered)[: 4 - len(options)]
        question = Question(
            id=self._question_seq,
            cluster_key=cluster_key,
            utterance_id=utterance_id,
            text=event.text.strip(),
            options=options,
            reason=decision.reason,
            asked_at=event.start_ms / 1000.0,
        )
        self.questions[question.id] = question
        self.stats.questions_asked += 1
        people.record_decision(
            self.conn,
            self.id,
            cluster_key,
            source="auto",
            chosen_slug=None,
            candidates=list(decision.candidates),
            utterance_id=utterance_id,
            top_p=decision.confidence,
            margin=decision.margin,
            confidence=decision.confidence,
        )
        return question

    def _meeting_people(self, embedding, exclude: set[str]) -> list[tuple[str, str, float]]:
        """People named in this meeting, ranked by how close this voice is to them."""
        probe = np.asarray(embedding, dtype=np.float32).ravel()
        probe = probe / (float(np.linalg.norm(probe)) or 1.0)
        best: dict[str, float] = {}
        for cluster in self.clusterer.clusters.values():
            slug = cluster.person_slug
            if cluster.count == 0 or not cluster.pinned or slug is None or slug in exclude:
                continue
            similarity = float(np.dot(probe, cluster.centroid))
            best[slug] = max(best.get(slug, -1.0), similarity)
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        return [(slug, self._display(slug), max(similarity, 0.0)) for slug, similarity in ranked]

    def _cluster_of(self, slug: str | None) -> str | None:
        """The live cluster this meeting has already bound to a person."""
        if slug is None:
            return None
        for key, cluster in self.clusterer.clusters.items():
            if cluster.count > 0 and cluster.person_slug == slug:
                return key
        return None

    # ── human authority ────────────────────────────────────────────────────

    def skip(self, question_id: int | None = None) -> bool:
        """Dismiss a question (the oldest by default) without naming anyone.

        The voice is not asked about again this meeting. Its lines keep `?`
        and it can still be named later with /name.
        """
        if question_id is None:
            if not self.questions:
                return False
            question_id = min(self.questions)
        question = self.questions.pop(question_id, None)
        if question is None:
            return False
        cluster = self.clusterer.clusters.get(question.cluster_key)
        if cluster is not None:
            now_ms = self.lines[-1].end_ms if self.lines else 0
            self._undo.append(_Skip(question, cluster.skipped_until_ms))
            cluster.skipped_until_ms = now_ms + self.policy.skip_for_ms
        return True

    def answer(self, question_id: int, name_or_slug: str) -> str:
        """The human names a cluster. Authoritative, retroactive, and taught."""
        question = self.questions.pop(question_id, None)
        if question is None:
            raise KeyError(f"no open question {question_id}")
        return self.name_cluster(question.cluster_key, name_or_slug, anchor=question.utterance_id)

    def name_cluster(self, cluster_key: str, name_or_slug: str, anchor: int | None = None) -> str:
        """Bind a cluster to a person and propagate the consequences.

        Three effects, in one call:
          1. the cluster's whole history relabels, because lines resolve through
             the cluster rather than storing a name of their own;
          2. the cluster is pinned, so automatic evidence can no longer rename it;
          3. every qualifying utterance already heard from this cluster becomes a
             human-trust voice sample, which is how the next meeting asks less.
        """
        cluster = self.clusterer.clusters.get(cluster_key)
        if cluster is None:
            raise KeyError(f"unknown cluster {cluster_key}")

        row = self.conn.execute(
            "SELECT slug, display_name FROM person WHERE slug=? OR display_name=? COLLATE NOCASE",
            (name_or_slug, name_or_slug),
        ).fetchone()
        slug = row["slug"] if row else db.upsert_person(self.conn, name_or_slug)
        display = row["display_name"] if row else name_or_slug.strip()
        self.roster.setdefault(slug, display)

        self._undo.append((cluster_key, cluster.person_slug))
        self.clusterer.pin(cluster_key, slug)
        self._write_cluster(cluster_key, slug, "human")
        people.record_decision(
            self.conn,
            self.id,
            cluster_key,
            source="human",
            chosen_slug=slug,
            candidates=[],
            confidence=1.0,
        )
        # Close any other question that was about this same voice.
        for qid in [q.id for q in self.questions.values() if q.cluster_key == cluster_key]:
            self.questions.pop(qid, None)

        self.stats.samples_learned += self._teach(cluster_key, slug, anchor)

        # The human used a name this meeting already has a voice for: the two
        # clusters are one person. Fold this one into the other so the voice
        # has one centroid; two pinned halves of the same person each drift
        # toward whoever speaks near them. The fold is part of this answer, so
        # one `undo` reverses both.
        for other_key, other in self.clusterer.clusters.items():
            if other_key != cluster_key and other.count > 0 and other.person_slug == slug:
                self._undo.append(self._fold(cluster_key, other_key))
                break
        return display

    def _fold(self, source_key: str, target_key: str) -> _Fold:
        """Merge two clusters of the same named person, keeping enough to undo it."""
        source = self.clusterer.clusters[source_key]
        target = self.clusterer.clusters[target_key]
        record = _Fold(
            source_key=source_key,
            target_key=target_key,
            moved=list(source.utterance_ids),
            source_state=(source.total.copy(), source.count, source.person_slug, source.pinned),
            target_state=(target.total.copy(), target.count, target.person_slug, target.pinned,
                          list(target.utterance_ids)),
        )
        self.clusterer.merge(source_key, target_key)
        self._move_lines(source_key, target_key, None)
        self.conn.execute(
            "INSERT INTO cluster(meeting_id, key, merged_into, decided_by, decided_at) VALUES (?,?,?,?,?) "
            "ON CONFLICT(meeting_id, key) DO UPDATE SET merged_into=excluded.merged_into",
            (self.id, source_key, target_key, "human", db.now()),
        )
        return record

    def _unfold(self, record: _Fold) -> None:
        source = self.clusterer.clusters[record.source_key]
        target = self.clusterer.clusters[record.target_key]
        source_total, source_count, source_slug, source_pinned = record.source_state
        # Subtract rather than restore a snapshot: lines heard since the fold
        # belong to the target and must stay counted there.
        target.total = target.total - source_total
        target.count -= source_count
        moved = set(record.moved)
        target.utterance_ids = [i for i in target.utterance_ids if i not in moved]
        _, _, target.person_slug, target.pinned, _ = record.target_state
        source.total, source.count = source_total, source_count
        source.person_slug, source.pinned = source_slug, source_pinned
        source.utterance_ids = list(record.moved)
        self._move_lines(record.target_key, record.source_key, set(record.moved))
        self.conn.execute(
            "UPDATE cluster SET merged_into=NULL WHERE meeting_id=? AND key=?",
            (self.id, record.source_key),
        )

    def _move_lines(self, from_key: str, to_key: str, only: set[int] | None) -> None:
        """Re-point stored and rendered lines from one cluster to another."""
        for line in self.lines:
            if line.cluster_key == from_key and (only is None or line.utterance_id in only):
                line.cluster_key = to_key
        if only is None:
            self.conn.execute(
                "UPDATE utterance SET cluster_key=? WHERE meeting_id=? AND cluster_key=?",
                (to_key, self.id, from_key),
            )
        else:
            self.conn.executemany(
                "UPDATE utterance SET cluster_key=? WHERE id=?", [(to_key, i) for i in sorted(only)]
            )

    def _teach(self, cluster_key: str, slug: str, anchor: int | None = None) -> int:
        """Turn this cluster's clean utterances into voice samples.

        Exactly one of them is direct human evidence: the line the human was
        looking at when they answered. That one is stored as 'human' and skips
        the consistency gate, because a human correcting a wrong profile must be
        able to move it. Every other line in the cluster was placed there by the
        clusterer, so it is stored as 'auto' and must agree with the anchor to be
        accepted. Without that split, one mis-clustered segment would be laundered
        into human-confirmed evidence by a single keystroke.
        """
        rows = self.conn.execute(
            "SELECT id, embedding, dim, model_id, speech_s, snr_db, clipping FROM utterance "
            "WHERE meeting_id=? AND cluster_key=? AND embedding IS NOT NULL ORDER BY seq",
            (self.id, cluster_key),
        ).fetchall()
        if not rows:
            return 0
        if anchor is None:
            # No specific line was asked about, so vouch for the cleanest one.
            anchor = max(rows, key=lambda r: (r["snr_db"] or 0.0, r["speech_s"] or 0.0))["id"]
        # The anchor goes first so the gate on the others has something to check.
        ordered = sorted(rows, key=lambda r: r["id"] != anchor)

        learned = 0
        for r in ordered:
            already = self.conn.execute(
                "SELECT 1 FROM voice_sample WHERE utterance_id=? AND person_slug=? AND revoked_at IS NULL",
                (r["id"], slug),
            ).fetchone()
            if already:
                continue
            outcome = people.learn(
                self.conn,
                slug,
                db.unpack(r["embedding"], r["dim"]),
                r["model_id"],
                trust="human" if r["id"] == anchor else "auto",
                speech_s=r["speech_s"] or 0.0,
                snr_db=r["snr_db"] or 0.0,
                clipping=r["clipping"] if r["clipping"] is not None else 1.0,
                # A segment that cleared the sidecar's own floor and carries a
                # real SNR is above the noise floor by construction.
                rms=self.policy.min_rms if (r["snr_db"] or 0.0) >= self.policy.min_snr_db else 0.0,
                meeting_id=self.id,
                cluster_key=cluster_key,
                utterance_id=r["id"],
                policy=self.policy,
            )
            if outcome.stored:
                learned += 1
        return learned

    def merge(self, source_key: str, target_key: str) -> bool:
        """Two clusters are one person. Rewrites the source's lines onto the target."""
        target = self.clusterer.clusters.get(target_key)
        named_before = target.person_slug if target is not None else None
        if not self.clusterer.merge(source_key, target_key):
            return False
        self.conn.execute(
            "UPDATE utterance SET cluster_key=? WHERE meeting_id=? AND cluster_key=?",
            (target_key, self.id, source_key),
        )
        for line in self.lines:
            if line.cluster_key == source_key:
                line.cluster_key = target_key
        # A cluster that was only ever asked about has no row yet; the merge
        # must still leave a record of where its lines went.
        self.conn.execute(
            "INSERT INTO cluster(meeting_id, key, merged_into, decided_by, decided_at) VALUES (?,?,?,?,?) "
            "ON CONFLICT(meeting_id, key) DO UPDATE SET merged_into=excluded.merged_into",
            (self.id, source_key, target_key, "human", db.now()),
        )
        target = self.clusterer.clusters[target_key]
        if target.person_slug != named_before:
            self._write_cluster(target_key, target.person_slug, "human")
        people.record_decision(
            self.conn,
            self.id,
            target_key,
            source="merge",
            chosen_slug=target.person_slug,
            candidates=[],
        )
        if target.person_slug:
            self.stats.samples_learned += self._teach(target_key, target.person_slug)
        self.stats.corrections += 1
        return True

    def correct_last(self, name_or_slug: str) -> str:
        """`:wrong <name>` — retarget the most recent line's cluster."""
        if not self.lines:
            raise IndexError("nothing has been said yet")
        # Only ever the line on screen: falling back to an earlier line would
        # silently rename someone the human already named.
        last = self.lines[-1]
        if last.cluster_key not in self.clusterer.clusters:
            raise IndexError("that line was too short to hold a voice; use /name <speaker#> <name>")
        named = self.name_cluster(last.cluster_key, name_or_slug)
        self.stats.corrections += 1
        return named

    def undo(self) -> str:
        """Reverse the last human naming. The learned samples stay, revoked."""
        if not self._undo:
            raise IndexError("nothing to undo")
        entry = self._undo.pop()
        if isinstance(entry, _Skip):
            # A stray Enter: put the question back on screen.
            cluster = self.clusterer.clusters.get(entry.question.cluster_key)
            if cluster is None or cluster.count == 0:
                return "that voice has since been merged; nothing to restore"
            cluster.skipped_until_ms = entry.previous_until_ms
            if cluster.pinned:
                return f"that voice has since been named {self._display(cluster.person_slug)}"
            self.questions[entry.question.id] = entry.question
            return "question restored"
        if isinstance(entry, _Fold):
            # The fold and the naming under it were one answer.
            self._unfold(entry)
            entry = self._undo.pop()
        cluster_key, previous = entry
        cluster = self.clusterer.clusters.get(cluster_key)
        if cluster is None:
            raise KeyError(cluster_key)
        wrong_slug = cluster.person_slug
        cluster.person_slug = previous
        cluster.pinned = previous is not None
        self._write_cluster(cluster_key, previous, "human")
        if wrong_slug:
            # Revoke rather than delete: the sample stays in the record, marked
            # withdrawn, and the centroid is rebuilt without it.
            self.conn.execute(
                "UPDATE voice_sample SET revoked_at=? WHERE meeting_id=? AND cluster_key=? "
                "AND person_slug=? AND revoked_at IS NULL",
                (db.now(), self.id, cluster_key, wrong_slug),
            )
            if self.model_id:
                people.rebuild_profile(self.conn, wrong_slug, self.model_id)
        return self._display(previous) if previous else UNKNOWN_LABEL

    def note(self, kind: str, text: str | None = None) -> str:
        """`:decision` / `:action` — mark the latest line without leaving the loop."""
        body = text or (self.lines[-1].text if self.lines else "")
        entry = f"{kind}: {body}"
        self.notes.append(entry)
        return entry

    # ── persistence ────────────────────────────────────────────────────────

    def _store_utterance(self, event: events.Utterance, cluster_key: str) -> int:
        cursor = self.conn.execute(
            "INSERT INTO utterance(meeting_id, cluster_key, seq, start_ms, end_ms, text, embedding, "
            "dim, model_id, speech_s, snr_db, clipping, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                self.id,
                cluster_key,
                event.seq,
                event.start_ms,
                event.end_ms,
                event.text.strip(),
                db.pack(np.asarray(event.embedding, dtype=np.float32)),
                event.dim,
                event.model_id,
                event.quality.speech_s,
                event.quality.snr_db,
                event.quality.clipping,
                db.now(),
            ),
        )
        return int(cursor.lastrowid)

    def _write_cluster(self, key: str, slug: str | None, decided_by: str) -> None:
        self.conn.execute(
            "INSERT INTO cluster(meeting_id, key, person_slug, decided_by, decided_at) "
            "VALUES (?,?,?,?,?) ON CONFLICT(meeting_id, key) DO UPDATE SET "
            "person_slug=excluded.person_slug, decided_by=excluded.decided_by, "
            "decided_at=excluded.decided_at",
            (self.id, key, slug, decided_by, db.now()),
        )

    def finish(self, wav_path: str | None, sample_rate: int) -> None:
        self.conn.execute(
            "UPDATE meeting SET ended_at=?, audio_path=?, sample_rate=? WHERE id=?",
            (db.now(), wav_path, sample_rate, self.id),
        )

    # ── output ─────────────────────────────────────────────────────────────

    def transcript(self) -> list[tuple[str, str, int]]:
        """(speaker, text, start_ms), resolved through clusters at render time."""
        return [(self.label_for(ln.cluster_key), ln.text, ln.start_ms) for ln in self.lines]

    def questions_per_hour(self) -> float:
        if not self.lines:
            return 0.0
        hours = max(self.lines[-1].end_ms / 3_600_000.0, 1e-6)
        return self.stats.questions_asked / hours
