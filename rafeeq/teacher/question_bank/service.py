
"""Question Bank service - M6."""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Callable, Iterable, NamedTuple, Optional

from pydantic import ValidationError
from sqlalchemy import create_engine, delete, inspect, select, text
from sqlalchemy.orm import Session

from rafeeq.quiz.contracts import ItemKind, QuizItem
from rafeeq.quiz.quiz.generator import generate_item, select_chunks
from rafeeq.teacher.question_bank.contracts import Grading, SearchIntent
from rafeeq.teacher.question_bank.difficulty import classify_difficulties
from rafeeq.teacher.question_bank.grading import (
    CHOICE_KINDS,
    choice_grading,
    generate_grading,
)
from rafeeq.teacher.question_bank.models import Base, BankItemRow
from rafeeq.teacher.question_bank.nlu import (
    DEFAULT_N,
    MAX_N,
    normalize,
    parse_query,
)

log = logging.getLogger(__name__)

LLM = Callable[[str], str]

# Weighted Rubric is supported only for essay and problem questions.
WEIGHTED_RUBRIC_KINDS = {"problem", "essay"}


class Entry(NamedTuple):
    item: QuizItem
    teacher_id: str
    approved: bool
    grading: Optional[Grading]
    difficulty: Optional[str] = None


class NluResult(NamedTuple):
    intent: SearchIntent
    entries: list[Entry]
    related_ids: frozenset = frozenset()
    missing: tuple = ()


REPEAT_SIMILARITY = 0.6
M2_MAX_TOP_K = 20
M6_SEARCH_CHUNKS = 8


class InvalidItem(ValueError):
    """An edit would produce a question that M4 could not use."""


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def _row_to_item(row: BankItemRow) -> QuizItem:
    """Reconstruct a QuizItem from a BankItemRow."""
    return QuizItem(
        item_id=row.item_id,
        topic_id=row.topic_id,
        stem=row.stem,
        options=row.options or [],
        right_choice=row.right_choice,
        explanation=row.explanation,
        file_name=row.file_name,
        page=row.page,
        source="lecturer_approved" if row.approved else "generated",
        kind=row.kind,  # type: ignore[arg-type]
        answer=row.answer or "",
        rubric=row.rubric or [],
        rubric_weights=row.rubric_weights or [],
    )


def _item_to_row(
    item: QuizItem,
    room_id: str,
    teacher_id: str,
    approved: bool = False,
    grading: Optional[Grading] = None,
    difficulty: Optional[str] = None,
) -> BankItemRow:
    return BankItemRow(
        item_id=item.item_id,
        room_id=room_id,
        teacher_id=teacher_id,
        kind=item.kind,
        stem=item.stem,
        options=item.options,
        right_choice=item.right_choice,
        answer=item.answer,
        rubric=item.rubric,
        rubric_weights=item.rubric_weights,
        explanation=item.explanation,
        grading=grading.model_dump() if grading else {},
        difficulty=difficulty or "",
        file_name=item.file_name,
        page=item.page,
        topic_id=item.topic_id,
        approved=approved,
    )


def _grading_of(row: BankItemRow) -> Optional[Grading]:
    if not row.grading:
        return None
    try:
        return Grading(**row.grading)
    except ValidationError:
        log.warning(
            "question_bank: stored grading for %s is malformed",
            row.item_id,
        )
        return None


def _entry(row: BankItemRow) -> Entry:
    return Entry(
        _row_to_item(row),
        row.teacher_id,
        row.approved,
        _grading_of(row),
        row.difficulty or None,
    )


_WORD = re.compile(r"\w+", re.UNICODE)


def _words(text: str) -> set[str]:
    found = set()
    for word in _WORD.findall(normalize(text)):
        if word.startswith("ال") and len(word) > 4:
            word = word[2:]
        if len(word) > 2:
            found.add(word)
    return found


def _repeats(stem: str, previous: list[set]) -> bool:
    mine = _words(stem)
    return bool(mine) and any(
        len(mine & other) / len(mine | other) >= REPEAT_SIMILARITY
        for other in previous
        if other
    )


def _driver_url(db_url: str) -> str:
    if db_url.startswith(("postgresql://", "postgres://")):
        return "postgresql+psycopg://" + db_url.split("://", 1)[1]
    return db_url


def _parse_llm_json(response: str) -> dict:
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


# ------------------------------------------------------------------
# QuestionBankService
# ------------------------------------------------------------------

class QuestionBankService:
    """Manages the question bank for one deployment."""

    def __init__(self, db_url: str) -> None:
        db_url = _driver_url(db_url)
        self._engine = create_engine(db_url, pool_pre_ping=True)
        Base.metadata.create_all(self._engine)
        self._ensure_columns()

    def _ensure_columns(self) -> None:
        """Add newer columns to existing databases without losing data."""
        table = BankItemRow.__tablename__

        existing = {
            c["name"]
            for c in inspect(self._engine).get_columns(table)
        }

        newer = {
            "grading": "JSON",
            "difficulty": "VARCHAR(8) DEFAULT ''",
            "rubric_weights": "JSON",
        }

        for name, sql_type in newer.items():
            if name not in existing:
                with self._engine.begin() as conn:
                    conn.execute(
                        text(
                            f"ALTER TABLE {table} "
                            f"ADD COLUMN {name} {sql_type}"
                        )
                    )
                log.info(
                    "question_bank: added the %s column to %s",
                    name,
                    table,
                )

    def _session(self) -> Session:
        return Session(self._engine)

    # ------------------------------------------------------------------
    # 1. Generate
    # ------------------------------------------------------------------

    def generate(
        self,
        *,
        room_id: str,
        teacher_id: str,
        chunks: list[dict],
        llm: LLM,
        n: int = 10,
        kinds: Iterable[ItemKind] = ("mcq",),
        topic_id: Optional[str] = None,
        verifier: Optional[LLM] = None,
    ) -> list[Entry]:
        """Generate questions from material chunks and save them."""
        kinds = list(dict.fromkeys(kinds)) or ["mcq"]

        pool = [
            c for c in chunks
            if topic_id is None or c.get("topic_id") == topic_id
        ]

        if not pool:
            log.warning(
                "question_bank.generate: no chunks for room=%s topic=%s",
                room_id,
                topic_id,
            )
            return []

        selected = select_chunks(pool, n * 2)[: n * 2]

        with self._session() as db:
            existing = db.execute(
                select(BankItemRow.stem)
                .where(BankItemRow.room_id == room_id)
                .order_by(BankItemRow.created_at.desc())
                .limit(200)
            ).scalars().all()

        previous = [_words(stem) for stem in existing]
        avoid: list[str] = list(reversed(existing[:30]))
        entries: list[Entry] = []

        for chunk in selected:
            if len(entries) >= n:
                break

            kind: ItemKind = kinds[len(entries) % len(kinds)]  # type: ignore[assignment]
            item_id = f"bank-{room_id[:8]}-{uuid.uuid4().hex[:8]}"

            item, _ = generate_item(
                llm,
                chunk,
                item_id,
                verifier=verifier,
                avoid=avoid,
                kind=kind,
            )

            if item is None and kind != "mcq":
                item, _ = generate_item(
                    llm,
                    chunk,
                    item_id,
                    verifier=verifier,
                    avoid=avoid,
                    kind="mcq",
                )

            if item is None:
                continue

            # Rubric is only supported for essay and problem questions.
            # Short questions keep their existing grading mechanism.
            if item.kind not in WEIGHTED_RUBRIC_KINDS:
                item = item.model_copy(
                    update={
                        "rubric": [],
                        "rubric_weights": [],
                    }
                )

            if _repeats(item.stem, previous):
                log.info(
                    "question_bank.generate: skipped a repeat: %s",
                    item.stem[:80],
                )
                continue

            entries.append(
                Entry(
                    item,
                    teacher_id,
                    False,
                    generate_grading(item, llm),
                    None,
                )
            )
            avoid.append(item.stem)

        levels = classify_difficulties(
            [e.item for e in entries],
            llm,
        )

        entries = [
            e._replace(difficulty=levels[e.item.item_id])
            for e in entries
        ]

        with self._session() as db:
            for entry in entries:
                db.add(
                    _item_to_row(
                        entry.item,
                        room_id,
                        teacher_id,
                        approved=False,
                        grading=entry.grading,
                        difficulty=entry.difficulty,
                    )
                )
            db.commit()

        log.info(
            "question_bank.generate: room=%s generated=%d requested=%d",
            room_id,
            len(entries),
            n,
        )
        return entries

    # ------------------------------------------------------------------
    # 2. CRUD
    # ------------------------------------------------------------------

    def list_items(
        self,
        room_id: str,
        *,
        teacher_id: Optional[str] = None,
        topic_id: Optional[str] = None,
        approved_only: bool = False,
    ) -> list[Entry]:
        """Return questions, newest first."""
        with self._session() as db:
            stmt = select(BankItemRow).where(
                BankItemRow.room_id == room_id
            )

            if teacher_id:
                stmt = stmt.where(
                    BankItemRow.teacher_id == teacher_id
                )
            if topic_id:
                stmt = stmt.where(
                    BankItemRow.topic_id == topic_id
                )
            if approved_only:
                stmt = stmt.where(
                    BankItemRow.approved.is_(True)
                )

            stmt = stmt.order_by(
                BankItemRow.created_at.desc(),
                BankItemRow.id.desc(),
            )
            rows = db.execute(stmt).scalars().all()

        return [_entry(row) for row in rows]

    def get_item(
        self,
        item_id: str,
        *,
        room_id: Optional[str] = None,
    ) -> Optional[Entry]:
        """Return an item, optionally restricted to one room."""
        with self._session() as db:
            stmt = select(BankItemRow).where(
                BankItemRow.item_id == item_id
            )
            if room_id is not None:
                stmt = stmt.where(
                    BankItemRow.room_id == room_id
                )
            row = db.execute(stmt).scalar_one_or_none()

        return None if row is None else _entry(row)

    def update_item(self, item_id: str, **fields) -> Optional[Entry]:
        """Update editable fields after validating the resulting QuizItem."""
        allowed = {
            "stem",
            "answer",
            "explanation",
            "rubric",
            "rubric_weights",
            "options",
            "right_choice",
            "grading",
            "difficulty",
            "approved",
        }

        changes = {
            key: value
            for key, value in fields.items()
            if key in allowed and value is not None
        }

        with self._session() as db:
            row = db.execute(
                select(BankItemRow).where(
                    BankItemRow.item_id == item_id
                )
            ).scalar_one_or_none()

            if row is None:
                return None

            if not changes:
                return _entry(row)

            # Weighted Rubric is only available for essay and problem questions.
            # Existing questions are not affected unless their rubric is edited.
            if (
                changes.keys() & {"rubric", "rubric_weights"}
                and row.kind not in WEIGHTED_RUBRIC_KINDS
            ):
                new_rubric = changes.get("rubric", row.rubric or [])
                new_weights = changes.get(
                    "rubric_weights",
                    row.rubric_weights or [],
                )

                if new_rubric or new_weights:
                    raise InvalidItem(
                        "Rubric and weights are only supported for "
                        "problem and essay questions."
                    )

            # If criteria change without new weights, clear old weights.
            # The grader will then use equal weights for the new criteria.
            if "rubric" in changes and "rubric_weights" not in changes:
                changes["rubric_weights"] = []

            if "grading" in changes:
                try:
                    changes["grading"] = Grading(
                        **changes["grading"]
                    ).model_dump()
                except ValidationError as exc:
                    raise InvalidItem(
                        "the grading guide needs correct, partial and wrong, each not empty"
                    ) from exc

            for key, value in changes.items():
                setattr(row, key, value)

            try:
                # Validate the complete updated question, including
                # rubric/weight count and the question's answer shape.
                item = _row_to_item(row)
            except ValidationError as exc:
                db.rollback()
                first = exc.errors()[0]
                raise InvalidItem(
                    first.get("msg", "invalid question")
                ) from exc

            if (
                row.kind in CHOICE_KINDS
                and "grading" not in changes
                and changes.keys() & {"options", "right_choice", "stem"}
            ):
                row.grading = choice_grading(item).model_dump()

            db.commit()
            return _entry(row)

    def approve_item(
        self,
        item_id: str,
        approved: bool,
    ) -> Optional[Entry]:
        """Approve or unapprove a question."""
        return self.update_item(item_id, approved=approved)

    def regenerate_grading(
        self,
        item_id: str,
        llm: Optional[LLM],
    ) -> Optional[Entry]:
        """Generate a fresh grading guide for one question."""
        with self._session() as db:
            row = db.execute(
                select(BankItemRow).where(
                    BankItemRow.item_id == item_id
                )
            ).scalar_one_or_none()

            if row is None:
                return None

            row.grading = generate_grading(
                _row_to_item(row),
                llm,
            ).model_dump()

            db.commit()
            return _entry(row)

    def backfill(
        self,
        room_id: str,
        teacher_id: str,
        llm: Optional[LLM],
    ) -> int:
        """Add missing grading guides and difficulty values."""
        with self._session() as db:
            rows = db.execute(
                select(BankItemRow)
                .where(BankItemRow.room_id == room_id)
                .where(BankItemRow.teacher_id == teacher_id)
            ).scalars().all()

            items: dict[str, QuizItem] = {}

            for row in rows:
                if row.grading and row.difficulty:
                    continue
                try:
                    items[row.item_id] = _row_to_item(row)
                except ValidationError:
                    log.warning(
                        "question_bank.backfill: skipping invalid item %s",
                        row.item_id,
                    )

            todo = [
                row for row in rows
                if row.item_id in items
            ]

            for row in todo:
                if not row.grading:
                    row.grading = generate_grading(
                        items[row.item_id],
                        llm,
                    ).model_dump()

            unrated = [
                row for row in todo
                if not row.difficulty
            ]

            levels = classify_difficulties(
                [items[row.item_id] for row in unrated],
                llm,
            )

            for row in unrated:
                row.difficulty = levels[row.item_id]

            db.commit()

        return len(todo)

    def delete_item(self, item_id: str) -> bool:
        """Delete an item."""
        with self._session() as db:
            result = db.execute(
                delete(BankItemRow).where(
                    BankItemRow.item_id == item_id
                )
            )
            db.commit()

        return result.rowcount > 0

    # ------------------------------------------------------------------
    # 3. Search
    # ------------------------------------------------------------------

    def search(
        self,
        room_id: str,
        query: str,
        *,
        teacher_id: Optional[str] = None,
        top_k: int = 10,
    ) -> list[Entry]:
        """Find bank items matching a free-text query."""
        return self.search_ranked(
            room_id,
            query,
            teacher_id=teacher_id,
            top_k=top_k,
        )[0]

    def search_ranked(
        self,
        room_id: str,
        query: str,
        *,
        teacher_id: Optional[str] = None,
        top_k: int = 10,
        llm: Optional[LLM] = None,
    ) -> tuple[list[Entry], int]:
        """Search and optionally re-rank results using the LLM."""
        matched, exact = self._search_via_m2(
            room_id,
            query,
            top_k,
            teacher_id,
        )

        if not matched:
            matched = self._search_fallback(
                room_id,
                query,
                top_k,
                teacher_id,
            )
            exact = len(matched)

        if not matched:
            return [], 0

        if llm is None:
            return matched, exact

        return self._rerank_entries(
            query,
            matched,
            llm,
            fallback_exact=exact,
        )

    def _rerank_entries(
        self,
        query: str,
        entries: list[Entry],
        llm: LLM,
        fallback_exact: int = 0,
    ) -> tuple[list[Entry], int]:
        """Classify and rank candidate questions by semantic relevance."""
        if not entries:
            return [], 0

        candidates = [
            {
                "id": str(entry.item.item_id),
                "question": entry.item.stem,
                "kind": str(entry.item.kind),
                "options": entry.item.options or [],
            }
            for entry in entries
        ]

        prompt = f"""
You are a question relevance evaluator for an educational Question Bank.

The teacher's request:
{json.dumps(query, ensure_ascii=False)}

Candidate questions:
{json.dumps(candidates, ensure_ascii=False)}

Evaluate each candidate by MEANING, not exact words.
The request and questions may use different languages.

Classify each candidate:
- direct: directly answers the teacher's request.
- related: same subject but does not directly answer the request.
- irrelevant: not meaningfully related.

Do not generate or rewrite questions.
Do not invent IDs.
Return every candidate exactly once.
Order results: direct first, related second, irrelevant last.

Return ONLY valid JSON:
{{
  "results": [
    {{"id": "candidate-id", "relevance": "direct"}},
    {{"id": "candidate-id", "relevance": "related"}},
    {{"id": "candidate-id", "relevance": "irrelevant"}}
  ]
}}
""".strip()

        try:
            parsed = _parse_llm_json(llm(prompt))
            results = parsed.get("results")

            if not isinstance(results, list):
                raise ValueError("LLM response has no valid results list")

            allowed = {
                str(entry.item.item_id): entry
                for entry in entries
            }

            direct: list[Entry] = []
            related: list[Entry] = []
            seen: set[str] = set()

            for result in results:
                if not isinstance(result, dict):
                    continue

                item_id = str(result.get("id", ""))
                relevance = result.get("relevance")

                if item_id not in allowed or item_id in seen:
                    continue
                if relevance not in {"direct", "related", "irrelevant"}:
                    continue

                seen.add(item_id)

                if relevance == "direct":
                    direct.append(allowed[item_id])
                elif relevance == "related":
                    related.append(allowed[item_id])
                # Irrelevant candidates are deliberately excluded.

            # Preserve omitted candidates as related.
            for entry in entries:
                if str(entry.item.item_id) not in seen:
                    related.append(entry)

            ranked = direct + related

            log.info(
                "question_bank.rerank: direct=%d related=%d",
                len(direct),
                len(related),
            )

            return ranked, len(direct)

        except Exception:
            log.warning(
                "question_bank.rerank: LLM ranking failed; "
                "keeping the original search order",
                exc_info=True,
            )
            return entries, fallback_exact

    def _room_rows(
        self,
        room_id: str,
        teacher_id: Optional[str],
    ) -> list[BankItemRow]:
        with self._session() as db:
            stmt = select(BankItemRow).where(
                BankItemRow.room_id == room_id
            )
            if teacher_id:
                stmt = stmt.where(
                    BankItemRow.teacher_id == teacher_id
                )
            return list(db.execute(stmt).scalars().all())

    def _search_via_m2(
        self,
        room_id: str,
        query: str,
        top_k: int,
        teacher_id: Optional[str],
    ) -> tuple[list[Entry], int]:
        """Return page-level matches first; use topic matches as fallback."""
        try:
            from rafeeq.knowledge import integration as ki

            if not ki.enabled():
                return [], 0

            from rafeeq.knowledge.service import get_service as get_ks

            result = get_ks().search(
                room_id,
                query,
                top_k=min(
                    top_k * 2,
                    M2_MAX_TOP_K,
                    M6_SEARCH_CHUNKS,
                ),
            )

            chunks = (
                result.get("chunks", [])
                if isinstance(result, dict)
                else list(result or [])
            )

        except Exception:
            log.debug(
                "question_bank.search: M2 unavailable, falling back",
                exc_info=True,
            )
            return [], 0

        if not chunks:
            return [], 0

        topic_ids = {
            chunk.get("topic_id")
            for chunk in chunks
        } - {None, ""}

        page_rank: dict[tuple, int] = {}

        for rank, chunk in enumerate(chunks):
            file_name = chunk.get("file_name")
            page = chunk.get("page")

            if file_name is None or page is None:
                continue

            page_rank.setdefault((file_name, page), rank)

        exact: list[tuple[int, BankItemRow]] = []
        topic_level: list[BankItemRow] = []

        for row in self._room_rows(room_id, teacher_id):
            key = (row.file_name, row.page)

            if key in page_rank:
                exact.append((page_rank[key], row))
            elif row.topic_id in topic_ids:
                topic_level.append(row)

        exact.sort(key=lambda pair: pair[0])
        exact_rows = [row for _, row in exact]

        # Prioritize exact page matches, then fill remaining slots
        # with questions from the same topic.
        selected_exact = exact_rows[:top_k]

        remaining = max(0, top_k - len(selected_exact))
        selected_topic = topic_level[:remaining]

        selected = selected_exact + selected_topic

        return (
            [_entry(row) for row in selected],
            len(selected_exact),
        )

    def _search_fallback(
        self,
        room_id: str,
        query: str,
        top_k: int,
        teacher_id: Optional[str],
    ) -> list[Entry]:
        """Rank by the number of query words appearing in each question."""
        query_words = _words(query)

        if not query_words:
            return []

        scored = []

        for row in self._room_rows(room_id, teacher_id):
            hits = len(query_words & _words(row.stem))
            if hits:
                scored.append((hits, row))

        scored.sort(
            key=lambda pair: pair[0],
            reverse=True,
        )

        return [
            _entry(row)
            for _, row in scored[:top_k]
        ]

    def nlu_search(
        self,
        room_id: str,
        text: str,
        *,
        teacher_id: Optional[str] = None,
        llm: Optional[LLM] = None,
    ) -> NluResult:
        """Understand the request, search, re-rank, and filter results."""
        intent = parse_query(text, llm)

        if intent.topic_query:
            pool, exact = self.search_ranked(
                room_id,
                intent.topic_query,
                teacher_id=teacher_id,
                top_k=MAX_N,
                llm=llm,
            )

            related = {
                entry.item.item_id
                for entry in pool[exact:]
            }
        else:
            pool = self.list_items(
                room_id,
                teacher_id=teacher_id,
            )
            related = set()

        def fits(entry: Entry) -> bool:
            return (
                (not intent.difficulty or entry.difficulty == intent.difficulty)
                and (not intent.kinds or entry.item.kind in intent.kinds)
                and (not intent.approved_only or entry.approved)
            )

        kept = [
            entry for entry in pool
            if fits(entry)
        ]

        if not intent.difficulties:
            chosen = kept[: intent.n or DEFAULT_N]
            missing: list[str] = []
        else:
            levels = intent.difficulties
            target = (
                intent.n
                if intent.n and intent.n >= len(levels)
                else len(levels)
            )

            buckets = {
                level: [
                    entry for entry in kept
                    if entry.difficulty == level
                ]
                for level in levels
            }

            chosen = []
            turn = 0

            while len(chosen) < target and any(buckets.values()):
                bucket = buckets[levels[turn % len(levels)]]
                turn += 1
                if bucket:
                    chosen.append(bucket.pop(0))

            missing = [
                level for level in levels
                if not any(
                    entry.difficulty == level
                    for entry in chosen
                )
            ]

        return NluResult(
            intent,
            chosen,
            frozenset(
                entry.item.item_id
                for entry in chosen
                if entry.item.item_id in related
            ),
            tuple(missing),
        )

    # ------------------------------------------------------------------
    # 4. M4 integration - approved items
    # ------------------------------------------------------------------

    def approved_items(
        self,
        room_id: str,
        topic_ids: list[str],
    ) -> list[QuizItem]:
        """Return approved bank items for M4's quiz builder."""
        with self._session() as db:
            stmt = (
                select(BankItemRow)
                .where(BankItemRow.room_id == room_id)
                .where(BankItemRow.approved.is_(True))
            )

            if topic_ids:
                stmt = stmt.where(
                    BankItemRow.topic_id.in_(topic_ids)
                )

            rows = db.execute(stmt).scalars().all()

        return [
            _row_to_item(row)
            for row in rows
        ]


# ------------------------------------------------------------------
# singleton
# ------------------------------------------------------------------

_service: Optional[QuestionBankService] = None


def get_service() -> Optional[QuestionBankService]:
    return _service


def set_service(svc: QuestionBankService) -> None:
    global _service
    _service = svc


def init_service(db_url: str) -> QuestionBankService:
    """Create or replace the module-level singleton."""
    svc = QuestionBankService(db_url)
    set_service(svc)
    return svc
