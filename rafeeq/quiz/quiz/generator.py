"""Generate quiz items from course chunks.

The LLM is injected as a plain function `llm(prompt: str) -> str`, so this
works with any provider (wrap M5's shared llm/ factory later) and is easy to
test with a fake.
"""
from __future__ import annotations

import json
import random
import re
import uuid
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from pydantic import ValidationError

from rafeeq.quiz.contracts import QuizItem
from rafeeq.quiz.quiz.validator import check_item

LLM = Callable[[str], str]

# Book sections that are not teachable content (used for the dev book).
DEFAULT_SKIP_TOPICS = frozenset(
    {"front-matter", "preface", "afterword", "index", "about-the-authors"}
)
WEAK_SHARE = 0.6
MAX_CHARS_IN_PROMPT = 3000


def build_prompt(chunk: dict, problems: Optional[list[str]] = None, avoid: Optional[list[str]] = None) -> str:
    text = chunk["text"][:MAX_CHARS_IN_PROMPT]
    prompt = (
        "You write multiple-choice questions from course material.\n"
        "Write ONE question that can be answered using only the text below.\n"
        "Rules:\n"
        "- Exactly 4 options and exactly one correct. The wrong options must be "
        "plausible and clearly different from each other.\n"
        "- All four options must be about the same length. Do not make the "
        "correct option the longest or the most detailed.\n"
        "- The question must make sense on its own. Never write 'the text', "
        "'the passage', 'the figure', 'provided', 'this example' or 'according to'.\n"
        "- Do not ask about code line numbers or small numeric constants.\n"
        "- Ask about ideas and concepts, not about variable names, code details "
        "or what a specific line does. Do not refer to 'the code', 'the list' or "
        "'the evaluator' as if the reader can see them.\n"
        "- The explanation says in one or two sentences why the answer is "
        "correct. It does not say 'the text states'.\n"
        'Return ONLY JSON: {"stem": str, "options": [4 strings], '
        '"right_choice": 0-3, "explanation": str}\n\n'
        f"Topic: {chunk.get('topic_name', chunk['topic_id'])}\n"
        f'Text:\n"""\n{text}\n"""\n'
    )
    if avoid:  # a short lesson gives several questions per passage: each one must test a different idea
        prompt += ("\nThe quiz already has these questions. Ask about a DIFFERENT idea from the text, not a rewording:\n"
                   + "\n".join(f"- {a}" for a in avoid[-8:]) + "\n")
    if problems:
        prompt += "\nYour previous attempt was rejected: " + "; ".join(problems) + ". Fix this.\n"
    return prompt


def _parse_json(raw: str) -> Optional[dict]:
    """Take the first {...} block, so code fences or chatter around it are fine."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _try_build(raw: str, chunk: dict, item_id: str) -> tuple[Optional[QuizItem], list[str]]:
    data = _parse_json(raw)
    if data is None:
        return None, ["the reply was not valid JSON"]
    try:
        # topic, file and page always come from the chunk, never from the LLM
        item = QuizItem(
            item_id=item_id,
            topic_id=chunk["topic_id"],
            stem=str(data.get("stem", "")),
            options=[str(o) for o in data.get("options", [])],
            right_choice=int(data.get("right_choice", -1)),
            explanation=str(data.get("explanation", "")),
            file_name=chunk["file_name"],
            page=chunk["page"],
            source="generated",
        )
    except (ValidationError, TypeError, ValueError) as exc:
        return None, [f"the item shape is wrong: {str(exc).splitlines()[0]}"]
    return item, check_item(item)


def shuffle_options(item: QuizItem, rng: random.Random) -> QuizItem:
    """LLMs put the right answer in the same slot far too often (mostly B).
    Shuffle the options and move right_choice with them."""
    order = list(range(4))
    rng.shuffle(order)  # order[new_position] = old_position
    return item.model_copy(
        update={
            "options": [item.options[i] for i in order],
            "right_choice": order.index(item.right_choice),
        }
    )


def verify_item(verifier: LLM, chunk: dict, item: QuizItem) -> tuple[bool, str]:
    """Blind check: a second model answers the question using only the source page.
    Returns (ok, reason). If the verifier is down (rate limit...) the item is kept."""
    from rafeeq.quiz.eval.judge_quiz import build_judge_prompt, parse_judgement

    try:
        raw = verifier(build_judge_prompt(chunk["text"][:MAX_CHARS_IN_PROMPT], item.stem, item.options))
    except Exception:
        return True, ""
    verdict = parse_judgement(raw)
    if verdict is None:
        return True, ""
    if verdict["answer"] != item.right_choice:
        return False, "a second reader chose a different option, so the correct answer is not clearly the only one"
    if not verdict["supported"]:
        return False, "the source does not clearly support exactly one answer: " + verdict["reason"]
    return True, ""


def generate_item(
    llm: LLM,
    chunk: dict,
    item_id: str,
    rng: Optional[random.Random] = None,
    verifier: Optional[LLM] = None,
    avoid: Optional[list[str]] = None,
) -> tuple[Optional[QuizItem], bool]:
    """Returns (item or None, passed_on_first_try). Regenerates once on failure."""
    rng = rng or random.Random()
    problems: list[str] = []
    for attempt in range(2):
        item, problems = _try_build(llm(build_prompt(chunk, problems or None, avoid)), chunk, item_id)
        if item is not None and not problems and verifier is not None:
            ok, why = verify_item(verifier, chunk, item)
            if not ok:
                problems = [why]
        if item is not None and not problems:
            return shuffle_options(item, rng), attempt == 0
    return None, False


def select_chunks(
    chunks: list[dict],
    n: int,
    weak_topic_ids: Iterable[str] = (),
    skip_topics: Iterable[str] = DEFAULT_SKIP_TOPICS,
    rng: Optional[random.Random] = None,
    min_chars: int = 400,
) -> list[dict]:
    """About 60% of the picks come from weak topics; the rest from the others."""
    rng = rng or random.Random()
    weak = set(weak_topic_ids)
    skip = set(skip_topics)
    pool = [c for c in chunks if c["topic_id"] not in skip and len(c["text"]) >= min_chars]
    weak_pool = [c for c in pool if c["topic_id"] in weak]
    other_pool = [c for c in pool if c["topic_id"] not in weak]

    picked = rng.sample(weak_pool, min(len(weak_pool), round(n * WEAK_SHARE)))
    picked += rng.sample(other_pool, min(len(other_pool), n - len(picked)))
    if len(picked) < n:  # not enough other topics: top up from weak ones
        rest = [c for c in weak_pool if c not in picked]
        picked += rng.sample(rest, min(len(rest), n - len(picked)))
    rng.shuffle(picked)
    return picked


@dataclass
class GenerationResult:
    quiz_id: str
    items: list[QuizItem] = field(default_factory=list)
    attempted: int = 0
    first_try_ok: int = 0

    @property
    def first_try_rate(self) -> float:
        return self.first_try_ok / self.attempted if self.attempted else 0.0


def generate_quiz(
    llm: LLM,
    chunks: list[dict],
    n: int,
    weak_topic_ids: Iterable[str] = (),
    skip_topics: Iterable[str] = DEFAULT_SKIP_TOPICS,
    rng: Optional[random.Random] = None,
    verifier: Optional[LLM] = None,
) -> GenerationResult:
    result = GenerationResult(quiz_id=f"quiz-{uuid.uuid4().hex[:8]}")
    # with a verifier some pages are dropped, so pick spare pages and stop at n
    picks = select_chunks(chunks, n * 2 if verifier else n, weak_topic_ids, skip_topics, rng)
    for i, chunk in enumerate(picks):
        if verifier is not None and len(result.items) >= n:
            break
        item, first_try = generate_item(llm, chunk, f"{result.quiz_id}-{i + 1}", rng, verifier)
        result.attempted += 1
        result.first_try_ok += int(first_try)
        if item is not None:
            result.items.append(item)
    _top_up(result, llm, chunks, n, weak_topic_ids, skip_topics, rng, verifier)
    return result


TOP_UP_MIN_CHARS = 150   # a short lesson: passages shorter than select_chunks' 400 characters still hold a question


def _norm_stem(stem: str) -> str:
    return " ".join(re.findall(r"\w+", stem.lower()))


def _top_up(result: GenerationResult, llm: LLM, chunks: list[dict], n: int, weak_topic_ids: Iterable[str],
            skip_topics: Iterable[str], rng: Optional[random.Random], verifier: Optional[LLM]) -> None:
    """Fewer questions than asked (a one- or two-page lesson): ask again on the same passages, each time for a
    different idea than the questions already made. At most two tries per missing question, duplicates dropped."""
    missing = n - len(result.items)
    if missing <= 0:
        return
    rng = rng or random.Random()
    weak, skip = set(weak_topic_ids), set(skip_topics)
    pool = [c for c in chunks if c["topic_id"] not in skip and len(c["text"]) >= TOP_UP_MIN_CHARS]
    if not pool:
        return
    pool.sort(key=lambda c: (c["topic_id"] not in weak, -len(c["text"])))   # weak topics, then the longest passages
    seen = {_norm_stem(it.stem) for it in result.items}
    for i in range(2 * missing):
        if len(result.items) >= n:
            break
        chunk = pool[i % len(pool)]
        item, first_try = generate_item(llm, chunk, f"{result.quiz_id}-x{i + 1}", rng, verifier,
                                        avoid=[it.stem for it in result.items])
        result.attempted += 1
        result.first_try_ok += int(first_try)
        if item is not None and _norm_stem(item.stem) not in seen:
            seen.add(_norm_stem(item.stem))
            result.items.append(item)