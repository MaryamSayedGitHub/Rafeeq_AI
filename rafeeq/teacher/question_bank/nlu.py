"""Understand a teacher's request in their own words (English or Arabic).

    "give me 5 medium questions about photosynthesis"  ->  n=5, difficulty=medium, topic="photosynthesis"
    "هاتلي ٣ أسئلة صعبة عن البناء الضوئي"                 ->  n=3, difficulty=hard,   topic="البناء الضوئي"

Two readers work together:
  * simple rules (words, digits, a few phrases) that need no LLM and never fail;
  * an LLM that handles everything the rules cannot (other wording, dialects, typos).
The LLM answer wins for the difficulty and the subject; for the count, the types and "approved" the rules fill
the gaps. If there is no LLM, or it fails, the rules alone are used.
"""
from __future__ import annotations

import logging
import re
from typing import Callable, Optional

from pydantic import ValidationError

from rafeeq.teacher.question_bank.contracts import SearchIntent
from rafeeq.teacher.question_bank.jsonreply import first_json_object

log = logging.getLogger(__name__)

LLM = Callable[[str], str]

DEFAULT_N = 10   # how many we return when the teacher does not say
MAX_N = 50
KINDS = ("mcq", "true_false", "short", "math", "problem", "essay")
LEVELS = ("easy", "medium", "hard")

# ---------------------------------------------------------------------------
# normalising: same length as the input, so positions carry over to the original text
# ---------------------------------------------------------------------------

_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه", "ئ": "ي", "ؤ": "و"})
_ARABIC = re.compile("[؀-ۿ]")


def normalize(text: str) -> str:
    return text.translate(_AR_DIGITS).translate(_FOLD).lower()


def _words(*groups: str) -> frozenset[str]:
    return frozenset(normalize(w) for g in groups for w in g.split())


def _rx(pattern: str) -> re.Pattern:
    return re.compile(normalize(pattern))


# ---------------------------------------------------------------------------
# word lists
# ---------------------------------------------------------------------------

NUMBER_WORDS = {
    **{normalize(w): n for n, w in enumerate("one two three four five six seven eight nine ten".split(), 1)},
    "fifteen": 15, "twenty": 20,
    **{normalize(w): n for w, n in {
        "واحد": 1, "اتنين": 2, "اثنين": 2, "تلاتة": 3, "ثلاثة": 3, "اربعة": 4, "أربعة": 4, "خمسة": 5,
        "سؤالين": 2, "ستة": 6, "سبعة": 7, "تمانية": 8, "ثمانية": 8, "تسعة": 9, "عشرة": 10, "عشر": 10,
        # the same numbers as spoken or written before a plural noun, without the final ة
        "تلات": 3, "ثلاث": 3, "اربع": 4, "أربع": 4, "خمس": 5, "ست": 6, "سبع": 7, "تمن": 8, "ثمان": 8, "تسع": 9}.items()},
}

EASY = _words("easy simple basic beginner", "سهل سهلة سهلين بسيط بسيطة مبتدئ")
MEDIUM = _words("medium moderate intermediate", "متوسط متوسطة متوسطين وسط")
HARD = _words("hard difficult challenging advanced", "صعب صعبة صعبين متقدم متقدمة معقد معقدة")

STRONG_LEVELS = {
    **{normalize(w): "easy" for w in ("easy", "سهل", "سهلة", "سهلين")},
    **{normalize(w): "medium" for w in ("medium", "moderate", "متوسط", "متوسطة", "متوسطين")},
    **{normalize(w): "hard" for w in ("hard", "difficult", "صعب", "صعبة", "صعبين")},
}

KIND_PATTERNS = [
    ("true_false", _rx(r"true\s*(?:/|or|and|&)?\s*false|صح\s*(?:و|او|/)?\s*(?:خطأ|غلط)")),
    ("mcq", _rx(r"\bmcqs?\b|multiple[\s-]*choice|اختيار(?:ي|ات)?\s*من\s*متعدد|اختيارات|اختياري")),
    ("short", _rx(r"short[\s-]*answers?|اجابة\s*قصيرة|اجابات\s*قصيرة|إجابة\s*قصيرة")),
    ("math", _rx(r"\bmaths?\b|\bcalculations?\b|رياضيات|حسابي|حساب")),
    ("problem", _rx(r"\bproblems?\b|مسألة|مسائل")),
    ("essay", _rx(r"\bessays?\b|مقال(?:ي|ية|ات)?")),
]
APPROVED = _rx(r"\bapproved\b|\bvetted\b|معتمد(?:ة|ين)?|موافق\s*عليه(?:ا|م)?")
TOPIC_MARKER = _rx(r"(?:^|\s)(?:about|on|regarding|concerning|covering|related to|في موضوع|عن موضوع|عن|حول|بخصوص|في|متعلقة ب|خاصة ب)\s+([^\s].*)$")

STOP = _words(
    "give me show find get fetch list i need want please some any all the a an of for to from my with and level "
    "type question questions quiz quizzes item items exercise exercises them other ones",
    "هات هاتلي هاتي ادي اديني ادينا عايز عايزة عاوز عاوزة اعرض اعرضلي ورني دور دورلي اسئلة سؤال أسئلة اسألة "
    "من مستوى مستوي نوع ال الـ بس كام كم عندك عندكم ممكن لو سمحت فضلك ياريت وواحد وسؤال والتاني التاني والاخر الاخر",
)


# ---------------------------------------------------------------------------
# 1. rules
# ---------------------------------------------------------------------------

def _blank(chars: list[str], span: tuple[int, int]) -> None:
    for i in range(*span):
        chars[i] = " "


def parse_with_rules(text: str) -> SearchIntent:
    original = text.translate(_AR_DIGITS).strip()
    norm = normalize(original)
    orig_chars, norm_chars = list(original), list(norm)

    # kinds and "approved" are phrases: find them, then blank them out of the topic text
    kinds: list[str] = []
    for kind, rx in KIND_PATTERNS:
        spans = [m.span() for m in rx.finditer(norm)]
        if spans:
            kinds.append(kind)
            for span in spans:
                _blank(orig_chars, span)
                _blank(norm_chars, span)
    approved_only = False
    for m in APPROVED.finditer(norm):
        approved_only = True
        _blank(orig_chars, m.span())
        _blank(norm_chars, m.span())

    tokens = [(m.group(0), normalize(m.group(0))) for m in re.finditer(r"[\w']+", "".join(orig_chars))]

    n: Optional[int] = None
    difficulty: Optional[str] = None
    for _, nt in tokens:
        if n is None:
            if nt.isdigit() and 1 <= int(nt) <= MAX_N:
                n = int(nt)
            elif nt in NUMBER_WORDS:
                n = NUMBER_WORDS[nt]
        if difficulty is None:
            bare = nt[2:] if nt.startswith("ال") else nt
            if nt in EASY or bare in EASY:
                difficulty = "easy"
            elif nt in MEDIUM or bare in MEDIUM:
                difficulty = "medium"
            elif nt in HARD or bare in HARD:
                difficulty = "hard"

    # the topic: what follows "about / on / عن / حول ..." or else everything that is not a command word
    blanked_norm = "".join(norm_chars)
    marker = TOPIC_MARKER.search(blanked_norm)
    topic_source = "".join(orig_chars)[marker.start(1):] if marker else "".join(orig_chars)
    topic_words = []
    for word in re.findall(r"[\w']+", topic_source):
        nt = normalize(word)
        bare = nt[2:] if nt.startswith("ال") and len(nt) > 4 else nt
        if nt in STOP or nt.isdigit() or nt in NUMBER_WORDS:
            continue
        if any(nt in group or bare in group for group in (EASY, MEDIUM, HARD)):
            continue
        topic_words.append(word)

    # "one hard and one easy": two or more different levels mean one question of each level. Only the clear
    # words count here (not "simple", "basic", "advanced"), because those often belong to the subject:
    # "hard questions about simple harmonic motion" is not a mix.
    mix: list[str] = []
    for m in re.finditer(r"[\w']+", "".join(orig_chars)):
        level = STRONG_LEVELS.get(normalize(m.group(0)).removeprefix("ال"))
        if level and level not in mix:
            mix.append(level)
    if len(mix) < 2:
        mix = []

    return SearchIntent(n=n, difficulty=None if mix else difficulty, difficulties=mix, kinds=kinds,
                        topic_query=" ".join(topic_words), approved_only=approved_only)


# ---------------------------------------------------------------------------
# 2. LLM
# ---------------------------------------------------------------------------

def _prompt(text: str) -> str:
    return (
        "A teacher asks for questions from their question bank. Read the request (English or Arabic) and return JSON only:\n"
        '{"n": integer or null, "difficulty": "easy" | "medium" | "hard" | null, "difficulty_word": string, '
        '"kinds": list from ["mcq","true_false","short","math","problem","essay"], '
        '"topic_query": string, "approved_only": boolean}\n'
        "- n: how many questions they want; null if they did not say.\n"
        "- difficulty: null unless the request itself says how hard the questions should be. Never guess it.\n"
        "- difficulty_word: the exact words copied from the request that show the difficulty; an empty string when difficulty is null.\n"
        "- kinds: the question types they ask for; an empty list means any type.\n"
        "- topic_query: ONLY the subject (for example: photosynthesis), in the language of the request, "
        "without the number, the difficulty or the question type; an empty string if no subject.\n"
        "- approved_only: true only if they ask for approved questions.\n\n"
        f"Request: {text}"
    )


def parse_with_llm(text: str, llm: LLM) -> Optional[SearchIntent]:
    data = first_json_object(llm(_prompt(text)))
    if data is None or not {"n", "difficulty", "kinds", "topic_query", "approved_only"} & data.keys():
        return None   # not an answer to our question (some other JSON): the rules read the request instead
    n = data.get("n")
    if isinstance(n, str) and n.strip().isdigit():
        n = int(n)
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= MAX_N:
        n = None
    difficulty = str(data.get("difficulty") or "").strip().lower()
    if difficulty in LEVELS:
        # the LLM must point at the words that say it, or the simple rules must agree: otherwise it guessed
        evidence = normalize(str(data.get("difficulty_word") or "")).strip()
        if not ((evidence and evidence in normalize(text)) or parse_with_rules(text).difficulty == difficulty):
            difficulty = ""
    kinds = data.get("kinds") if isinstance(data.get("kinds"), list) else []
    try:
        return SearchIntent(
            n=n,
            difficulty=difficulty if difficulty in LEVELS else None,
            kinds=[k for k in kinds if k in KINDS],
            topic_query=str(data.get("topic_query") or "").strip(),
            approved_only=data.get("approved_only") is True,
        )
    except (ValidationError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# 3. both together
# ---------------------------------------------------------------------------

def parse_query(text: str, llm: Optional[LLM] = None) -> SearchIntent:
    rules = parse_with_rules(text)
    if llm is None:
        return rules
    try:
        smart = parse_with_llm(text, llm)
    except Exception:
        log.warning("nlu: LLM could not read the request, using the rules only", exc_info=True)
        smart = None
    if smart is None:
        return rules
    # When the LLM read the request, its answer about the difficulty and the subject stands, even when it is
    # empty: the rules treat every leftover word as the subject, which is wrong for "5 approved mcq" and for noisy speech.
    return SearchIntent(
        n=smart.n or rules.n,
        difficulty=None if rules.difficulties else smart.difficulty,
        difficulties=rules.difficulties,   # "one hard and one easy" is read by the rules only
        kinds=smart.kinds or rules.kinds,
        topic_query=smart.topic_query,
        approved_only=smart.approved_only or rules.approved_only,
    )


def result_message(query: str, requested: Optional[int], found: int, related: int = 0,
                   missing: Optional[list] = None) -> str:
    """One line for the teacher, in the language they wrote in.

    `related`: how many of the `found` questions are only close to the subject. `missing`: levels of a mix
    ("one hard and one easy") that the bank has no question for."""
    missing = missing or []
    exact = found - related
    ar = bool(_ARABIC.search(query))
    if found == 0:
        text = "لم أجد أسئلة مطابقة." if ar else "No matching questions found."
    elif related:
        if ar:
            text = (f"وجدت {exact} سؤال عن الموضوع، و{related} سؤال قريب منه." if exact
                    else f"لا يوجد سؤال مطابق تماماً. هذه {related} سؤال قريب من الموضوع.")
        else:
            text = (f"Found {exact} question{'s' if exact != 1 else ''} on the subject, plus {related} related."
                    if exact else f"No exact match. Here {'is' if related == 1 else 'are'} {related} related question{'s' if related != 1 else ''}.")
    elif requested and found < requested:
        text = f"وجدت {found} من {requested} أسئلة طلبتها." if ar else f"Found {found} of the {requested} questions you asked for."
    else:
        text = f"وجدت {found} سؤال." if ar else f"Found {found} question{'s' if found != 1 else ''}."
    if missing:
        names = {"easy": "سهل", "medium": "متوسط", "hard": "صعب"}
        text += (" لا يوجد سؤال " + " ولا سؤال ".join(names[m] for m in missing) + ".") if ar \
            else " No " + " or ".join(missing) + " question available."
    return text
