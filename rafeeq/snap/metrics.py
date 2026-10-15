"""CER / WER with explicit text normalisation (decide the rules ONCE, apply to ground truth and output)."""
import re
import unicodedata

try:
    from rapidfuzz.distance import Levenshtein
except ImportError:                                            # pure-python fallback (slower, same result)
    class Levenshtein:                                         # noqa: N801
        @staticmethod
        def distance(a, b):
            prev = list(range(len(b) + 1))
            for i, x in enumerate(a, 1):
                cur = [i]
                for j, y in enumerate(b, 1):
                    cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
                prev = cur
            return prev[-1]

_AR_DIACRITICS = re.compile(r"[ؗ-ًؚ-ْٰـ]")  # harakat + tatweel
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")  # Arabic-Indic / Persian -> 0-9
_ALEF = str.maketrans("أإآٱ", "اااا")


def normalize(text: str, strip_diacritics=True, unify_digits=True, unify_alef=False, collapse_ws=True) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    if strip_diacritics:
        t = _AR_DIACRITICS.sub("", t)
    if unify_digits:
        t = t.translate(_DIGITS)
    if unify_alef:
        t = t.translate(_ALEF)
    if collapse_ws:
        t = re.sub(r"\s+", " ", t)
    return t.strip()


def cer(ref: str, hyp: str, **norm) -> float:
    r, h = normalize(ref, **norm), normalize(hyp, **norm)
    return Levenshtein.distance(r, h) / max(1, len(r))


def wer(ref: str, hyp: str, **norm) -> float:
    r, h = normalize(ref, **norm).split(), normalize(hyp, **norm).split()
    return Levenshtein.distance(r, h) / max(1, len(r))
