"""The web is never searched for illegal or adult (18+) things.

Every Tavily call goes through web_search.multi_search, which asks check() first. Two layers:
  1. rules: explicit words in Arabic and English (pornography, explicit sexual content, making weapons / explosives / drugs, buying illegal goods,
     hacking other people's accounts, fraud) block at once, with no model;
  2. the model (when it is reachable) judges the rest, and is told that study questions are fine: biology, medicine, law, history, security as a
     subject. Anything it cannot judge is allowed (the rules above already caught the obvious).
A blocked search says so, in both languages, and nothing is sent to Tavily. The student still gets answers from the course material.
"""
from __future__ import annotations

import re

from . import llm

MESSAGE = {
    "ar": "لا أستطيع البحث في الويب عن هذا الموضوع لأنه غير قانوني أو غير مناسب. اسألني عن مادتك وسأساعدك.",
    "en": "I cannot search the web for this: it is illegal or not appropriate. Ask me about your course and I will help.",
}

_ADULT = re.compile(
    r"\b(?:porn\w*|xxx|nsfw|hentai|erotic\w*|nude\s+(?:photos?|pics?|videos?)|naked\s+(?:women|men|girls?|boys?)|sex\s+(?:videos?|tapes?|chat)|onlyfans|escorts?|camgirls?|blowjob|"
    r"fetish\s+(?:videos?|porn))\b|سكس|إباحي|اباحي|بورن|افلام\s+(?:جنسية|اباحية|إباحية)|أفلام\s+(?:جنسية|إباحية)|صور\s+عاري\w*|فيديوهات?\s+عاري\w*|نيك|مواقع\s+(?:جنس|سكس)", re.I)
_ILLEGAL = re.compile(
    r"\b(?:how\s+to\s+(?:make|build|cook|synthesi[sz]e|manufacture)\s+(?:a\s+)?(?:bomb|explosives?|nerve\s+gas|meth\w*|cocaine|heroin|fentanyl|ghost\s+gun)|"
    r"(?:buy|order|sell|where\s+to\s+get)\s+(?:illegal\s+)?(?:drugs?|cocaine|heroin|meth\w*|weed|cannabis|fake\s+(?:passports?|ids?|money)|stolen\s+cards?|guns?\s+illegally)|"
    r"(?:hack|crack|steal)\s+(?:into\s+)?(?:someone'?s?|a\s+person'?s?|my\s+(?:ex|girlfriend|boyfriend)'?s?|their)\s+(?:accounts?|passwords?|wi-?fi|phones?|facebook|instagram|whatsapp)|"
    r"carding\s+(?:tutorials?|methods?|forums?|guides?|cc|shops?)|counterfeit\s+(?:money|currency)|ransomware\s+(?:builder|source|kit)|ddos\s+(?:for\s+hire|booter)|child\s+(?:porn|abuse\s+material)|hitman|launder\s+money)\b|"
    r"تصنيع\s+(?:قنبل\w*|متفجرات|مخدرات|سلاح)|(?:شراء|بيع|اشتري|أين\s+أجد)\s+(?:مخدرات|حشيش|هيروين|كوكايين|بانجو|سلاح\s+بدون\s+ترخيص|فيزا\s+مسروق\w*)|"
    r"(?:اختراق|تهكير|سرقة|هكر|هاك)\s+(?:حساب|باسورد|كلمة\s+سر|واتس\w*|فيسبوك|انستجرام|هاتف|موبايل)|تزوير\s+(?:شهادة|جواز|فلوس|عملة|هوية)|غسيل\s+(?:اموال|أموال)", re.I)


class Blocked(RuntimeError):
    """Raised before a search that is illegal or 18+. `lang` picks the message."""

    def __init__(self, why: str):
        super().__init__(MESSAGE["en"])
        self.why = why

    def message(self, lang: str = "en") -> str:
        return MESSAGE.get(lang, MESSAGE["en"])


def _by_rules(text: str) -> str | None:
    if _ADULT.search(text):
        return "adult"
    if _ILLEGAL.search(text):
        return "illegal"
    return None


def _by_model(text: str) -> str | None:
    res = llm.ask_json(f"""A student of a course wants to search the web for: {text[:400]}
Decide if the web search must be BLOCKED. Block only: pornography or explicit sexual content (18+), sexual content about minors, instructions or help to commit
a crime (making weapons, explosives or drugs, buying illegal goods, hacking or stealing from other people, fraud, forgery), or evading the police.
ALLOW everything that is a study subject: biology, medicine and anatomy, sex education, law and criminology as subjects, history, chemistry, how security
works in general, news, definitions of any term.
Return JSON only: {{"block": false, "category": "none"}} or {{"block": true, "category": "adult" | "illegal"}}""", max_tokens=60)
    if isinstance(res, dict) and res.get("block") is True:
        return "adult" if res.get("category") == "adult" else "illegal"
    return None


def check(queries: list[str]) -> str | None:
    """-> None when the searches may go ahead, else "adult" | "illegal"."""
    text = " ; ".join(q for q in queries if q)
    return _by_rules(text) or _by_model(text)


def assert_allowed(queries: list[str]) -> None:
    why = check(queries)
    if why:
        raise Blocked(why)
