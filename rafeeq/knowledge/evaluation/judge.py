"""Optional LLM pieces for the evaluation: an OpenAI-compatible client and an LLM judge.

Nothing here runs unless the caller configures it; no key is ever read implicitly or stored.

``OpenAICompatibleLLM``
    Implements the ``generate(system, prompt)`` protocol the Answerer expects, over any
    "/chat/completions" endpoint (Groq, OpenAI, a local vLLM / Ollama server...), using only the
    standard library. Temperature defaults to 0 for repeatability (it does NOT guarantee identical
    outputs on every provider). Reasoning blocks (``<think>...</think>``, emitted by some Qwen models)
    are stripped so they cannot be mistaken for the answer or for citations.

``LLMJudge``
    Grades (a) correctness against the verified reference answer and (b) faithfulness to the excerpts
    the answerer was given. Known limitations, also stated in the report whenever it is used:
      - it is an LLM opinion, not ground truth; judges are noisy and can disagree with humans;
      - a judge from the same model family as the answerer tends to favour its answers
        (use a different model for judging if you can);
      - faithfulness is judged from the excerpts only; it cannot tell whether the excerpts themselves
        were the right ones (that is what the retrieval metrics are for);
      - results from different judge models are not comparable.
    Unparseable judge output is recorded as an error and excluded, never guessed.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping, Sequence

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
RETRY_STATUSES = (429, 500, 502, 503, 504)


class LLMError(RuntimeError):
    """The LLM endpoint failed or returned something unusable."""


def _default_post(url: str, headers: Mapping[str, str], body: bytes, timeout: float) -> tuple[int, str]:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LLMError(f"cannot reach {url}: {exc}") from exc


class OpenAICompatibleLLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str = "LLM_API_KEY",
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        min_interval: float = 0.0,
        post: Callable[[str, Mapping[str, str], bytes, float], tuple[int, str]] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        if not base_url.strip() or not model.strip():
            raise ValueError("base_url and model must be non-empty")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_interval = min_interval
        self._last_call = 0.0
        self._clock = clock
        self._post = post or _default_post
        self._sleep = sleep
        self._environ = os.environ if environ is None else environ

    def describe(self) -> dict[str, Any]:
        """Run configuration for the results file (never includes the key)."""
        return {"base_url": self.base_url, "model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens}  # no key

    def generate(self, system: str, prompt: str) -> str:
        key = self._environ.get(self.api_key_env, "")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "KnowledgeBaseRAG/1.0",
        }
        if key:
            headers["Authorization"] = f"Bearer {key}"
        elif "localhost" not in self.base_url and "127.0.0.1" not in self.base_url:
            raise LLMError(f"environment variable {self.api_key_env} is not set (needed for {self.base_url})")
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        body = json.dumps(payload).encode("utf-8")

        wait = self.min_interval - (self._clock() - self._last_call)
        if self._last_call and wait > 0:
            self._sleep(wait)  # stay under provider rate limits (free tiers are strict)
        for attempt in range(self.max_retries + 1):
            self._last_call = self._clock()
            status, text = self._post(f"{self.base_url}/chat/completions", headers, body, self.timeout)
            if status == 200:
                break
            if status in RETRY_STATUSES and attempt < self.max_retries:
                self._sleep(min(2 ** attempt * 2, 30))
                continue
            raise LLMError(f"HTTP {status} from {self.base_url}: {text[:300]}")
        try:
            content = json.loads(text)["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response shape from {self.base_url}: {text[:300]}") from exc
        if not isinstance(content, str):
            raise LLMError("the response has no text content")
        return _THINK.sub("", content).strip()


# ----------------------------------------------------------------------- judge

JUDGE_SYSTEM = """You are a strict grader of a question-answering system. You are given a question, a trusted reference answer, the excerpts the system was allowed to use, and the system's answer.

Grade two things independently:
1. correctness: does the system answer agree with the reference answer on the substance?
   "correct" = same substance, nothing important missing or wrong; "partially_correct" = some key points right, others missing or wrong; "incorrect" = wrong, off-topic, or contradicts the reference.
2. faithfulness: is every claim in the system answer supported by the excerpts?
   "faithful" = all claims supported; "partially_faithful" = some claims not supported; "unfaithful" = most claims not supported or contradicted.
   Judge only against the excerpts, not against your own knowledge or the reference.

Reply with ONLY a JSON object: {"correctness": "...", "faithfulness": "...", "reason": "one short sentence"}"""

CORRECTNESS_LABELS = ("correct", "partially_correct", "incorrect")
FAITHFULNESS_LABELS = ("faithful", "partially_faithful", "unfaithful")


def build_judge_prompt(question: str, reference: str, excerpts: Sequence[Mapping[str, Any]], answer: str) -> str:
    blocks = "\n\n".join(f"[{i}] (page {c['page']})\n{c['text']}" for i, c in enumerate(excerpts, start=1)) or "(none)"
    return f"Question:\n{question}\n\nReference answer:\n{reference}\n\nExcerpts given to the system:\n{blocks}\n\nSystem answer:\n{answer}\n"


def parse_verdict(raw: str) -> dict[str, str]:
    """Extract and validate the judge's JSON. Raises LLMError if it is unusable."""
    match = re.search(r"\{.*\}", _THINK.sub("", raw), re.DOTALL)
    if not match:
        raise LLMError(f"judge reply has no JSON object: {raw[:200]!r}")
    try:
        data = json.loads(match.group(0))
    except ValueError as exc:
        raise LLMError(f"judge reply is not valid JSON: {raw[:200]!r}") from exc
    correctness, faithfulness = data.get("correctness"), data.get("faithfulness")
    if correctness not in CORRECTNESS_LABELS or faithfulness not in FAITHFULNESS_LABELS:
        raise LLMError(f"judge returned unknown labels: {correctness!r} / {faithfulness!r}")
    return {"correctness": correctness, "faithfulness": faithfulness, "reason": str(data.get("reason", ""))[:300]}


class LLMJudge:
    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def describe(self) -> dict[str, Any]:
        describe = getattr(self._llm, "describe", None)
        return describe() if callable(describe) else {"model": type(self._llm).__name__}

    def judge(self, question: str, reference: str, excerpts: Sequence[Mapping[str, Any]], answer: str) -> dict[str, str]:
        raw = self._llm.generate(JUDGE_SYSTEM, build_judge_prompt(question, reference, excerpts, answer))
        return parse_verdict(raw)