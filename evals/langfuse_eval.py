"""Evaluation IN Langfuse: Langfuse's own LLM-as-a-judge evaluators score every feature's traces.

    uv run python evals/langfuse_eval.py            # create / check the evaluators and rules (safe to repeat)

What it sets up in the Langfuse project (through the Langfuse API, nothing is computed here):

    LLM connection   "google-ai-studio" with GOOGLE_API_KEY, only when the project has no Google AI Studio connection
                     yet: Langfuse calls the judge model itself with it.
    Evaluators       one LLM-as-a-judge per instructor feature (prompt + numeric 0-1 score, model EVAL_JUDGE_MODEL /
                     GEMINI_MODEL), named "rafeeq · <feature>".
    Rules            one per feature: which traces it scores (root observation + the feature's trace name or tag),
                     sampling 1 = every trace, in every environment (the running app AND the evaluation runs).

From then on Langfuse scores each new trace by itself, a few seconds after it arrives. The report reads those
scores back with the Metrics API (evals/langfuse_monitor.py): for this evaluation run (environment "evaluation") and
for real use (environment "default").
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rafeeq import observability

PREFIX = "rafeeq · "
PROVIDER = "google-ai-studio"

_SCALE = ("Return a score from 0 to 1 (0 = unusable, 0.5 = needs real fixes, 1 = a teacher could use it as is) and one "
          "sentence of reasoning. A factual error or an answer in the wrong language caps the score at 0.4. An empty "
          "or error output scores 0.")

# feature suite -> (evaluator name, how its traces are found, judge prompt with {{input}} and {{output}})
JUDGES = {
    "lesson_studio": ("lesson draft quality", ("tags", ["lecturer-assistant"]),
                      "You review a lesson draft that an AI assistant prepared for a teacher.\nThe teacher's brief (topic, "
                      "level, language, objectives, what was asked for):\n{{input}}\n\nThe draft (slides, code examples, "
                      "quiz):\n{{output}}\n\nJudge: the content is correct and fits the level; it covers the objectives; "
                      "every quiz answer key is right; it is in the brief's language. " + _SCALE),
    "video_discovery": ("video moments relevance", ("traceName", ["video_discovery.search"]),
                        "A teacher searched for YouTube moments.\nThe question:\n{{input}}\n\nThe moments returned "
                        "(title, start time, transcript text):\n{{output}}\n\nJudge: the share of returned moments whose "
                        "transcript really answers the question; the first one matters most. No results for an "
                        "answerable question scores 0. " + _SCALE),
    "story_video": ("animated video plan", ("traceName", ["story_video.plan"]),
                    "An AI planned a short animated explainer video for school students.\nThe request (topic, scenes, "
                    "language):\n{{input}}\n\nThe plan (character, setting, scenes with visual + narration):\n{{output}}\n\n"
                    "Judge: scene 1 hooks, the middle scenes teach one correct idea each in order, the last recaps; the "
                    "narration teaches (no filler) in the requested language; visuals show the real thing. " + _SCALE),
    "voice_visual": ("voice to visual moments", ("traceName", ["voice_visual.plan"]),
                     "An AI chose moments of a recorded lecture where a picture would help.\nThe caption lines (time: "
                     "text):\n{{input}}\n\nThe chosen moments (start, kind, label, image prompt):\n{{output}}\n\nJudge: "
                     "each moment is on a line that explains something visual (not a greeting or announcement); the "
                     "picture matches what was said; prompts ask for no written words. " + _SCALE),
    "inbox": ("inbox reply draft", ("traceName", ["inbox.draft", "inbox.assistant"]),
              "An AI drafted a reply for a teacher to a student's question.\nThe question and context:\n{{input}}\n\n"
              "The draft:\n{{output}}\n\nJudge: correct, answers what was asked, polite, short enough for a chat, in the "
              "student's language. " + _SCALE),
}


def judge_model() -> str:
    return os.getenv("EVAL_JUDGE_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"


def _all(page_fn) -> list:
    out, cursor = [], None
    while True:
        page = page_fn(limit=100, cursor=cursor) if cursor else page_fn(limit=100)
        out += list(page.data)
        cursor = getattr(page.meta, "next_cursor", None) or getattr(page.meta, "cursor", None)
        if not cursor:
            return out


def ensure_connection(api) -> str:
    """The provider name of a Google AI Studio connection in the project (created from GOOGLE_API_KEY if missing)."""
    from langfuse.api.llm_connections.types.llm_adapter import LlmAdapter

    conns = list(api.llm_connections.list(limit=100).data)
    for c in conns:
        if str(c.adapter) in ("google-ai-studio", "LlmAdapter.GOOGLE_AI_STUDIO"):
            return c.provider
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Langfuse needs a judge model: add GOOGLE_API_KEY to .env, or add an LLM connection in "
                           "Langfuse (Settings > LLM Connections)")
    api.llm_connections.upsert(provider=PROVIDER, adapter=LlmAdapter.GOOGLE_AI_STUDIO, secret_key=key,
                               custom_models=[judge_model()], with_default_models=True)
    return PROVIDER


def _filters(column: str, values: list[str]) -> list:
    from langfuse.api.evaluation_commons.types.evaluation_rule_filter import (
        EvaluationRuleFilter_ArrayOptions, EvaluationRuleFilter_Boolean, EvaluationRuleFilter_StringOptions)

    feature = (EvaluationRuleFilter_ArrayOptions(column="tags", operator="any of", value=values) if column == "tags"
               else EvaluationRuleFilter_StringOptions(column="traceName", operator="any of", value=values))
    return [EvaluationRuleFilter_Boolean(column="isRootObservation", operator="=", value=True), feature]


def setup(only: list[str] | None = None) -> dict:
    """Create what is missing (matched by name); never duplicates. Returns {suite: {evaluator, rule}}."""
    from langfuse.api.evaluation_commons.types.evaluator_output_definition import EvaluatorOutputDefinition_Numeric
    from langfuse.api.evaluation_commons.types.prompt_variable_mapping_input import PromptVariableMappingInput
    from langfuse.api.evaluation_rules.types.evaluation_rule_evaluator_assignment_input import (
        EvaluationRuleEvaluatorAssignmentInput)
    from langfuse.api.evaluators.types.create_evaluator_request import CreateEvaluatorRequest_LlmAsJudge
    from langfuse.api.evaluators.types.evaluator_model_config import EvaluatorModelConfig

    client = observability.langfuse_client()
    if client is None:
        raise RuntimeError("Langfuse is not configured (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY, `uv add langfuse`)")
    api = client.api
    provider = ensure_connection(api)
    evaluators = {e.name: e for e in _all(api.evaluators.list)}
    rules = {r.name: r for r in _all(api.evaluation_rules.list)}
    mapping = [PromptVariableMappingInput(variable="input", source="input"),
               PromptVariableMappingInput(variable="output", source="output")]
    done = {}
    for suite, (title, (column, values), prompt) in JUDGES.items():
        if only and suite not in only:
            continue
        name = PREFIX + title
        ev = evaluators.get(name)
        if ev is None:
            ev = api.evaluators.create(request=CreateEvaluatorRequest_LlmAsJudge(
                name=name, description=f"Rafeeq M5 instructor feature '{suite}': LLM-as-a-judge, 0-1",
                prompt=prompt, model_config_=EvaluatorModelConfig(provider=provider, model=judge_model()),
                variable_mapping=mapping,
                output_definition=EvaluatorOutputDefinition_Numeric(
                    min_value=0.0, max_value=1.0,
                    score_value_instructions="0 = unusable, 1 = usable as is",
                    score_reasoning_instructions="One sentence: the main reason for the score.")))
        rule = rules.get(name)
        if rule is None:
            rule = api.evaluation_rules.create(
                name=name, enabled=True, sampling=1.0, filter=_filters(column, values),
                evaluator_assignments=[EvaluationRuleEvaluatorAssignmentInput(evaluator_id=ev.id, variable_mapping=mapping)])
        done[suite] = {"evaluator": name, "evaluator_id": ev.id, "rule_id": rule.id, "traces": f"{column} = {', '.join(values)}"}
    return done


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ImportError:
        pass
    for suite, d in setup().items():
        print(f"{suite:16} evaluator '{d['evaluator']}' -> rule on {d['traces']}")
