"""Small helpers shared by all agents, so the same formatting code is not repeated."""

import logging
import re

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda

from rafeeq.teacher.lecturer.llm import get_llm


def _strip_thinking(message) -> str:
    """Model text without <think>...</think> (Qwen-style reasoning) so only the JSON is parsed."""
    text = getattr(message, "content", message)
    if isinstance(text, list):
        text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in text)
    return re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL).strip()


def build_chain(prompt, schema, temperature: float = 0.3, attempts: int = 3):
    """prompt | llm with structured output, retried automatically if the model fails.

    First choice: the provider's tool calling (with_structured_output). Some models (e.g. Qwen on Groq) sometimes
    call a tool name that does not exist ("tool call validation failed"), and that repeats on retry. Then the
    same prompt is asked again for plain JSON with the schema written in, and parsed with Pydantic."""
    llm = get_llm(temperature=temperature)
    with_tools = (prompt | llm.with_structured_output(schema)).with_retry(stop_after_attempt=2)

    parser = PydanticOutputParser(pydantic_object=schema)
    instructions = parser.get_format_instructions() + "\nAnswer with the JSON object only, no other text."

    def add_instructions(prompt_value):
        return prompt_value.to_messages() + [HumanMessage(content=instructions)]

    as_json = (prompt | RunnableLambda(add_instructions) | llm | RunnableLambda(_strip_thinking) | parser
               ).with_retry(stop_after_attempt=attempts)
    return with_tools.with_fallbacks([as_json])

def format_outline(outline: list[str] | None) -> str:
    """['A', 'B'] -> '- A\\n- B' (the prompts expect text, the state stores a list)."""
    if not outline:
        return "(no outline available)"
    return "\n".join(f"- {section}" for section in outline)


def format_optional(value) -> str:
    """
    Turn an optional state value into prompt text.
    None / "" / [] -> "None" (the prompts are written for this exact word)
    list[str]      -> bullet lines
    str            -> unchanged
    """
    if not value:
        return "None"
    if isinstance(value, (list, tuple)):
        return "\n".join(f"- {item}" for item in value)
    return str(value)


def warn_if_count_differs(name: str, got: int, expected: int) -> None:
    """Models sometimes miss the requested count by one or two. Warn, do not crash."""
    if got != expected:
        logging.getLogger(name).warning("expected %d items, got %d", expected, got)