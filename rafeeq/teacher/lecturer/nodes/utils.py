"""Small helpers shared by all agents, so the same formatting code is not repeated."""

import logging

from rafeeq.teacher.lecturer.llm import get_llm


def build_chain(prompt, schema, temperature: float = 0.3, attempts: int = 3):
    """prompt | llm with structured output, retried automatically if the model fails."""
    llm = get_llm(temperature=temperature).with_structured_output(schema)
    return (prompt | llm).with_retry(stop_after_attempt=attempts)

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