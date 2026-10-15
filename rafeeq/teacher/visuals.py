"""LLM step of slide-picture generation: turn a slide into a good image prompt.

The lecturer assistant's LLM writes the prompt; the diffusion model (image_gen.py) draws it.
Without an LLM (no key, provider error) a plain template is used, and the caller is told.
"""

import logging

log = logging.getLogger(__name__)

_INSTRUCTION = (
    "Write ONE prompt (in English, at most 60 words) for an image-generation model. The image is an "
    "illustration on a teaching slide.\nSlide title: {title}\nSlide points: {points}\n"
    "Style: clean flat educational illustration, simple shapes, plenty of empty space, no text or letters "
    "in the image, no people's faces. Return only the prompt, nothing else."
)


def fallback_prompt(title: str, bullets: list[str]) -> str:
    topic = ", ".join([title] + [b for b in bullets[:2]])[:300]
    return (f"Clean flat educational illustration about: {topic}. Simple shapes, soft colours, plenty of empty "
            "space, no text or letters in the image.")


def suggest_prompt(title: str, bullets: list[str]) -> tuple[str, bool]:
    """(prompt, used_llm)."""
    try:
        from rafeeq.teacher.lecturer.llm import get_llm

        reply = get_llm(temperature=0.4).invoke(_INSTRUCTION.format(title=title, points="; ".join(bullets[:5])))
        content = getattr(reply, "content", reply)
        if isinstance(content, list):  # some providers return content blocks
            content = " ".join(str(b.get("text", "")) if isinstance(b, dict) else str(b) for b in content)
        text = str(content).strip().strip('"')
        if text:
            return text[:600], True
    except Exception as e:
        log.info("image prompt: no LLM (%s: %s), using the template", type(e).__name__, str(e)[:120])
    return fallback_prompt(title, bullets), False
