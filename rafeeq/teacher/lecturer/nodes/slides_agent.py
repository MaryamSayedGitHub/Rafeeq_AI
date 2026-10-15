from rafeeq.teacher.lecturer.nodes.utils import build_chain, format_optional, format_outline, warn_if_count_differs
from rafeeq.teacher.lecturer.prompts import SLIDES_PROMPT
from rafeeq.teacher.lecturer.schemas import SlidesOutput


def slides_agent(state) -> dict:
    """Reads: brief, outline, memory_context, feedback.   Writes: slides."""
    brief = state["brief"]

    # The number of slides is computed in code, then passed to the prompt ready-made.
    
    # was: max(5, min(12, duration // 5))
    num_slides = max(5, min(8, brief["duration_minutes"] // 8))

    chain = build_chain(SLIDES_PROMPT, SlidesOutput, temperature=0.3)
    result = chain.invoke({
        "topic": brief["topic"],
        "outline": format_outline(state.get("outline")),
        "level": brief["student_level"],
        "num_slides": num_slides,
        "language": brief["language"],
        "notes": format_optional(brief.get("notes")),
        "memory_context": format_optional(state.get("memory_context")),
        "feedback": format_optional(state.get("feedback")),
    })

    warn_if_count_differs("slides_agent", len(result.slides), num_slides)

    return {"slides": [s.model_dump() for s in result.slides]}


if __name__ == "__main__":
    fake_state = {
        "brief": {
            "topic": "Python decorators",
            "student_level": "intermediate",
            "duration_minutes": 30,
            "language": "English",
        },
        "outline": ["Functions are objects", "Closures", "Writing a decorator"],
        "memory_context": [],
        "feedback": "",
    }

    out = slides_agent(fake_state)

    assert list(out.keys()) == ["slides"], f"unexpected keys: {list(out.keys())}"
    assert len(out["slides"]) > 0
    for s in out["slides"]:
        assert s["title"] and s["bullets"] and s["notes"]

    for i, s in enumerate(out["slides"], 1):
        print(f"{i}. {s['title']}  ({len(s['bullets'])} bullets)")
    print("slides_agent: OK")

