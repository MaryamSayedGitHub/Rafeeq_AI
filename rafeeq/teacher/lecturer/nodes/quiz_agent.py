from rafeeq.teacher.lecturer.nodes.utils import build_chain, format_optional, format_outline, warn_if_count_differs
from rafeeq.teacher.lecturer.prompts import QUIZ_PROMPT
from rafeeq.teacher.lecturer.schemas import QuizOutput


def quiz_agent(state) -> dict:
    """Reads: brief, outline, feedback.   Writes: quiz."""
    brief = state["brief"]

    # Build the chain inside the function, not at import time: get_llm() needs .env to be loaded.
    chain = build_chain(QUIZ_PROMPT, QuizOutput, temperature=0.3)
    result = chain.invoke({
        "topic": brief["topic"],
        "outline": format_outline(state.get("outline")),
        "level": brief["student_level"],
        "num_questions": brief["num_quiz_questions"],
        "language": brief["language"],
        "feedback": format_optional(state.get("feedback")),
    })

    warn_if_count_differs("quiz_agent", len(result.questions), brief["num_quiz_questions"])

    # A partial update: only the key this node owns. model_dump() -> plain dicts (JSON friendly).
    return {"quiz": [q.model_dump() for q in result.questions]}


if __name__ == "__main__":
    fake_state = {
        "brief": {
            "topic": "Python decorators",
            "student_level": "intermediate",
            "num_quiz_questions": 3,
            "language": "English",
        },
        "outline": ["Functions are objects", "Closures", "Writing a decorator"],
        "feedback": "",
    }

    out = quiz_agent(fake_state)

    assert list(out.keys()) == ["quiz"], f"unexpected keys: {list(out.keys())}"
    assert len(out["quiz"]) == 3, f"expected 3 questions, got {len(out['quiz'])}"
    for q in out["quiz"]:
        assert len(q["options"]) == 4
        assert 0 <= q["answer_index"] < 4

    positions = [q["answer_index"] for q in out["quiz"]]
    print("correct answer positions:", positions)
    for q in out["quiz"]:
        print("-", q["question"])
    print("quiz_agent: OK")