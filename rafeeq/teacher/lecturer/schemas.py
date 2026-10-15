"""
Pydantic models = the shape of the output we ask the LLM to produce.

Why Pydantic and not TypedDict?
- with_structured_output(Model) sends the schema (field names + descriptions) to the model.
- After the reply, Pydantic validates it. If the model returns something wrong, an error is
  raised here instead of the bad data flowing on through the graph.

Every Field(description=...) is sent to the model, so write it as if you were explaining
to the model what the field should contain.

Note: RoutingDecision (the orchestrator's decision) is created in Phase 4 inside
nodes/orchestrator.py. It needs AgentName from state.py, and putting it here
would cause a circular import.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------- research
class OutlineOutput(BaseModel):
    sections: list[str] = Field(
        description="Ordered section titles of the session, from the first to the last."
    )


# ------------------------------------------------------------------ slides
class Slide(BaseModel):
    title: str = Field(description="Short slide title.")
    bullets: list[str] = Field(
        description="3 to 5 short bullet points. Each bullet is at most 12 words."
    )
    notes: str = Field(
        description="Speaker notes: 2 to 4 sentences the lecturer can say while showing this slide."
    )
    diagram_steps: list[str] = Field(
        default_factory=list,
        description=(
            "Optional small flow diagram drawn next to the bullets: 3 to 5 steps in order, "
            "each at most 5 words. Use it only when the slide explains a process, a sequence "
            "or a cause-and-effect chain. Otherwise leave it empty."
        ),
    )


class SlidesOutput(BaseModel):
    slides: list[Slide]


# -------------------------------------------------------------------- code
class CodeExample(BaseModel):
    title: str = Field(description="What this example demonstrates.")
    code: str = Field(
        description="Complete, runnable code. Self-contained: no input(), no files, no network."
    )
    explanation: str = Field(
        description=(
            "Plain text, no markdown and no code fences. "
            "At most 5 short sentences: what the code does, then what it prints."
        )
    )


class CodeOutput(BaseModel):
    examples: list[CodeExample]


# -------------------------------------------------------------------- quiz
class QuizQuestion(BaseModel):
    question: str
    options: list[str] = Field(description="Exactly 4 answer options.")
    answer_index: int = Field(description="0-based index of the single correct option.")
    explanation: str = Field(description="1 to 2 sentences explaining why the answer is correct.")

    @field_validator("options")
    @classmethod
    def must_have_four_options(cls, v: list[str]) -> list[str]:
        if len(v) != 4:
            raise ValueError("options must contain exactly 4 items")
        return v

    @model_validator(mode="after")
    def answer_in_range(self):
        if not 0 <= self.answer_index < len(self.options):
            raise ValueError("answer_index must point to one of the options")
        return self


class QuizOutput(BaseModel):
    # Wrapper around the list: some providers do not accept a list as the schema root.
    questions: list[QuizQuestion]

# ===================================================================== API
# The shapes below are what the browser sends to FastAPI (not what the LLM returns).


class SessionRequest(BaseModel):
    """LecturerSessionRequest in contracts/openapi.yaml."""
    lecturer_id: str = Field(min_length=1, max_length=80)
    lecturer_name: str = Field(default="", max_length=120)  # shown on every slide; default = lecturer_id
    room_id: str = Field(default="", max_length=80)
    topic: str = Field(min_length=2, max_length=200)
    course_name: str = Field(default="", max_length=200)
    student_level: Literal["beginner", "intermediate", "advanced"] = "intermediate"
    duration_minutes: int = Field(default=60, ge=15, le=240)
    language: str = Field(default="English", max_length=40)
    programming_language: str = Field(default="Python", max_length=40)
    needs: list[Literal["slides", "code", "quiz"]] = Field(min_length=1)
    num_quiz_questions: int = Field(default=5, ge=1, le=20)
    learning_objectives: list[str] = Field(default_factory=list)
    notes: str = Field(default="", max_length=2000)

    @field_validator("learning_objectives", mode="before")
    @classmethod
    def objectives_as_list(cls, v):
        """The form sends one text box. Accept text (one objective per line) or a list."""
        if isinstance(v, str):
            v = v.splitlines()
        return [line.strip(" -\t") for line in (v or []) if line and line.strip(" -\t")]


class ApproveRequest(BaseModel):
    """LecturerDecision in contracts/openapi.yaml (session_id comes from the URL)."""
    decision: Literal["approve", "revise"]
    feedback: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def revise_needs_feedback(self):
        if self.decision == "revise" and not self.feedback.strip():
            raise ValueError("feedback is required when requesting changes")
        return self
