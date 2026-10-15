"""Sample JSON for every M4 endpoint, so M1 and M5 can call us from Day 1."""
from rafeeq.quiz.contracts import (
    ClassProfileRow,
    GradeSheetResponse,
    DetectedAnswer,
    ItemResult,
    Profile,
    QuizCreateResponse,
    QuizItem,
    SubmitResponse,
    TopicProfile,
    WrongAnswerRow,
)

SAMPLE_ITEM = QuizItem(
    item_id="item-1",
    topic_id="topic-1",
    stem="What does a tokenizer do?",
    options=[
        "Splits text into tokens",
        "Trains the model",
        "Stores embeddings",
        "Generates images",
    ],
    right_choice=0,
    explanation="A tokenizer converts raw text into tokens the model can read.",
    file_name="sample.pdf",
    page=12,
    source="generated",
)

SAMPLE_ITEM_2 = SAMPLE_ITEM.model_copy(
    update={
        "item_id": "item-2",
        "stem": "Which layer gives contextual embeddings?",
        "options": ["Tokenizer", "Transformer", "Softmax", "Dropout"],
        "right_choice": 1,
        "page": 14,
    }
)


def profile() -> Profile:
    return Profile(
        topics=[
            TopicProfile(topic_id="topic-1", name="Tokenization", mastery=0.45,
                         attempts=3, questions_asked=4, weak=True),
            TopicProfile(topic_id="topic-2", name="Embeddings", mastery=0.82,
                         attempts=2, questions_asked=1, weak=False),
        ]
    )


def quiz(n: int = 2, mode: str = "screen") -> QuizCreateResponse:
    return QuizCreateResponse(
        quiz_id="quiz-1",
        items=[SAMPLE_ITEM, SAMPLE_ITEM_2][:max(1, min(n, 2))],
        sheet_pdf="/sheets/quiz-1.pdf" if mode == "paper" else None,
    )


def submit() -> SubmitResponse:
    return SubmitResponse(
        score=0.5,
        items=[
            ItemResult(item_id="item-1", correct=True, right_choice=0,
                       explanation=SAMPLE_ITEM.explanation,
                       file_name="sample.pdf", page=12),
            ItemResult(item_id="item-2", correct=False, right_choice=1,
                       explanation=SAMPLE_ITEM_2.explanation,
                       file_name="sample.pdf", page=14),
        ],
    )


def grade_sheet() -> GradeSheetResponse:
    return GradeSheetResponse(
        answers=[
            DetectedAnswer(item_id="item-1", choice=0, confidence=0.97),
            DetectedAnswer(item_id="item-2", choice=None, confidence=0.2),
        ],
        score=0.5,
        needs_review=["item-2"],
    )


def class_profile() -> list[ClassProfileRow]:
    return [
        ClassProfileRow(user_id="u-1", name="Student One", topic_id="topic-1",
                        mastery=0.45, attempts=3),
        ClassProfileRow(user_id="u-2", name="Student Two", topic_id="topic-1",
                        mastery=0.7, attempts=2),
    ]


def wrong_answers() -> list[WrongAnswerRow]:
    return [
        WrongAnswerRow(item_id="item-2", student_ref="s-a1", topic_id="topic-2",
                       stem=SAMPLE_ITEM_2.stem, chosen=0, right=1),
    ]
