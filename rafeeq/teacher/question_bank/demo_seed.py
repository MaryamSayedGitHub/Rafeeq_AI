"""Sample questions for trying the question bank without a PDF or an LLM (the "Add sample questions" button)."""
from __future__ import annotations

from sqlalchemy import select

from rafeeq.quiz.contracts import QuizItem
from rafeeq.teacher.question_bank.contracts import Grading
from rafeeq.teacher.question_bank.models import BankItemRow
from rafeeq.teacher.question_bank.service import QuestionBankService, _item_to_row

TOPIC = "demo"
FILE = "sample-course.pdf"

# (kind, difficulty, approved, stem, options, right_choice, answer, explanation, rubric, page)
SAMPLES = [
    ("mcq", "easy", True,
     "What does \"multimodal\" mean in a multimodal AI model?",
     ["It handles more than one type of data, such as text and images", "It runs on several GPUs", "It has several output languages", "It was trained several times"], 0,
     "", "A multimodal model takes in and relates several kinds of data (modalities), for example text, images and audio.", [], 1),
    ("mcq", "medium", False,
     "In a multimodal model such as CLIP, what is the image encoder and the text encoder trained to do?",
     ["Translate captions between languages", "Place an image and its caption close together in one shared space", "Compress images into smaller files", "Count the objects in an image"], 1,
     "", "CLIP is trained with a contrastive loss so matching image-text pairs end up close in the same embedding space.", [], 2),
    ("true_false", "easy", True,
     "True or false: a multimodal model can only work with images.",
     ["True", "False"], 1,
     "", "False. \"Multi\" means several modalities, not only images.", [], 1),
    ("short", "medium", False,
     "Give one example of a task that needs a multimodal model, and say which modalities it uses.",
     [], 0,
     "Visual question answering: the model reads an image and a text question, and answers in text.",
     "Any task that joins two kinds of input counts, as long as both modalities are named.", [], 3),
    ("essay", "hard", False,
     "Explain how a multimodal model can fuse text and image information, and compare early fusion with late fusion.",
     [], 0,
     "Early fusion joins the features of the modalities at the input or first layers, so the model learns their interaction from the start. Late fusion processes each modality with its own model and combines the outputs at the end.",
     "A full answer defines both fusion types and states one advantage or cost of each.",
     ["Defines early fusion", "Defines late fusion", "Compares them (interaction vs. simplicity or cost)"], 4),
    ("mcq", "hard", True,
     "Why does a Transformer need positional encoding when it reads a sequence of tokens?",
     ["Attention alone does not know the order of the tokens", "It makes the model smaller", "It removes the need for embeddings", "It translates tokens into pixels"], 0,
     "", "Self-attention treats the input as a set; positional encoding adds the order information.", [], 5),
]

GRADING = {
    "mcq": ("The student picks the correct option.", "No partial credit for a multiple-choice question.", "Any other option."),
    "true_false": ("The student picks the correct value.", "No partial credit.", "The other value."),
    "short": ("Names a valid task and its modalities.", "Names a valid task but not the modalities, or the reverse.", "An example that does not combine modalities."),
    "essay": ("Defines both fusion types and compares them.", "Defines only one, or compares without defining.", "Mixes up the two or gives no comparison."),
}


def seed(svc: QuestionBankService, room_id: str, teacher_id: str) -> int:
    """Add the sample questions to this room for this teacher. Safe to press twice: it adds nothing the second time."""
    added = 0
    with svc._session() as db:
        for number, (kind, level, approved, stem, options, right, answer, explanation, rubric, page) in enumerate(SAMPLES, 1):
            item_id = f"demo-{room_id}-{teacher_id}-{number}"
            if db.scalar(select(BankItemRow).where(BankItemRow.item_id == item_id)) is not None:
                continue
            item = QuizItem(item_id=item_id, topic_id=TOPIC, stem=stem, options=options, right_choice=right,
                            explanation=explanation, file_name=FILE, page=page, source="generated",
                            kind=kind, answer=answer, rubric=rubric)
            correct, partial, wrong = GRADING[kind]
            db.add(_item_to_row(item, room_id, teacher_id, approved=approved,
                                grading=Grading(correct=correct, partial=partial, wrong=wrong), difficulty=level))
            added += 1
        db.commit()
    return added
