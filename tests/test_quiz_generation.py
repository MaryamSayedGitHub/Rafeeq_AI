import json
import random

from rafeeq.quiz.contracts import QuizItem
from rafeeq.quiz.quiz.generator import generate_item, generate_quiz, select_chunks, shuffle_options
from rafeeq.quiz.quiz.validator import check_item

GOOD = {
    "stem": "What does a tokenizer do?",
    "options": ["Splits text into tokens", "Trains the model", "Stores vectors", "Draws images"],
    "right_choice": 0,
    "explanation": "It converts raw text into tokens.",
}


def chunk(topic="ch2", page=10, text="x" * 500):
    return {"chunk_id": f"b-p{page}", "text": text, "file_name": "book.pdf",
            "page": page, "topic_id": topic, "topic_name": topic}


def llm_returning(*replies):
    calls = iter(replies)
    return lambda prompt: next(calls)


def make_item(**over):
    base = dict(item_id="i1", topic_id="t", stem="What does a tokenizer do?",
                options=["a", "b", "c", "d"], right_choice=0, explanation="why",
                file_name="book.pdf", page=3)
    base.update(over)
    return QuizItem(**base)


def test_valid_item_has_no_problems():
    assert check_item(make_item()) == []


def test_duplicate_options_are_caught():
    assert "two options are duplicates" in check_item(make_item(options=["a", "A ", "c", "d"]))


def test_empty_explanation_and_missing_page_are_caught():
    problems = check_item(make_item(explanation=" ", page=0))
    assert "the explanation is empty" in problems
    assert "the item has no source page" in problems


def test_good_reply_passes_first_try():
    item, first = generate_item(llm_returning(json.dumps(GOOD)), chunk(page=42), "q-1")
    assert first is True and item is not None
    assert item.page == 42 and item.file_name == "book.pdf"  # from the chunk


def test_json_inside_code_fence_is_parsed():
    item, first = generate_item(llm_returning("```json\n" + json.dumps(GOOD) + "\n```"), chunk(), "q-1")
    assert first is True and item is not None


def test_bad_first_then_good_regenerates_once():
    bad = dict(GOOD, options=["same", "same", "c", "d"])
    item, first = generate_item(llm_returning(json.dumps(bad), json.dumps(GOOD)), chunk(), "q-1")
    assert item is not None and first is False


def test_bad_twice_returns_none():
    bad = dict(GOOD, options=["only", "three", "options"])
    item, first = generate_item(llm_returning(json.dumps(bad), "not json"), chunk(), "q-1")
    assert item is None and first is False


def test_llm_cannot_set_page_or_topic():
    sneaky = dict(GOOD, page=999, topic_id="hacked")
    item, _ = generate_item(llm_returning(json.dumps(sneaky)), chunk(topic="ch2", page=7), "q-1")
    assert item.page == 7 and item.topic_id == "ch2"


def test_select_chunks_aims_at_weak_topics_and_skips_front_matter():
    chunks = ([chunk("weak", p) for p in range(1, 21)]
              + [chunk("other", p) for p in range(1, 21)]
              + [chunk("index", p) for p in range(1, 21)])
    picked = select_chunks(chunks, 10, weak_topic_ids=["weak"], rng=random.Random(0))
    assert len(picked) == 10
    assert sum(c["topic_id"] == "weak" for c in picked) == 6
    assert all(c["topic_id"] != "index" for c in picked)


def test_select_chunks_works_with_no_weak_topics():
    chunks = [chunk("a", p) for p in range(1, 11)]
    assert len(select_chunks(chunks, 5, rng=random.Random(1))) == 5


def test_generate_quiz_reports_first_try_rate():
    chunks = [chunk("a", p) for p in range(1, 11)]
    llm = lambda prompt: json.dumps(GOOD)
    result = generate_quiz(llm, chunks, 4, rng=random.Random(0))
    assert len(result.items) == 4 and result.first_try_rate == 1.0
    assert len({i.item_id for i in result.items}) == 4
    
def test_shuffle_keeps_the_right_answer():
    item = make_item(options=["w", "x", "y", "z"], right_choice=2)
    for seed in range(30):
        s = shuffle_options(item, random.Random(seed))
        assert s.options[s.right_choice] == "y"
        assert sorted(s.options) == ["w", "x", "y", "z"]


def test_shuffle_spreads_the_right_answer_over_all_slots():
    item = make_item(options=["w", "x", "y", "z"], right_choice=1)
    slots = {shuffle_options(item, random.Random(seed)).right_choice for seed in range(40)}
    assert slots == {0, 1, 2, 3}


def test_context_dependent_questions_are_caught():
    for stem in ["According to the text, what is a tokenizer?",
                 "In the provided dataset example, which key holds labels?",
                 "What does the figure show about attention?"]:
        assert any("on its own" in p for p in check_item(make_item(stem=stem)))
    assert check_item(make_item(stem="What does a tokenizer do?")) == []
    
    
def test_normal_context_wording_is_allowed():
    assert check_item(make_item(stem="In the context of NLP, what is a token?")) == []
    
    
def test_references_to_unseen_context_are_caught():
    for stem in [
        "Which hardware was used, as referenced in the context of high computing costs?",
        "What is the purpose of the four combinations described in relation to predictions?",
        "In the specific architecture described using two networks, what happens?",
    ]:
        assert any("on its own" in p for p in check_item(make_item(stem=stem)))


def test_verifier_drops_items_a_second_reader_disagrees_with():
    import json

    from rafeeq.quiz.quiz.generator import generate_item

    chunk = {"topic_id": "t", "topic_name": "T", "file_name": "b.pdf", "page": 3, "text": "x" * 500}
    reply = json.dumps({
        "stem": "What does a tokenizer do to raw text?",
        "options": ["Splits it into tokens", "Trains the model", "Stores vectors", "Draws images"],
        "right_choice": 0,
        "explanation": "It splits text into tokens.",
    })
    llm = lambda prompt: reply
    agree = lambda prompt: '{"answer": "A", "supported": true, "reason": "ok"}'
    disagree = lambda prompt: '{"answer": "B", "supported": true, "reason": "no"}'

    item, first_try = generate_item(llm, chunk, "q-1", verifier=agree)
    assert item is not None and first_try
    item, _ = generate_item(llm, chunk, "q-1", verifier=disagree)
    assert item is None
    
    
def test_option_that_repeats_the_question_is_caught():
    stem = "A model that has gone through pretraining is called a pretrained model, even after fine-tuning."
    options = [stem, "Only encoder-only models are pretrained", "Pretrained means never fine-tuned", "Pretraining needs labeled data"]
    assert any("repeats the question" in p for p in check_item(make_item(stem=stem, options=options)))


def test_described_architecture_phrase_is_caught():
    stem = "What is the main purpose of the encoding step in the described architecture?"
    assert any("on its own" in p for p in check_item(make_item(stem=stem)))

def test_a_short_lesson_still_gives_the_questions_asked_for_each_on_a_different_idea():
    stems = iter(["What does a tokenizer do?", "Why are tokens mapped to ids?", "What is a vocabulary?", "What is BPE?"])
    prompts = []

    def llm(prompt):
        prompts.append(prompt)
        return json.dumps(dict(GOOD, stem=next(stems)))

    one_page = [chunk(text="y" * 300)]                       # shorter than the 400 characters a normal pick needs
    result = generate_quiz(llm, one_page, 3, rng=random.Random(0))
    assert len(result.items) == 3 and len({i.stem for i in result.items}) == 3
    assert "DIFFERENT idea" in prompts[-1] and "What does a tokenizer do?" in prompts[-1]


def test_the_top_up_drops_a_repeated_question_and_stops_after_two_tries_each():
    llm = lambda prompt: json.dumps(GOOD)                    # the model keeps writing the same question
    result = generate_quiz(llm, [chunk(text="y" * 300)], 2, rng=random.Random(0))
    assert len(result.items) == 1 and result.attempted == 4     # 1 kept, then 2 tries for each of the 2 missing... capped
