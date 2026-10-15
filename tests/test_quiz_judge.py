import json

from rafeeq.quiz.eval import judge_quiz as jq

ITEM = {"item_id": "q1", "topic_id": "t", "page": 3, "stem": "What does a tokenizer do?",
        "options": ["Splits text into tokens", "Trains the model", "Stores vectors", "Draws images"],
        "right_choice": 0}


def reply(**kw):
    base = {"answer": "A", "supported": True, "reason": "page says so"}
    return lambda prompt: json.dumps({**base, **kw})


def test_prompt_contains_page_question_options_but_not_the_key():
    p = jq.build_judge_prompt("PAGE TEXT HERE", ITEM["stem"], ITEM["options"])
    assert "PAGE TEXT HERE" in p and ITEM["stem"] in p and "A) Splits text into tokens" in p
    assert "right_choice" not in p and "key" not in p.lower().replace("keyword", "")


def test_parse_accepts_letters_numbers_and_fenced_json():
    assert jq.parse_judgement('{"answer": "c", "supported": false, "reason": "x"}')["answer"] == 2
    assert jq.parse_judgement('{"answer": 3, "supported": true}')["answer"] == 3
    fenced = '```json\n{"answer": "B", "supported": true, "reason": "r"}\n```'
    assert jq.parse_judgement(fenced)["answer"] == 1
    assert jq.parse_judgement('Sure! {"answer": "D", "supported": true} done')["answer"] == 3


def test_parse_rejects_garbage():
    for bad in ["", "no json", '{"answer": "E", "supported": true}', '{"answer": 7, "supported": true}',
                '{"answer": "A"}', '{"answer": "A", "supported": "yes"}', "[1, 2]"]:
        assert jq.parse_judgement(bad) is None


def test_agreement_is_computed_against_the_answer_key():
    agree = jq.judge_item(reply(answer="A"), "page", ITEM)
    disagree = jq.judge_item(reply(answer="B"), "page", ITEM)
    assert agree["agree"] is True and disagree["agree"] is False
    assert jq.judge_item(lambda p: "garbage", "page", ITEM)["error"]


def test_summary_counts_solid_and_lists_what_to_review():
    rows = [
        jq.judge_item(reply(answer="A", supported=True), "p", ITEM),   # solid
        jq.judge_item(reply(answer="A", supported=False), "p", ITEM),  # agrees but page unclear
        jq.judge_item(reply(answer="C", supported=True), "p", ITEM),   # disagrees
        {"item_id": "x", "error": "judge answer could not be read"},
    ]
    s = jq.summarize(rows, first_try_rate=0.8)
    assert s["asked"] == 4 and s["judged"] == 3
    assert round(s["agree"], 2) == 0.67 and round(s["supported"], 2) == 0.67
    assert round(s["solid"], 2) == 0.33
    assert len(s["to_review"]) == 2 and s["first_try_validation"] == 0.8


def test_summary_of_nothing_does_not_crash():
    assert jq.summarize([])["solid"] == 0.0


def test_pacing_waits_between_calls():
    import time

    calls = []
    slow = jq.paced(lambda p: calls.append(time.time()) or "ok", 0.15)
    slow("a"), slow("b")
    assert calls[1] - calls[0] >= 0.14