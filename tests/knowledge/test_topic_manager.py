from types import MappingProxyType

import pytest

from rafeeq.knowledge.topics import (
    DEFAULT_TOPICS,
    GENERAL_TOPIC_ID,
    Topic,
    TopicManager,
    list_topics,
    tag_topic,
)
from rafeeq.knowledge.topics.topic_manager import MAX_TOPICS_PER_ROOM

ROOM = "llm_101"

SELF_ATTENTION = "Self-attention allows each token to attend to other tokens."
CNN = "Convolutional neural networks are useful for image processing."
GRADIENT = "Gradient descent is used to optimize neural networks."
POSITIONAL = "Transformers use positional encoding to represent token positions."
LLM_TEXT = "Large language models are trained on large collections of text."


def chunk(text, room=ROOM, n=1):
    return {"chunk_id": f"c{n}", "room_id": room, "file_name": "b.pdf", "page": 1,
            "block_type": "text", "text": text}


# ---------- taxonomy ----------

def test_list_topics_format():
    topics = list_topics(ROOM)
    assert topics and all(set(t) == {"topic_id", "name"} for t in topics)
    assert all(isinstance(t["topic_id"], str) and isinstance(t["name"], str) for t in topics)


def test_topic_count_is_within_8_to_15_including_general():
    assert 8 <= len(list_topics(ROOM)) <= 15


def test_topic_ids_are_unique_stable_slugs_and_include_general():
    ids = [t["topic_id"] for t in list_topics(ROOM)]
    assert len(ids) == len(set(ids))
    assert all(i.startswith("topic_") and i == i.lower() and " " not in i for i in ids)
    assert GENERAL_TOPIC_ID in ids
    assert "topic_attention" in ids


def test_list_topics_is_identical_across_calls_and_instances():
    assert list_topics(ROOM) == list_topics(ROOM) == TopicManager().list_topics(ROOM)


def test_unknown_room_gets_the_default_taxonomy():
    assert list_topics("brand-new-room") == list_topics(ROOM)


# ---------- tagging ----------

def test_attention_text_is_tagged_attention():
    assert tag_topic(ROOM, SELF_ATTENTION) == "topic_attention"


def test_sample_course_sentences_get_sensible_topics():
    assert tag_topic(ROOM, CNN) == "topic_deep_learning_basics"
    assert tag_topic(ROOM, GRADIENT) == "topic_deep_learning_basics"
    assert tag_topic(ROOM, POSITIONAL) == "topic_transformers"
    assert tag_topic(ROOM, LLM_TEXT) == "topic_language_models"


def test_more_topic_examples():
    assert tag_topic(ROOM, "BPE and WordPiece are subword tokenization algorithms.") == "topic_tokenization"
    assert tag_topic(ROOM, "We use few-shot prompting and chain-of-thought prompts.") == "topic_prompting"
    assert tag_topic(ROOM, "Semantic search with a vector database enables RAG.") == "topic_semantic_search_rag"
    assert tag_topic(ROOM, "LoRA makes fine-tuning cheaper than full fine-tuning.") == "topic_fine_tuning"


def test_arabic_keywords_are_recognised():
    assert tag_topic(ROOM, "يعتمد الترانسفورمر على الانتباه الذاتي بين الكلمات") == "topic_attention"


def test_matching_is_case_insensitive_and_hyphen_or_space_tolerant():
    assert tag_topic(ROOM, "SELF ATTENTION lets tokens look at each other") == "topic_attention"


def test_whole_words_only():
    # "rag" must not match inside "storage" / "fragment", "agent" not inside "reagent"
    assert tag_topic(ROOM, "storage fragment fragmentation reagent") == GENERAL_TOPIC_ID


def test_unrelated_text_falls_back_to_general():
    assert tag_topic(ROOM, "The cafeteria opens at nine o'clock on Sundays.") == GENERAL_TOPIC_ID


def test_empty_and_whitespace_text_is_general():
    assert tag_topic(ROOM, "") == GENERAL_TOPIC_ID
    assert tag_topic(ROOM, "  \n\t ") == GENERAL_TOPIC_ID


def test_one_stray_keyword_is_not_enough():
    assert tag_topic(ROOM, "Today we mention an encoder once, in passing.") == GENERAL_TOPIC_ID


def test_tagging_is_deterministic_across_calls_and_instances():
    results = {TopicManager().tag_topic(ROOM, SELF_ATTENTION) for _ in range(5)}
    results |= {tag_topic(ROOM, SELF_ATTENTION) for _ in range(5)}
    assert results == {"topic_attention"}


def test_tagging_one_text_does_not_depend_on_other_texts_seen_before():
    manager = TopicManager()
    before = manager.tag_topic(ROOM, SELF_ATTENTION)
    for text in (CNN, GRADIENT, POSITIONAL, LLM_TEXT, "unrelated words"):  # "adding more files"
        manager.tag_topic(ROOM, text)
    assert manager.tag_topic(ROOM, SELF_ATTENTION) == before
    assert manager.list_topics(ROOM) == TopicManager().list_topics(ROOM)


def test_tie_goes_to_the_topic_listed_first():
    topics = (Topic("topic_first", "First", ("alpha beta",)), Topic("topic_second", "Second", ("alpha beta",)))
    assert TopicManager(topics).tag_topic(ROOM, "alpha beta") == "topic_first"
    assert TopicManager(topics[::-1]).tag_topic(ROOM, "alpha beta") == "topic_second"


def test_tag_topic_rejects_bad_input():
    with pytest.raises(TypeError):
        tag_topic(ROOM, None)
    with pytest.raises(ValueError, match="room_id"):
        tag_topic("", "text")
    with pytest.raises(ValueError, match="room_id"):
        list_topics("  ")


# ---------- per-room taxonomies ----------

def test_custom_room_taxonomy_is_isolated_from_other_rooms():
    manager = TopicManager()
    manager.register_room_topics("bio_101", [Topic("topic_cells", "Cells", ("mitochondria", "cell membrane"))])
    assert [t["topic_id"] for t in manager.list_topics("bio_101")] == ["topic_cells", GENERAL_TOPIC_ID]
    assert manager.tag_topic("bio_101", "The mitochondria sits inside the cell membrane.") == "topic_cells"
    assert "topic_cells" not in [t["topic_id"] for t in manager.list_topics(ROOM)]
    assert manager.tag_topic(ROOM, "The mitochondria sits inside the cell membrane.") == GENERAL_TOPIC_ID


def test_registering_the_same_taxonomy_twice_is_fine_but_changing_it_is_refused():
    manager = TopicManager()
    topics = [Topic("topic_cells", "Cells", ("mitochondria",))]
    manager.register_room_topics("bio_101", topics)
    manager.register_room_topics("bio_101", topics)
    with pytest.raises(ValueError, match="stable"):
        manager.register_room_topics("bio_101", [Topic("topic_other", "Other", ("x",))])


def test_taxonomy_validation():
    manager = TopicManager()
    with pytest.raises(ValueError, match="at least one"):
        manager.register_room_topics("r", [])
    with pytest.raises(ValueError, match="reserved"):
        manager.register_room_topics("r", [Topic(GENERAL_TOPIC_ID, "General", ("x",))])
    with pytest.raises(ValueError, match="duplicate"):
        manager.register_room_topics("r", [Topic("topic_a", "A", ("a",)), Topic("topic_a", "A2", ("b",))])
    with pytest.raises(ValueError, match="too many"):
        manager.register_room_topics("r", [Topic(f"topic_t{i}", f"T{i}", ("k",)) for i in range(15)])
    with pytest.raises(TypeError):
        manager.register_room_topics("r", ["topic_a"])


@pytest.mark.parametrize("bad_id", ["attention", "Topic_Attention", "topic-attention", "topic_", "topic__x", ""])
def test_topic_id_must_be_a_lowercase_slug(bad_id):
    with pytest.raises(ValueError, match="topic_id"):
        Topic(bad_id, "Name", ("k",))


def test_default_topics_are_valid_and_within_limits():
    assert 8 <= len(DEFAULT_TOPICS) + 1 <= 15
    assert all(t.keywords for t in DEFAULT_TOPICS)


# ---------- replaceable classifier ----------

class FixedClassifier:
    def __init__(self, topic_id):
        self.topic_id, self.calls = topic_id, []

    def classify(self, text, topics):
        self.calls.append((text, [t.topic_id for t in topics]))
        return self.topic_id


def test_a_different_classifier_can_be_plugged_in():
    clf = FixedClassifier("topic_prompting")
    manager = TopicManager(classifier=clf)
    assert manager.tag_topic(ROOM, SELF_ATTENTION) == "topic_prompting"
    assert clf.calls and "topic_attention" in clf.calls[0][1]


def test_classifier_returning_an_unknown_topic_is_rejected():
    manager = TopicManager(classifier=FixedClassifier("topic_made_up"))
    with pytest.raises(ValueError, match="not a topic of room"):
        manager.tag_topic(ROOM, SELF_ATTENTION)


# ---------- tagging chunks ----------

def test_tag_chunks_adds_topic_id_and_preserves_everything_else():
    originals = [chunk(SELF_ATTENTION, n=1), chunk(CNN, n=2)]
    tagged = TopicManager().tag_chunks(ROOM, originals)
    assert [c["topic_id"] for c in tagged] == ["topic_attention", "topic_deep_learning_basics"]
    for old, new in zip(originals, tagged):
        assert {k: v for k, v in new.items() if k != "topic_id"} == old


def test_tag_chunks_does_not_mutate_its_input():
    originals = [chunk(SELF_ATTENTION)]
    TopicManager().tag_chunks(ROOM, originals)
    assert "topic_id" not in originals[0]


def test_every_tagged_chunk_has_a_topic_id_that_is_listed_for_the_room():
    manager = TopicManager()
    texts = [SELF_ATTENTION, CNN, GRADIENT, POSITIONAL, LLM_TEXT, "nothing relevant here at all", ""]
    tagged = manager.tag_chunks(ROOM, [chunk(t, n=i) for i, t in enumerate(texts)])
    listed = {t["topic_id"] for t in manager.list_topics(ROOM)}
    assert all(c["topic_id"] in listed for c in tagged)


def test_existing_topic_id_is_kept_unless_overwrite_is_requested():
    manager = TopicManager()
    pre = dict(chunk(SELF_ATTENTION), topic_id="topic_prompting")
    assert manager.tag_chunks(ROOM, [pre])[0]["topic_id"] == "topic_prompting"
    assert manager.tag_chunks(ROOM, [pre], overwrite=True)[0]["topic_id"] == "topic_attention"


def test_tag_chunks_refuses_chunks_from_another_room():
    with pytest.raises(ValueError, match="room_id"):
        TopicManager().tag_chunks(ROOM, [chunk(SELF_ATTENTION, room="other_room")])


def test_tag_chunks_refuses_chunks_without_text():
    with pytest.raises(ValueError, match="text"):
        TopicManager().tag_chunks(ROOM, [{"room_id": ROOM}])


# =====================================================================
# Hardening: validated existing topic ids, validated classifier results
# =====================================================================

class ReturningClassifier:
    """Classifier that returns whatever it is told to (to test result validation)."""

    def __init__(self, value):
        self.value, self.calls = value, 0

    def classify(self, text, topics):
        self.calls += 1
        return self.value


# ---------- existing topic_id in tag_chunks ----------

def test_valid_existing_topic_id_is_preserved_without_calling_the_classifier():
    clf = FixedClassifier("topic_evaluation")
    pre = dict(chunk(SELF_ATTENTION), topic_id="topic_prompting")
    out = TopicManager(classifier=clf).tag_chunks(ROOM, [pre])
    assert out[0]["topic_id"] == "topic_prompting"
    assert clf.calls == []


def test_general_topic_id_counts_as_valid_and_is_preserved():
    pre = dict(chunk(SELF_ATTENTION), topic_id=GENERAL_TOPIC_ID)
    assert TopicManager().tag_chunks(ROOM, [pre])[0]["topic_id"] == GENERAL_TOPIC_ID


@pytest.mark.parametrize("bad", [
    "topic_made_up", "Topic_Attention", "attention", " topic_attention", "topic_attention ", "   ", 0, 5, False, ["topic_attention"],
])
def test_unknown_or_invalid_existing_topic_id_raises_value_error(bad):
    pre = dict(chunk(SELF_ATTENTION), topic_id=bad)
    with pytest.raises(ValueError, match="topic_id"):
        TopicManager().tag_chunks(ROOM, [pre])


def test_unknown_existing_topic_id_error_names_the_chunk_the_id_and_the_room():
    chunks = [chunk(CNN, n=1), dict(chunk(SELF_ATTENTION, n=2), topic_id="topic_made_up")]
    with pytest.raises(ValueError, match=r"chunks\[1\].*topic_made_up.*not a topic of room.*llm_101"):
        TopicManager().tag_chunks(ROOM, chunks)


def test_existing_id_must_belong_to_that_rooms_taxonomy_not_just_the_default_one():
    manager = TopicManager()
    manager.register_room_topics("bio_101", [Topic("topic_cells", "Cells", ("mitochondria",))])
    foreign = dict(chunk("mitochondria", room="bio_101"), topic_id="topic_attention")  # valid elsewhere only
    with pytest.raises(ValueError, match="not a topic of room"):
        manager.tag_chunks("bio_101", [foreign])
    own = dict(chunk("mitochondria", room="bio_101"), topic_id="topic_cells")
    assert manager.tag_chunks("bio_101", [own])[0]["topic_id"] == "topic_cells"


@pytest.mark.parametrize("missing", [None, ""])
def test_missing_none_or_empty_topic_id_is_classified(missing):
    out = TopicManager().tag_chunks(ROOM, [dict(chunk(SELF_ATTENTION), topic_id=missing)])
    assert out[0]["topic_id"] == "topic_attention"


def test_overwrite_replaces_a_valid_existing_topic_id_with_the_classifier_result():
    clf = FixedClassifier("topic_evaluation")
    pre = dict(chunk(SELF_ATTENTION), topic_id="topic_prompting")
    out = TopicManager(classifier=clf).tag_chunks(ROOM, [pre], overwrite=True)
    assert out[0]["topic_id"] == "topic_evaluation"
    assert len(clf.calls) == 1


def test_overwrite_also_replaces_an_invalid_existing_topic_id():
    pre = dict(chunk(SELF_ATTENTION), topic_id="topic_made_up")
    assert TopicManager().tag_chunks(ROOM, [pre], overwrite=True)[0]["topic_id"] == "topic_attention"


def test_a_failure_returns_nothing_instead_of_a_partial_result():
    chunks = [chunk(SELF_ATTENTION, n=1), dict(chunk(CNN, n=2), topic_id="topic_made_up")]
    result = None
    with pytest.raises(ValueError):
        result = TopicManager().tag_chunks(ROOM, chunks)
    assert result is None


def test_every_returned_chunk_has_a_valid_topic_id_for_mixed_inputs():
    manager = TopicManager()
    chunks = [
        chunk(SELF_ATTENTION, n=1),                                      # classified
        dict(chunk(CNN, n=2), topic_id="topic_prompting"),               # preserved
        dict(chunk("", n=3), topic_id=""),                               # empty text -> general
        dict(chunk(GRADIENT, n=4), topic_id="topic_evaluation"),         # preserved
    ]
    valid = {t["topic_id"] for t in manager.list_topics(ROOM)}
    for overwrite in (False, True):
        out = manager.tag_chunks(ROOM, chunks, overwrite=overwrite)
        assert len(out) == len(chunks)
        assert all(isinstance(c["topic_id"], str) and c["topic_id"] in valid for c in out)


# ---------- classifier results in tag_topic ----------

@pytest.mark.parametrize("bad", ["topic_made_up", "Topic_Attention", " topic_attention"])
def test_classifier_returning_an_unknown_id_raises_and_is_not_mapped_to_general(bad):
    with pytest.raises(ValueError, match="classifier returned .* not a topic of room"):
        TopicManager(classifier=ReturningClassifier(bad)).tag_topic(ROOM, SELF_ATTENTION)


@pytest.mark.parametrize("bad", ["", "   "])
def test_classifier_returning_an_empty_string_raises(bad):
    with pytest.raises(ValueError, match="classifier returned an invalid topic id"):
        TopicManager(classifier=ReturningClassifier(bad)).tag_topic(ROOM, SELF_ATTENTION)


@pytest.mark.parametrize("bad", [None, 5, 1.5, True, b"topic_attention", ["topic_attention"], {"topic_attention"}])
def test_classifier_returning_a_non_string_raises_value_error_even_if_unhashable(bad):
    with pytest.raises(ValueError, match="classifier returned an invalid topic id"):
        TopicManager(classifier=ReturningClassifier(bad)).tag_topic(ROOM, SELF_ATTENTION)


def test_classifier_result_is_checked_against_the_rooms_own_taxonomy():
    manager = TopicManager(classifier=ReturningClassifier("topic_attention"))  # valid in the default room only
    manager.register_room_topics("bio_101", [Topic("topic_cells", "Cells", ("mitochondria",))])
    assert manager.tag_topic(ROOM, SELF_ATTENTION) == "topic_attention"
    with pytest.raises(ValueError, match="not a topic of room"):
        manager.tag_topic("bio_101", "anything")


def test_invalid_classifier_result_also_fails_inside_tag_chunks():
    manager = TopicManager(classifier=ReturningClassifier(None))
    with pytest.raises(ValueError, match="classifier returned"):
        manager.tag_chunks(ROOM, [chunk(SELF_ATTENTION)])


def test_a_valid_classifier_result_is_returned_unchanged():
    assert TopicManager(classifier=ReturningClassifier("topic_prompting")).tag_topic(ROOM, SELF_ATTENTION) == "topic_prompting"


# ---------- empty text ----------

def test_empty_text_gets_topic_general_without_consulting_the_classifier():
    clf = ReturningClassifier("topic_prompting")
    manager = TopicManager(classifier=clf)
    assert manager.tag_topic(ROOM, "") == GENERAL_TOPIC_ID
    assert manager.tag_topic(ROOM, " \n\t") == GENERAL_TOPIC_ID
    assert clf.calls == 0


def test_empty_text_chunk_gets_topic_general_in_tag_chunks():
    out = TopicManager().tag_chunks(ROOM, [chunk("   ")])
    assert out[0]["topic_id"] == GENERAL_TOPIC_ID


# ---------- taxonomy registration stays stable ----------

def test_a_rejected_registration_leaves_the_original_taxonomy_untouched():
    manager = TopicManager()
    original = [Topic("topic_cells", "Cells", ("mitochondria",))]
    manager.register_room_topics("bio_101", original)
    before = manager.list_topics("bio_101")
    with pytest.raises(ValueError, match="stable"):
        manager.register_room_topics("bio_101", [Topic("topic_other", "Other", ("x",))])
    assert manager.list_topics("bio_101") == before
    assert manager.tag_topic("bio_101", "The mitochondria sits here, mitochondria again.") == "topic_cells"


def test_same_ids_with_different_names_keywords_or_order_are_a_different_taxonomy():
    manager = TopicManager()
    a, b = Topic("topic_a", "A", ("alpha",)), Topic("topic_b", "B", ("beta",))
    manager.register_room_topics("r", [a, b])
    for different in (
        [Topic("topic_a", "Renamed A", ("alpha",)), b],
        [Topic("topic_a", "A", ("alpha", "extra")), b],
        [b, a],
    ):
        with pytest.raises(ValueError, match="stable"):
            manager.register_room_topics("r", different)
    manager.register_room_topics("r", [a, b])  # the identical taxonomy is still accepted


def test_registering_a_room_does_not_change_other_rooms_or_the_default():
    manager = TopicManager()
    default_before = manager.list_topics(ROOM)
    manager.register_room_topics("bio_101", [Topic("topic_cells", "Cells", ("mitochondria",))])
    assert manager.list_topics(ROOM) == default_before
    assert manager.list_topics("another_room") == default_before


# ---------- inputs are never mutated ----------

def test_original_chunk_mappings_are_unchanged_on_success_and_on_failure():
    good = chunk(SELF_ATTENTION, n=1)
    preset = dict(chunk(CNN, n=2), topic_id="topic_prompting")
    bad = dict(chunk(GRADIENT, n=3), topic_id="topic_made_up")
    snapshots = [dict(good), dict(preset), dict(bad)]
    manager = TopicManager()
    manager.tag_chunks(ROOM, [good, preset])
    manager.tag_chunks(ROOM, [good, preset], overwrite=True)
    with pytest.raises(ValueError):
        manager.tag_chunks(ROOM, [good, preset, bad])
    assert [good, preset, bad] == snapshots
    assert "topic_id" not in good


def test_read_only_mappings_are_accepted_and_returned_as_new_plain_dicts():
    read_only = MappingProxyType(chunk(SELF_ATTENTION))
    out = TopicManager().tag_chunks(ROOM, [read_only])
    assert out[0] is not read_only and isinstance(out[0], dict)
    assert out[0]["topic_id"] == "topic_attention" and "topic_id" not in read_only


def test_returned_chunks_are_new_dicts_even_when_nothing_changes():
    pre = dict(chunk(SELF_ATTENTION), topic_id="topic_attention")
    out = TopicManager().tag_chunks(ROOM, [pre])
    assert out[0] == pre and out[0] is not pre


# ---------- maximum topic limit ----------

def test_default_taxonomy_respects_the_maximum_topic_limit():
    assert len(DEFAULT_TOPICS) + 1 <= MAX_TOPICS_PER_ROOM
    assert len(list_topics(ROOM)) <= MAX_TOPICS_PER_ROOM
    TopicManager(DEFAULT_TOPICS)  # constructing with the default taxonomy must succeed


def test_custom_taxonomy_at_the_limit_is_accepted_and_one_over_is_rejected():
    at_limit = [Topic(f"topic_t{i}", f"T{i}", ("kw",)) for i in range(MAX_TOPICS_PER_ROOM - 1)]
    manager = TopicManager()
    manager.register_room_topics("full_room", at_limit)
    assert len(manager.list_topics("full_room")) == MAX_TOPICS_PER_ROOM
    over = at_limit + [Topic("topic_one_too_many", "Too many", ("kw",))]
    with pytest.raises(ValueError, match="too many"):
        manager.register_room_topics("over_room", over)
    assert manager.list_topics("over_room") == manager.list_topics(ROOM)  # rejected: room keeps the default


def test_the_limit_also_applies_to_the_default_taxonomy_passed_to_the_constructor():
    over = [Topic(f"topic_t{i}", f"T{i}", ("kw",)) for i in range(MAX_TOPICS_PER_ROOM)]
    with pytest.raises(ValueError, match="too many"):
        TopicManager(default_topics=over)
    TopicManager(default_topics=over[:-1])  # exactly at the limit once the general topic is added