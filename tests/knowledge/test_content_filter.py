"""The index-page test used by Retriever(exclude_non_content=True)."""

import pytest

from rafeeq.knowledge.retrieval.content_filter import is_non_content_page, page_features

# Real-looking shapes. Each is built from the property being tested, not from the thresholds' output.
INDEX_ENTRIES = [
    ("self-attention", "Self-Attention Mechanism, Attention Heads"),
    ("tokenizer", "Tokenizer Vocabulary, Token Embeddings"),
    ("Transformer", "Transformer Blocks, Encoder Decoder Stacks"),
    ("retrieval", "Retrieval Augmented Generation, Dense Retrieval"),
    ("fine-tuning", "Full Fine-Tuning, Parameter Efficient Tuning"),
    ("embedding", "Text Embeddings, Sentence Embedding Models"),
]


def index_page(entries=INDEX_ENTRIES * 3):
    """An index-like page: "term, Section Title, Section Title" entries, 102 words with the default entries."""
    return " ".join(f"{term}, {titles}" for term, titles in entries)


PROSE = (
    "Attention is a mechanism that helps the model incorporate context as it is processing a specific token. "
    "For the model to predict what comes after the word, it needs to know what the word refers to. "
    "In a trained Transformer the attention mechanism makes that determination, and it adds information from the "
    "context into the representation of the token. We can see a simple version of that in the figure, which shows "
    "how the layer works. The model does that based on the patterns that it learned from the training data, and "
    "this is why the mechanism is so important for language models. It can also be extended to many positions."
)

PROMPT_TEMPLATE = (
    'prompt = """ I have a topic that contains the following documents: [DOCUMENTS] The topic is described by the '
    "following keywords: [KEYWORDS] Based on the information above extract a short topic label in the following "
    "format topic: <short topic label> Return only the label Documents Keywords Format Label Topic Short Name "
    "Description Output Result Input Title Summary Context Answer Question"
)


# ---------------------------------------------------------------- features, by hand

def test_features_match_a_hand_count():
    # words: The cat sat We can see it too -> 8; "sat." ends a sentence -> 1 per 8 words
    # capitalised: The, We -> 2/8; function words: The, We, can, it -> 4/8; commas: 1 -> 1/8
    f = page_features("The cat sat. We can see it, too")
    assert f == {"words": 8, "sentence_ends_per_100": 12.5, "capitalised": 0.25, "function_words": 0.5, "commas_per_word": 0.125}


def test_features_ignore_line_wrapping():
    wrapped = "The cat sat.\nWe can\n  see it,\ttoo"
    assert page_features(wrapped) == page_features("The cat sat. We can see it, too")


def test_text_without_latin_words_has_zero_features_and_is_never_flagged():
    arabic = "الانتباه الذاتي هو آلية تساعد النموذج على فهم السياق " * 20
    assert page_features(arabic)["words"] == 0
    assert is_non_content_page(arabic) is False
    assert is_non_content_page("") is False
    assert is_non_content_page("   \n ") is False


def test_non_string_is_rejected():
    with pytest.raises(TypeError):
        page_features(None)
    with pytest.raises(TypeError):
        is_non_content_page(123)


# ---------------------------------------------------------------- decisions

def test_an_index_like_page_is_flagged():
    text = index_page()
    assert page_features(text)["words"] >= 60
    assert is_non_content_page(text) is True


def test_ordinary_teaching_prose_is_not_flagged():
    assert is_non_content_page(PROSE) is False


def test_a_prompt_template_page_is_not_flagged():
    # no sentences and title-case labels like an index, but hardly any commas
    assert page_features(PROMPT_TEMPLATE)["sentence_ends_per_100"] < 1
    assert is_non_content_page(PROMPT_TEMPLATE) is False


def test_short_text_is_never_flagged():
    short = index_page(INDEX_ENTRIES[:2])
    assert page_features(short)["words"] < 60
    assert is_non_content_page(short) is False


def test_each_condition_alone_can_keep_an_index_like_page():
    base = index_page()
    assert is_non_content_page(base) is True
    with_sentences = base + " " + " ".join(["This is one."] * 8)             # sentence ends rise above the limit
    assert page_features(with_sentences)["sentence_ends_per_100"] >= 1 and is_non_content_page(with_sentences) is False
    no_commas = base.replace(",", "")                                          # comma density falls below the limit
    assert page_features(no_commas)["commas_per_word"] == 0 and is_non_content_page(no_commas) is False
    lowercase = base.lower()                                                   # no title-case terms
    assert page_features(lowercase)["capitalised"] < 0.25 and is_non_content_page(lowercase) is False
    wordy = base + " the of to is are we this that it can which a" * 10         # function words take over
    assert page_features(wordy)["function_words"] > 0.15 and is_non_content_page(wordy) is False


def test_decision_does_not_depend_on_unicode_spaces_or_wrapping():
    text = index_page().replace(" ", " \n ", 5)
    assert is_non_content_page(text) is True