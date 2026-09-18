# -*- coding: utf-8 -*-
"""
Tests for the BGE-M3 semantic-relevance layer and the surrounding
rank_evidence() contract.

Tests that actually need the encoder are marked `@requires_model` and skip
with a clear reason when `sentence-transformers` or the BAAI/bge-m3 weights
are unavailable (offline / air-gapped machine). Everything that can be
checked without the model - config merging, weight validation, empty-text
handling, import purity - always runs.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

import task4_rank_evidence as t4
from task4_rank_evidence import (
    DEFAULT_CONFIG,
    SemanticModelUnavailableError,
    merge_config,
    rank_evidence,
    semantic_relevance,
    validate_relevance_weights,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _model_available() -> bool:
    try:
        t4._get_semantic_model()
        return True
    except SemanticModelUnavailableError:
        return False


requires_model = pytest.mark.skipif(
    not _model_available(),
    reason="BAAI/bge-m3 weights or sentence-transformers unavailable "
           "(offline environment) - semantic tests skipped",
)


# ---------------------------------------------------------------------
# Fixtures / shared data
# ---------------------------------------------------------------------

EN_ON_TOPIC = {
    "passage_id": "EN-ON",
    "text": "Archaeological excavations near Ayodhya have documented occupation "
            "layers consistent with the historical narrative associated with "
            "Rama in the Ramayana epic.",
}
EN_OFF_TOPIC = {
    "passage_id": "EN-OFF",
    "text": "The weather today is sunny with a chance of light rain in the evening.",
}
HI_ON_TOPIC = {
    "passage_id": "HI-ON",
    "text": "अयोध्या में राम के जन्मस्थान से जुड़े पुरातात्विक साक्ष्य रामायण की "
            "परंपरा से मेल खाते हैं।",
}
HI_OFF_TOPIC = {
    "passage_id": "HI-OFF",
    "text": "आज मौसम साफ़ रहेगा और शाम को हल्की बारिश होने की संभावना है।",
}

EN_QUERY = "ancient archaeological evidence about Rama in Ayodhya"
HI_QUERY = "राम अयोध्या से जुड़ी प्राचीन जानकारी"
MIXED_QUERY = "Ayodhya में राम से जुड़ी ancient archaeological evidence"


# ---------------------------------------------------------------------
# 1. English
# ---------------------------------------------------------------------

@requires_model
def test_english_query_and_passages():
    passages = [EN_ON_TOPIC, EN_OFF_TOPIC]
    scores = semantic_relevance(EN_QUERY, passages)

    assert len(scores) == 2
    assert scores[0] > scores[1], "on-topic English passage must outscore off-topic"
    assert rank_evidence(EN_QUERY, passages)[0]["passage_id"] == "EN-ON"


# ---------------------------------------------------------------------
# 2. Hindi
# ---------------------------------------------------------------------

@requires_model
def test_hindi_query_and_passages():
    passages = [HI_ON_TOPIC, HI_OFF_TOPIC]
    scores = semantic_relevance(HI_QUERY, passages)

    assert len(scores) == 2
    assert scores[0] > scores[1], "on-topic Hindi passage must outscore off-topic"
    assert rank_evidence(HI_QUERY, passages)[0]["passage_id"] == "HI-ON"


@requires_model
def test_hindi_query_matches_english_passage_cross_lingually():
    """The point of a multilingual dense model: a Devanagari query must match
    an English passage on meaning, with no shared surface strings."""
    scores = semantic_relevance(HI_QUERY, [EN_ON_TOPIC, EN_OFF_TOPIC])
    assert scores[0] > scores[1]


# ---------------------------------------------------------------------
# 3. Mixed Hindi-English
# ---------------------------------------------------------------------

@requires_model
def test_mixed_hindi_english_query():
    passages = [HI_ON_TOPIC, EN_ON_TOPIC, HI_OFF_TOPIC, EN_OFF_TOPIC]
    scores = semantic_relevance(MIXED_QUERY, passages)

    assert len(scores) == 4
    on_topic = min(scores[0], scores[1])
    off_topic = max(scores[2], scores[3])
    assert on_topic > off_topic, (
        "a code-mixed query must rank both on-topic passages above both "
        "off-topic ones, in either script"
    )


@requires_model
def test_mixed_script_passage_is_handled():
    mixed_passage = {
        "passage_id": "MIX",
        "text": "ASI की report के अनुसार Ayodhya के excavation में प्राचीन "
                "occupation layers मिले हैं।",
    }
    scores = semantic_relevance(MIXED_QUERY, [mixed_passage, EN_OFF_TOPIC])
    assert scores[0] > scores[1]


# ---------------------------------------------------------------------
# 4. Empty passage text
# ---------------------------------------------------------------------

@requires_model
def test_empty_passage_text_scores_zero():
    passages = [
        EN_ON_TOPIC,
        {"passage_id": "EMPTY", "text": ""},
        {"passage_id": "BLANK", "text": "   \n\t "},
        {"passage_id": "MISSING"},                  # no "text" key at all
        {"passage_id": "NONE", "text": None},       # explicit None
    ]
    scores = semantic_relevance(EN_QUERY, passages)

    assert len(scores) == 5
    assert scores[0] > 0.0
    for i in (1, 2, 3, 4):
        assert scores[i] == 0.0, "empty passages must score exactly 0.0"


def test_all_empty_passages_need_no_model():
    """Only-empty input returns zeros in order without touching the encoder."""
    scores = semantic_relevance(
        HI_QUERY,
        [{"passage_id": "A", "text": ""}, {"passage_id": "B", "text": "  "}],
    )
    assert list(scores) == [0.0, 0.0]


@requires_model
def test_empty_passage_survives_rank_evidence():
    ranked = rank_evidence(EN_QUERY, [EN_ON_TOPIC, {"passage_id": "EMPTY", "text": ""}])
    by_id = {r["passage_id"]: r for r in ranked}
    assert len(ranked) == 2
    assert by_id["EMPTY"]["query_relevance"] == 0.0


# ---------------------------------------------------------------------
# 5. Lazy loading + model reuse
# ---------------------------------------------------------------------

def test_import_does_not_load_model():
    """Requirement: no model load and no network request at import time."""
    code = (
        "import sys, task4_rank_evidence as t4;"
        " assert t4._SEMANTIC_MODEL is None, 'model loaded at import';"
        " assert 'sentence_transformers' not in sys.modules, 'ST imported at import';"
        " assert 'torch' not in sys.modules, 'torch imported at import';"
        " print('clean')"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "clean" in proc.stdout


@requires_model
def test_model_loaded_once_and_reused_across_calls(monkeypatch):
    import sentence_transformers

    model = t4._get_semantic_model()          # make sure it is cached
    assert model is not None

    constructions = []
    real_ctor = sentence_transformers.SentenceTransformer

    def spy(*args, **kwargs):
        constructions.append(args)
        return real_ctor(*args, **kwargs)

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", spy)

    rank_evidence(EN_QUERY, [EN_ON_TOPIC, EN_OFF_TOPIC])
    rank_evidence(HI_QUERY, [HI_ON_TOPIC, HI_OFF_TOPIC])
    semantic_relevance(MIXED_QUERY, [EN_ON_TOPIC])

    assert constructions == [], "encoder was re-instantiated instead of reused"
    assert t4._SEMANTIC_MODEL is model


@requires_model
def test_repeated_calls_are_deterministic():
    a = semantic_relevance(EN_QUERY, [EN_ON_TOPIC, EN_OFF_TOPIC])
    b = semantic_relevance(EN_QUERY, [EN_ON_TOPIC, EN_OFF_TOPIC])
    assert list(a) == pytest.approx(list(b), abs=1e-6)


def test_unavailable_model_raises_actionable_error():
    """A missing model must fail loudly with install instructions - never
    fall back to the old character n-gram placeholder."""
    cached = t4._SEMANTIC_MODEL
    with pytest.raises(SemanticModelUnavailableError) as excinfo:
        semantic_relevance(
            EN_QUERY, [EN_ON_TOPIC],
            model_config={"name": "not-a-real-org/not-a-real-model-xyz"},
        )
    message = str(excinfo.value)
    assert "sentence-transformers" in message
    assert "BAAI/bge-m3" in message
    assert "pip install" in message
    # a failed load must not clobber an already-cached encoder
    assert t4._SEMANTIC_MODEL is cached


# ---------------------------------------------------------------------
# 6. Configuration: defaults, deep merge, partial override
# ---------------------------------------------------------------------

def test_default_config_contains_required_sections():
    assert DEFAULT_CONFIG["relevance_weights"] == {
        "lexical_weight": 0.30, "semantic_weight": 0.70,
    }
    assert DEFAULT_CONFIG["semantic_model"] == {
        "name": "BAAI/bge-m3", "batch_size": 16, "device": None,
    }


def test_partial_config_does_not_drop_unrelated_sections():
    merged = merge_config(DEFAULT_CONFIG, {"output": {"include_provenance": False}})

    # the named key changed...
    assert merged["output"]["include_provenance"] is False
    # ...its sibling inside the same section survived...
    assert merged["output"]["include_feature_breakdown"] is True
    # ...and every unrelated section survived
    for section in ("relevance_weights", "context_weights", "ranking_weights",
                    "filter", "semantic_model"):
        assert merged[section] == DEFAULT_CONFIG[section]


def test_partial_config_merges_nested_weights():
    merged = merge_config(DEFAULT_CONFIG, {"relevance_weights": {"lexical_weight": 0.5}})
    assert merged["relevance_weights"] == {"lexical_weight": 0.5, "semantic_weight": 0.70}


def test_merge_config_never_mutates_defaults():
    before = DEFAULT_CONFIG["output"]["include_provenance"]
    merged = merge_config(DEFAULT_CONFIG, {"output": {"include_provenance": False}})
    merged["context_weights"]["w_script"] = 999
    assert DEFAULT_CONFIG["output"]["include_provenance"] is before
    assert DEFAULT_CONFIG["context_weights"]["w_script"] != 999


@requires_model
def test_partial_config_override_through_rank_evidence():
    ranked = rank_evidence(EN_QUERY, [EN_ON_TOPIC, EN_OFF_TOPIC],
                           config={"output": {"include_provenance": False}})
    assert "provenance" not in ranked[0]
    assert "feature_breakdown" in ranked[0], "unrelated default was dropped"


@requires_model
def test_semantic_model_batch_size_override_is_honoured():
    passages = [EN_ON_TOPIC, HI_ON_TOPIC, EN_OFF_TOPIC, HI_OFF_TOPIC]
    big = semantic_relevance(MIXED_QUERY, passages, model_config={"batch_size": 16})
    small = semantic_relevance(MIXED_QUERY, passages, model_config={"batch_size": 1})
    assert list(big) == pytest.approx(list(small), abs=1e-5)


# ---------------------------------------------------------------------
# 7. Relevance-weight validation
# ---------------------------------------------------------------------

@pytest.mark.parametrize("weights", [
    {"lexical_weight": -0.1, "semantic_weight": 0.9},    # negative lexical
    {"lexical_weight": 0.3, "semantic_weight": -0.7},    # negative semantic
    {"lexical_weight": 0.0, "semantic_weight": 0.0},     # sum == 0
    {"lexical_weight": 0.3},                             # missing key
    {"semantic_weight": 0.7},                            # missing key
    {"lexical_weight": "a lot", "semantic_weight": 0.7},  # not a number
    {"lexical_weight": None, "semantic_weight": 0.7},    # not a number
    {"lexical_weight": float("nan"), "semantic_weight": 0.7},
    {"lexical_weight": float("inf"), "semantic_weight": 0.7},
    "not-a-dict",
])
def test_invalid_relevance_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        validate_relevance_weights(weights)


@pytest.mark.parametrize("weights", [
    {"lexical_weight": -0.1, "semantic_weight": 0.9},
    {"lexical_weight": 0.3, "semantic_weight": -0.7},
    {"lexical_weight": 0.0, "semantic_weight": 0.0},
    {"lexical_weight": "a lot", "semantic_weight": 0.7},
])
def test_rank_evidence_rejects_invalid_weights_before_loading_model(weights, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("weights must be validated before the encoder loads")

    monkeypatch.setattr(t4, "_get_semantic_model", boom)
    with pytest.raises(ValueError):
        rank_evidence(EN_QUERY, [EN_ON_TOPIC], config={"relevance_weights": weights})


def test_partial_relevance_weights_are_backfilled_not_rejected(monkeypatch):
    """A half-specified relevance_weights section is completed from the
    defaults by the deep merge, so it stays valid rather than erroring."""
    captured = {}

    def fake_semantic(query, passages, model_config=None):
        import numpy as np
        return np.zeros(len(passages))

    monkeypatch.setattr(t4, "semantic_relevance", fake_semantic)
    merged = merge_config(DEFAULT_CONFIG, {"relevance_weights": {"lexical_weight": 0.3}})
    captured["w"] = validate_relevance_weights(merged["relevance_weights"])

    assert captured["w"] == pytest.approx({"lexical_weight": 0.3, "semantic_weight": 0.7})
    ranked = rank_evidence(EN_QUERY, [EN_ON_TOPIC],
                           config={"relevance_weights": {"lexical_weight": 0.3}})
    assert ranked[0]["passage_id"] == "EN-ON"


@pytest.mark.parametrize("given,expected", [
    ({"lexical_weight": 3.0, "semantic_weight": 7.0}, (0.3, 0.7)),
    ({"lexical_weight": 30, "semantic_weight": 70}, (0.3, 0.7)),
    ({"lexical_weight": 1.0, "semantic_weight": 0.0}, (1.0, 0.0)),
    ({"lexical_weight": 0.30, "semantic_weight": 0.70}, (0.3, 0.7)),
])
def test_valid_relevance_weights_are_normalized(given, expected):
    w = validate_relevance_weights(given)
    assert (w["lexical_weight"], w["semantic_weight"]) == pytest.approx(expected)
    assert w["lexical_weight"] + w["semantic_weight"] == pytest.approx(1.0)


def test_lexical_only_config_skips_the_encoder(monkeypatch):
    """The A1 ablation (semantic_weight=0) must not require model weights."""
    def boom(*args, **kwargs):
        raise AssertionError("encoder must not be loaded when semantic_weight == 0")

    monkeypatch.setattr(t4, "_get_semantic_model", boom)
    ranked = rank_evidence(
        EN_QUERY, [EN_ON_TOPIC, EN_OFF_TOPIC],
        config={"relevance_weights": {"lexical_weight": 1.0, "semantic_weight": 0.0}},
    )
    assert ranked[0]["passage_id"] == "EN-ON"


# ---------------------------------------------------------------------
# 8. rank_evidence() API compatibility
# ---------------------------------------------------------------------

DEMO_PASSAGES = [
    {
        "passage_id": "S1-P01",
        "text": "अयोध्या में राम के जन्मस्थान से जुड़े पुरातात्विक साक्ष्य रामायण की "
                "परंपरा से मेल खाते हैं। यह शोध भारतीय पुरातत्व सर्वेक्षण (ASI) "
                "द्वारा प्रकाशित किया गया।",
        "source_metadata": {"url": "https://asi.nic.in/report", "author": "Dr. R. Sharma",
                            "organization": "Archaeological Survey of India",
                            "source_type": "academic", "publication_date": "2023-02-10",
                            "has_citations": True},
    },
    dict(EN_ON_TOPIC, source_metadata={"url": "https://cambridge.org/journal",
                                       "author": "J. Smith", "source_type": "academic",
                                       "publication_date": "2021-06-01",
                                       "has_citations": True}),
    {"passage_id": "S4-P02",
     "text": "Ayodhya is the birthplace of Rama according to the Ramayana, an ancient Indian epic.",
     "source_metadata": {"source_type": "blog"}},
    {"passage_id": "S4-P03",
     "text": "Ayodhya is the birthplace of Rama according to the Ramayana, an ancient Indian epic.",
     "source_metadata": {"source_type": "blog"}},
    dict(EN_OFF_TOPIC, source_metadata={"source_type": "blog"}),
    {"passage_id": "S7-P01",
     "text": "Ram Mandir construction in Ayodhya was completed after decades of legal proceedings."},
]

EXPECTED_KEYS = {
    "rank", "passage_id", "text", "query_relevance", "indian_context_relevance",
    "source_signal", "final_score", "feature_breakdown", "provenance",
}


@requires_model
def test_rank_evidence_output_format_unchanged():
    ranked = rank_evidence(HI_QUERY, DEMO_PASSAGES)

    assert len(ranked) == len(DEMO_PASSAGES)
    assert [r["rank"] for r in ranked] == list(range(1, len(DEMO_PASSAGES) + 1))

    for row in ranked:
        assert set(row) == EXPECTED_KEYS
        for field in ("query_relevance", "indian_context_relevance",
                      "source_signal", "final_score"):
            assert 0.0 <= row[field] <= 1.0
        assert set(row["feature_breakdown"]) == {"indian_context", "evidence"}
        assert set(row["provenance"]) == {"source_metadata", "extraction_metadata"}

    scores = [r["final_score"] for r in ranked]
    assert scores == sorted(scores, reverse=True), "results must be sorted descending"


@requires_model
def test_indian_context_and_evidence_breakdowns_preserved():
    top = rank_evidence(HI_QUERY, DEMO_PASSAGES)[0]
    assert set(top["feature_breakdown"]["indian_context"]) == {
        "script_feature", "entity_feature", "place_feature", "institution_feature",
        "primary_text_feature", "contextual_vocab_feature", "query_context_match",
    }
    assert set(top["feature_breakdown"]["evidence"]) == {
        "source_type_score", "attribution_score", "date_score", "citation_score",
        "ocr_score", "provenance_score", "duplication_score",
    }


@requires_model
def test_duplicate_detection_still_penalizes_near_duplicates():
    ranked = rank_evidence(HI_QUERY, DEMO_PASSAGES)
    dup_scores = {r["passage_id"]: r["feature_breakdown"]["evidence"]["duplication_score"]
                  for r in ranked if r["passage_id"] in ("S4-P02", "S4-P03")}
    assert len(dup_scores) == 2
    assert all(v < 1.0 for v in dup_scores.values())


@requires_model
def test_top_k_behavior():
    full = rank_evidence(HI_QUERY, DEMO_PASSAGES)
    assert len(rank_evidence(HI_QUERY, DEMO_PASSAGES, top_k=3)) == 3
    assert rank_evidence(HI_QUERY, DEMO_PASSAGES, top_k=3) == full[:3]
    assert len(rank_evidence(HI_QUERY, DEMO_PASSAGES, top_k=99)) == len(DEMO_PASSAGES)
    assert rank_evidence(HI_QUERY, DEMO_PASSAGES, top_k=None) == full


@requires_model
def test_filter_config_still_applies():
    ranked = rank_evidence(HI_QUERY, DEMO_PASSAGES,
                           config={"filter": {"enabled": True, "threshold": 0.99}})
    assert ranked == []


def test_input_validation_unchanged():
    assert rank_evidence(EN_QUERY, []) == []
    with pytest.raises(ValueError):
        rank_evidence("", [EN_ON_TOPIC])
    with pytest.raises(ValueError):
        rank_evidence("   ", [EN_ON_TOPIC])
    with pytest.raises(ValueError):
        rank_evidence(EN_QUERY, "not a list")
    with pytest.raises(ValueError):
        rank_evidence(EN_QUERY, [{"text": "no passage_id here"}])


@requires_model
def test_missing_metadata_does_not_crash():
    ranked = rank_evidence(EN_QUERY, [{"passage_id": "BARE", "text": "Ayodhya and Rama."}])
    assert ranked[0]["passage_id"] == "BARE"
    assert 0.0 <= ranked[0]["final_score"] <= 1.0
