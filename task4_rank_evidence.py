"""
TASK 4 - Indian-Context Relevance & Evidence Prioritization
=============================================================

Implements the callable API required by the project spec:

    rank_evidence(query, passages, top_k=None, config=None) -> list[dict]

Pipeline (per Appendix A pseudocode):
    1. validate_inputs
    2. lexical_relevance      (word TF-IDF cosine - Layer A)
    3. semantic_relevance     (BAAI/bge-m3 multilingual dense embeddings,
                               cosine similarity - Layer B)
    4. compute_indian_context (script + entity + place + institution +
                               primary-text + vocabulary + query-conditioned
                               match)
    5. compute_source_signal  (source type, attribution, date, citations,
                               OCR confidence, corroboration/duplication)
    6. normalize_features     (clip to [0,1])
    7. combine_scores         (configurable weighted sum)
    8. sort + rank + return top_k

Hybrid query relevance
----------------------
    QueryRelevance = 0.30 * LexicalScore + 0.70 * BGE_M3Score

Layer B is BAAI/bge-m3, a multilingual *dense embedding* model
(XLM-RoBERTa backbone, 1024-d) served through `sentence-transformers`.
It projects Hindi, English and code-mixed Hindi-English text into a single
shared vector space, so a Devanagari query can match an English passage on
meaning rather than on surface form. Embeddings are L2-normalized, which
makes cosine similarity a plain dot product.

Layer A (word-level TF-IDF) is deliberately kept alongside it. Dense
embeddings generalize across paraphrase and script but blur exact surface
forms; lexical overlap still pins down what must match literally - named
entities, person and place names, dates, numbers, acronyms and technical
terms (e.g. "ASI", "1998", "Chandrayaan"). The 30/70 split keeps that
literal anchor without letting it dominate the semantic signal.

Model loading
-------------
The encoder is loaded lazily - never at import time - and cached for the
lifetime of the process, so it is initialized once and reused across every
call to `rank_evidence()`. Importing this module performs no network
access and no model load.

If `sentence-transformers` or the model weights are unavailable, semantic
scoring raises `SemanticModelUnavailableError` with install/download
instructions. It does not silently fall back to a weaker similarity
measure (an earlier revision of this file used character n-gram TF-IDF as
an offline stand-in; that placeholder has been removed).
"""

from __future__ import annotations
import copy
import os
import re
import math
import unicodedata
from datetime import datetime, date
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------
# 0. Curated Indian-context resources (Section 7 — first-baseline lexicons)
#    In the real repo these live in resources/*.json; kept inline here so
#    the script is self-contained and runnable as-is.
# ----------------------------------------------------------------------

INDIAN_ENTITIES = {
    "rama", "ram", "राम", "sita", "सीता", "krishna", "कृष्ण", "buddha", "बुद्ध",
    "gandhi", "गांधी", "ashoka", "अशोक", "isro", "इसरो", "chandrayaan", "आर्यभट्ट",
    "aryabhata", "akbar", "अकबर", "tagore", "टैगोर", "nehru", "नेहरू",
}

INDIAN_PLACES = {
    "varanasi", "वाराणसी", "ayodhya", "अयोध्या", "kerala", "केरल", "delhi", "दिल्ली",
    "mumbai", "मुंबई", "kashi", "काशी", "ganga", "गंगा", "himalaya", "हिमालय",
    "kumbh mela", "कुंभ मेला", "prayagraj", "प्रयागराज", "madhya pradesh", "मध्य प्रदेश",
}

INDIAN_INSTITUTIONS = {
    "iit", "आईआईटी", "asi", "archaeological survey of india", "isro",
    "sahitya akademi", "banaras hindu university", "bhu", "sangeet natak akademi",
    "national museum", "nehru memorial",
}

PRIMARY_TEXTS = {
    "ramayana", "रामायण", "mahabharata", "महाभारत", "veda", "वेद", "vedas",
    "upanishad", "उपनिषद्", "gita", "गीता", "puranas", "पुराण",
}

CONTEXTUAL_VOCAB = {
    "dharma", "धर्म", "puja", "पूजा", "panchayat", "पंचायत", "yatra", "यात्रा",
    "mela", "मेला", "ashram", "आश्रम", "guru", "गुरु", "mandir", "मंदिर",
}

FEATURE_LEXICONS = {
    "entity_feature": INDIAN_ENTITIES,
    "place_feature": INDIAN_PLACES,
    "institution_feature": INDIAN_INSTITUTIONS,
    "primary_text_feature": PRIMARY_TEXTS,
    "contextual_vocab_feature": CONTEXTUAL_VOCAB,
}

SOURCE_TYPE_PRIOR = {
    "academic": 0.95, "institutional": 0.9, "scholarly": 0.9,
    "book": 0.85, "manuscript": 0.8, "news": 0.6, "blog": 0.35,
    "anonymous": 0.15, None: 0.4,
}

DEFAULT_CONFIG = {
    # QueryRelevance = 0.30 * lexical TF-IDF + 0.70 * BGE-M3 dense cosine
    "relevance_weights": {"lexical_weight": 0.30, "semantic_weight": 0.70},
    # Layer B encoder. device=None lets sentence-transformers choose
    # (CUDA when available, otherwise CPU).
    "semantic_model": {"name": "BAAI/bge-m3", "batch_size": 16, "device": None},
    "context_weights": {
        "w_script": 0.15, "w_entity": 0.20, "w_place": 0.15,
        "w_institution": 0.15, "w_primary": 0.15, "w_vocab": 0.10,
        "w_query_ctx": 0.10,
    },
    "ranking_weights": {
        "query_relevance_weight": 0.109,
        "indian_context_weight": 0.349,
        "evidence_weight": 0.543,
    },
    "filter": {"enabled": False, "threshold": 0.0},
    "output": {"include_feature_breakdown": True, "include_provenance": True},
}


# ----------------------------------------------------------------------
# 1. Validation
# ----------------------------------------------------------------------

def validate_inputs(query: str, passages: list[dict]) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(passages, list):
        raise ValueError("passages must be a list of dicts")
    for p in passages:
        if "passage_id" not in p:
            raise ValueError("every passage needs a passage_id")


def merge_config(base: dict, override: Optional[dict]) -> dict:
    """Recursively merge `override` onto a deep copy of `base`.

    A partial config only replaces the keys it actually names: unrelated
    DEFAULT_CONFIG sections - and unrelated keys *inside* a named section -
    survive untouched. `base` is never mutated.
    """
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _text_of(p: dict) -> str:
    return (p.get("text") or "").strip()


# ----------------------------------------------------------------------
# 2. Query-passage relevance (Section 6)
# ----------------------------------------------------------------------

def _tfidf_cosine(query: str, texts: list[str], analyzer="word", ngram_range=(1, 1)) -> np.ndarray:
    """Generic TF-IDF cosine helper. Empty texts get score 0."""
    docs = [query] + texts
    non_empty_idx = [i for i, t in enumerate(texts) if t.strip()]
    scores = np.zeros(len(texts))
    if not non_empty_idx:
        return scores
    try:
        vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=1)
        matrix = vec.fit_transform(docs)
        sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        scores = sims
    except ValueError:
        # e.g. vocabulary is empty after tokenization (very short/odd text)
        pass
    return scores


def lexical_relevance(query: str, passages: list[dict]) -> np.ndarray:
    """Layer A — TF-IDF word-overlap baseline (BM25-style, simplified)."""
    texts = [_text_of(p) for p in passages]
    return _tfidf_cosine(query, texts, analyzer="word", ngram_range=(1, 1))


# --- Layer B: BAAI/bge-m3 multilingual dense embeddings ----------------

class SemanticModelUnavailableError(RuntimeError):
    """Raised when the BGE-M3 encoder cannot be imported or loaded."""


_MODEL_HELP = """Could not load the multilingual semantic encoder {name!r}.
Reason: {reason}

How to fix:
  1. Install the runtime:
       pip install -r requirements.txt
       (or: pip install "sentence-transformers>=2.7" torch)
  2. Download the weights once (~2.2 GB, needs internet):
       python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
       (or: hf download BAAI/bge-m3)
  3. Offline / air-gapped machine: copy the HuggingFace cache directory
     (~/.cache/huggingface/hub) from a machine that already has the model,
     or point config['semantic_model']['name'] at a local directory.

Semantic relevance is a required component of this pipeline, so it will
not silently fall back to a weaker similarity measure."""

# Process-wide singleton: loaded on first use, then reused by every
# subsequent rank_evidence() call.
_SEMANTIC_MODEL = None
_SEMANTIC_MODEL_KEY = None


def _semantic_model_config(model_config: Optional[dict] = None) -> dict:
    """Overlay a partial semantic-model config on the defaults."""
    return {**DEFAULT_CONFIG["semantic_model"], **(model_config or {})}


def _get_semantic_model(model_config: Optional[dict] = None):
    """Lazily load the BGE-M3 encoder once, then reuse it.

    `sentence_transformers` is imported *inside* this function so that
    importing this module never triggers a torch import, a model load or a
    network request.
    """
    global _SEMANTIC_MODEL, _SEMANTIC_MODEL_KEY

    cfg = _semantic_model_config(model_config)
    name, device = cfg["name"], cfg.get("device")
    key = (name, device)
    if _SEMANTIC_MODEL is not None and _SEMANTIC_MODEL_KEY == key:
        return _SEMANTIC_MODEL

    # This pipeline is torch-only. Tell transformers not to probe for
    # TensorFlow / Flax backends: importing them is pure startup cost here,
    # and a stale TF build in the environment would abort the import
    # outright. setdefault, so an explicit USE_TF from the caller wins.
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # not installed, or a broken torch install
        raise SemanticModelUnavailableError(_MODEL_HELP.format(
            name=name,
            reason="sentence-transformers is not importable (%s: %s)"
                   % (type(exc).__name__, exc),
        )) from exc

    try:
        model = SentenceTransformer(name, device=device)
    except Exception as exc:  # weights missing, offline, bad name, OOM, ...
        raise SemanticModelUnavailableError(_MODEL_HELP.format(
            name=name,
            reason="loading the model failed (%s: %s)"
                   % (type(exc).__name__, exc),
        )) from exc

    _SEMANTIC_MODEL = model
    _SEMANTIC_MODEL_KEY = key
    return model


def semantic_relevance(query: str, passages: list[dict],
                       model_config: Optional[dict] = None) -> np.ndarray:
    """
    Layer B - multilingual semantic similarity via BAAI/bge-m3.

    BGE-M3 is a multilingual dense embedding model: the query and every
    passage are encoded into one shared 1024-d space, so Hindi, English and
    code-mixed Hindi-English text are directly comparable and a match no
    longer depends on shared surface strings.

    Query and passages are encoded in batches (`batch_size`), embeddings are
    L2-normalized, and each score is the cosine similarity - a dot product
    on unit vectors - between the query and that passage.

    Returns one score per passage, in the order the passages were given.
    Passages whose text is empty or whitespace-only are never sent to the
    encoder and score exactly 0.0; if every passage is empty the encoder is
    not loaded at all.
    """
    texts = [_text_of(p) for p in passages]
    scores = np.zeros(len(texts), dtype=float)

    filled = [i for i, t in enumerate(texts) if t]
    if not filled:
        return scores

    cfg = _semantic_model_config(model_config)
    model = _get_semantic_model(cfg)
    batch_size = max(1, int(cfg.get("batch_size") or 16))

    embeddings = model.encode(
        [query] + [texts[i] for i in filled],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    scores[filled] = embeddings[1:] @ embeddings[0]
    return scores


def validate_relevance_weights(weights: dict) -> dict:
    """Validate and normalize the hybrid relevance weights.

    Both weights must be present, finite and non-negative, and must sum to
    strictly more than zero. Weights that do not already sum to 1 are
    normalized internally, so e.g. {3, 7} behaves exactly like {0.3, 0.7}.
    """
    if not isinstance(weights, dict):
        raise ValueError(
            "relevance_weights must be a dict with 'lexical_weight' and "
            "'semantic_weight'")
    try:
        lex = float(weights["lexical_weight"])
        sem = float(weights["semantic_weight"])
    except KeyError as exc:
        raise ValueError("relevance_weights is missing key %s" % exc) from exc
    except (TypeError, ValueError) as exc:
        raise ValueError("relevance weights must be numbers") from exc

    if not (math.isfinite(lex) and math.isfinite(sem)):
        raise ValueError("relevance weights must be finite numbers")
    if lex < 0 or sem < 0:
        raise ValueError(
            "relevance weights must be non-negative (got lexical_weight=%r, "
            "semantic_weight=%r)" % (lex, sem))

    total = lex + sem
    if total <= 0:
        raise ValueError("relevance weights must sum to more than zero")
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        lex, sem = lex / total, sem / total
    return {"lexical_weight": lex, "semantic_weight": sem}


def score_query_relevance(query: str, passages: list[dict], weights: dict,
                          model_config: Optional[dict] = None) -> np.ndarray:
    """QueryRelevance = lexical_weight * TF-IDF + semantic_weight * BGE-M3."""
    w = validate_relevance_weights(weights)
    lex = np.clip(lexical_relevance(query, passages), 0, 1)
    if w["semantic_weight"] > 0:
        sem = np.clip(semantic_relevance(query, passages, model_config), 0, 1)
    else:
        # lexical-only configs (e.g. the A1 ablation) must not pay to load a
        # 2 GB encoder whose contribution is multiplied by zero
        sem = np.zeros_like(lex)
    combined = w["lexical_weight"] * lex + w["semantic_weight"] * sem
    return np.clip(combined, 0, 1)


# ----------------------------------------------------------------------
# 3. Indian-context relevance (Section 7)
# ----------------------------------------------------------------------

def _devanagari_ratio(text: str) -> float:
    if not text:
        return 0.0
    dev_chars = sum(1 for ch in text if "\u0900" <= ch <= "\u097F")
    letters = sum(1 for ch in text if ch.isalpha())
    if letters == 0:
        return 0.0
    return dev_chars / letters


def _keyword_hits(text: str, lexicon: set[str]) -> int:
    t = text.lower()
    return sum(1 for term in lexicon if term.lower() in t)


def compute_indian_context(query: str, passage: dict, weights: dict) -> dict:
    text = _text_of(passage)
    script_feature = _clip01(_devanagari_ratio(text) * 2)  # any real presence saturates fast

    feature_hits = {}
    for feat_name, lexicon in FEATURE_LEXICONS.items():
        hits = _keyword_hits(text, lexicon)
        feature_hits[feat_name] = _clip01(hits / 2.0)  # 2+ hits => saturated

    # query-conditioned context: does the query itself invoke an Indian
    # context term that also appears in the passage?
    query_terms = set()
    for lexicon in FEATURE_LEXICONS.values():
        query_terms |= {t for t in lexicon if t.lower() in query.lower()}
    if query_terms:
        overlap = sum(1 for t in query_terms if t.lower() in text.lower())
        query_context_match = _clip01(overlap / max(1, len(query_terms)))
    else:
        query_context_match = 0.0

    context_score = (
        weights["w_script"] * script_feature
        + weights["w_entity"] * feature_hits["entity_feature"]
        + weights["w_place"] * feature_hits["place_feature"]
        + weights["w_institution"] * feature_hits["institution_feature"]
        + weights["w_primary"] * feature_hits["primary_text_feature"]
        + weights["w_vocab"] * feature_hits["contextual_vocab_feature"]
        + weights["w_query_ctx"] * query_context_match
    )

    breakdown = {"script_feature": round(script_feature, 3), **{k: round(v, 3) for k, v in feature_hits.items()},
                 "query_context_match": round(query_context_match, 3)}
    return {"score": _clip01(context_score), "breakdown": breakdown}


# ----------------------------------------------------------------------
# 4. Source / evidence signal (Section 8)
# ----------------------------------------------------------------------

def _recency_score(pub_date: Optional[str]) -> float:
    if not pub_date:
        return 0.5  # unknown date: neutral, don't punish
    try:
        d = datetime.strptime(pub_date, "%Y-%m-%d").date()
    except ValueError:
        return 0.5
    years_old = (date.today() - d).days / 365.25
    # gentle decay; never below 0.4 — old primary/historical sources aren't punished hard
    return _clip01(1.0 - min(years_old, 15) / 30)


def _duplication_penalty(idx: int, passages: list[dict], sim_matrix: np.ndarray, threshold=0.92) -> float:
    """1.0 = fully independent, lower = near-duplicate of other passages."""
    dup_count = sum(1 for j in range(len(passages)) if j != idx and sim_matrix[idx][j] >= threshold)
    if dup_count == 0:
        return 1.0
    return _clip01(1.0 / (1 + dup_count))


def compute_source_signal(passage: dict, dup_penalty: float) -> dict:
    meta = passage.get("source_metadata", {}) or {}
    extraction = passage.get("extraction_metadata", {}) or {}

    source_type_score = SOURCE_TYPE_PRIOR.get(meta.get("source_type"), SOURCE_TYPE_PRIOR[None])
    attribution_score = 0.7 if (meta.get("author") or meta.get("organization")) else 0.3
    date_score = _recency_score(meta.get("publication_date"))
    citation_score = 0.8 if meta.get("has_citations") else 0.3
    ocr_conf = extraction.get("ocr_confidence")
    ocr_score = float(ocr_conf) if ocr_conf is not None else 1.0  # non-OCR text: full confidence
    provenance_score = 1.0 if meta.get("url") or extraction.get("page_number") else 0.4

    evidence_score = np.mean([
        source_type_score, attribution_score, date_score,
        citation_score, ocr_score, provenance_score, dup_penalty,
    ])

    breakdown = {
        "source_type_score": round(source_type_score, 3),
        "attribution_score": round(attribution_score, 3),
        "date_score": round(date_score, 3),
        "citation_score": round(citation_score, 3),
        "ocr_score": round(ocr_score, 3),
        "provenance_score": round(provenance_score, 3),
        "duplication_score": round(dup_penalty, 3),
    }
    return {"score": _clip01(evidence_score), "breakdown": breakdown}


# ----------------------------------------------------------------------
# 5. Normalization + combination (Section 9)
# ----------------------------------------------------------------------

def normalize_features(raw: dict) -> dict:
    return {k: _clip01(v) for k, v in raw.items()}


def combine_scores(features: dict, ranking_weights: dict) -> float:
    return _clip01(
        ranking_weights["query_relevance_weight"] * features["query_relevance"]
        + ranking_weights["indian_context_weight"] * features["indian_context_relevance"]
        + ranking_weights["evidence_weight"] * features["source_signal"]
    )


# ----------------------------------------------------------------------
# 6. Orchestration — the required public API
# ----------------------------------------------------------------------

def rank_evidence(query: str, passages: list[dict], top_k: Optional[int] = None,
                   config: Optional[dict] = None) -> list[dict]:
    """Return ranked evidence with interpretable feature scores."""
    cfg = merge_config(DEFAULT_CONFIG, config)
    validate_inputs(query, passages)
    # fail fast on bad weights, before any encoder is loaded
    cfg["relevance_weights"] = validate_relevance_weights(cfg["relevance_weights"])

    if not passages:
        return []

    query_rel = score_query_relevance(query, passages, cfg["relevance_weights"],
                                      cfg["semantic_model"])

    texts = [_text_of(p) for p in passages]
    # pairwise similarity for duplication/corroboration detection
    non_empty = [t if t.strip() else " " for t in texts]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    try:
        mat = vec.fit_transform(non_empty)
        sim_matrix = cosine_similarity(mat)
    except ValueError:
        sim_matrix = np.eye(len(passages))

    results = []
    for i, p in enumerate(passages):
        ctx = compute_indian_context(query, p, cfg["context_weights"])
        dup_penalty = _duplication_penalty(i, passages, sim_matrix)
        ev = compute_source_signal(p, dup_penalty)

        features = normalize_features({
            "query_relevance": query_rel[i],
            "indian_context_relevance": ctx["score"],
            "source_signal": ev["score"],
        })
        final = combine_scores(features, cfg["ranking_weights"])

        if cfg["filter"]["enabled"] and features["query_relevance"] < cfg["filter"]["threshold"]:
            continue

        item = {
            "passage_id": p["passage_id"],
            "text": _text_of(p),
            "query_relevance": round(features["query_relevance"], 3),
            "indian_context_relevance": round(features["indian_context_relevance"], 3),
            "source_signal": round(features["source_signal"], 3),
            "final_score": round(final, 3),
        }
        if cfg["output"]["include_feature_breakdown"]:
            item["feature_breakdown"] = {
                "indian_context": ctx["breakdown"],
                "evidence": ev["breakdown"],
            }
        if cfg["output"]["include_provenance"]:
            item["provenance"] = {
                "source_metadata": p.get("source_metadata", {}),
                "extraction_metadata": p.get("extraction_metadata", {}),
            }
        results.append(item)

    results.sort(key=lambda r: r["final_score"], reverse=True)
    # put 'rank' first for readability
    results = [{"rank": rank, **item} for rank, item in enumerate(results, start=1)]

    if top_k is not None:
        results = results[:top_k]
    return results


# ----------------------------------------------------------------------
# 7. Demo — mirrors the testing matrix in Section 16 of the spec
# ----------------------------------------------------------------------

if __name__ == "__main__":
    query = "राम अयोध्या से जुड़ी प्राचीन जानकारी"  # "ancient information related to Rama and Ayodhya"

    passages = [
        {  # Indian scholarly Hindi source — should rank near top
            "passage_id": "S1-P01",
            "text": "अयोध्या में राम के जन्मस्थान से जुड़े पुरातात्विक साक्ष्य रामायण की परंपरा से मेल खाते हैं। "
                    "यह शोध भारतीय पुरातत्व सर्वेक्षण (ASI) द्वारा प्रकाशित किया गया।",
            "language": "hi", "script": "Devanagari",
            "source_metadata": {"url": "https://asi.nic.in/report", "author": "Dr. R. Sharma",
                                 "organization": "Archaeological Survey of India",
                                 "source_type": "academic", "publication_date": "2023-02-10",
                                 "has_citations": True},
        },
        {  # strong external academic English source — corroborating
            "passage_id": "S2-P01",
            "text": "Archaeological excavations near Ayodhya have documented occupation layers consistent "
                    "with the historical narrative associated with Rama in the Ramayana epic.",
            "language": "en",
            "source_metadata": {"url": "https://cambridge.org/journal", "author": "J. Smith",
                                 "source_type": "academic", "publication_date": "2021-06-01",
                                 "has_citations": True},
        },
        {  # scanned Sanskrit/Hindi page via OCR — high context, medium evidence
            "passage_id": "S3-P12",
            "text": "प्राचीन ग्रन्थ के अनुसार अयोध्या नगरी राम की जन्मभूमि रही है, यह वेदों में भी वर्णित है।",
            "language": "hi", "script": "Devanagari",
            "source_metadata": {"source_type": "manuscript", "publication_date": "1998-01-01"},
            "extraction_metadata": {"method": "ocr", "ocr_confidence": 0.62, "page_number": 45},
        },
        {  # anonymous high-overlap blog — near duplicate of passage below, weak evidence
            "passage_id": "S4-P02",
            "text": "Ayodhya is the birthplace of Rama according to the Ramayana, an ancient Indian epic.",
            "language": "en",
            "source_metadata": {"source_type": "blog"},
        },
        {  # exact-copy duplicate of S4-P02 — should be down-weighted for corroboration
            "passage_id": "S4-P03",
            "text": "Ayodhya is the birthplace of Rama according to the Ramayana, an ancient Indian epic.",
            "language": "en",
            "source_metadata": {"source_type": "blog"},
        },
        {  # "India mention trap" — contains India-related word but irrelevant to query
            "passage_id": "S5-P01",
            "text": "India's IT exports grew significantly in 2023 due to demand for Python-based backend "
                    "services and cloud computing.",
            "language": "en",
            "source_metadata": {"source_type": "news", "publication_date": "2023-11-01"},
        },
        {  # completely irrelevant passage — should rank lowest
            "passage_id": "S6-P01",
            "text": "The weather today is sunny with a chance of light rain in the evening.",
            "language": "en",
            "source_metadata": {"source_type": "blog"},
        },
        {  # missing metadata entirely — must not crash, gets neutral defaults
            "passage_id": "S7-P01",
            "text": "Ram Mandir construction in Ayodhya was completed after decades of legal proceedings.",
            "language": "en",
        },
    ]

    ranked = rank_evidence(query, passages, top_k=None,
                            config={"output": {"include_feature_breakdown": True, "include_provenance": False}})

    print(f"Query: {query}\n")
    header = f"{'Rank':<5}{'Passage ID':<12}{'QueryRel':<10}{'IndiaCtx':<10}{'Evidence':<10}{'Final':<8}"
    print(header)
    print("-" * len(header))
    for r in ranked:
        print(f"{r['rank']:<5}{r['passage_id']:<12}{r['query_relevance']:<10}"
              f"{r['indian_context_relevance']:<10}{r['source_signal']:<10}{r['final_score']:<8}")

    print("\nDetailed breakdown for the top-ranked passage:")
    top = ranked[0]
    print(f"  passage_id: {top['passage_id']}")
    print(f"  text: {top['text'][:80]}...")
    print(f"  final_score: {top['final_score']}")
    print(f"  feature_breakdown: {top['feature_breakdown']}")

    print("\nNote on duplicates (S4-P02 vs S4-P03) — duplication_score should be < 1.0 for both:")
    for r in ranked:
        if r["passage_id"] in ("S4-P02", "S4-P03"):
            print(f"  {r['passage_id']}: duplication_score = "
                  f"{r['feature_breakdown']['evidence']['duplication_score']}")