"""
Section 13 — ablation.py

Runs the ablation variants your spec calls for (A0-A5) on the SAME
annotated dataset and config machinery as evaluate.py, so results are
directly comparable. Each variant only changes the ranking/context
weights passed into rank_evidence's `config` argument — the underlying
feature-computation code never changes, which is what makes this a fair
ablation rather than a different pipeline.

Variants
--------
A0  Full model                    default weights (query 0.50 / context 0.30 / evidence 0.20)
A1  Remove semantic similarity    relevance = 100% lexical, 0% semantic
A2  Remove Indian-context         context weight -> 0, redistributed to query+evidence
A3  Remove evidence/source        evidence weight -> 0, redistributed to query+context
A4  Context as keywords only      context score uses only entity/place/institution/
                                  primary-text/vocab hits (script + query-conditioned
                                  match zeroed out) -> tests whether the richer
                                  context model beats a plain keyword dictionary
A5  Different weight set          an alternative manually-chosen weight split, to show
                                  sensitivity to the weights (33/33/34 instead of 50/30/20)
"""
from __future__ import annotations
import copy
import sys

from task4_rank_evidence import DEFAULT_CONFIG
from evaluation.evaluate import load_dataset, evaluate_ranker

def _renormalize(weights: dict, zero_key: str) -> dict:
    """Zero out one ranking weight and redistribute it proportionally
    across the remaining weights so they still sum to 1."""
    w = dict(weights)
    removed = w[zero_key]
    w[zero_key] = 0.0
    remaining_keys = [k for k in w if k != zero_key]
    remaining_total = sum(w[k] for k in remaining_keys)
    if remaining_total > 0:
        for k in remaining_keys:
            w[k] += removed * (w[k] / remaining_total)
    return w


def build_variants() -> dict[str, dict]:
    base = copy.deepcopy(DEFAULT_CONFIG)

    variants = {}

    # A0 — full model, unmodified defaults
    variants["A0_full_model"] = copy.deepcopy(base)

    # A1 — remove semantic similarity (lexical-only relevance)
    a1 = copy.deepcopy(base)
    a1["relevance_weights"] = {"lexical_weight": 1.0, "semantic_weight": 0.0}
    variants["A1_no_semantic"] = a1

    # A2 — remove Indian-context features from the final ranking
    a2 = copy.deepcopy(base)
    a2["ranking_weights"] = _renormalize(base["ranking_weights"], "indian_context_weight")
    variants["A2_no_indian_context"] = a2

    # A3 — remove evidence/source signal from the final ranking
    a3 = copy.deepcopy(base)
    a3["ranking_weights"] = _renormalize(base["ranking_weights"], "evidence_weight")
    variants["A3_no_evidence"] = a3

    # A4 — context reduced to a plain keyword dictionary (no script cue,
    # no query-conditioned matching — just entity/place/institution/
    # primary-text/vocab keyword hits)
    a4 = copy.deepcopy(base)
    a4["context_weights"] = _renormalize(
        _renormalize(base["context_weights"], "w_script"), "w_query_ctx"
    )
    variants["A4_context_as_keywords"] = a4

    # A5 — an alternative, more evenly split weight set
    a5 = copy.deepcopy(base)
    a5["ranking_weights"] = {
        "query_relevance_weight": 0.34,
        "indian_context_weight": 0.33,
        "evidence_weight": 0.33,
    }
    variants["A5_alt_weights_33_33_34"] = a5

    return variants


def run_ablation(dataset_path: str, ks: tuple[int, ...] = (5, 10)) -> list[dict]:
    grouped = load_dataset(dataset_path)
    variants = build_variants()
    rows = []
    for name, cfg in variants.items():
        summary = evaluate_ranker(grouped, config=cfg, ks=ks)
        row = {"variant": name}
        for k in ks:
            row[f"P@{k}"] = summary[f"mean_P@{k}"]
            row[f"nDCG@{k}"] = summary[f"mean_nDCG@{k}"]
        row["context_agreement"] = summary["mean_context_agreement_spearman"]
        rows.append(row)
    return rows


def print_table(rows: list[dict]) -> None:
    cols = list(rows[0].keys())
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) + 2 for c in cols}
    print("".join(c.ljust(widths[c]) for c in cols))
    print("-" * sum(widths.values()))
    for r in rows:
        print("".join(str(r[c]).ljust(widths[c]) for c in cols))


if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "sample_dataset.jsonl"
    rows = run_ablation(dataset_path)
    print_table(rows)