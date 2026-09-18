"""
Section 12 — Evaluation metrics: Precision@K, Recall@K, nDCG@K.

All functions take:
    ranked_ids : list[str]   passage_ids in the order the system ranked them
    labels     : dict[str,int]   passage_id -> human relevance label (0..3)
    k          : int
    rel_threshold : label >= this value counts as "relevant" for
                    precision/recall (nDCG uses the raw graded label).
"""
from __future__ import annotations
import math


def precision_at_k(ranked_ids: list[str], labels: dict[str, int], k: int, rel_threshold: int = 2) -> float:
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for pid in top_k if labels.get(pid, 0) >= rel_threshold)
    return relevant / len(top_k)


def recall_at_k(ranked_ids: list[str], labels: dict[str, int], k: int, rel_threshold: int = 2) -> float:
    total_relevant = sum(1 for lbl in labels.values() if lbl >= rel_threshold)
    if total_relevant == 0:
        return 0.0
    top_k = ranked_ids[:k]
    hit = sum(1 for pid in top_k if labels.get(pid, 0) >= rel_threshold)
    return hit / total_relevant


def dcg_at_k(ranked_ids: list[str], labels: dict[str, int], k: int) -> float:
    dcg = 0.0
    for i, pid in enumerate(ranked_ids[:k], start=1):
        rel = labels.get(pid, 0)
        dcg += (2 ** rel - 1) / math.log2(i + 1)
    return dcg


def ndcg_at_k(ranked_ids: list[str], labels: dict[str, int], k: int) -> float:
    dcg = dcg_at_k(ranked_ids, labels, k)
    ideal_order = sorted(labels.values(), reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / math.log2(i + 1) for i, rel in enumerate(ideal_order, start=1))
    if idcg == 0:
        return 0.0
    return dcg / idcg