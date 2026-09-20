"""
Section 12 — evaluate.py

Purpose
-------
Takes a human-annotated dataset (built by your teammate, per Section 11.2
of the spec: query_id, query, passage_id, text, query_relevance label,
indian_context_relevance label, + optional source/extraction metadata),
runs `rank_evidence()` on every query's passage group, and reports:

    - Precision@5, Precision@10
    - Recall@5, Recall@10
    - nDCG@5, nDCG@10
    - Agreement between the system's indian_context_relevance score and
      the human indian_context_relevance label (Spearman rank correlation)

Dataset format expected (JSON Lines, one passage-row per line):
{
  "query_id": "Q001",
  "query": "...",
  "query_language": "hi",
  "passage_id": "S1-P01",
  "text": "...",
  "source_metadata": {...},          # optional
  "extraction_metadata": {...},      # optional
  "query_relevance": 3,              # human label, 0-3
  "indian_context_relevance": 3      # human label, 0-3
}

All rows sharing the same query_id are grouped into one rank_evidence()
call. This file has no dependency on any specific dataset — swap in the
real annotated file once your teammate produces it; nothing else changes.
"""
from __future__ import annotations
import json
import os
import sys
from collections import defaultdict
from statistics import mean

# Make the project root importable so that `python evaluation/evaluate.py ...`
# works from the project root, not just `python -m evaluation.evaluate`.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from scipy.stats import spearmanr

from task4_rank_evidence import rank_evidence, DEFAULT_CONFIG
from evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k

def load_dataset(path: str) -> dict[str, dict]:
    """Group dataset rows by query_id.

    Returns: {query_id: {"query": str, "passages": list[dict],
                          "query_relevance_labels": {pid: int},
                          "context_labels": {pid: int}}}
    """
    grouped: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row["query_id"]
            if qid not in grouped:
                grouped[qid] = {
                    "query": row["query"],
                    "passages": [],
                    "query_relevance_labels": {},
                    "context_labels": {},
                }
            passage = {
                "passage_id": row["passage_id"],
                "text": row.get("text", ""),
                "source_metadata": row.get("source_metadata", {}),
                "extraction_metadata": row.get("extraction_metadata", {}),
            }
            grouped[qid]["passages"].append(passage)
            grouped[qid]["query_relevance_labels"][row["passage_id"]] = row.get("query_relevance", 0)
            grouped[qid]["context_labels"][row["passage_id"]] = row.get("indian_context_relevance", 0)
    return grouped


def evaluate_ranker(grouped_dataset: dict[str, dict], config: dict | None = None,
                     ks: tuple[int, ...] = (5, 10), rel_threshold: int = 2) -> dict:
    """Run rank_evidence on every query and average metrics across queries."""
    per_k_precision = {k: [] for k in ks}
    per_k_recall = {k: [] for k in ks}
    per_k_ndcg = {k: [] for k in ks}
    context_corr_scores = []
    per_query_report = []

    for qid, data in grouped_dataset.items():
        query, passages = data["query"], data["passages"]
        qrel_labels = data["query_relevance_labels"]
        ctx_labels = data["context_labels"]

        ranked = rank_evidence(query, passages, config=config)
        ranked_ids = [r["passage_id"] for r in ranked]

        row_metrics = {"query_id": qid}
        for k in ks:
            p = precision_at_k(ranked_ids, qrel_labels, k, rel_threshold)
            r = recall_at_k(ranked_ids, qrel_labels, k, rel_threshold)
            n = ndcg_at_k(ranked_ids, qrel_labels, k)
            per_k_precision[k].append(p)
            per_k_recall[k].append(r)
            per_k_ndcg[k].append(n)
            row_metrics[f"P@{k}"] = round(p, 3)
            row_metrics[f"R@{k}"] = round(r, 3)
            row_metrics[f"nDCG@{k}"] = round(n, 3)

        # human-agreement: system's indian_context_relevance vs human label
        sys_ctx = [item["indian_context_relevance"] for item in ranked]
        human_ctx = [ctx_labels.get(item["passage_id"], 0) for item in ranked]
        if len(set(human_ctx)) > 1 and len(set(sys_ctx)) > 1:
            corr, _ = spearmanr(sys_ctx, human_ctx)
            context_corr_scores.append(corr)
            row_metrics["context_agreement_spearman"] = round(float(corr), 3)
        else:
            row_metrics["context_agreement_spearman"] = None

        per_query_report.append(row_metrics)

    summary = {"per_query": per_query_report, "config_used": config or DEFAULT_CONFIG}
    for k in ks:
        summary[f"mean_P@{k}"] = round(mean(per_k_precision[k]), 3) if per_k_precision[k] else 0.0
        summary[f"mean_R@{k}"] = round(mean(per_k_recall[k]), 3) if per_k_recall[k] else 0.0
        summary[f"mean_nDCG@{k}"] = round(mean(per_k_ndcg[k]), 3) if per_k_ndcg[k] else 0.0
    summary["mean_context_agreement_spearman"] = (
        round(float(np.mean(context_corr_scores)), 3) if context_corr_scores else None
    )
    return summary


def print_report(summary: dict) -> None:
    print("Per-query metrics:")
    for row in summary["per_query"]:
        print(" ", row)
    print("\nAggregate metrics across all queries:")
    for key, val in summary.items():
        if key in ("per_query", "config_used"):
            continue
        print(f"  {key}: {val}")


if __name__ == "__main__":
    dataset_path = (sys.argv[1] if len(sys.argv) > 1
                    else os.path.join(_ROOT, "evaluation", "sample_dataset.jsonl"))
    grouped = load_dataset(dataset_path)
    result = evaluate_ranker(grouped)
    print_report(result)